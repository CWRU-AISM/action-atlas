#!/usr/bin/env python3
"""
Compute experiment statistics from experiment_results JSONs AND on-disk data.

Scans both:
1. experiment_results_*.json files for structured episode counts and success rates
2. Actual data directories on disk for file-based episode counts (mp4, json, npz, scene.json)

Outputs: action_atlas/data/experiment_stats.json

Usage:
    python scripts/compute_experiment_stats.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "action_atlas" / "data"
OUTPUT_PATH = DATA_DIR / "experiment_stats.json"

# Data directories per model
DISK_PATHS = {
    "pi05": {
        "rollouts": "/data/robotsteering/pi05_rollouts",
        "concept_ablation_jsons": str(BASE_DIR / "results" / "experiment_results" / "pi05_concept_ablation"),
    },
    "oft": {
        "rollouts": "/data/openvla_rollouts/openvla_oft",
        "concept_ablation_jsons": str(BASE_DIR / "results" / "experiment_results" / "oft_concept_ablation"),
    },
    "xvla": {
        "libero": "/data/batch_1/xvla_libero",
        "simplerenv": "/data/batch_1/xvla_SIMPLERENV",
        "concept_ablation": "/data/batch_1/xvla_concept_ablation",
        "concept_steering": "/data/batch_1/xvla_concept_steering",
        "reconstruction": "/data/batch_1/xvla_reconstruction",
    },
    "smolvla": {
        "batch1": "/data/smolvla_rollouts/smolvla",
        "batch2": "/data/smolvla_rollouts",
    },
    "groot": {
        "batch1": "/data/groot_rollouts",
        "batch2": "/data/groot_rollouts_batch2",
    },
    "act": {
        "rollouts": "/data/robotsteering/aloha_rollouts",
    },
}

MODELS = ["pi05", "oft", "xvla", "smolvla", "groot", "act"]


# ── Utility functions ─────────────────────────────────────────────────────────

def count_files(directory, extension, recursive=True):
    """Count files with given extension. Fast os.walk-based approach."""
    if not os.path.isdir(directory):
        return 0
    count = 0
    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith(extension):
                    count += 1
    else:
        for f in os.listdir(directory):
            if f.endswith(extension):
                count += 1
    return count


def count_files_in_subdirs(directory, extension, subdirs):
    total = 0
    for sd in subdirs:
        path = os.path.join(directory, sd)
        if os.path.isdir(path):
            total += count_files(path, extension)
    return total


def count_scene_json(directory):
    """Count *_scene.json or scene.json files (GR00T episode proxy)."""
    if not os.path.isdir(directory):
        return 0
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f == "scene.json" or f.endswith("_scene.json"):
                count += 1
    return count


def parse_concept_ablation_json(filepath):
    """
    Parse a concept ablation/steering JSON to count episodes.

    Handles the standard structure:
      {n_episodes: N, tasks: {concept: {tasks: {task_id: {results}}}}}
    For steering files with strengths:
      {n_episodes: N, strengths: [...], tasks: {concept: {tasks: ...}}}
    Episodes = n_concepts * n_tasks * n_episodes [* n_strengths]
    """
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0, None

    if not isinstance(data, dict):
        return 0, None

    n_eps = data.get("n_episodes", 0)
    strengths = data.get("strengths", [])
    n_strengths = len(strengths) if strengths else 1

    concepts = data.get("tasks", {})
    if not isinstance(concepts, dict):
        return 0, None

    total_task_slots = 0
    success_rates = []

    for ck, cv in concepts.items():
        if not isinstance(cv, dict):
            continue
        inner_tasks = cv.get("tasks", {})
        if isinstance(inner_tasks, dict):
            total_task_slots += len(inner_tasks)
            for tk, tv in inner_tasks.items():
                if isinstance(tv, dict):
                    sr = tv.get("success_rate")
                    if sr is not None:
                        success_rates.append(sr)

    episodes = total_task_slots * n_eps * n_strengths
    avg_sr = (sum(success_rates) / len(success_rates)
              if success_rates else None)
    return episodes, avg_sr


def recursive_episode_count(obj):
    """Recursively find and sum episode counts from nested dicts/lists."""
    total = 0
    sr_list = []

    if isinstance(obj, dict):
        # Direct episode count keys
        for ep_key in ("n_episodes", "episodes", "num_episodes",
                       "n_total_episodes"):
            if ep_key in obj and isinstance(obj[ep_key], (int, float)):
                total += int(obj[ep_key])
                sr = obj.get("success_rate",
                             obj.get("overall_success_rate"))
                if sr is not None and isinstance(sr, (int, float)):
                    sr_list.append((int(obj[ep_key]), sr))
                return total, sr_list

        # Recurse into sub-dicts
        for key, val in obj.items():
            if key in ("summary", "metadata", "config", "description",
                       "timestamp", "sae_models", "sae_validation",
                       "activations"):
                continue
            if key == "runs" and isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        n = item.get("n_episodes", 0)
                        total += n
                        sr = item.get("success_rate")
                        if sr is not None and isinstance(sr, (int, float)):
                            sr_list.append((n, sr))
            elif key == "tasks" and isinstance(val, dict):
                for tk, tv in val.items():
                    if isinstance(tv, dict):
                        n = tv.get("n_episodes", 0)
                        total += n
                        sr = tv.get("success_rate")
                        if sr is not None and isinstance(sr, (int, float)):
                            sr_list.append((n, sr))
            elif isinstance(val, dict):
                sub_total, sub_sr = recursive_episode_count(val)
                total += sub_total
                sr_list.extend(sub_sr)
    elif isinstance(obj, list):
        for item in obj:
            sub_total, sub_sr = recursive_episode_count(item)
            total += sub_total
            sr_list.extend(sub_sr)

    return total, sr_list


def weighted_avg_sr(sr_list):
    """Compute weighted average success rate from [(n_episodes, sr), ...]."""
    if not sr_list:
        return None
    total_eps = sum(n for n, _ in sr_list)
    if total_eps == 0:
        return None
    return sum(n * sr for n, sr in sr_list) / total_eps


# ── JSON-based counting ───────────────────────────────────────────────────────

def count_from_json(model):
    """Extract episode counts and success rates from experiment_results JSON."""
    json_path = DATA_DIR / f"experiment_results_{model}.json"
    if not json_path.exists():
        return {}

    with open(json_path) as f:
        data = json.load(f)

    skip_keys = {"model", "model_name", "description", "architecture",
                 "params", "timestamp", "environments",
                 # SmolVLA aggregate sections that overlap with per-section
                 "libero_experiments", "metaworld_experiments", "summary",
                 # Metadata sections (not episode-producing experiments)
                 "scene_state", "concept_id"}

    sections = {}
    for key, val in data.items():
        if key in skip_keys or not isinstance(val, dict):
            continue

        episodes, sr_list = recursive_episode_count(val)
        avg_sr = weighted_avg_sr(sr_list)

        sections[key] = {
            "episodes_json": episodes,
            "success_rate_json": (round(avg_sr, 4)
                                  if avg_sr is not None else None),
            "n_entries": len(val),
        }

    return sections


# ── Disk-based counting ───────────────────────────────────────────────────────

def count_pi05_disk():
    rollouts = DISK_PATHS["pi05"]["rollouts"]
    ca_dir = DISK_PATHS["pi05"]["concept_ablation_jsons"]
    sections = {}

    # Baselines
    baseline_dir = os.path.join(rollouts, "baseline")
    sections["baselines"] = {
        "episodes_disk": count_files(baseline_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Cross-task experiments
    ct_dirs = ["cross_task_10", "cross_task_goal", "cross_task_spatial"]
    sections["cross_task"] = {
        "episodes_disk": count_files_in_subdirs(rollouts, ".mp4", ct_dirs),
        "source": "disk_mp4",
    }

    # Counterfactual (mp4 = reliable proxy, not npz which are inflated)
    cf_dir = os.path.join(rollouts, "counterfactual")
    cf_mp4 = 0
    if os.path.isdir(cf_dir):
        for suite in ["10", "goal", "object", "spatial"]:
            suite_dir = os.path.join(cf_dir, suite)
            if os.path.isdir(suite_dir):
                cf_mp4 += count_files(suite_dir, ".mp4")
    sections["counterfactual"] = {
        "episodes_disk": cf_mp4,
        "source": "disk_mp4",
    }

    # Vision perturbation
    vp_dirs = ["vision_perturbation", "vision_perturbation_batch2"]
    sections["vision_perturbation"] = {
        "episodes_disk": count_files_in_subdirs(rollouts, ".mp4", vp_dirs),
        "source": "disk_mp4",
    }

    # Temporal
    temp_dir = os.path.join(rollouts, "cheng_libero10_5-9_temporal")
    sections["temporal_injection"] = {
        "episodes_disk": count_files(temp_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Transfer
    transfer_dir = os.path.join(rollouts, "transfer_20260130")
    sections["transfer"] = {
        "episodes_disk": count_files(transfer_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Displacement
    disp_dir = os.path.join(rollouts, "groot_feb13_pi05")
    sections["displacement"] = {
        "episodes_disk": count_files(disp_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Archive
    archive_dir = os.path.join(rollouts, "archive")
    sections["archive"] = {
        "episodes_disk": count_files(archive_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Concept ablation: parse JSONs for episode counts
    ca_episodes_from_json = 0
    n_ca_json = 0
    if os.path.isdir(ca_dir):
        for f in os.listdir(ca_dir):
            if f.endswith(".json"):
                n_ca_json += 1
                eps, _ = parse_concept_ablation_json(
                    os.path.join(ca_dir, f))
                ca_episodes_from_json += eps

    sections["concept_ablation"] = {
        "episodes_disk": ca_episodes_from_json,
        "source": "concept_ablation_json_parse",
        "json_files": n_ca_json,
    }

    return sections


def count_oft_disk():
    rollouts = DISK_PATHS["oft"]["rollouts"]
    ca_dir = DISK_PATHS["oft"]["concept_ablation_jsons"]
    sections = {}

    # Main rollouts: count mp4s per suite
    suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    total_mp4 = 0
    for suite in suites:
        suite_dir = os.path.join(rollouts, suite)
        total_mp4 += count_files(suite_dir, ".mp4")
    sections["main_rollouts"] = {
        "episodes_disk": total_mp4,
        "source": "disk_mp4",
    }

    # Trajectories
    traj_dir = os.path.join(rollouts, "trajectories")
    sections["trajectories"] = {
        "episodes_disk": count_files(traj_dir, ".mp4"),
        "source": "disk_mp4",
    }

    # Concept ablation: parse JSONs + count mp4s in sibling dirs
    ca_episodes_from_json = 0
    n_json = 0
    if os.path.isdir(ca_dir):
        for f in os.listdir(ca_dir):
            if f.endswith(".json"):
                n_json += 1
                eps, _ = parse_concept_ablation_json(
                    os.path.join(ca_dir, f))
                ca_episodes_from_json += eps

    sections["concept_ablation"] = {
        "episodes_disk": ca_episodes_from_json,
        "source": "concept_ablation_json_parse",
        "json_files": n_json,
    }

    return sections


def count_xvla_disk():
    libero_dir = DISK_PATHS["xvla"]["libero"]
    simplerenv_dir = DISK_PATHS["xvla"]["simplerenv"]
    ca_dir = DISK_PATHS["xvla"]["concept_ablation"]
    cs_dir = DISK_PATHS["xvla"]["concept_steering"]
    recon_dir = DISK_PATHS["xvla"]["reconstruction"]
    sections = {}

    # LIBERO experiments (json = 1 per episode is reliable proxy)
    exp_dir = os.path.join(libero_dir, "experiments")
    if os.path.isdir(exp_dir):
        exp_types = defaultdict(int)
        for subdir in os.listdir(exp_dir):
            full = os.path.join(exp_dir, subdir)
            if not os.path.isdir(full):
                continue
            # Extract experiment type from dir name
            # e.g. counterfactual_libero_goal_v2 -> counterfactual
            parts = subdir.split("_libero_")
            exp_type = parts[0] if parts else subdir
            json_count = count_files(full, ".json")
            exp_types[exp_type] += json_count

        for exp_type, count in exp_types.items():
            sections[f"libero_{exp_type}"] = {
                "episodes_disk": count,
                "source": "disk_json",
            }

    # LIBERO baselines
    lb_dir = os.path.join(libero_dir, "baselines")
    sections["libero_baselines"] = {
        "episodes_disk": count_files(lb_dir, ".json"),
        "source": "disk_json",
    }

    # SimplerEnv experiments
    se_exp_dir = os.path.join(simplerenv_dir, "experiments")
    if os.path.isdir(se_exp_dir):
        sections["simplerenv_experiments"] = {
            "episodes_disk": count_files(se_exp_dir, ".json"),
            "source": "disk_json",
        }

    # SimplerEnv baselines
    se_base_dir = os.path.join(simplerenv_dir, "baselines")
    sections["simplerenv_baselines"] = {
        "episodes_disk": count_files(se_base_dir, ".json"),
        "source": "disk_json",
    }

    # Concept ablation (parse JSONs)
    ca_episodes = 0
    n_ca_json = 0
    if os.path.isdir(ca_dir):
        for f in os.listdir(ca_dir):
            if f.endswith(".json"):
                n_ca_json += 1
                eps, _ = parse_concept_ablation_json(
                    os.path.join(ca_dir, f))
                ca_episodes += eps
    sections["concept_ablation"] = {
        "episodes_disk": ca_episodes,
        "source": "concept_ablation_json_parse",
        "json_files": n_ca_json,
    }

    # Concept steering (parse JSONs)
    cs_episodes = 0
    cs_n_json = 0
    if os.path.isdir(cs_dir):
        for f in os.listdir(cs_dir):
            if f.endswith(".json"):
                cs_n_json += 1
                eps, _ = parse_concept_ablation_json(
                    os.path.join(cs_dir, f))
                cs_episodes += eps
    sections["concept_steering"] = {
        "episodes_disk": cs_episodes,
        "source": "concept_steering_json_parse",
        "json_files": cs_n_json,
    }

    # Reconstruction (mp4s)
    if os.path.isdir(recon_dir):
        sections["reconstruction"] = {
            "episodes_disk": count_files(recon_dir, ".mp4"),
            "source": "disk_mp4",
        }

    return sections


def count_smolvla_disk():
    """Count SmolVLA episodes from disk, with batch deduplication."""
    b1 = DISK_PATHS["smolvla"]["batch1"]
    b2 = DISK_PATHS["smolvla"]["batch2"]
    sections = {}

    # ── LIBERO (batch1 primary, batch2 baselines are duplicates) ──

    # Baselines: batch1 only (batch2 baselines are duplicates per
    # DATA_VERIFICATION.md)
    b1_baselines = os.path.join(b1, "baselines")
    sections["libero_baselines"] = {
        "episodes_disk": count_files(b1_baselines, ".mp4"),
        "source": "disk_mp4",
        "note": "batch2 baselines are duplicates, counted once",
    }

    # Counterfactual (batch1)
    b1_cf = os.path.join(b1, "counterfactual")
    sections["libero_counterfactual"] = {
        "episodes_disk": count_files(b1_cf, ".npz"),
        "source": "disk_npz",
    }

    # Concept ablation (batch2 LIBERO, npz = 1 per episode, skip .pt)
    b2_ca = os.path.join(b2, "concept_ablation")
    sections["libero_concept_ablation"] = {
        "episodes_disk": count_files(b2_ca, ".npz"),
        "source": "disk_npz",
    }

    # Grid ablation (batch1 LIBERO)
    b1_ga = os.path.join(b1, "grid_ablation")
    ga_mp4 = count_files(b1_ga, ".mp4")
    ga_npz = count_files(b1_ga, ".npz")
    sections["libero_grid_ablation"] = {
        "episodes_disk": max(ga_mp4, ga_npz),
        "source": "disk_mp4" if ga_mp4 >= ga_npz else "disk_npz",
    }

    # ── MetaWorld (batch2) ──

    # MetaWorld baselines
    mw_base = os.path.join(b2, "metaworld_baseline")
    sections["metaworld_baselines"] = {
        "episodes_disk": count_files(mw_base, ".npz"),
        "source": "disk_npz",
    }

    # MetaWorld grid ablation (json = 1 per episode)
    mw_ga = os.path.join(b2, "metaworld_grid_ablation")
    sections["metaworld_grid_ablation"] = {
        "episodes_disk": count_files(mw_ga, ".json"),
        "source": "disk_json",
    }

    # MetaWorld cross-task
    mw_ct = os.path.join(b2, "metaworld_cross_task")
    sections["metaworld_cross_task"] = {
        "episodes_disk": count_files(mw_ct, ".json"),
        "source": "disk_json",
    }

    # MetaWorld vision perturbation
    mw_vp = os.path.join(b2, "metaworld_vision_perturbation")
    sections["metaworld_vision_perturbation"] = {
        "episodes_disk": count_files(mw_vp, ".json"),
        "source": "disk_json",
    }

    # MetaWorld counterfactual (all v2 variants, npz = episodes)
    mw_cf_dirs = [
        "metaworld_counterfactual",
        "metaworld_counterfactual_v2",
        "metaworld_counterfactual_v2_hard",
        "metaworld_counterfactual_v2_medium",
        "metaworld_counterfactual_v2_very_hard",
    ]
    mw_cf_total = 0
    for d in mw_cf_dirs:
        mw_cf_total += count_files(os.path.join(b2, d), ".npz")
    sections["metaworld_counterfactual"] = {
        "episodes_disk": mw_cf_total,
        "source": "disk_npz",
    }

    # Concept ablation results from batch1 - track count but as metadata
    # (not episodes). The actual episode counts come from the
    # experiment_results JSON sections for steering/temporal/FTF.
    ca_results = os.path.join(b1, "concept_ablation", "results")
    if os.path.isdir(ca_results):
        n_result_files = len([f for f in os.listdir(ca_results)
                              if f.endswith(".json")])
        sections["libero_sae_result_files"] = {
            "episodes_disk": 0,  # Not episode counts
            "source": "metadata",
            "n_result_files": n_result_files,
            "note": "SAE result JSONs (ablation/steering/temporal/FTF)",
        }

    return sections


def count_groot_disk():
    b1 = DISK_PATHS["groot"]["batch1"]
    b2 = DISK_PATHS["groot"]["batch2"]
    sections = {}
    suites = ["libero_goal", "libero_long", "libero_object"]

    # ── Batch 1: Suite experiments ──
    # Use mp4 as primary proxy; npz for baselines (which have no mp4)
    suite_exp_types = [
        "baseline", "counterfactual", "cross_task",
        "grid_ablation", "null_injection", "visual_perturbation",
    ]
    for exp_type in suite_exp_types:
        total = 0
        for suite in suites:
            exp_dir = os.path.join(b1, suite, exp_type)
            if os.path.isdir(exp_dir):
                mp4 = count_files(exp_dir, ".mp4")
                npz = count_files(exp_dir, ".npz")
                # Use mp4 if available, else npz
                total += mp4 if mp4 > 0 else npz
        source = "disk_mp4" if exp_type != "baseline" else "disk_npz"
        sections[f"suite_{exp_type}"] = {
            "episodes_disk": total,
            "source": source,
        }

    # ── Batch 1: SAE experiments (npz = 1 per episode, paired with
    #    *_scene.json) ──
    sae_b1_types = [
        "sae_feature_ablation",
        "sae_fraction_to_failure",
        "sae_temporal_ablation",
    ]
    for sae_type in sae_b1_types:
        sae_dir = os.path.join(b1, sae_type)
        name = sae_type.replace("sae_", "")
        sections[f"b1_{name}"] = {
            "episodes_disk": count_files(sae_dir, ".npz"),
            "source": "disk_npz",
        }

    # ── Batch 2: SAE experiments ──
    sae_b2_types = [
        "sae_cross_suite_ablation",
        "sae_fraction_to_failure",
        "sae_steering",
        "sae_temporal_ablation",
    ]
    for sae_type in sae_b2_types:
        sae_dir = os.path.join(b2, sae_type)
        name = sae_type.replace("sae_", "")
        sections[f"b2_{name}"] = {
            "episodes_disk": count_files(sae_dir, ".npz"),
            "source": "disk_npz",
        }

    return sections


def count_act_disk():
    rollouts = DISK_PATHS["act"]["rollouts"]
    interp_dir = os.path.join(rollouts, "act_aloha_interp")
    sections = {}

    if not os.path.isdir(interp_dir):
        return sections

    # Baselines
    for env in ["AlohaInsertion-v0", "AlohaTransferCube-v0"]:
        base_dir = os.path.join(interp_dir, f"baseline_{env}", "videos")
        mp4 = count_files(base_dir, ".mp4")
        sections[f"baseline_{env}"] = {
            "episodes_disk": mp4,
            "source": "disk_mp4",
        }

    # Grid ablation
    for env in ["AlohaInsertion-v0", "AlohaTransferCube-v0"]:
        ga_dir = os.path.join(interp_dir, f"grid_ablation_{env}")
        mp4 = count_files(ga_dir, ".mp4")
        sections[f"grid_ablation_{env}"] = {
            "episodes_disk": mp4,
            "source": "disk_mp4",
        }

    # Injection
    for inj in ["injection_cross_task", "injection_same_task_ins",
                 "injection_same_task_tc"]:
        inj_dir = os.path.join(interp_dir, inj)
        mp4 = count_files(inj_dir, ".mp4")
        sections[inj] = {
            "episodes_disk": mp4,
            "source": "disk_mp4",
        }

    return sections


# ── Merge and output ──────────────────────────────────────────────────────────

def merge_sections(json_sections, disk_sections, dedup_map=None):
    """
    Merge JSON-based and disk-based sections into unified section list.

    Uses max(json, disk) for the canonical episode count per section.
    dedup_map: dict mapping disk_key -> json_key for overlapping sections.
    When a disk section duplicates a JSON section, only the one with the
    higher count is kept (under the json_key name).
    """
    # Handle deduplication: merge disk into matching JSON key
    if dedup_map:
        for disk_key, json_key in dedup_map.items():
            if disk_key in disk_sections and json_key in json_sections:
                # Disk section overlaps with JSON section - pick the max
                disk_eps = disk_sections[disk_key].get("episodes_disk", 0)
                json_eps = json_sections[json_key].get("episodes_json", 0)
                if disk_eps > json_eps:
                    # Merge disk info into json section
                    json_sections[json_key]["episodes_json"] = disk_eps
                    json_sections[json_key]["_source_override"] = (
                        disk_sections[disk_key].get("source", "disk"))
                del disk_sections[disk_key]
            elif disk_key in disk_sections and json_key not in json_sections:
                # Rename disk key to json key
                disk_sections[json_key] = disk_sections.pop(disk_key)

    all_keys = sorted(set(list(json_sections.keys()) +
                          list(disk_sections.keys())))
    merged = {}

    for key in all_keys:
        entry = {}
        if key in json_sections:
            js = json_sections[key]
            entry["episodes_json"] = js.get("episodes_json", 0)
            if js.get("success_rate_json") is not None:
                entry["success_rate"] = js["success_rate_json"]
            entry["n_entries_json"] = js.get("n_entries", 0)
            if "_source_override" in js:
                entry["source"] = js["_source_override"]

        if key in disk_sections:
            ds = disk_sections[key]
            entry["episodes_disk"] = ds.get("episodes_disk", 0)
            entry["source"] = ds.get("source", "unknown")
            if "json_files" in ds:
                entry["json_files"] = ds["json_files"]
            if "note" in ds:
                entry["note"] = ds["note"]

        # Best available episode count
        eps_json = entry.get("episodes_json", 0)
        eps_disk = entry.get("episodes_disk", 0)
        entry["episodes"] = max(eps_json, eps_disk)

        merged[key] = entry

    return merged


def compute_model_stats(model):
    """Compute full stats for a model combining JSON and disk sources."""
    print(f"  Processing {model}...")

    # 1. Count from experiment_results JSON
    json_sections = count_from_json(model)

    # 2. Count from disk
    disk_counter = {
        "pi05": count_pi05_disk,
        "oft": count_oft_disk,
        "xvla": count_xvla_disk,
        "smolvla": count_smolvla_disk,
        "groot": count_groot_disk,
        "act": count_act_disk,
    }
    disk_sections = disk_counter[model]()

    # 3. Merge with deduplication maps for overlapping sections
    # disk_key -> json_key: sections that represent the same data
    DEDUP_MAPS = {
        "pi05": {},
        "oft": {},
        "xvla": {
            # Disk counts per experiment type overlap with JSON sections
            "libero_baselines": "baselines",
            "libero_counterfactual": "counterfactual",
            "libero_cross_task": "cross_task",
            "libero_grid_ablation": "grid_ablation",
            "libero_vision": "vision_perturbation",
            "libero_temporal": "temporal",
        },
        "smolvla": {
            # JSON libero_experiments/metaworld_experiments are aggregates
            # that overlap with disk counts and other JSON sections
            "libero_baselines": "baselines",
            "libero_counterfactual": "counterfactual",
        },
        "groot": {
            # Disk suite_ counts overlap with JSON sections
            "suite_baseline": "baselines",
            "suite_counterfactual": "counterfactual",
            "suite_cross_task": "cross_task",
            "suite_grid_ablation": "grid_ablation",
            "suite_null_injection": "null_injection",
            "suite_visual_perturbation": "vision_perturbation",
        },
        "act": {
            # Disk per-env baselines/grid_ablation overlap with JSON totals
            "baseline_AlohaInsertion-v0": "baselines",
            "baseline_AlohaTransferCube-v0": "baselines",
            "grid_ablation_AlohaInsertion-v0": "grid_ablation",
            "grid_ablation_AlohaTransferCube-v0": "grid_ablation",
        },
    }
    dedup = DEDUP_MAPS.get(model, {})
    sections = merge_sections(json_sections, disk_sections, dedup)

    # 4. Compute totals
    total = sum(s.get("episodes", 0) for s in sections.values())

    # 5. Get baseline success rate
    baseline_sr = None
    for key in ("baselines", "libero_baselines",
                "baseline_AlohaInsertion-v0"):
        if key in sections and "success_rate" in sections[key]:
            baseline_sr = sections[key]["success_rate"]
            break

    return {
        "total": total,
        "label": f"{total:,}",
        "baseline_success_rate": baseline_sr,
        "sections": sections,
    }


def main():
    start = time.time()
    print("Computing experiment statistics from JSONs + disk...")

    results = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "description": ("Dynamically computed episode counts from "
                         "experiment_results JSONs and on-disk data"),
    }

    grand_total = 0
    for model in MODELS:
        stats = compute_model_stats(model)
        results[model] = stats
        grand_total += stats["total"]
        n_secs = len(stats["sections"])
        print(f"    {model}: {stats['label']} episodes "
              f"({n_secs} sections)")

    # Also create openvla alias for backend compatibility
    if "oft" in results:
        results["openvla"] = results["oft"]

    results["grand_total"] = grand_total
    results["grand_total_label"] = f"{grand_total:,}"

    # Verified totals from DATA_VERIFICATION.md for comparison
    results["verified_reference"] = {
        "pi05_conservative": 30119,
        "pi05_raw_mp4": 63024,
        "oft": 32474,
        "xvla": 49561,
        "smolvla": 38038,
        "groot": 164191,
        "act": 990,
    }

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Grand total: {results['grand_total_label']} episodes")

    return results


if __name__ == "__main__":
    main()
