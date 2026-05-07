# Aggregate SmolVLA experiment results
from pathlib import Path
import json

def aggregate_smolvla() -> dict:
    result = {
        "model": "smolvla",
        "model_name": "SmolVLA",
        "description": "SmolVLA 450M with interleaved VLM + Expert pathways",
        "architecture": "dual_pathway_interleaved",
        "params": "450M",
        "timestamp": datetime.now().isoformat(),
        "environments": ["libero", "metaworld"],
        "baselines": {},
        "grid_ablation": {},
        "counterfactual": {},
        "cross_task": {},
        "vision_perturbation": {},
        "displacement": {},
    }

    # --- LIBERO Baselines ---
    for suite in SMOLVLA_LIBERO_SUITES:
        baseline_path = SMOLVLA_LIBERO_DIR / "baselines" / suite / "results.json"
        data = safe_load_json(baseline_path)
        if data:
            tasks_summary = {}
            for task_key, task_data in data.get("tasks", {}).items():
                episodes = task_data.get("episodes", [])
                tasks_summary[task_key] = {
                    "task_description": task_data.get("task_description", ""),
                    "success_rate": round_val(compute_success_rate(episodes)),
                    "n_episodes": len(episodes),
                    "avg_steps": round_val(
                        sum(ep.get("steps", 0) for ep in episodes) / max(len(episodes), 1)
                    ),
                }
            result["baselines"][suite] = {
                "environment": "libero",
                "suite": suite,
                "n_episodes": data.get("n_episodes", 0),
                "tasks": tasks_summary,
                "overall_success_rate": round_val(
                    sum(t["success_rate"] for t in tasks_summary.values()) / max(len(tasks_summary), 1)
                ),
            }

    # --- MetaWorld Baselines ---
    mw_baseline_path = SMOLVLA_METAWORLD_DIR / "metaworld_baseline" / "results.json"
    mw_data = safe_load_json(mw_baseline_path)
    if mw_data:
        tasks_summary = {}
        for task_name, task_info in mw_data.get("tasks", {}).items():
            tasks_summary[task_name] = {
                "task_description": task_info.get("task_description", ""),
                "success_rate": round_val(task_info.get("success_rate", 0)),
                "n_episodes": task_info.get("n_episodes", 0),
            }
        result["baselines"]["metaworld"] = {
            "environment": "metaworld",
            "n_episodes": mw_data.get("n_episodes", 0),
            "tasks": tasks_summary,
            "overall_success_rate": round_val(
                sum(t["success_rate"] for t in tasks_summary.values()) / max(len(tasks_summary), 1)
            ),
        }

    # --- LIBERO Grid Ablation ---
    for suite in SMOLVLA_LIBERO_SUITES:
        grid_path = SMOLVLA_LIBERO_DIR / "grid_ablation" / suite / "results.json"
        data = safe_load_json(grid_path)
        if data:
            grid_summary = _extract_smolvla_grid(data)
            result["grid_ablation"][suite] = grid_summary

    # --- MetaWorld Grid Ablation ---
    for diff in SMOLVLA_MW_DIFFICULTIES:
        grid_path = SMOLVLA_METAWORLD_DIR / "metaworld_grid_ablation" / diff / "results.json"
        data = safe_load_json(grid_path)
        if data:
            grid_summary = _extract_smolvla_grid(data)
            grid_summary["difficulty"] = diff
            result["grid_ablation"][f"metaworld_{diff}"] = grid_summary
    # Also check hard_v2, very_hard_v2
    for variant in ["hard_v2", "very_hard_v2"]:
        grid_path = SMOLVLA_METAWORLD_DIR / "metaworld_grid_ablation" / variant / "results.json"
        data = safe_load_json(grid_path)
        if data:
            grid_summary = _extract_smolvla_grid(data)
            grid_summary["difficulty"] = variant
            result["grid_ablation"][f"metaworld_{variant}"] = grid_summary

    # --- LIBERO Counterfactual ---
    for suite in SMOLVLA_LIBERO_SUITES:
        cf_dir = SMOLVLA_LIBERO_DIR / "counterfactual" / suite
        metadata_path = cf_dir / "metadata.jsonl"
        entries = load_jsonl(metadata_path)
        if entries:
            result["counterfactual"][suite] = _aggregate_counterfactual_entries(
                entries, environment="libero", suite=suite
            )

    # --- MetaWorld Counterfactual ---
    for suffix in ["", "_medium", "_hard", "_very_hard"]:
        cf_dir = SMOLVLA_METAWORLD_DIR / f"metaworld_counterfactual_v2{suffix}"
        metadata_path = cf_dir / "metadata.jsonl"
        entries = load_jsonl(metadata_path)
        if entries:
            difficulty = suffix.lstrip("_") if suffix else "easy"
            key = f"metaworld_{difficulty}"
            result["counterfactual"][key] = _aggregate_counterfactual_entries(
                entries, environment="metaworld", difficulty=difficulty
            )

    # --- MetaWorld Cross-Task ---
    for diff in SMOLVLA_MW_DIFFICULTIES:
        ct_path = SMOLVLA_METAWORLD_DIR / "metaworld_cross_task" / diff / "results.json"
        data = safe_load_json(ct_path)
        if data:
            result["cross_task"][f"metaworld_{diff}"] = _extract_cross_task(data, environment="metaworld")

    # --- LIBERO Cross-Task (activation-level NPZ data, summarize directory structure) ---
    for suite in SMOLVLA_LIBERO_SUITES:
        ct_dir = SMOLVLA_LIBERO_DIR / "cross_task" / suite
        if ct_dir.is_dir():
            # Count NPZ files to indicate data availability
            npz_count = len(list(ct_dir.rglob("*.npz")))
            if npz_count > 0:
                result["cross_task"][suite] = {
                    "environment": "libero",
                    "suite": suite,
                    "data_type": "activation_npz",
                    "n_files": npz_count,
                    "status": "raw_data_available",
                }

    # --- MetaWorld Vision Perturbation ---
    for diff in SMOLVLA_MW_DIFFICULTIES:
        vp_path = SMOLVLA_METAWORLD_DIR / "metaworld_vision_perturbation" / diff / "results.json"
        data = safe_load_json(vp_path)
        if data:
            result["vision_perturbation"][f"metaworld_{diff}"] = _extract_vision_perturbation(
                data, environment="metaworld", difficulty=diff
            )

    # --- Displacement Analysis ---
    disp_path = SMOLVLA_METAWORLD_DIR / "metaworld_cross_task" / "smolvla_displacement_analysis.json"
    data = safe_load_json(disp_path)
    if data:
        result["displacement"]["metaworld"] = _extract_displacement(data)

    return result


def _extract_smolvla_grid(data: dict) -> dict:
    # Extract grid ablation summary from SmolVLA grid results
    conditions = data.get("conditions", [])
    tasks = data.get("tasks", [])
    grid = data.get("grid", {})

    summary = {
        "n_episodes": data.get("n_episodes", 0),
        "conditions": conditions,
        "n_conditions": len(conditions),
        "n_tasks": len(tasks),
        "baseline": {},
        "per_condition": {},
    }

    # Extract baseline
    baseline = grid.get("baseline", {})
    for task_key, task_data in baseline.items():
        sr = task_data.get("success_rate", 0)
        if isinstance(sr, (int, float)):
            summary["baseline"][str(task_key)] = {
                "success_rate": round_val(sr),
                "task_description": task_data.get("task_description", ""),
            }

    # Extract per-condition results
    for condition, cond_data in grid.items():
        if condition == "baseline":
            continue
        per_task = {}
        for task_key, task_data in cond_data.items():
            sr = task_data.get("success_rate", 0)
            if isinstance(sr, (int, float)):
                per_task[str(task_key)] = round_val(sr)
        if per_task:
            overall = sum(per_task.values()) / max(len(per_task), 1)
            summary["per_condition"][condition] = {
                "overall_success_rate": round_val(overall),
                "per_task": per_task,
            }

    return summary


def _aggregate_counterfactual_entries(entries: list[dict], **extra) -> dict:
    # Aggregate counterfactual JSONL entries into summary statistics
    by_category = defaultdict(list)
    for entry in entries:
        cat = entry.get("category", "unknown")
        by_category[cat].append(entry)

    summary = {
        "n_total_episodes": len(entries),
        "categories": {},
    }
    summary.update(extra)

    for cat, cat_entries in by_category.items():
        n = len(cat_entries)
        successes = sum(1 for e in cat_entries if e.get("success", False))
        avg_steps = sum(e.get("n_steps", 0) for e in cat_entries) / max(n, 1)
        summary["categories"][cat] = {
            "n_episodes": n,
            "success_rate": round_val(successes / max(n, 1)),
            "avg_steps": round_val(avg_steps),
        }

        # For cross_prompt, track per source_task
        if cat == "cross_prompt":
            by_source = defaultdict(list)
            for e in cat_entries:
                src = e.get("source_task", "unknown")
                by_source[str(src)].append(e)
            source_summary = {}
            for src, src_entries in by_source.items():
                ns = len(src_entries)
                ss = sum(1 for e in src_entries if e.get("success", False))
                source_summary[src] = {
                    "n_episodes": ns,
                    "success_rate": round_val(ss / max(ns, 1)),
                }
            summary["categories"][cat]["per_source"] = source_summary

    return summary


def _extract_cross_task(data: dict, environment: str = "metaworld") -> dict:
    # Extract cross-task injection summary
    pairs = data.get("pairs", {})
    injection_groups = data.get("injection_groups", [])

    pair_summaries = {}
    for pair_key, pair_data in pairs.items():
        pair_info = {
            "task_a": pair_data.get("task_a", ""),
            "task_b": pair_data.get("task_b", ""),
            "task_a_desc": pair_data.get("task_a_desc", ""),
            "task_b_desc": pair_data.get("task_b_desc", ""),
            "baseline_A_success": pair_data.get("baseline_A", {}).get("success"),
            "baseline_B_success": pair_data.get("baseline_B", {}).get("success"),
        }
        # Extract injection results (skip raw actions/trajectories)
        for group in injection_groups:
            inj_key = f"inject_A_into_B_{group}"
            inj_data = pair_data.get(inj_key, {})
            if inj_data:
                pair_info[group] = {
                    "success": inj_data.get("success"),
                    "n_steps": inj_data.get("n_steps"),
                    "cosine_to_A": round_val(inj_data.get("cosine_to_A")) if "cosine_to_A" in inj_data else None,
                    "cosine_to_B": round_val(inj_data.get("cosine_to_B")) if "cosine_to_B" in inj_data else None,
                }
        pair_summaries[pair_key] = pair_info

    # Compute aggregate stats per injection group
    group_stats = {}
    for group in injection_groups:
        successes = 0
        total = 0
        for pair_data in pair_summaries.values():
            gdata = pair_data.get(group, {})
            if gdata and gdata.get("success") is not None:
                total += 1
                if gdata["success"]:
                    successes += 1
        if total > 0:
            group_stats[group] = {
                "success_rate": round_val(successes / total),
                "n_pairs": total,
            }

    return {
        "environment": environment,
        "injection_groups": injection_groups,
        "n_pairs": len(pair_summaries),
        "group_stats": group_stats,
        "pairs": pair_summaries,
    }


def _extract_vision_perturbation(data: dict, environment: str = "metaworld",
                                  difficulty: str = "") -> dict:
    perturbations = data.get("perturbations", [])
    tasks = data.get("tasks", [])
    grid = data.get("grid", {})

    per_perturbation = {}
    for pert in perturbations:
        pert_data = grid.get(pert, {})
        task_results = {}
        total_sr = 0
        n_tasks = 0
        for task_name, task_info in pert_data.items():
            sr = task_info.get("success_rate", 0)
            task_results[task_name] = round_val(sr)
            total_sr += sr
            n_tasks += 1
        per_perturbation[pert] = {
            "overall_success_rate": round_val(total_sr / max(n_tasks, 1)),
            "n_tasks": n_tasks,
            "per_task": task_results,
        }

    # Compute baseline success rate
    baseline_sr = per_perturbation.get("baseline", {}).get("overall_success_rate", 0)

    # Compute deltas from baseline
    for pert, pdata in per_perturbation.items():
        if pert != "baseline" and baseline_sr > 0:
            pdata["delta_from_baseline"] = round_val(
                pdata["overall_success_rate"] - baseline_sr
            )

    return {
        "environment": environment,
        "difficulty": difficulty,
        "perturbations": perturbations,
        "n_perturbations": len(perturbations),
        "n_tasks": len(tasks),
        "n_episodes": data.get("n_episodes", 0),
        "per_perturbation": per_perturbation,
    }


def _extract_displacement(data: dict) -> dict:
    summary = {
        "model": data.get("model", ""),
        "difficulties": {},
    }
    for diff, diff_data in data.get("difficulties", {}).items():
        per_group = {}
        for group, gdata in diff_data.get("per_group", {}).items():
            per_group[group] = {
                "total": gdata.get("total", 0),
                "source_behavior": gdata.get("source_behavior", 0),
                "destination_behavior": gdata.get("destination_behavior", 0),
                "ambiguous": gdata.get("ambiguous", 0),
                "successes": gdata.get("successes", 0),
                "mean_cos_to_src": round_val(gdata.get("mean_cos_to_src", 0)),
                "mean_cos_to_dst": round_val(gdata.get("mean_cos_to_dst", 0)),
                "override_rate": round_val(
                    gdata.get("source_behavior", 0) / max(gdata.get("total", 1), 1)
                ),
            }
        summary["difficulties"][diff] = {
            "total_episodes": diff_data.get("total_episodes", 0),
            "n_pairs": diff_data.get("n_pairs", 0),
            "per_group": per_group,
        }
    return summary
# X-VLA aggregation
XVLA_LIBERO_DIR = Path("/data/xvla_rollouts")
XVLA_SIMPLERENV_DIR = Path("/data/xvla_simplerenv")
XVLA_CONCEPT_ABLATION_DIR = Path("/data/batch_1/xvla_concept_ablation")
XVLA_CONCEPT_STEERING_DIR = Path("/data/batch_1/xvla_concept_steering")
XVLA_LIBERO_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]


