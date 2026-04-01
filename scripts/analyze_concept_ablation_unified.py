#!/usr/bin/env python3
"""
Unified Concept Ablation & Steering Analysis across all 6 VLA models.

Reads concept ablation and steering results from all models (OFT, Pi0.5,
X-VLA, SmolVLA, GR00T, ACT), normalizes them into a common schema, and
produces:
  1. Per-model width-resilience profiles (zero-effect %, destruction %, mean delta)
  2. Kill-switch feature identification (concepts causing >50pp drops)
  3. Layer gradient analysis (effect severity by layer depth)
  4. Steering dose-response curves
  5. Cross-model comparison table (for paper Table X)
  6. JSON output for Action Atlas integration
  7. LaTeX table snippet for direct paper inclusion

Handles 3 different JSON schemas:
  - OFT/Pi0.5/X-VLA: {tasks: {concept: {tasks: {task_id: {delta, success_rate}}}}}
  - SmolVLA: {ablation: {concept: {overall_rate, overall_delta}}, fraction_to_failure: {...}}
  - GR00T: displacement-based (separate analysis)

Usage:
    python scripts/analyze_concept_ablation_unified.py [--output-dir results/concept_ablation_analysis]
    python scripts/analyze_concept_ablation_unified.py --latex   # emit LaTeX table
    python scripts/analyze_concept_ablation_unified.py --json    # emit Action Atlas JSON

Wilson CI from analyze_rollouts.py for consistency.
"""

import json
import math
import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ============================================================
# Configuration: where each model's data lives
# ============================================================

DATA_SOURCES = {
    "oft": {
        "ablation_dir": Path("results/experiment_results/oft_concept_ablation"),
        "ablation_glob": "ablation_*.json",
        "steering_glob": "steering_*.json",
        "schema": "oft",  # {tasks: {concept: {tasks: {id: {delta}}}}}
        "dim": 4096,
        "params": "7B",
    },
    "pi05_expert": {
        "ablation_dir": Path("results/experiment_results/pi05_concept_ablation"),
        "ablation_glob": "ablation_expert_*.json",
        "steering_glob": "steering_expert_*.json",
        "schema": "oft",
        "dim": 1024,
        "params": "3B",
    },
    "xvla": {
        "ablation_dir": Path("results/xvla_concept_ablation"),
        "ablation_glob": "ablation_*.json",
        "steering_glob": "steering_*.json",
        "schema": "xvla",  # same as OFT but model field differs
        "dim": 1024,
        "params": "1B",
    },
    "smolvla_expert": {
        "ablation_dir": Path("rollouts/smolvla/concept_ablation/results"),
        "ablation_glob": "expert_*_all_*.json",
        "ablation_glob_alt": "expert_*_ablation_*.json",  # richer per-task format
        "steering_glob": None,
        "schema": "smolvla",  # {ablation: {concept: {overall_delta}}}
        "dim": 480,
        "params": "450M",
    },
    "smolvla_vlm": {
        "ablation_dir": Path("rollouts/smolvla/concept_ablation/results"),
        "ablation_glob": "vlm_*_all_*.json",
        "ablation_glob_alt": "vlm_*_ablation_*.json",
        "steering_glob": None,
        "schema": "smolvla",
        "dim": 960,
        "params": "450M",
    },
    "groot": {
        "ablation_dir": Path("/data/groot_rollouts/sae_feature_ablation"),
        "ablation_dir_b2": Path("/data/groot_rollouts_batch2/sae_feature_ablation") if Path("/data/groot_rollouts_batch2/sae_feature_ablation").exists() else None,
        "ablation_glob": "**/ablation_results.json",
        "steering_glob": None,
        "schema": "groot",
        "dim": "varies",
        "params": "3B",
    },
}

# Also check smolvla batch2 for SmolVLA data
SMOLVLA_smolvla = Path("/data/smolvla_rollouts/concept_ablation/results")


# ============================================================
# Wilson CI (matches analyze_rollouts.py)
# ============================================================

def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) / denom
    return p_hat * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


# ============================================================
# Schema Parsers: normalize to common format
# ============================================================

def parse_oft_schema(data):
    """Parse OFT/Pi0.5/X-VLA schema into normalized records.

    Returns list of dicts:
        {concept, layer, suite, task_id, delta_pp, success_rate, baseline_rate, n_episodes}
    """
    records = []
    layer = data.get("layer", -1)
    suite = data.get("suite", "unknown")
    n_episodes = data.get("n_episodes", 3)
    mode = data.get("mode", "ablation")
    baseline_data = data.get("baseline", {})

    tasks_block = data.get("tasks", {})
    for concept, cdata in tasks_block.items():
        if not isinstance(cdata, dict) or "tasks" not in cdata:
            continue
        for task_id, tdata in cdata["tasks"].items():
            if not isinstance(tdata, dict):
                continue
            delta = tdata.get("delta", 0)
            sr = tdata.get("success_rate", 0)
            # delta is in fraction (0-1), convert to pp
            delta_pp = delta * 100 if abs(delta) <= 1.0 else delta
            sr_pct = sr * 100 if sr <= 1.0 else sr

            # Baseline for this task
            bl = 0
            if isinstance(baseline_data, dict):
                bl_task = baseline_data.get(task_id, baseline_data.get(str(task_id), {}))
                if isinstance(bl_task, dict):
                    bl = bl_task.get("success_rate", 0)
                    bl = bl * 100 if bl <= 1.0 else bl

            records.append({
                "concept": concept,
                "layer": layer,
                "suite": suite,
                "task_id": str(task_id),
                "delta_pp": delta_pp,
                "success_rate": sr_pct,
                "baseline_rate": bl,
                "n_episodes": n_episodes,
                "mode": mode,
            })
    return records


def parse_smolvla_schema(data):
    """Parse SmolVLA schema into normalized records.

    SmolVLA has per-concept overall_delta (already in fraction).
    No per-task breakdown in this format.
    """
    records = []
    layer = data.get("layer", -1)
    suite = data.get("suite", "unknown")
    component = data.get("component", "expert")
    baseline = data.get("baseline_overall", 0)
    baseline_pct = baseline * 100 if baseline <= 1.0 else baseline

    ablation_block = data.get("ablation", {})
    for concept, cdata in ablation_block.items():
        if not isinstance(cdata, dict):
            continue
        delta = cdata.get("overall_delta", 0)
        rate = cdata.get("overall_rate", 0)
        delta_pp = delta * 100 if abs(delta) <= 1.0 else delta
        rate_pct = rate * 100 if rate <= 1.0 else rate

        records.append({
            "concept": concept,
            "layer": layer,
            "suite": suite,
            "task_id": "overall",
            "delta_pp": delta_pp,
            "success_rate": rate_pct,
            "baseline_rate": baseline_pct,
            "n_episodes": 2,
            "mode": "ablation",
        })
    return records


def parse_steering_oft(data):
    """Parse OFT/Pi0.5 steering schema.

    Format: {tasks: {concept: {strengths: {"-3.0": {task_id: {delta, success_rate}}}}}}
    Strengths are typically "-3.0" and "3.0" (suppression and amplification).
    """
    records = []
    layer = data.get("layer", -1)
    suite = data.get("suite", "unknown")
    n_episodes = data.get("n_episodes", 3)

    tasks_block = data.get("tasks", {})
    for concept, cdata in tasks_block.items():
        if not isinstance(cdata, dict):
            continue
        strengths = cdata.get("strengths", {})
        if not isinstance(strengths, dict):
            continue

        for strength_str, strength_data in strengths.items():
            if not isinstance(strength_data, dict):
                continue
            # strength_data keys are task IDs directly: {"0": {delta, success_rate}, ...}
            for task_id, tdata in strength_data.items():
                if not isinstance(tdata, dict):
                    continue
                delta = tdata.get("delta", 0)
                sr = tdata.get("success_rate", 0)
                records.append({
                    "concept": concept,
                    "layer": layer,
                    "suite": suite,
                    "task_id": str(task_id),
                    "delta_pp": delta * 100 if abs(delta) <= 1.0 else delta,
                    "success_rate": sr * 100 if sr <= 1.0 else sr,
                    "multiplier": float(strength_str),
                    "mode": "steering",
                })
    return records


def parse_smolvla_pertask_schema(data):
    """Parse SmolVLA per-task ablation format.

    Format: {concepts: {concept: {tasks: {task_id: {rate, delta}}, overall_rate, overall_delta}}}
    This format has per-task breakdowns, enabling proper zero-effect counting.
    """
    records = []
    layer = data.get("layer", -1)
    suite = data.get("suite", "unknown")
    baseline = data.get("baseline_overall", 0)
    baseline_pct = baseline * 100 if baseline <= 1.0 else baseline

    concepts_block = data.get("concepts", {})
    for concept, cdata in concepts_block.items():
        if not isinstance(cdata, dict) or "tasks" not in cdata:
            continue
        tasks = cdata.get("tasks", {})
        for task_id, tdata in tasks.items():
            if not isinstance(tdata, dict):
                continue
            delta = tdata.get("delta", 0)
            rate = tdata.get("rate", 0)
            delta_pp = delta * 100 if abs(delta) <= 1.0 else delta
            rate_pct = rate * 100 if rate <= 1.0 else rate
            records.append({
                "concept": concept,
                "layer": layer,
                "suite": suite,
                "task_id": str(task_id),
                "delta_pp": delta_pp,
                "success_rate": rate_pct,
                "baseline_rate": baseline_pct,
                "n_episodes": 10,  # ablation files use 10 episodes
                "mode": "ablation",
            })
    return records


def parse_smolvla_ftf(data):
    """Parse SmolVLA fraction-to-failure data."""
    records = []
    layer = data.get("layer", -1)
    suite = data.get("suite", "unknown")
    baseline = data.get("baseline_overall", 0)
    baseline_pct = baseline * 100 if baseline <= 1.0 else baseline

    ftf_block = data.get("fraction_to_failure", {})
    for concept, cdata in ftf_block.items():
        if not isinstance(cdata, dict):
            continue
        fractions = cdata.get("fractions", {})
        for frac_str, fdata in fractions.items():
            if not isinstance(fdata, dict):
                continue
            delta = fdata.get("overall_delta", 0)
            rate = fdata.get("overall_rate", 0)
            records.append({
                "concept": concept,
                "layer": layer,
                "suite": suite,
                "fraction": int(frac_str),
                "delta_pp": delta * 100 if abs(delta) <= 1.0 else delta,
                "success_rate": rate * 100 if rate <= 1.0 else rate,
                "baseline_rate": baseline_pct,
                "mode": "fraction_to_failure",
            })
    return records


def parse_groot_schema(data):
    """Parse GR00T feature ablation schema into normalized records.

    Format: {layer, suite, baseline: {task_id: {success_rate}},
             groups: {group_name: {tasks: {task_id: {delta, success_rate}}}}}
    Deltas are in [0,1] fraction format. Groups are feature categories
    (task_specific, success_predictive, universal, frequent, disc_*).
    """
    records = []
    layer = data.get("layer", "unknown")
    suite = data.get("suite", "unknown")
    n_episodes = data.get("n_episodes", 3)

    groups = data.get("groups", {})
    for group_name, group_data in groups.items():
        if not isinstance(group_data, dict) or "tasks" not in group_data:
            continue
        tasks = group_data.get("tasks", {})
        for task_id, tdata in tasks.items():
            if not isinstance(tdata, dict) or "delta" not in tdata:
                continue
            delta = tdata["delta"]
            sr = tdata.get("success_rate", 0)
            records.append({
                "concept": group_name,
                "layer": layer,
                "suite": suite,
                "task_id": str(task_id),
                "delta_pp": delta * 100,
                "success_rate": sr * 100,
                "baseline_rate": 0,
                "n_episodes": n_episodes,
                "mode": "ablation",
            })
    return records


# ============================================================
# Data Loading
# ============================================================

def load_all_data():
    """Load and normalize all concept ablation + steering data."""
    all_records = {}  # model_key -> list of records
    all_steering = {}
    all_ftf = {}
    file_counts = {}

    for model_key, cfg in DATA_SOURCES.items():
        abl_dir = cfg["ablation_dir"]
        records = []
        steering = []
        ftf = []
        n_files = 0

        if not abl_dir.exists():
            print(f"  SKIP {model_key}: {abl_dir} not found", file=sys.stderr)
            continue

        # Ablation files
        for f in sorted(abl_dir.glob(cfg["ablation_glob"])):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, IOError):
                continue
            n_files += 1
            if cfg["schema"] in ("oft", "xvla"):
                records.extend(parse_oft_schema(data))
            elif cfg["schema"] == "smolvla":
                records.extend(parse_smolvla_schema(data))
                ftf.extend(parse_smolvla_ftf(data))
            elif cfg["schema"] == "groot":
                records.extend(parse_groot_schema(data))

        # Alt ablation files (SmolVLA per-task format)
        alt_glob = cfg.get("ablation_glob_alt")
        if alt_glob:
            for f in sorted(abl_dir.glob(alt_glob)):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, IOError):
                    continue
                n_files += 1
                if "concepts" in data:
                    records.extend(parse_smolvla_pertask_schema(data))
                else:
                    records.extend(parse_smolvla_schema(data))

        # Steering files
        if cfg["steering_glob"]:
            for f in sorted(abl_dir.glob(cfg["steering_glob"])):
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, IOError):
                    continue
                n_files += 1
                steering.extend(parse_steering_oft(data))

        all_records[model_key] = records
        all_steering[model_key] = steering
        all_ftf[model_key] = ftf
        file_counts[model_key] = n_files
        print(f"  {model_key}: {n_files} files, {len(records)} ablation records, "
              f"{len(steering)} steering records, {len(ftf)} FTF records", file=sys.stderr)

    # Also check smolvla for SmolVLA
    if SMOLVLA_smolvla.exists():
        for f in sorted(SMOLVLA_smolvla.glob("expert_*_all_*.json")):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, IOError):
                continue
            recs = parse_smolvla_schema(data)
            ftf_recs = parse_smolvla_ftf(data)
            if "smolvla_expert" not in all_records:
                all_records["smolvla_expert"] = []
                all_ftf["smolvla_expert"] = []
            all_records["smolvla_expert"].extend(recs)
            all_ftf["smolvla_expert"].extend(ftf_recs)

    return all_records, all_steering, all_ftf, file_counts


# ============================================================
# Analysis Functions
# ============================================================

def compute_width_resilience(records, dim):
    """Compute width-resilience profile from ablation records.

    Returns dict with:
        zero_effect_pct: fraction of (concept, task) pairs with |delta| < 1pp
        destruction_pct: fraction with delta < -50pp
        mean_delta: mean delta across all pairs
        median_delta: median delta
        n_pairs: total pairs analyzed
        kill_switches: concepts with any task showing > 50pp drop
    """
    if not records:
        return None

    deltas = [r["delta_pp"] for r in records]
    n = len(deltas)
    zero_count = sum(1 for d in deltas if abs(d) < 1.0)
    destroy_count = sum(1 for d in deltas if d < -50)
    severe_count = sum(1 for d in deltas if d < -25)
    mean_d = sum(deltas) / n
    sorted_d = sorted(deltas)
    median_d = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2

    # Kill-switch identification
    concept_worst = defaultdict(float)
    concept_tasks_affected = defaultdict(int)
    for r in records:
        if r["delta_pp"] < concept_worst[r["concept"]]:
            concept_worst[r["concept"]] = r["delta_pp"]
        if r["delta_pp"] < -25:
            concept_tasks_affected[r["concept"]] += 1

    kill_switches = []
    for concept, worst in sorted(concept_worst.items(), key=lambda x: x[1]):
        if worst < -50:
            kill_switches.append({
                "concept": concept,
                "worst_delta_pp": round(worst, 1),
                "tasks_affected_gt25pp": concept_tasks_affected.get(concept, 0),
            })

    return {
        "dim": dim,
        "n_pairs": n,
        "zero_effect_pct": round(zero_count / n * 100, 1),
        "destruction_pct": round(destroy_count / n * 100, 1),
        "severe_pct": round(severe_count / n * 100, 1),
        "mean_delta_pp": round(mean_d, 1),
        "median_delta_pp": round(median_d, 1),
        "min_delta_pp": round(min(deltas), 1),
        "max_delta_pp": round(max(deltas), 1),
        "kill_switches": kill_switches[:10],
    }


def compute_layer_gradient(records):
    """Compute how effect severity varies by layer depth.

    Returns dict: layer -> {mean_delta, destruction_pct, n_pairs}
    """
    by_layer = defaultdict(list)
    for r in records:
        by_layer[r["layer"]].append(r["delta_pp"])

    gradient = {}
    for layer in sorted(by_layer.keys()):
        deltas = by_layer[layer]
        n = len(deltas)
        gradient[layer] = {
            "mean_delta_pp": round(sum(deltas) / n, 1),
            "destruction_pct": round(sum(1 for d in deltas if d < -50) / n * 100, 1),
            "zero_effect_pct": round(sum(1 for d in deltas if abs(d) < 1) / n * 100, 1),
            "n_pairs": n,
        }
    return gradient


def compute_per_suite(records):
    """Break down by suite."""
    by_suite = defaultdict(list)
    for r in records:
        by_suite[r["suite"]].append(r["delta_pp"])

    result = {}
    for suite, deltas in sorted(by_suite.items()):
        n = len(deltas)
        result[suite] = {
            "mean_delta_pp": round(sum(deltas) / n, 1),
            "zero_effect_pct": round(sum(1 for d in deltas if abs(d) < 1) / n * 100, 1),
            "destruction_pct": round(sum(1 for d in deltas if d < -50) / n * 100, 1),
            "n_pairs": n,
        }
    return result


def compute_steering_summary(steering_records):
    """Summarize steering dose-response per multiplier."""
    by_mult = defaultdict(list)
    for r in steering_records:
        mult = r.get("multiplier", 0)
        by_mult[mult].append(r["delta_pp"])

    result = {}
    for mult in sorted(by_mult.keys()):
        deltas = by_mult[mult]
        n = len(deltas)
        result[str(mult)] = {
            "mean_delta_pp": round(sum(deltas) / n, 1),
            "zero_effect_pct": round(sum(1 for d in deltas if abs(d) < 1) / n * 100, 1),
            "destruction_pct": round(sum(1 for d in deltas if d < -50) / n * 100, 1),
            "n_pairs": n,
        }
    return result


def classify_width_resilience(profile):
    """Classify as narrow/wide/mixed based on profile."""
    if profile is None:
        return "N/A"
    zero = profile["zero_effect_pct"]
    destroy = profile["destruction_pct"]
    if zero > 80:
        return "wide (resilient)"
    elif destroy > 10 or zero < 30:
        return "narrow (fragile)"
    else:
        return "mixed"


# ============================================================
# Output Formatters
# ============================================================

def print_report(all_records, all_steering, all_ftf, file_counts):
    """Print text report."""
    sep = "=" * 100
    thin = "-" * 100

    print(sep)
    print("UNIFIED CONCEPT ABLATION & STEERING ANALYSIS")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    # Per-model analysis
    all_profiles = {}
    for model_key, records in all_records.items():
        cfg = DATA_SOURCES.get(model_key, {})
        dim = cfg.get("dim", 0)
        params = cfg.get("params", "?")

        print(f"\n{thin}")
        print(f"MODEL: {model_key} ({params}, {dim}-dim)")
        print(thin)
        print(f"  Files: {file_counts.get(model_key, '?')}")
        print(f"  Ablation records: {len(records)}")
        print(f"  Steering records: {len(all_steering.get(model_key, []))}")
        print(f"  FTF records: {len(all_ftf.get(model_key, []))}")

        profile = compute_width_resilience(records, dim)
        all_profiles[model_key] = profile
        if profile is None:
            print("  NO DATA")
            continue

        classification = classify_width_resilience(profile)
        print(f"\n  WIDTH-RESILIENCE PROFILE: {classification}")
        print(f"    Zero effect (<1pp):     {profile['zero_effect_pct']}% ({int(profile['n_pairs'] * profile['zero_effect_pct'] / 100)}/{profile['n_pairs']})")
        print(f"    Severe (>25pp drop):    {profile['severe_pct']}%")
        print(f"    Destruction (>50pp):    {profile['destruction_pct']}%")
        print(f"    Mean delta:             {profile['mean_delta_pp']:+.1f}pp")
        print(f"    Median delta:           {profile['median_delta_pp']:+.1f}pp")
        print(f"    Range:                  [{profile['min_delta_pp']:+.1f}, {profile['max_delta_pp']:+.1f}]pp")

        # Kill-switches
        if profile["kill_switches"]:
            print(f"\n  KILL-SWITCH FEATURES (>50pp drop on any task):")
            for ks in profile["kill_switches"]:
                print(f"    {ks['concept']:30s}  worst={ks['worst_delta_pp']:+.1f}pp  "
                      f"tasks_affected(>25pp)={ks['tasks_affected_gt25pp']}")
        else:
            print(f"\n  NO kill-switch features found (no concept causes >50pp drop)")

        # Layer gradient
        gradient = compute_layer_gradient(records)
        if len(gradient) > 1:
            print(f"\n  LAYER GRADIENT:")
            print(f"    {'Layer':>6}  {'Mean Delta':>10}  {'Zero%':>6}  {'Destroy%':>8}  {'N':>5}")
            for layer, g in sorted(gradient.items(), key=lambda x: (int(x[0]) if str(x[0]).isdigit() else float('inf'), str(x[0]))):
                layer_label = f"L{layer:02d}" if isinstance(layer, int) else f"L{layer}"
                print(f"    {layer_label:<8}  {g['mean_delta_pp']:+7.1f}pp  {g['zero_effect_pct']:5.1f}%  {g['destruction_pct']:7.1f}%  {g['n_pairs']:5d}")

        # Per-suite
        suite_data = compute_per_suite(records)
        if len(suite_data) > 1:
            print(f"\n  PER-SUITE BREAKDOWN:")
            print(f"    {'Suite':>20}  {'Mean Delta':>10}  {'Zero%':>6}  {'Destroy%':>8}  {'N':>5}")
            for suite, s in sorted(suite_data.items()):
                print(f"    {suite:>20}  {s['mean_delta_pp']:+7.1f}pp  {s['zero_effect_pct']:5.1f}%  {s['destruction_pct']:7.1f}%  {s['n_pairs']:5d}")

        # Steering
        steering = all_steering.get(model_key, [])
        if steering:
            steer_summary = compute_steering_summary(steering)
            print(f"\n  STEERING DOSE-RESPONSE:")
            print(f"    {'Multiplier':>10}  {'Mean Delta':>10}  {'Zero%':>6}  {'Destroy%':>8}  {'N':>5}")
            for mult, s in steer_summary.items():
                print(f"    {mult:>10}  {s['mean_delta_pp']:+7.1f}pp  {s['zero_effect_pct']:5.1f}%  {s['destruction_pct']:7.1f}%  {s['n_pairs']:5d}")

    # Cross-model comparison table
    print(f"\n{sep}")
    print("CROSS-MODEL COMPARISON TABLE")
    print(sep)
    print(f"  {'Model':>18}  {'Dim':>5}  {'Params':>6}  {'N Pairs':>8}  "
          f"{'Zero%':>6}  {'Destroy%':>8}  {'Mean Delta':>10}  {'Classification':>22}  {'Kill-Switches':>13}")
    print(thin)
    for model_key in ["pi05_expert", "oft", "xvla", "smolvla_expert", "smolvla_vlm", "groot"]:
        p = all_profiles.get(model_key)
        cfg = DATA_SOURCES.get(model_key, {})
        if p is None:
            print(f"  {model_key:>18}  {cfg.get('dim','?'):>5}  {cfg.get('params','?'):>6}  {'NO DATA':>8}")
            continue
        cl = classify_width_resilience(p)
        n_ks = len(p["kill_switches"])
        print(f"  {model_key:>18}  {p['dim']:>5}  {cfg.get('params','?'):>6}  {p['n_pairs']:>8}  "
              f"{p['zero_effect_pct']:>5.1f}%  {p['destruction_pct']:>7.1f}%  {p['mean_delta_pp']:>+9.1f}pp  "
              f"{cl:>22}  {n_ks:>13}")

    print(f"\n{sep}")

    return all_profiles


def emit_latex_table(all_profiles):
    """Emit LaTeX table for direct inclusion in paper."""
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Width-dependent causal profiles across architectures. Zero effect: fraction of concept-task pairs with $|\Delta| < 1$pp. Destruction: fraction with $\Delta < -50$pp. Kill-switches: concepts causing $> 50$pp drop on any task.}")
    print(r"\label{tab:width-profiles}")
    print(r"\small")
    print(r"\begin{tabular}{lrcrrrl}")
    print(r"\toprule")
    print(r"Model & Dim & Pairs & Zero\% & Destroy\% & Mean $\Delta$ & Classification \\")
    print(r"\midrule")

    display_names = {
        "pi05_expert": r"$\pi_{0.5}$ Expert",
        "oft": "OpenVLA-OFT",
        "xvla": "X-VLA",
        "smolvla_expert": "SmolVLA Expert",
        "smolvla_vlm": "SmolVLA VLM",
        "groot": r"GR00T N1.5",
    }

    for model_key in ["pi05_expert", "oft", "xvla", "smolvla_expert", "smolvla_vlm", "groot"]:
        p = all_profiles.get(model_key)
        name = display_names.get(model_key, model_key)
        if p is None:
            print(rf"{name} & -- & -- & -- & -- & -- & N/A \\")
            continue
        cl = classify_width_resilience(p)
        cl_short = cl.split("(")[0].strip()
        print(rf"{name} & {p['dim']} & {p['n_pairs']} & {p['zero_effect_pct']:.1f} & "
              rf"{p['destruction_pct']:.1f} & {p['mean_delta_pp']:+.1f} & {cl_short} \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def emit_action_atlas_json(all_records, all_steering, all_ftf, all_profiles, output_path):
    """Emit JSON for Action Atlas integration."""
    output = {
        "generated": datetime.now().isoformat(),
        "profiles": {},
        "per_model": {},
    }

    for model_key, profile in all_profiles.items():
        if profile is None:
            continue
        output["profiles"][model_key] = profile
        output["profiles"][model_key]["classification"] = classify_width_resilience(profile)

    for model_key, records in all_records.items():
        if not records:
            continue
        output["per_model"][model_key] = {
            "layer_gradient": compute_layer_gradient(records),
            "per_suite": compute_per_suite(records),
            "n_records": len(records),
        }
        # Add steering if available
        steering = all_steering.get(model_key, [])
        if steering:
            output["per_model"][model_key]["steering"] = compute_steering_summary(steering)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nAction Atlas JSON written to: {output_path}", file=sys.stderr)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Unified concept ablation analysis")
    parser.add_argument("--output-dir", type=str,
                        default="results/concept_ablation_analysis")
    parser.add_argument("--latex", action="store_true",
                        help="Emit LaTeX table to stdout")
    parser.add_argument("--json", action="store_true",
                        help="Emit Action Atlas JSON")
    args = parser.parse_args()

    print("Loading data from all models...", file=sys.stderr)
    all_records, all_steering, all_ftf, file_counts = load_all_data()

    total_records = sum(len(r) for r in all_records.values())
    total_steering = sum(len(r) for r in all_steering.values())
    print(f"\nTotal: {total_records} ablation records, {total_steering} steering records\n",
          file=sys.stderr)

    if total_records == 0:
        print("ERROR: No ablation data found. Check DATA_SOURCES paths.", file=sys.stderr)
        sys.exit(1)

    all_profiles = print_report(all_records, all_steering, all_ftf, file_counts)

    if args.latex:
        print("\n\n% === LATEX TABLE ===\n")
        emit_latex_table(all_profiles)

    if args.json:
        out_path = Path(args.output_dir) / "concept_ablation_profiles.json"
        emit_action_atlas_json(all_records, all_steering, all_ftf, all_profiles, out_path)

    # Always save the profiles JSON
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "concept_ablation_profiles.json"
    emit_action_atlas_json(all_records, all_steering, all_ftf, all_profiles, out_path)


if __name__ == "__main__":
    main()
