#!/usr/bin/env python3
"""
Cross-Task Displacement Analysis for Pi0.5 and OpenVLA-OFT

Measures whether injected activations steer behavior toward source or destination
task trajectories. Uses the same methodology as the X-VLA displacement analysis:

For each cross-task injection episode:
1. Compute cosine similarity between injected trajectory and source baseline
2. Compute cosine similarity between injected trajectory and destination baseline
3. Classify as source/destination/ambiguous behavior (threshold=0.05)
4. Compute override rate (% where source similarity > destination similarity)

Pi0.5: Uses pre-computed cos_to_src_baseline / cos_to_dst_baseline from results.json
OFT: Computes cosine similarity from EEF velocity trajectories (action deltas of
     end-effector positions), since only cos_to_baseline_b was stored during rollouts.

Usage:
    python scripts/displacement_analysis_pi05_oft.py

Output:
    results/experiment_results/displacement_analysis_pi05.json
    results/experiment_results/displacement_analysis_oft.json
"""

import json
import math
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

# Classification threshold (same as X-VLA analysis)
THRESHOLD = 0.05

# Pi0.5 data locations
PI05_CROSS_TASK_DIRS = {
    "libero_10": [
        "/data/robotsteering/pi05_rollouts/cross_task_10/comprehensive_cross_task_libero_10_20260127_051354",
        "/data/robotsteering/pi05_rollouts/cross_task_10/comprehensive_cross_task_libero_10_20260127_085109",
    ],
    "libero_goal": [
        "/data/robotsteering/pi05_rollouts/cross_task_goal/comprehensive_cross_task_libero_goal_20260126_234343",
        "/data/robotsteering/pi05_rollouts/cross_task_goal/comprehensive_cross_task_libero_goal_seed42_20260127_195312",
        "/data/robotsteering/pi05_rollouts/cross_task_goal/comprehensive_cross_task_libero_goal_seed123_20260127_195312",
        "/data/robotsteering/pi05_rollouts/cross_task_goal/comprehensive_cross_task_libero_goal_seed456_20260127_195312",
    ],
    "libero_spatial": [
        "/data/robotsteering/pi05_rollouts/cross_task_spatial/comprehensive_cross_task_libero_spatial_20260126_225619",
    ],
}

# OFT data location
OFT_TRAJ_DIR = "/data/openvla_rollouts/openvla_oft/trajectories"
OFT_SUITES = ["libero_goal", "libero_object", "libero_spatial", "libero_10"]

# Output
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'experiment_results'


# ============================================================
# Utilities
# ============================================================

def wilson_ci(successes, total, z=1.96):
    """Wilson score confidence interval."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p_hat = successes / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * ((p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return p_hat * 100, max(0, center - margin) * 100, min(1, center + margin) * 100


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def classify_behavior(cos_src, cos_dst, threshold=THRESHOLD):
    """Classify episode as source, destination, or ambiguous."""
    diff = cos_src - cos_dst
    if diff > threshold:
        return "source"
    elif diff < -threshold:
        return "destination"
    else:
        return "ambiguous"


def eef_velocity(eef_trajectory):
    """Compute velocity (deltas) from EEF position trajectory."""
    arr = np.array(eef_trajectory)
    return np.diff(arr, axis=0).flatten()


# ============================================================
# Pi0.5 Analysis
# ============================================================

def load_pi05_episodes():
    """Load all Pi0.5 cross-task injection episodes from results.json files."""
    episodes = []

    for suite, exp_dirs in PI05_CROSS_TASK_DIRS.items():
        for exp_dir_str in exp_dirs:
            exp_dir = Path(exp_dir_str)
            if not exp_dir.exists():
                print(f"  SKIP (not found): {exp_dir}")
                continue

            # Check for merged results first
            merged = exp_dir / "merged_results.json"
            if merged.exists():
                batch_files = [merged]
            else:
                # Load individual batches
                batch_files = sorted(exp_dir.glob("batch_*/results.json"))

            for bf in batch_files:
                try:
                    with open(bf) as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"  ERROR loading {bf}: {e}")
                    continue

                results = data.get("results", {})
                for pair_key, pair_data in results.items():
                    task_a = pair_data.get("task_a")
                    task_b = pair_data.get("task_b")
                    prompt_a = pair_data.get("prompt_a", "")
                    prompt_b = pair_data.get("prompt_b", "")

                    # Process both injection directions
                    for inject_key in [f"inject_{task_a}_into_{task_b}",
                                       f"inject_{task_b}_into_{task_a}"]:
                        inject_data = pair_data.get(inject_key)
                        if inject_data is None:
                            continue

                        # Parse source/dest from key
                        parts = inject_key.split("_")
                        src_task = int(parts[1])
                        dst_task = int(parts[3])

                        for cond_name, cond_data in inject_data.items():
                            if not isinstance(cond_data, dict):
                                continue

                            cos_src = cond_data.get("cos_to_src_baseline")
                            cos_dst = cond_data.get("cos_to_dst_baseline")

                            if cos_src is None or cos_dst is None:
                                continue

                            # Determine if this is an injection condition
                            is_injection = cond_data.get("injections", 0) > 0
                            inject_pathway = cond_data.get("inject_pathway")

                            # Extract layer info from condition name
                            layer = "none"
                            if "pali_ALL" in cond_name:
                                layer = "pali_ALL"
                            elif "pali_L0" in cond_name:
                                layer = "pali_L0"
                            elif "expert_L16_L17" in cond_name:
                                layer = "expert_L16_L17"
                            elif "expert_L16" in cond_name:
                                layer = "expert_L16"

                            # Determine prompt type
                            if "own_prompt" in cond_name:
                                prompt_type = "own_prompt"
                            elif "cross_prompt" in cond_name:
                                prompt_type = "cross_prompt"
                            else:
                                prompt_type = "unknown"

                            episodes.append({
                                "suite": suite,
                                "src_task": src_task,
                                "dst_task": dst_task,
                                "condition": cond_name,
                                "layer": layer,
                                "prompt_type": prompt_type,
                                "is_injection": is_injection,
                                "inject_pathway": inject_pathway,
                                "cos_to_src": cos_src,
                                "cos_to_dst": cos_dst,
                                "success": cond_data.get("success", False),
                                "steps": cond_data.get("steps", 0),
                                "xyz_to_src": cond_data.get("xyz_to_src_baseline"),
                                "xyz_to_dst": cond_data.get("xyz_to_dst_baseline"),
                                "source": bf.parent.parent.name,
                            })

    return episodes


def analyze_pi05():
    """Run displacement analysis on Pi0.5 cross-task injection data."""
    print("\n" + "=" * 60)
    print("Pi0.5 CROSS-TASK DISPLACEMENT ANALYSIS")
    print("=" * 60)

    episodes = load_pi05_episodes()
    print(f"\nTotal episodes loaded: {len(episodes)}")

    if not episodes:
        print("ERROR: No episodes found!")
        return None

    # Separate injection vs non-injection
    injection_eps = [e for e in episodes if e["is_injection"]]
    no_inject_eps = [e for e in episodes if not e["is_injection"]]
    print(f"  Injection episodes: {len(injection_eps)}")
    print(f"  No-injection episodes: {len(no_inject_eps)}")

    results = {
        "meta": {
            "model": "Pi0.5",
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(episodes),
            "injection_episodes": len(injection_eps),
            "threshold": THRESHOLD,
        },
        "grand_summary": {},
        "by_suite": {},
        "by_condition": {},
        "by_layer": {},
        "by_prompt_type": {},
        "by_pathway": {},
    }

    # Grand summary (injection episodes only)
    results["grand_summary"] = _compute_displacement_stats(injection_eps, "all_injection")

    # By suite
    suites = set(e["suite"] for e in injection_eps)
    for suite in sorted(suites):
        suite_eps = [e for e in injection_eps if e["suite"] == suite]
        results["by_suite"][suite] = _compute_displacement_stats(suite_eps, suite)

    # By condition (include no-inject for comparison)
    conditions = set(e["condition"] for e in episodes)
    for cond in sorted(conditions):
        cond_eps = [e for e in episodes if e["condition"] == cond]
        results["by_condition"][cond] = _compute_displacement_stats(cond_eps, cond)

    # By layer (injection only)
    layers = set(e["layer"] for e in injection_eps)
    for layer in sorted(layers):
        layer_eps = [e for e in injection_eps if e["layer"] == layer]
        results["by_layer"][layer] = _compute_displacement_stats(layer_eps, layer)

    # By prompt type
    for pt in ["own_prompt", "cross_prompt"]:
        pt_eps = [e for e in injection_eps if e["prompt_type"] == pt]
        if pt_eps:
            results["by_prompt_type"][pt] = _compute_displacement_stats(pt_eps, pt)

    # By pathway (pali vs expert)
    pathways = set(e["inject_pathway"] for e in injection_eps if e["inject_pathway"])
    for pw in sorted(pathways):
        pw_eps = [e for e in injection_eps if e["inject_pathway"] == pw]
        results["by_pathway"][pw] = _compute_displacement_stats(pw_eps, pw)

    # Per-suite per-condition breakdown
    results["per_suite_condition"] = {}
    for suite in sorted(suites):
        results["per_suite_condition"][suite] = {}
        suite_eps_all = [e for e in episodes if e["suite"] == suite]
        for cond in sorted(set(e["condition"] for e in suite_eps_all)):
            cond_eps = [e for e in suite_eps_all if e["condition"] == cond]
            results["per_suite_condition"][suite][cond] = _compute_displacement_stats(
                cond_eps, f"{suite}/{cond}"
            )

    _print_summary(results, "Pi0.5")
    return results


# ============================================================
# OFT Analysis
# ============================================================

def load_oft_episodes():
    """Load all OFT cross-task injection episodes.

    OFT data only has cos_to_baseline_b (cosine to destination task).
    We compute cos_to_baseline_a (cosine to source) from EEF velocity trajectories.
    """
    episodes = []
    traj_dir = Path(OFT_TRAJ_DIR)

    for suite in OFT_SUITES:
        suite_dir = traj_dir / suite
        if not suite_dir.exists():
            print(f"  SKIP (not found): {suite_dir}")
            continue

        # Load all baseline EEF trajectories
        baselines = {}
        for bf in sorted(suite_dir.glob("baseline_task_*.json")):
            try:
                with open(bf) as f:
                    bdata = json.load(f)
                task_idx = int(bf.stem.split("_")[-1])
                # Average EEF velocity across trials
                velocities = []
                for trial in bdata.get("trials", []):
                    scene = trial.get("scene", {})
                    eef = scene.get("robot_eef_trajectory", [])
                    if len(eef) > 1:
                        velocities.append(eef_velocity(eef))
                if velocities:
                    # Use first successful trial, or first trial
                    success_trials = [i for i, t in enumerate(bdata.get("trials", []))
                                      if t.get("success", False)]
                    if success_trials:
                        baselines[task_idx] = velocities[success_trials[0]]
                    else:
                        baselines[task_idx] = velocities[0]
            except Exception as e:
                print(f"  ERROR loading baseline {bf}: {e}")

        print(f"  {suite}: loaded {len(baselines)} baseline trajectories")

        # Load cross-task pair files
        for pf in sorted(suite_dir.glob("cross_task_pair_*.json")):
            try:
                with open(pf) as f:
                    pdata = json.load(f)
            except Exception as e:
                print(f"  ERROR loading {pf}: {e}")
                continue

            task_a = pdata["task_a"]
            task_b = pdata["task_b"]

            # Get baseline velocities
            vel_a = baselines.get(task_a)
            vel_b = baselines.get(task_b)

            # Also get EEF from the pair's own baselines as fallback
            for bl_key, bl_task in [("baseline_task_0", task_a), ("baseline_task_1", task_b)]:
                bl_data = pdata.get(bl_key, pdata.get(f"baseline_task_{bl_task}", {}))
                if bl_data and bl_task not in baselines:
                    eef = bl_data.get("scene", {}).get("robot_eef_trajectory", [])
                    if len(eef) > 1:
                        if bl_task == task_a and vel_a is None:
                            vel_a = eef_velocity(eef)
                        elif bl_task == task_b and vel_b is None:
                            vel_b = eef_velocity(eef)

            # Also get the in-pair baselines for comparison
            bl_a_scene = pdata.get("baseline_task_0", pdata.get(f"baseline_task_{task_a}", {}))
            bl_b_scene = pdata.get("baseline_task_1", pdata.get(f"baseline_task_{task_b}", {}))

            if bl_a_scene:
                eef_a = bl_a_scene.get("scene", {}).get("robot_eef_trajectory", [])
                if len(eef_a) > 1:
                    vel_a_pair = eef_velocity(eef_a)
                else:
                    vel_a_pair = vel_a
            else:
                vel_a_pair = vel_a

            if bl_b_scene:
                eef_b = bl_b_scene.get("scene", {}).get("robot_eef_trajectory", [])
                if len(eef_b) > 1:
                    vel_b_pair = eef_velocity(eef_b)
                else:
                    vel_b_pair = vel_b
            else:
                vel_b_pair = vel_b

            # Process injection: inject task_a activations into task_b environment
            inject_key = f"inject_{task_a}_into_{task_b}"
            inject_data = pdata.get(inject_key)
            if inject_data is None:
                # Try the old naming convention
                inject_key = "inject_0_into_1"
                inject_data = pdata.get(inject_key)

            if inject_data is None:
                continue

            for layer_name, layer_data in inject_data.items():
                if not isinstance(layer_data, dict):
                    continue

                # Get EEF trajectory of injection episode
                inj_scene = layer_data.get("scene", {})
                inj_eef = inj_scene.get("robot_eef_trajectory", [])

                if len(inj_eef) < 2:
                    continue

                inj_vel = eef_velocity(inj_eef)

                # Compute cosine similarities using EEF velocities
                # We need to handle different trajectory lengths
                if vel_a_pair is not None:
                    min_len = min(len(inj_vel), len(vel_a_pair))
                    cos_src = cosine_sim(inj_vel[:min_len], vel_a_pair[:min_len])
                else:
                    cos_src = None

                if vel_b_pair is not None:
                    min_len = min(len(inj_vel), len(vel_b_pair))
                    cos_dst = cosine_sim(inj_vel[:min_len], vel_b_pair[:min_len])
                else:
                    cos_dst = layer_data.get("cos_to_baseline_b")

                if cos_src is None or cos_dst is None:
                    continue

                # EEF displacement metrics
                eef_disp_to_src = None
                eef_disp_to_dst = None
                if vel_a_pair is not None:
                    min_len = min(len(inj_vel), len(vel_a_pair))
                    eef_disp_to_src = float(np.linalg.norm(
                        inj_vel[:min_len] - vel_a_pair[:min_len]
                    ))
                if vel_b_pair is not None:
                    min_len = min(len(inj_vel), len(vel_b_pair))
                    eef_disp_to_dst = float(np.linalg.norm(
                        inj_vel[:min_len] - vel_b_pair[:min_len]
                    ))

                episodes.append({
                    "suite": suite,
                    "src_task": task_a,
                    "dst_task": task_b,
                    "layer": layer_name,
                    "cos_to_src": cos_src,
                    "cos_to_dst": cos_dst,
                    "cos_to_baseline_b_stored": layer_data.get("cos_to_baseline_b"),
                    "success": layer_data.get("success", False),
                    "n_steps": layer_data.get("n_steps", 0),
                    "eef_disp_to_src": eef_disp_to_src,
                    "eef_disp_to_dst": eef_disp_to_dst,
                    "pair_file": pf.name,
                })

    return episodes


def analyze_oft():
    """Run displacement analysis on OFT cross-task injection data."""
    print("\n" + "=" * 60)
    print("OpenVLA-OFT CROSS-TASK DISPLACEMENT ANALYSIS")
    print("=" * 60)

    episodes = load_oft_episodes()
    print(f"\nTotal episodes loaded: {len(episodes)}")

    if not episodes:
        print("ERROR: No episodes found!")
        return None

    results = {
        "meta": {
            "model": "OpenVLA-OFT",
            "timestamp": datetime.now().isoformat(),
            "total_episodes": len(episodes),
            "threshold": THRESHOLD,
            "method": "EEF velocity cosine similarity",
        },
        "grand_summary": {},
        "by_suite": {},
        "by_layer": {},
        "per_suite_layer": {},
    }

    # Grand summary
    results["grand_summary"] = _compute_displacement_stats(episodes, "all")

    # By suite
    suites = set(e["suite"] for e in episodes)
    for suite in sorted(suites):
        suite_eps = [e for e in episodes if e["suite"] == suite]
        results["by_suite"][suite] = _compute_displacement_stats(suite_eps, suite)

    # By layer
    layers = set(e["layer"] for e in episodes)
    for layer in sorted(layers):
        layer_eps = [e for e in episodes if e["layer"] == layer]
        results["by_layer"][layer] = _compute_displacement_stats(layer_eps, layer)

    # Per suite per layer
    for suite in sorted(suites):
        results["per_suite_layer"][suite] = {}
        suite_eps = [e for e in episodes if e["suite"] == suite]
        for layer in sorted(set(e["layer"] for e in suite_eps)):
            layer_eps = [e for e in suite_eps if e["layer"] == layer]
            results["per_suite_layer"][suite][layer] = _compute_displacement_stats(
                layer_eps, f"{suite}/{layer}"
            )

    _print_summary(results, "OpenVLA-OFT")
    return results


# ============================================================
# Shared Statistics
# ============================================================

def _compute_displacement_stats(episodes, label=""):
    """Compute displacement statistics for a group of episodes."""
    if not episodes:
        return {"n": 0, "label": label}

    cos_srcs = [e["cos_to_src"] for e in episodes]
    cos_dsts = [e["cos_to_dst"] for e in episodes]

    n_source = sum(1 for e in episodes if classify_behavior(e["cos_to_src"], e["cos_to_dst"]) == "source")
    n_dest = sum(1 for e in episodes if classify_behavior(e["cos_to_src"], e["cos_to_dst"]) == "destination")
    n_ambig = sum(1 for e in episodes if classify_behavior(e["cos_to_src"], e["cos_to_dst"]) == "ambiguous")
    n_total = len(episodes)

    # Override rate: cos_to_src > cos_to_dst (regardless of threshold)
    n_override = sum(1 for e in episodes if e["cos_to_src"] > e["cos_to_dst"])

    # Success rate under injection
    n_success = sum(1 for e in episodes if e.get("success", False))

    src_rate, src_lo, src_hi = wilson_ci(n_source, n_total)
    dst_rate, dst_lo, dst_hi = wilson_ci(n_dest, n_total)
    override_rate, override_lo, override_hi = wilson_ci(n_override, n_total)
    success_rate, _, _ = wilson_ci(n_success, n_total)

    return {
        "label": label,
        "n": n_total,
        "source_behavior": {
            "count": n_source,
            "rate_pct": round(src_rate, 1),
            "ci_lo": round(src_lo, 1),
            "ci_hi": round(src_hi, 1),
        },
        "destination_behavior": {
            "count": n_dest,
            "rate_pct": round(dst_rate, 1),
            "ci_lo": round(dst_lo, 1),
            "ci_hi": round(dst_hi, 1),
        },
        "ambiguous": {
            "count": n_ambig,
            "rate_pct": round(n_ambig / n_total * 100, 1),
        },
        "override_rate": {
            "count": n_override,
            "rate_pct": round(override_rate, 1),
            "ci_lo": round(override_lo, 1),
            "ci_hi": round(override_hi, 1),
        },
        "success_rate_pct": round(success_rate, 1),
        "mean_cos_to_src": round(float(np.mean(cos_srcs)), 4),
        "mean_cos_to_dst": round(float(np.mean(cos_dsts)), 4),
        "std_cos_to_src": round(float(np.std(cos_srcs)), 4),
        "std_cos_to_dst": round(float(np.std(cos_dsts)), 4),
        "median_cos_to_src": round(float(np.median(cos_srcs)), 4),
        "median_cos_to_dst": round(float(np.median(cos_dsts)), 4),
        "cos_src_gt_dst": f"{n_override}/{n_total}",
    }


def _print_summary(results, model_name):
    """Print a human-readable summary."""
    print(f"\n{'='*60}")
    print(f"{model_name} DISPLACEMENT ANALYSIS RESULTS")
    print(f"{'='*60}")

    gs = results["grand_summary"]
    print(f"\n--- GRAND SUMMARY ---")
    print(f"Total episodes: {gs['n']}")
    print(f"Source behavior (cos_src - cos_dst > {THRESHOLD}): "
          f"{gs['source_behavior']['rate_pct']}% ({gs['source_behavior']['count']}/{gs['n']}) "
          f"[CI: {gs['source_behavior']['ci_lo']}-{gs['source_behavior']['ci_hi']}%]")
    print(f"Destination behavior: "
          f"{gs['destination_behavior']['rate_pct']}% ({gs['destination_behavior']['count']}/{gs['n']}) "
          f"[CI: {gs['destination_behavior']['ci_lo']}-{gs['destination_behavior']['ci_hi']}%]")
    print(f"Ambiguous: {gs['ambiguous']['rate_pct']}% ({gs['ambiguous']['count']}/{gs['n']})")
    print(f"Override rate (cos_src > cos_dst): "
          f"{gs['override_rate']['rate_pct']}% ({gs['override_rate']['count']}/{gs['n']}) "
          f"[CI: {gs['override_rate']['ci_lo']}-{gs['override_rate']['ci_hi']}%]")
    print(f"Dest task success rate: {gs['success_rate_pct']}%")
    print(f"Mean cos->src: {gs['mean_cos_to_src']:.4f} (+/- {gs['std_cos_to_src']:.4f})")
    print(f"Mean cos->dst: {gs['mean_cos_to_dst']:.4f} (+/- {gs['std_cos_to_dst']:.4f})")

    print(f"\n--- BY SUITE ---")
    print(f"{'Suite':<25} {'N':>5} {'Source%':>8} {'Dest%':>7} {'Ambig%':>7} {'Override%':>10} {'cos->src':>9} {'cos->dst':>9} {'Succ%':>6}")
    for suite, stats in sorted(results["by_suite"].items()):
        print(f"{suite:<25} {stats['n']:>5} "
              f"{stats['source_behavior']['rate_pct']:>7.1f}% "
              f"{stats['destination_behavior']['rate_pct']:>6.1f}% "
              f"{stats['ambiguous']['rate_pct']:>6.1f}% "
              f"{stats['override_rate']['rate_pct']:>9.1f}% "
              f"{stats['mean_cos_to_src']:>9.4f} "
              f"{stats['mean_cos_to_dst']:>9.4f} "
              f"{stats['success_rate_pct']:>5.1f}%")

    if "by_layer" in results:
        print(f"\n--- BY LAYER ---")
        print(f"{'Layer':<25} {'N':>5} {'Source%':>8} {'Dest%':>7} {'Ambig%':>7} {'Override%':>10} {'cos->src':>9} {'cos->dst':>9}")
        for layer, stats in sorted(results["by_layer"].items()):
            print(f"{layer:<25} {stats['n']:>5} "
                  f"{stats['source_behavior']['rate_pct']:>7.1f}% "
                  f"{stats['destination_behavior']['rate_pct']:>6.1f}% "
                  f"{stats['ambiguous']['rate_pct']:>6.1f}% "
                  f"{stats['override_rate']['rate_pct']:>9.1f}% "
                  f"{stats['mean_cos_to_src']:>9.4f} "
                  f"{stats['mean_cos_to_dst']:>9.4f}")

    if "by_prompt_type" in results:
        print(f"\n--- BY PROMPT TYPE ---")
        for pt, stats in sorted(results.get("by_prompt_type", {}).items()):
            print(f"{pt}: N={stats['n']}, "
                  f"Source={stats['source_behavior']['rate_pct']}%, "
                  f"Dest={stats['destination_behavior']['rate_pct']}%, "
                  f"Override={stats['override_rate']['rate_pct']}%, "
                  f"Success={stats['success_rate_pct']}%")

    if "by_pathway" in results:
        print(f"\n--- BY PATHWAY ---")
        for pw, stats in sorted(results.get("by_pathway", {}).items()):
            print(f"{pw}: N={stats['n']}, "
                  f"Source={stats['source_behavior']['rate_pct']}%, "
                  f"Dest={stats['destination_behavior']['rate_pct']}%, "
                  f"Override={stats['override_rate']['rate_pct']}%, "
                  f"cos->src={stats['mean_cos_to_src']:.4f}, "
                  f"cos->dst={stats['mean_cos_to_dst']:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pi0.5 analysis
    pi05_results = analyze_pi05()
    if pi05_results:
        out_path = OUTPUT_DIR / "displacement_analysis_pi05.json"
        with open(out_path, "w") as f:
            json.dump(pi05_results, f, indent=2)
        print(f"\nSaved Pi0.5 results: {out_path}")

    # OFT analysis
    oft_results = analyze_oft()
    if oft_results:
        out_path = OUTPUT_DIR / "displacement_analysis_oft.json"
        with open(out_path, "w") as f:
            json.dump(oft_results, f, indent=2)
        print(f"\nSaved OFT results: {out_path}")

    # Print comparison summary
    if pi05_results and oft_results:
        print("\n" + "=" * 60)
        print("CROSS-MODEL COMPARISON")
        print("=" * 60)
        print(f"\n{'Model':<15} {'N':>6} {'Source%':>8} {'Dest%':>7} {'Ambig%':>7} {'Override%':>10} {'cos->src':>9} {'cos->dst':>9}")
        for name, res in [("Pi0.5", pi05_results), ("OFT", oft_results)]:
            gs = res["grand_summary"]
            print(f"{name:<15} {gs['n']:>6} "
                  f"{gs['source_behavior']['rate_pct']:>7.1f}% "
                  f"{gs['destination_behavior']['rate_pct']:>6.1f}% "
                  f"{gs['ambiguous']['rate_pct']:>6.1f}% "
                  f"{gs['override_rate']['rate_pct']:>9.1f}% "
                  f"{gs['mean_cos_to_src']:>9.4f} "
                  f"{gs['mean_cos_to_dst']:>9.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
