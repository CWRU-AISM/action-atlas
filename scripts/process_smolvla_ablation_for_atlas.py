#!/usr/bin/env python3
"""
Process SmolVLA concept ablation results into Action Atlas format.

Reads all result JSONs from rollouts/smolvla/concept_ablation/results/
and produces:
1. Updated experiment_results_smolvla.json with concept_ablation section
2. Summary statistics for the findings tab
3. Scene state data from trajectories

Usage:
    python scripts/process_smolvla_ablation_for_atlas.py [--dry-run]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("rollouts/smolvla/concept_ablation/results")
TRAJ_DIR = Path("rollouts/smolvla/concept_ablation/trajectories")
ATLAS_DATA = Path("action_atlas/data/experiment_results_smolvla.json")
OUTPUT_DIR = Path("action_atlas/data")


def load_all_results():
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if "partial" in f.name:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
                d["_filename"] = f.name
                results.append(d)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  WARN: {f.name}: {e}")
    return results


def compute_ablation_summary(results):
    total_pairs = 0
    zero_effect = 0
    destructive = 0  # >50pp drop
    kill_switches = []  # 100pp drop
    by_component = defaultdict(lambda: {"pairs": 0, "zero": 0, "destructive": 0})
    by_suite = defaultdict(lambda: {"pairs": 0, "zero": 0, "destructive": 0})
    by_layer = defaultdict(lambda: {"pairs": 0, "zero": 0, "destructive": 0})

    for r in results:
        if "ablation" not in r:
            continue
        comp = r.get("component", "unknown")
        suite = r.get("suite", "unknown")
        layer = r.get("layer", -1)
        baseline = r.get("baseline_overall", 0)

        ablation_data = r["ablation"]
        items = ablation_data.items() if isinstance(ablation_data, dict) else []

        for concept, entry in items:
            if isinstance(entry, dict):
                rate = entry.get("overall_rate", 0)
                delta_pp = entry.get("overall_delta", 0) * 100  # convert to pp
            else:
                continue

            total_pairs += 1
            by_component[comp]["pairs"] += 1
            by_suite[suite]["pairs"] += 1
            by_layer[layer]["pairs"] += 1

            if abs(delta_pp) < 1:  # <1pp change
                zero_effect += 1
                by_component[comp]["zero"] += 1
                by_suite[suite]["zero"] += 1
                by_layer[layer]["zero"] += 1

            if delta_pp <= -50:
                destructive += 1
                by_component[comp]["destructive"] += 1
                by_suite[suite]["destructive"] += 1
                by_layer[layer]["destructive"] += 1

            if delta_pp <= -90:
                kill_switches.append({
                    "component": comp,
                    "suite": suite,
                    "layer": layer,
                    "concept": concept,
                    "baseline": round(baseline * 100, 1),
                    "ablated": round(rate * 100, 1),
                    "delta_pp": round(delta_pp, 1),
                })

    return {
        "total_pairs": total_pairs,
        "zero_effect_pct": round(100 * zero_effect / max(total_pairs, 1), 1),
        "destructive_pct": round(100 * destructive / max(total_pairs, 1), 1),
        "kill_switches": sorted(kill_switches, key=lambda x: x["delta_pp"])[:20],
        "by_component": dict(by_component),
        "by_suite": dict(by_suite),
        "by_layer": {str(k): v for k, v in sorted(by_layer.items())},
    }


def compute_steering_summary(results):
    entries = []
    for r in results:
        if "steering" not in r:
            continue
        comp = r.get("component", "unknown")
        suite = r.get("suite", "unknown")
        layer = r.get("layer", -1)
        baseline = r.get("baseline_overall", 0)

        steering_data = r["steering"]
        items = steering_data.items() if isinstance(steering_data, dict) else []

        for concept, concept_data in items:
            if not isinstance(concept_data, dict):
                continue
            multipliers = concept_data.get("multipliers", concept_data)
            if isinstance(multipliers, dict):
                for mult_str, mult_data in multipliers.items():
                    if isinstance(mult_data, dict):
                        rate = mult_data.get("overall_rate", 0)
                        delta_pp = mult_data.get("overall_delta", 0) * 100
                    else:
                        continue
                    entries.append({
                        "component": comp,
                        "suite": suite,
                        "layer": layer,
                        "concept": concept,
                        "multiplier": float(mult_str),
                        "success_rate": round(rate * 100, 1),
                        "delta_pp": round(delta_pp, 1),
                    })

    if not entries:
        return {"count": 0}

    if not entries:
        return {"count": 0}

    deltas = [e["delta_pp"] for e in entries]
    return {
        "count": len(entries),
        "mean_delta_pp": round(float(np.mean(deltas)), 1),
        "min_delta_pp": round(float(min(deltas)), 1),
        "max_delta_pp": round(float(max(deltas)), 1),
        "suppression_entries": sum(1 for d in deltas if d < -10),
        "enhancement_entries": sum(1 for d in deltas if d > 10),
    }


def compute_ftf_summary(results):
    entries = []
    for r in results:
        if "fraction_to_failure" not in r:
            continue
        comp = r.get("component", "unknown")
        suite = r.get("suite", "unknown")
        layer = r.get("layer", -1)

        ftf_data = r["fraction_to_failure"]
        items = ftf_data.items() if isinstance(ftf_data, dict) else []

        for concept, concept_data in items:
            if isinstance(concept_data, dict):
                fractions = concept_data.get("fractions", {})
                for frac, frac_data in fractions.items():
                    entries.append({
                        "component": comp,
                        "suite": suite,
                        "layer": layer,
                        "concept": concept,
                        "fraction": int(frac),
                    })

    return {"count": len(entries)}


def compute_temporal_summary(results):
    entries = []
    for r in results:
        if "temporal" not in r:
            continue
        comp = r.get("component", "unknown")
        suite = r.get("suite", "unknown")
        layer = r.get("layer", -1)

        temporal_data = r["temporal"]
        items = temporal_data.items() if isinstance(temporal_data, dict) else []

        for concept, concept_data in items:
            if isinstance(concept_data, dict):
                for window, window_data in concept_data.items():
                    if isinstance(window_data, dict):
                        entries.append({
                            "component": comp,
                            "suite": suite,
                            "layer": layer,
                            "concept": concept,
                            "window": window,
                        })

    return {"count": len(entries)}


def count_trajectories():
    # Count saved trajectory and video files
    n_traj = sum(1 for _ in TRAJ_DIR.rglob("*.npz")) if TRAJ_DIR.exists() else 0
    n_video = sum(1 for _ in TRAJ_DIR.rglob("*.mp4")) if TRAJ_DIR.exists() else 0
    return n_traj, n_video


def update_atlas_json(ablation_summary, steering_summary, ftf_summary, temporal_summary, n_traj, n_video, dry_run=False):
    # Update experiment_results_smolvla.json with concept ablation data
    with open(ATLAS_DATA) as f:
        atlas = json.load(f)

    atlas["concept_ablation"] = {
        "description": "Concept ablation experiments across dual pathways (VLM + Expert)",
        "total_pairs": ablation_summary["total_pairs"],
        "zero_effect_pct": ablation_summary["zero_effect_pct"],
        "destructive_pct": ablation_summary["destructive_pct"],
        "kill_switches": ablation_summary["kill_switches"],
        "by_component": ablation_summary["by_component"],
        "by_suite": ablation_summary["by_suite"],
        "trajectories": n_traj,
        "videos": n_video,
    }

    atlas["steering"] = steering_summary
    atlas["fraction_to_failure"] = ftf_summary
    atlas["temporal_ablation"] = temporal_summary
    atlas["timestamp"] = __import__("datetime").datetime.now().isoformat()

    if dry_run:
        print(f"\n[DRY RUN] Would write to {ATLAS_DATA}")
        print(f"  concept_ablation: {ablation_summary['total_pairs']} pairs")
        print(f"  steering: {steering_summary.get('count', 0)} entries")
        print(f"  FTF: {ftf_summary['count']} entries")
        print(f"  temporal: {temporal_summary['count']} entries")
        print(f"  trajectories: {n_traj}, videos: {n_video}")
    else:
        with open(ATLAS_DATA, "w") as f:
            json.dump(atlas, f, indent=2, default=str)
        print(f"\nUpdated {ATLAS_DATA}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=== SmolVLA Concept Ablation → Action Atlas ===")
    results = load_all_results()
    print(f"Loaded {len(results)} result files")

    modes = defaultdict(int)
    for r in results:
        for k in ["ablation", "fraction_to_failure", "temporal", "steering"]:
            if k in r:
                modes[k] += 1
    print(f"  ablation: {modes['ablation']}, FTF: {modes['fraction_to_failure']}, "
          f"temporal: {modes['temporal']}, steering: {modes['steering']}")

    ablation_summary = compute_ablation_summary(results)
    print(f"\nAblation: {ablation_summary['total_pairs']} pairs, "
          f"{ablation_summary['zero_effect_pct']}% zero, "
          f"{ablation_summary['destructive_pct']}% destructive, "
          f"{len(ablation_summary['kill_switches'])} kill-switches")

    steering_summary = compute_steering_summary(results)
    print(f"Steering: {steering_summary.get('count', 0)} entries")

    ftf_summary = compute_ftf_summary(results)
    print(f"FTF: {ftf_summary['count']} entries")

    temporal_summary = compute_temporal_summary(results)
    print(f"Temporal: {temporal_summary['count']} entries")

    n_traj, n_video = count_trajectories()
    print(f"Trajectories: {n_traj}, Videos: {n_video}")

    by_comp = ablation_summary["by_component"]
    for comp, stats in by_comp.items():
        z_pct = round(100 * stats["zero"] / max(stats["pairs"], 1), 1)
        d_pct = round(100 * stats["destructive"] / max(stats["pairs"], 1), 1)
        print(f"  {comp}: {stats['pairs']} pairs, {z_pct}% zero, {d_pct}% destructive")

    update_atlas_json(ablation_summary, steering_summary, ftf_summary,
                      temporal_summary, n_traj, n_video, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
