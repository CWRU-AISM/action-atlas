#!/usr/bin/env python3
"""
Bake VP (Vision Perturbation) experiment results into summary JSONs for Action Atlas.

Parses:
- Pi0.5: video index entries with success metadata + OFT-style results
- OFT: 24 individual results.json files from visual_perturbation experiments

Outputs:
- data/vp_results_pi05.json
- data/vp_results_openvla.json
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
VIDEO_DIR = SCRIPT_DIR.parent / "data" / "videos"
PROJECT_ROOT = SCRIPT_DIR.parent.parent  


def parse_perturbation_from_filename(fname: str) -> str:
    # Strip extension and _combined suffix
    base = fname.replace(".mp4", "").replace("_combined", "")
    # OFT format: task0_blur_light
    m = re.match(r"task\d+_(.*)", base)
    if m:
        return m.group(1)
    # Pi0.5 format: libero_goal_task0_s42_blur_heavy_after_100
    m = re.match(r"libero_\w+_task\d+_s\d+_(.*)", base)
    if m:
        return m.group(1)
    return base


def bake_openvla():
    """Bake OFT VP results from 24 individual results.json files."""
    oft_base = PROJECT_ROOT / "videos" / "openvla_rollouts" / "openvla_oft"
    results_files = list(oft_base.rglob("visual_perturbations/results.json"))
    print(f"[OFT] Found {len(results_files)} VP results files")

    # Also load the existing summary for cross-reference
    summary_path = SCRIPT_DIR.parent / "openvla_oft_vision_perturbation_summary.json"
    existing_summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            existing_summary = json.load(f)
        print(f"[OFT] Loaded existing summary: {existing_summary.get('total_episodes')} episodes")

    # Load OFT video index for video path matching
    oft_index_path = VIDEO_DIR / "openvla" / "index.json"
    oft_videos_by_key = {}  # (suite, task, perturbation) -> video_path
    if oft_index_path.exists():
        with open(oft_index_path) as f:
            idx = json.load(f)
        for v in idx.get("videos", []):
            if v.get("experiment_type") != "vision_perturbation":
                continue
            fname = v["path"].split("/")[-1]
            ptype = parse_perturbation_from_filename(fname)
            suite = v.get("suite", "")
            # Extract task from filename: task0_xxx
            tm = re.search(r"task(\d+)", fname)
            task = int(tm.group(1)) if tm else -1
            key = (suite, task, ptype)
            oft_videos_by_key[key] = v["path"]
        print(f"[OFT] Indexed {len(oft_videos_by_key)} VP video paths")

    # Parse all results files
    # Structure: results_file parent dir name tells us the suite
    all_episodes = []  # list of (suite, task_idx, condition, success, n_steps)
    for rf in sorted(results_files):
        # Path: .../openvla_oft/{suite}/{timestamp}/visual_perturbations/results.json
        parts = rf.parts
        # Find suite from path
        suite = None
        for i, p in enumerate(parts):
            if p == "openvla_oft" and i + 1 < len(parts):
                suite = parts[i + 1]
                break
        if not suite:
            print(f"  Warning: can't determine suite from {rf}")
            continue

        with open(rf) as f:
            data = json.load(f)
        conditions = data.get("conditions", {})
        for task_key, task_conditions in conditions.items():
            # task_key: "task_0", "task_1", etc.
            tm = re.match(r"task_?(\d+)", task_key)
            if not tm:
                continue
            task_idx = int(tm.group(1))
            for condition, result in task_conditions.items():
                all_episodes.append({
                    "suite": suite,
                    "task": task_idx,
                    "condition": condition,
                    "success": result.get("success", False),
                    "n_steps": result.get("n_steps", 0),
                })

    print(f"[OFT] Parsed {len(all_episodes)} total episodes from results files")

    # Aggregate by suite -> task -> condition
    suites_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ep in all_episodes:
        suites_data[ep["suite"]][ep["task"]][ep["condition"]].append(ep)

    # Build output structure
    output = {
        "model": "openvla",
        "total_episodes": len(all_episodes),
        "suites": {},
        "perturbation_types": sorted(set(ep["condition"] for ep in all_episodes)),
    }

    for suite in sorted(suites_data.keys()):
        suite_tasks = suites_data[suite]
        suite_output = {"tasks": {}, "overall": {}}

        # Per-condition overall stats for this suite
        suite_by_condition = defaultdict(list)

        for task_idx in sorted(suite_tasks.keys()):
            task_conditions = suite_tasks[task_idx]
            task_output = {}

            # Get baseline stats for delta computation
            baseline_eps = task_conditions.get("baseline", [])
            baseline_sr = (
                100 * sum(1 for e in baseline_eps if e["success"]) / len(baseline_eps)
                if baseline_eps else None
            )
            baseline_steps = (
                sum(e["n_steps"] for e in baseline_eps) / len(baseline_eps)
                if baseline_eps else None
            )

            for condition in sorted(task_conditions.keys()):
                eps = task_conditions[condition]
                n = len(eps)
                successes = sum(1 for e in eps if e["success"])
                sr = 100 * successes / n if n > 0 else 0
                avg_steps = sum(e["n_steps"] for e in eps) / n if n > 0 else 0

                entry = {
                    "success_rate": round(sr, 1),
                    "avg_n_steps": round(avg_steps, 1),
                    "n_episodes": n,
                }

                # Add video paths
                vkey = (suite, task_idx, condition)
                if vkey in oft_videos_by_key:
                    entry["video_path"] = oft_videos_by_key[vkey]

                # Delta from baseline
                if condition != "baseline" and baseline_sr is not None:
                    entry["delta_success_rate"] = round(sr - baseline_sr, 1)
                if condition != "baseline" and baseline_steps is not None:
                    entry["delta_n_steps"] = round(avg_steps - baseline_steps, 1)

                task_output[condition] = entry
                suite_by_condition[condition].extend(eps)

            suite_output["tasks"][str(task_idx)] = task_output

        # Suite overall stats
        baseline_overall = suite_by_condition.get("baseline", [])
        baseline_overall_sr = (
            100 * sum(1 for e in baseline_overall if e["success"]) / len(baseline_overall)
            if baseline_overall else None
        )
        for condition in sorted(suite_by_condition.keys()):
            eps = suite_by_condition[condition]
            n = len(eps)
            successes = sum(1 for e in eps if e["success"])
            sr = 100 * successes / n if n > 0 else 0
            overall_entry = {
                "success_rate": round(sr, 1),
                "n_episodes": n,
            }
            if condition != "baseline" and baseline_overall_sr is not None:
                overall_entry["delta_success_rate"] = round(sr - baseline_overall_sr, 1)
            suite_output["overall"][condition] = overall_entry

        output["suites"][suite] = suite_output

    return output


def bake_pi05():
    """Bake Pi0.5 VP results from video index metadata."""
    pi05_index_path = VIDEO_DIR / "pi05" / "index.json"
    if not pi05_index_path.exists():
        print(f"[Pi0.5] Video index not found at {pi05_index_path}")
        return None

    with open(pi05_index_path) as f:
        idx = json.load(f)

    # Collect all VP entries
    vp_entries = []
    for v in idx.get("videos", []):
        if v.get("experiment_type") != "vision_perturbation":
            continue
        fname = v["path"].split("/")[-1]
        ptype = parse_perturbation_from_filename(fname)

        suite = v.get("suite", "")
        task = v.get("task")
        seed = v.get("seed")
        success = v.get("success")

        vp_entries.append({
            "path": v["path"],
            "suite": suite,
            "task": task,
            "seed": seed,
            "success": success,
            "perturbation": ptype,
            "temporal_window": v.get("temporal_window"),
        })

    print(f"[Pi0.5] Found {len(vp_entries)} VP entries in video index")
    print(f"[Pi0.5] Entries with success field: {sum(1 for e in vp_entries if e['success'] is not None)}")

    # Only use entries that have success metadata
    vp_with_success = [e for e in vp_entries if e["success"] is not None]
    print(f"[Pi0.5] Using {len(vp_with_success)} entries with success data")

    # Aggregate by suite -> task -> perturbation
    suites_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for ep in vp_with_success:
        suites_data[ep["suite"]][ep["task"]][ep["perturbation"]].append(ep)

    # Also build video path lookup: (suite, task, perturbation) -> path
    video_paths = {}
    for ep in vp_entries:
        key = (ep["suite"], ep["task"], ep["perturbation"])
        if key not in video_paths:
            video_paths[key] = ep["path"]

    # Build output
    output = {
        "model": "pi05",
        "total_episodes": len(vp_with_success),
        "suites": {},
        "perturbation_types": sorted(set(ep["perturbation"] for ep in vp_with_success)),
    }

    for suite in sorted(suites_data.keys()):
        suite_tasks = suites_data[suite]
        suite_output = {"tasks": {}, "overall": {}}
        suite_by_condition = defaultdict(list)

        for task_idx in sorted(suite_tasks.keys()):
            task_conditions = suite_tasks[task_idx]
            task_output = {}

            baseline_eps = task_conditions.get("baseline", [])
            baseline_sr = (
                100 * sum(1 for e in baseline_eps if e["success"]) / len(baseline_eps)
                if baseline_eps else None
            )

            for condition in sorted(task_conditions.keys()):
                eps = task_conditions[condition]
                n = len(eps)
                successes = sum(1 for e in eps if e["success"])
                sr = 100 * successes / n if n > 0 else 0

                entry = {
                    "success_rate": round(sr, 1),
                    "n_episodes": n,
                }

                # Video path
                vkey = (suite, task_idx, condition)
                if vkey in video_paths:
                    entry["video_path"] = video_paths[vkey]

                # Delta from baseline
                if condition != "baseline" and baseline_sr is not None:
                    entry["delta_success_rate"] = round(sr - baseline_sr, 1)

                task_output[condition] = entry
                suite_by_condition[condition].extend(eps)

            suite_output["tasks"][str(task_idx)] = task_output

        # Suite overall
        baseline_overall = suite_by_condition.get("baseline", [])
        baseline_overall_sr = (
            100 * sum(1 for e in baseline_overall if e["success"]) / len(baseline_overall)
            if baseline_overall else None
        )
        for condition in sorted(suite_by_condition.keys()):
            eps = suite_by_condition[condition]
            n = len(eps)
            successes = sum(1 for e in eps if e["success"])
            sr = 100 * successes / n if n > 0 else 0
            overall_entry = {
                "success_rate": round(sr, 1),
                "n_episodes": n,
            }
            if condition != "baseline" and baseline_overall_sr is not None:
                overall_entry["delta_success_rate"] = round(sr - baseline_overall_sr, 1)
            suite_output["overall"][condition] = overall_entry

        output["suites"][suite] = suite_output

    return output


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Bake OFT
    print("=" * 60)
    print("Baking OFT VP results...")
    print("=" * 60)
    oft_data = bake_openvla()
    oft_path = DATA_DIR / "vp_results_openvla.json"
    with open(oft_path, "w") as f:
        json.dump(oft_data, f, indent=2)
    print(f"[OFT] Written to {oft_path}")
    print(f"[OFT] {oft_data['total_episodes']} episodes, {len(oft_data['perturbation_types'])} perturbation types")
    print(f"[OFT] Suites: {list(oft_data['suites'].keys())}")
    for s, sd in oft_data["suites"].items():
        n_tasks = len(sd["tasks"])
        n_conditions = len(sd["overall"])
        print(f"  {s}: {n_tasks} tasks, {n_conditions} conditions")

    # Bake Pi0.5
    print()
    print("=" * 60)
    print("Baking Pi0.5 VP results...")
    print("=" * 60)
    pi05_data = bake_pi05()
    if pi05_data:
        pi05_path = DATA_DIR / "vp_results_pi05.json"
        with open(pi05_path, "w") as f:
            json.dump(pi05_data, f, indent=2)
        print(f"[Pi0.5] Written to {pi05_path}")
        print(f"[Pi0.5] {pi05_data['total_episodes']} episodes, {len(pi05_data['perturbation_types'])} perturbation types")
        print(f"[Pi0.5] Suites: {list(pi05_data['suites'].keys())}")
        for s, sd in pi05_data["suites"].items():
            n_tasks = len(sd["tasks"])
            n_conditions = len(sd["overall"])
            print(f"  {s}: {n_tasks} tasks, {n_conditions} conditions")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
