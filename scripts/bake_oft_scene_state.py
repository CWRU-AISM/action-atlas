#!/usr/bin/env python3
"""
Bake OFT trajectory collection results into scene state JSON for Action Atlas.

Reads results.json from OFT trajectory collection experiments and produces
compact baked JSON files compatible with the existing Scene State visualization.

Usage:
    python scripts/bake_oft_scene_state.py
    python scripts/bake_oft_scene_state.py --input-dir rollouts/oft_trajectories/
    python scripts/bake_oft_scene_state.py --output-dir action_atlas/data/oft_scene_state/
"""

import argparse
import json
from pathlib import Path

MAX_TRAJ_POINTS = 100


def subsample(points, max_n=MAX_TRAJ_POINTS):
    # Subsample a list of points to at most max_n points, preserving first and last
    if not points or len(points) <= max_n:
        return points
    n = len(points)
    indices = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    return [points[i] for i in indices]


def extract_condition_data(cond_data):
    # Extract trajectory data from a condition (baseline or injection)
    scene = cond_data.get("scene")
    if scene is None:
        return None

    result = {
        "n_steps": scene.get("n_steps"),
        "success": cond_data.get("success"),
    }

    robot_traj = scene.get("robot_eef_trajectory")
    if robot_traj is not None:
        result["robot_eef_trajectory"] = subsample(robot_traj)

    obj_trajs = scene.get("object_trajectories")
    if obj_trajs is not None:
        result["object_trajectories"] = {
            obj_name: subsample(obj_traj)
            for obj_name, obj_traj in obj_trajs.items()
        }

    obj_disps = scene.get("object_displacements")
    if obj_disps is not None:
        result["object_displacements"] = obj_disps

    return result


def process_baseline_results(results, suite_name):
    # Process baseline results into baked format
    baseline = results.get("baseline", {})
    if not baseline:
        return None

    tasks = []
    for task_key in sorted(baseline.keys(), key=lambda k: int(k.split("_")[1])):
        task_data = baseline[task_key]
        task_id = int(task_key.split("_")[1])

        trials = []
        for trial in task_data.get("trials", []):
            cond = extract_condition_data(trial)
            if cond:
                cond["trial"] = trial.get("trial", 0)
                trials.append(cond)

        tasks.append({
            "task_id": task_id,
            "task_description": task_data.get("task_description", ""),
            "success_rate": task_data.get("success_rate", 0),
            "trials": trials,
        })

    return {"suite": suite_name, "type": "baseline", "tasks": tasks}


def process_cross_task_results(results, suite_name):
    # Process cross-task results into baked format (compatible with Pi0.5 scene state)
    cross_task = results.get("cross_task", {})
    if not cross_task:
        return None

    pairs = []
    for pair_key in sorted(cross_task.keys()):
        pair_data = cross_task[pair_key]

        pair_out = {
            "key": pair_key,
            "task_a": pair_data.get("task_a"),
            "task_b": pair_data.get("task_b"),
            "prompt_a": pair_data.get("prompt_a", ""),
            "prompt_b": pair_data.get("prompt_b", ""),
            "conditions": {},
        }

        # Process baselines
        for key in [f"baseline_task_{pair_data.get('task_a')}", f"baseline_task_{pair_data.get('task_b')}"]:
            if key in pair_data:
                cond = extract_condition_data(pair_data[key])
                if cond:
                    pair_out["conditions"][key] = cond

        # Process injection conditions
        for key, value in pair_data.items():
            if key.startswith("inject_") and isinstance(value, dict):
                for sub_key, sub_data in value.items():
                    if isinstance(sub_data, dict) and "scene" in sub_data:
                        flat_key = f"{key}/{sub_key}"
                        cond = extract_condition_data(sub_data)
                        if cond:
                            cond["cos_to_baseline"] = sub_data.get("cos_to_baseline_b")
                            pair_out["conditions"][flat_key] = cond

        pairs.append(pair_out)

    return {"suite": suite_name, "type": "cross_task", "pairs": pairs}


def process_null_injection_results(results, suite_name):
    # Process null injection results into baked format
    null_inj = results.get("null_injection", {})
    if not null_inj:
        return None

    tasks = []
    for task_key in sorted(null_inj.keys(), key=lambda k: int(k.split("_")[1])):
        task_data = null_inj[task_key]
        task_id = int(task_key.split("_")[1])

        conditions = {}
        # Baseline
        if "baseline" in task_data:
            cond = extract_condition_data(task_data["baseline"])
            if cond:
                conditions["baseline"] = cond

        # Injections
        for inj_key, inj_data in task_data.get("injections", {}).items():
            if isinstance(inj_data, dict):
                cond = extract_condition_data(inj_data)
                if cond:
                    conditions[f"null_{inj_key}"] = cond

        tasks.append({
            "task_id": task_id,
            "task_description": task_data.get("task_description", ""),
            "conditions": conditions,
        })

    return {"suite": suite_name, "type": "null_injection", "tasks": tasks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="rollouts/oft_trajectories/",
                        help="Directory containing OFT trajectory results")
    parser.add_argument("--output-dir", default="action_atlas/data/oft_scene_state/",
                        help="Output directory for baked JSON files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # Find all results.json files
    result_files = sorted(input_dir.glob("*/results.json"))
    if not result_files:
        result_files = sorted(input_dir.glob("results.json"))

    if not result_files:
        print("No results.json files found!")
        return

    # Process each suite
    for result_file in result_files:
        print(f"\n{'='*60}")
        print(f"Processing: {result_file}")

        with open(result_file) as f:
            results = json.load(f)

        # Merge per-pair cross_task_pair_*.json files into results
        # (these are saved incrementally and may not be in results.json)
        suite_dir = result_file.parent
        pair_files = sorted(suite_dir.glob("cross_task_pair_*.json"))
        if pair_files:
            if "cross_task" not in results:
                results["cross_task"] = {}
            for pf in pair_files:
                pair_key = pf.stem.replace("cross_task_", "")  # e.g. "pair_0_2"
                if pair_key not in results["cross_task"]:
                    with open(pf) as f:
                        results["cross_task"][pair_key] = json.load(f)
            print(f"  Merged {len(pair_files)} per-pair files -> {len(results['cross_task'])} total cross-task pairs")

        # Merge per-task baseline_task_*.json and null_injection_task_*.json
        for prefix, key in [("baseline_task_", "baseline"), ("null_injection_task_", "null_injection")]:
            task_files = sorted(suite_dir.glob(f"{prefix}*.json"))
            if task_files and key not in results:
                results[key] = {}
                for tf in task_files:
                    task_id = tf.stem.replace(prefix.rstrip("_"), "").lstrip("_")
                    task_key = f"task_{task_id}"
                    with open(tf) as f:
                        results[key][task_key] = json.load(f)
                print(f"  Loaded {len(task_files)} {key} task files")

        suite_name = results.get("suite", result_file.parent.name.split("_20")[0])

        # Process baseline
        baseline_data = process_baseline_results(results, suite_name)
        if baseline_data:
            out_path = output_dir / f"{suite_name}_baseline.json"
            with open(out_path, "w") as f:
                json.dump(baseline_data, f, separators=(",", ":"))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            n_tasks = len(baseline_data["tasks"])
            print(f"  Baseline: {n_tasks} tasks -> {out_path} ({size_mb:.2f} MB)")

        # Process cross-task (compatible with existing Pi0.5 scene state format)
        cross_data = process_cross_task_results(results, suite_name)
        if cross_data:
            out_path = output_dir / f"{suite_name}_cross_task.json"
            with open(out_path, "w") as f:
                json.dump(cross_data, f, separators=(",", ":"))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            n_pairs = len(cross_data["pairs"])
            n_cond = sum(len(p["conditions"]) for p in cross_data["pairs"])
            print(f"  Cross-task: {n_pairs} pairs, {n_cond} conditions -> {out_path} ({size_mb:.2f} MB)")

        # Process null injection
        null_data = process_null_injection_results(results, suite_name)
        if null_data:
            out_path = output_dir / f"{suite_name}_null_injection.json"
            with open(out_path, "w") as f:
                json.dump(null_data, f, separators=(",", ":"))
            size_mb = out_path.stat().st_size / (1024 * 1024)
            n_tasks = len(null_data["tasks"])
            print(f"  Null injection: {n_tasks} tasks -> {out_path} ({size_mb:.2f} MB)")

    print(f"\n{'='*60}")
    print(f"Done! Baked data in: {output_dir}")


if __name__ == "__main__":
    main()
