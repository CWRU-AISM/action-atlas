#!/usr/bin/env python3
"""
Bake merged_results.json files into scene state JSON for the concept viz app.

Reads the full merged_results.json for goal and spatial cross-task injection
experiments, extracts trajectory data for both baseline and injection conditions,
and writes compact baked JSON files.
"""

import json
from pathlib import Path

# --- Config ---
SOURCES = {
    "goal": "/data/robotsteering/pi05_rollouts/cross_task_goal/comprehensive_cross_task_libero_goal_seed123_20260127_195312/merged_results.json",
    "spatial": "/data/robotsteering/pi05_rollouts/cross_task_spatial/comprehensive_cross_task_libero_spatial_20260126_225619/merged_results.json",
    # groot Feb 13-14 Pi0.5 cross-task injection (45 pairs each, 8 conditions × 2 directions)
    "object": "/data/robotsteering/pi05_rollouts/groot_feb13_pi05/object/results.json",
    "10": "/data/robotsteering/pi05_rollouts/groot_feb13_pi05/10/results.json",
}

OUTPUT_DIR = Path(__file__).parent.parent / 'action_atlas' / 'data' / 'pi05_scene_state'

MAX_TRAJ_POINTS = 100


def subsample(points, max_n=MAX_TRAJ_POINTS):
    # Subsample a list of points to at most max_n points, preserving first and last
    if not points or len(points) <= max_n:
        return points
    n = len(points)
    indices = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    return [points[i] for i in indices]


def extract_condition_data(cond_data):
    """
    Extract trajectory data from a condition (baseline or injection sub-condition).
    
    Returns None if no scene data is available.
    """
    scene = cond_data.get("scene")
    if scene is None:
        return None

    result = {
        "n_steps": scene.get("n_steps"),
        "success": cond_data.get("success"),
    }

    # Robot EEF trajectory
    robot_traj = scene.get("robot_eef_trajectory")
    if robot_traj is not None:
        result["robot_eef_trajectory"] = subsample(robot_traj)
    
    # Object trajectories - subsample each object's trajectory independently
    obj_trajs = scene.get("object_trajectories")
    if obj_trajs is not None:
        result["object_trajectories"] = {
            obj_name: subsample(obj_traj)
            for obj_name, obj_traj in obj_trajs.items()
        }

    # Object displacements (already scalar per object, no subsampling needed)
    obj_disps = scene.get("object_displacements")
    if obj_disps is not None:
        result["object_displacements"] = obj_disps

    return result


def process_file(name, src_path):
    # Process one merged_results.json file and return baked data + stats
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"Source: {src_path}")

    with open(src_path) as f:
        raw = json.load(f)

    # Handle both merged_results.json (has "results" key) and raw results.json (pairs at top level)
    results = raw.get("results", raw) if isinstance(raw, dict) else raw
    pairs_out = []

    total_conditions = 0
    conditions_with_traj = 0

    for pair_key in sorted(results.keys(), key=lambda k: tuple(int(x) for x in k.replace("pair_", "").split("_"))):
        pair_data = results[pair_key]

        pair_out = {
            "key": pair_key,
            "task_a": pair_data["task_a"],
            "task_b": pair_data["task_b"],
            "prompt_a": pair_data["prompt_a"],
            "prompt_b": pair_data["prompt_b"],
            "conditions": {},
        }

        # Dynamically discover all keys in this pair
        for key, value in pair_data.items():
            if not isinstance(value, dict):
                continue

            # Skip metadata keys
            if key in ("task_a", "task_b", "prompt_a", "prompt_b"):
                continue
            # Skip object list keys like task_2_objects
            if key.endswith("_objects"):
                continue

            if key.startswith("baseline_"):
                # Baseline condition - has scene directly
                cond = extract_condition_data(value)
                total_conditions += 1
                if cond is not None:
                    pair_out["conditions"][key] = cond
                    if cond.get("robot_eef_trajectory"):
                        conditions_with_traj += 1

            elif key.startswith("inject_"):
                # Injection block - contains sub-conditions
                for sub_key, sub_data in value.items():
                    if not isinstance(sub_data, dict):
                        continue
                    flat_key = f"{key}/{sub_key}"
                    total_conditions += 1
                    cond = extract_condition_data(sub_data)
                    if cond is not None:
                        pair_out["conditions"][flat_key] = cond
                        if cond.get("robot_eef_trajectory"):
                            conditions_with_traj += 1

        pairs_out.append(pair_out)

    output = {"pairs": pairs_out}

    # --- Write output ---
    out_path = OUTPUT_DIR / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = out_path.stat().st_size / (1024 * 1024)

    print(f"\nStats for {name}:")
    print(f"  Pairs: {len(pairs_out)}")
    print(f"  Total conditions: {total_conditions}")
    print(f"  Conditions with trajectories: {conditions_with_traj}")
    print(f"  Output: {out_path}")
    print(f"  File size: {file_size_mb:.2f} MB")

    # Per-pair breakdown
    for p in pairs_out[:3]:
        n_cond = len(p["conditions"])
        n_traj = sum(1 for c in p["conditions"].values() if c.get("robot_eef_trajectory"))
        print(f"    {p['key']}: {n_cond} conditions, {n_traj} with trajectories")
    if len(pairs_out) > 3:
        print(f"    ... ({len(pairs_out) - 3} more pairs)")

    return len(pairs_out), total_conditions, conditions_with_traj


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    grand_pairs = 0
    grand_conditions = 0
    grand_with_traj = 0

    for name, src_path in SOURCES.items():
        if not Path(src_path).exists():
            print(f"WARNING: {src_path} does not exist, skipping {name}")
            continue
        n_pairs, n_cond, n_traj = process_file(name, src_path)
        grand_pairs += n_pairs
        grand_conditions += n_cond
        grand_with_traj += n_traj

    print(f"\n{'='*60}")
    print(f"GRAND TOTALS:")
    print(f"  Pairs: {grand_pairs}")
    print(f"  Total conditions: {grand_conditions}")
    print(f"  Conditions with trajectories: {grand_with_traj}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
