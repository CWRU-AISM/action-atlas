#!/usr/bin/env python3
"""
Bake MetaWorld scene state JSONs for the Action Atlas visualization platform.

Reads MetaWorld rollout data from multiple experiment types (baseline, grid_ablation,
cross_task, vision_perturbation, counterfactual) and produces scene state JSONs
compatible with the Action Atlas SceneState viewer.

Data sources:
  - /data/smolvla_rollouts/metaworld_baseline/         (baselines, 50 tasks x 20 eps)
  - /data/smolvla_rollouts/metaworld_grid_ablation/     (grid ablation, 4 difficulties)
  - /data/smolvla_rollouts/metaworld_cross_task/        (cross-task transfer, 4 difficulties)
  - /data/smolvla_rollouts/metaworld_vision_perturbation/ (vision perturbation, 4 difficulties)
  - /data/smolvla_rollouts/metaworld_counterfactual_v2/ (counterfactual, 4 difficulties)

Output: action_atlas/data/smolvla_scene_state/metaworld_{type}.json

Episode JSON format has: task, task_description, episode, success, n_steps,
    actions, agent_pos_trajectory, scene_states (list of {tcp_pos, obj_pos, goal_pos})

The LIBERO baseline format is:
  { suite, type, model, tasks: [
      { task_id, task_description, success_rate, trials: [
          { n_steps, success, trial, robot_eef_trajectory, object_displacements }
      ]}
  ]}

For MetaWorld we produce:
  - baseline: same format but with tcp_pos trajectory, obj/goal displacements
  - grid_ablation: { suite, type, model, difficulty, tasks: [
      { task_id, task_name, task_description, conditions: {
          cond_name: { success_rate, n_episodes, successes }
      }}
  ]}
  - cross_task: { suite, type, model, difficulty, pairs: [...] }
  - vision_perturbation: { suite, type, model, difficulty, tasks: [...] }
  - counterfactual: { suite, type, model, difficulty, tasks: [...] }

Usage:
    python scripts/bake_metaworld_scene_state.py
    python scripts/bake_metaworld_scene_state.py --dry-run
    python scripts/bake_metaworld_scene_state.py --only baseline
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BATCH2_DIR = Path("/data/smolvla_rollouts")
OUTPUT_DIR = Path("action_atlas/data/smolvla_scene_state")

DIFFICULTIES = ["easy", "medium", "hard_v2", "very_hard_v2"]
DIFFICULTY_LABELS = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
    "hard_v2": "hard",
    "very_hard": "very_hard",
    "very_hard_v2": "very_hard",
}

# MetaWorld task descriptions for reference
TASK_DESCRIPTIONS = {}  # Will be populated from results.json


def subsample_trajectory(traj, max_points=100):
    # Subsample a trajectory list to at most max_points evenly spaced entries
    if len(traj) <= max_points:
        return traj
    step = max(1, len(traj) // max_points)
    return traj[::step]


def extract_eef_from_scene_states(scene_states, max_points=100):
    # Extract tcp_pos trajectory from scene_states list
    traj = []
    for s in scene_states:
        if isinstance(s, dict) and "tcp_pos" in s:
            traj.append(s["tcp_pos"])
        elif isinstance(s, dict) and "full_obs" in s:
            # full_obs[0:3] is agent position
            obs = s["full_obs"]
            traj.append(obs[:3])
    return subsample_trajectory(traj, max_points)


def extract_obj_displacement(scene_states):
    # Compute object displacement from first to last scene state
    if not scene_states or len(scene_states) < 2:
        return {}

    first = scene_states[0]
    last = scene_states[-1]

    result = {}
    if isinstance(first, dict):
        for key in ["obj_pos", "goal_pos"]:
            if key in first and key in last:
                start = first[key]
                end = last[key]
                disp = sum((a - b) ** 2 for a, b in zip(start, end)) ** 0.5
                result[key] = {
                    "start_pos": start,
                    "end_pos": end,
                    "displacement": disp,
                }
    return result


def extract_eef_from_agent_pos(agent_pos_traj, max_points=100):
    # Extract EEF xyz from agent_pos_trajectory (first 3 elements)
    traj = [[p[0], p[1], p[2]] for p in agent_pos_traj if len(p) >= 3]
    return subsample_trajectory(traj, max_points)
# 1. BASELINE
def bake_baseline(dry_run=False):
    # Bake baseline scene state with EEF trajectories from episode JSONs
    base_dir = BATCH2_DIR / "metaworld_baseline"
    results_path = base_dir / "results.json"
    traj_dir = base_dir / "trajectories"

    if not results_path.exists():
        print("  [SKIP] No baseline results.json")
        return

    with open(results_path) as f:
        results = json.load(f)

    tasks_data = results.get("tasks", {})
    task_names = sorted(tasks_data.keys())

    tasks = []
    total_trials = 0
    total_with_traj = 0

    for task_idx, task_name in enumerate(task_names):
        task_info = tasks_data[task_name]
        task_desc = task_info.get("task_description", task_name)
        TASK_DESCRIPTIONS[task_name] = task_desc

        trials = []
        task_traj_dir = traj_dir / task_name
        if task_traj_dir.exists():
            for ep_json in sorted(task_traj_dir.glob("ep*.json")):
                try:
                    with open(ep_json) as f:
                        ep_data = json.load(f)

                    trial = {
                        "n_steps": ep_data.get("n_steps", 0),
                        "success": ep_data.get("success", False),
                        "trial": len(trials),
                    }

                    # Extract EEF trajectory from scene_states or agent_pos_trajectory
                    if "scene_states" in ep_data and ep_data["scene_states"]:
                        trial["robot_eef_trajectory"] = extract_eef_from_scene_states(
                            ep_data["scene_states"]
                        )
                        trial["object_displacements"] = extract_obj_displacement(
                            ep_data["scene_states"]
                        )
                        total_with_traj += 1
                    elif "agent_pos_trajectory" in ep_data:
                        trial["robot_eef_trajectory"] = extract_eef_from_agent_pos(
                            ep_data["agent_pos_trajectory"]
                        )
                        total_with_traj += 1

                    trials.append(trial)
                    total_trials += 1
                except Exception as e:
                    print(f"    Error reading {ep_json}: {e}")
                    continue

        n_success = sum(1 for t in trials if t.get("success"))
        success_rate = n_success / len(trials) if trials else 0.0

        tasks.append({
            "task_id": task_idx,
            "task_name": task_name,
            "task_description": task_desc,
            "success_rate": round(success_rate, 4),
            "n_episodes": len(trials),
            "successes": n_success,
            "trials": trials,
        })

    output = {
        "suite": "metaworld",
        "type": "baseline",
        "model": "smolvla",
        "overall_success_rate": round(
            results.get("overall_success_rate", 0.0), 4
        ),
        "tasks": tasks,
    }

    print(f"  Baseline: {len(tasks)} tasks, {total_trials} trials, {total_with_traj} with trajectories")

    if not dry_run:
        out_path = OUTPUT_DIR / "metaworld_baseline.json"
        with open(out_path, "w") as f:
            json.dump(output, f)
        sz = out_path.stat().st_size
        print(f"  Wrote {out_path} ({sz / 1024:.0f} KB)")
    return output
# 2. GRID ABLATION
def bake_grid_ablation(dry_run=False):
    """
    Bake grid ablation scene state for all difficulty levels.

    Produces one file per difficulty AND one combined file.
    Includes EEF trajectories from episode JSONs where available.
    """
    grid_dir = BATCH2_DIR / "metaworld_grid_ablation"
    all_difficulties = {}

    for diff in DIFFICULTIES:
        diff_dir = grid_dir / diff
        results_path = diff_dir / "results.json"
        if not results_path.exists():
            print(f"  [SKIP] No grid_ablation/{diff}/results.json")
            continue

        with open(results_path) as f:
            results = json.load(f)

        conditions = results.get("conditions", [])
        task_names = results.get("tasks", [])
        grid = results.get("grid", {})

        # Build task-level data with condition success rates
        tasks = []
        for task_idx, task_name in enumerate(task_names):
            task_conditions = {}
            for cond in conditions:
                if cond in grid and task_name in grid[cond]:
                    cond_data = grid[cond][task_name]
                    episodes = cond_data.get("episodes", [])
                    n_eps = len(episodes)
                    n_success = sum(1 for e in episodes if e.get("success"))
                    task_conditions[cond] = {
                        "success_rate": round(n_success / n_eps, 4) if n_eps > 0 else 0.0,
                        "n_episodes": n_eps,
                        "successes": n_success,
                    }

            # Try to load trajectory data for a sample of conditions
            traj_dir = diff_dir / "trajectories"
            trials_with_traj = {}
            if traj_dir.exists():
                for cond_dir in sorted(traj_dir.iterdir()):
                    if not cond_dir.is_dir():
                        continue
                    cond_name = cond_dir.name
                    task_dir = cond_dir / task_name
                    if not task_dir.exists():
                        continue
                    cond_trials = []
                    for ep_json in sorted(task_dir.glob("ep*.json"))[:3]:
                        try:
                            with open(ep_json) as f:
                                ep_data = json.load(f)
                            trial = {
                                "trial": len(cond_trials),
                                "success": ep_data.get("success", False),
                                "n_steps": ep_data.get("n_steps", 0),
                            }
                            if "scene_states" in ep_data and ep_data["scene_states"]:
                                trial["robot_eef_trajectory"] = extract_eef_from_scene_states(
                                    ep_data["scene_states"]
                                )
                            elif "agent_pos_trajectory" in ep_data:
                                trial["robot_eef_trajectory"] = extract_eef_from_agent_pos(
                                    ep_data["agent_pos_trajectory"]
                                )
                            cond_trials.append(trial)
                        except Exception:
                            continue
                    if cond_trials:
                        trials_with_traj[cond_name] = cond_trials

            task_desc = grid.get("baseline", {}).get(task_name, {}).get(
                "task_description", task_name
            )
            TASK_DESCRIPTIONS.setdefault(task_name, task_desc)

            task_entry = {
                "task_id": task_idx,
                "task_name": task_name,
                "task_description": task_desc,
                "conditions": task_conditions,
            }
            if trials_with_traj:
                task_entry["condition_trials"] = trials_with_traj

            tasks.append(task_entry)

        diff_label = DIFFICULTY_LABELS.get(diff, diff)
        output = {
            "suite": "metaworld",
            "type": "grid_ablation",
            "model": "smolvla",
            "difficulty": diff_label,
            "n_conditions": len(conditions),
            "conditions_list": conditions,
            "tasks": tasks,
        }
        all_difficulties[diff_label] = output

        n_conds = len(conditions)
        n_tasks = len(tasks)
        n_traj_tasks = sum(1 for t in tasks if t.get("condition_trials"))
        print(f"  Grid ablation {diff_label}: {n_tasks} tasks, {n_conds} conditions, {n_traj_tasks} tasks with trajectories")

        if not dry_run:
            out_path = OUTPUT_DIR / f"metaworld_{diff_label}_grid_ablation.json"
            with open(out_path, "w") as f:
                json.dump(output, f)
            sz = out_path.stat().st_size
            print(f"    Wrote {out_path} ({sz / 1024:.0f} KB)")

    # Combined file
    if all_difficulties and not dry_run:
        combined = {
            "suite": "metaworld",
            "type": "grid_ablation",
            "model": "smolvla",
            "difficulties": all_difficulties,
        }
        out_path = OUTPUT_DIR / "metaworld_grid_ablation.json"
        with open(out_path, "w") as f:
            json.dump(combined, f)
        sz = out_path.stat().st_size
        print(f"  Combined grid ablation: {out_path} ({sz / 1024:.0f} KB)")
# 3. CROSS-TASK TRANSFER
def bake_cross_task(dry_run=False):
    # Bake cross-task transfer scene state for all difficulty levels
    cross_dir = BATCH2_DIR / "metaworld_cross_task"
    all_difficulties = {}

    for diff in ["easy", "medium", "hard", "very_hard"]:
        diff_dir = cross_dir / diff
        if not diff_dir.exists():
            continue

        cross_files = sorted(diff_dir.glob("cross_task_*.json"))
        if not cross_files:
            print(f"  [SKIP] No cross-task files in {diff}")
            continue

        pairs = []
        for cf in cross_files:
            try:
                with open(cf) as f:
                    data = json.load(f)

                task_a = data.get("task_a", "")
                task_b = data.get("task_b", "")

                pair = {
                    "task_a": task_a,
                    "task_b": task_b,
                    "task_a_desc": data.get("task_a_desc", task_a),
                    "task_b_desc": data.get("task_b_desc", task_b),
                }

                # Extract success/n_steps for each condition
                conditions = {}
                for key in data:
                    if key.startswith(("baseline_", "inject_")):
                        cond_data = data[key]
                        cond_entry = {
                            "success": cond_data.get("success", False),
                            "n_steps": cond_data.get("n_steps", 0),
                        }
                        if "total_injections" in cond_data:
                            cond_entry["total_injections"] = cond_data["total_injections"]
                        if "cosine_sim_with_target_baseline" in cond_data:
                            cond_entry["cosine_sim"] = round(
                                cond_data["cosine_sim_with_target_baseline"], 4
                            )
                        # Extract EEF trajectory from actions if available
                        # (cross-task files only have actions, not scene_states)
                        conditions[key] = cond_entry

                pair["conditions"] = conditions
                pairs.append(pair)
            except Exception as e:
                print(f"    Error reading {cf}: {e}")
                continue

        diff_label = DIFFICULTY_LABELS.get(diff, diff)
        output = {
            "suite": "metaworld",
            "type": "cross_task",
            "model": "smolvla",
            "difficulty": diff_label,
            "n_pairs": len(pairs),
            "pairs": pairs,
        }
        all_difficulties[diff_label] = output

        print(f"  Cross-task {diff_label}: {len(pairs)} pairs")

        if not dry_run:
            out_path = OUTPUT_DIR / f"metaworld_{diff_label}_cross_task.json"
            with open(out_path, "w") as f:
                json.dump(output, f)
            sz = out_path.stat().st_size
            print(f"    Wrote {out_path} ({sz / 1024:.0f} KB)")

    if all_difficulties and not dry_run:
        combined = {
            "suite": "metaworld",
            "type": "cross_task",
            "model": "smolvla",
            "difficulties": all_difficulties,
        }
        out_path = OUTPUT_DIR / "metaworld_cross_task.json"
        with open(out_path, "w") as f:
            json.dump(combined, f)
        sz = out_path.stat().st_size
        print(f"  Combined cross-task: {out_path} ({sz / 1024:.0f} KB)")
# 4. VISION PERTURBATION
def bake_vision_perturbation(dry_run=False):
    """
    Bake vision perturbation scene state for all difficulty levels.

    Includes EEF trajectories from episode JSONs.
    """
    vp_dir = BATCH2_DIR / "metaworld_vision_perturbation"
    all_difficulties = {}

    for diff in ["easy", "medium", "hard", "very_hard"]:
        diff_dir = vp_dir / diff
        results_path = diff_dir / "results.json"
        if not results_path.exists():
            print(f"  [SKIP] No vision_perturbation/{diff}/results.json")
            continue

        with open(results_path) as f:
            results = json.load(f)

        perturbations = results.get("perturbations", [])
        task_names = results.get("tasks", [])
        grid = results.get("grid", {})
        traj_dir = diff_dir / "trajectories"

        tasks = []
        for task_idx, task_name in enumerate(task_names):
            task_conditions = {}
            for pert in perturbations:
                if pert in grid and task_name in grid[pert]:
                    pert_data = grid[pert][task_name]
                    episodes = pert_data.get("episodes", [])
                    n_eps = len(episodes)
                    n_success = sum(1 for e in episodes if e.get("success"))
                    task_conditions[pert] = {
                        "success_rate": round(n_success / n_eps, 4) if n_eps > 0 else 0.0,
                        "n_episodes": n_eps,
                        "successes": n_success,
                    }

            # Load trajectory data
            trials_with_traj = {}
            if traj_dir.exists():
                for pert_dir in sorted(traj_dir.iterdir()):
                    if not pert_dir.is_dir():
                        continue
                    pert_name = pert_dir.name
                    task_dir_path = pert_dir / task_name
                    if not task_dir_path.exists():
                        continue
                    pert_trials = []
                    for ep_json in sorted(task_dir_path.glob("ep*.json"))[:3]:
                        try:
                            with open(ep_json) as f:
                                ep_data = json.load(f)
                            trial = {
                                "trial": len(pert_trials),
                                "success": ep_data.get("success", False),
                                "n_steps": ep_data.get("n_steps", 0),
                            }
                            if "scene_states" in ep_data and ep_data["scene_states"]:
                                trial["robot_eef_trajectory"] = extract_eef_from_scene_states(
                                    ep_data["scene_states"]
                                )
                            elif "agent_pos_trajectory" in ep_data:
                                trial["robot_eef_trajectory"] = extract_eef_from_agent_pos(
                                    ep_data["agent_pos_trajectory"]
                                )
                            pert_trials.append(trial)
                        except Exception:
                            continue
                    if pert_trials:
                        trials_with_traj[pert_name] = pert_trials

            task_desc = grid.get("baseline", {}).get(task_name, {}).get(
                "task_description", task_name
            )
            TASK_DESCRIPTIONS.setdefault(task_name, task_desc)

            task_entry = {
                "task_id": task_idx,
                "task_name": task_name,
                "task_description": task_desc,
                "conditions": task_conditions,
            }
            if trials_with_traj:
                task_entry["condition_trials"] = trials_with_traj

            tasks.append(task_entry)

        diff_label = DIFFICULTY_LABELS.get(diff, diff)
        output = {
            "suite": "metaworld",
            "type": "vision_perturbation",
            "model": "smolvla",
            "difficulty": diff_label,
            "n_perturbations": len(perturbations),
            "perturbations_list": perturbations,
            "tasks": tasks,
        }
        all_difficulties[diff_label] = output

        n_tasks = len(tasks)
        n_traj = sum(1 for t in tasks if t.get("condition_trials"))
        print(f"  Vision perturbation {diff_label}: {n_tasks} tasks, {len(perturbations)} perturbations, {n_traj} tasks with trajectories")

        if not dry_run:
            out_path = OUTPUT_DIR / f"metaworld_{diff_label}_vision_perturbation.json"
            with open(out_path, "w") as f:
                json.dump(output, f)
            sz = out_path.stat().st_size
            print(f"    Wrote {out_path} ({sz / 1024:.0f} KB)")

    if all_difficulties and not dry_run:
        combined = {
            "suite": "metaworld",
            "type": "vision_perturbation",
            "model": "smolvla",
            "difficulties": all_difficulties,
        }
        out_path = OUTPUT_DIR / "metaworld_vision_perturbation.json"
        with open(out_path, "w") as f:
            json.dump(combined, f)
        sz = out_path.stat().st_size
        print(f"  Combined vision perturbation: {out_path} ({sz / 1024:.0f} KB)")
# 5. COUNTERFACTUAL
def bake_counterfactual(dry_run=False):
    """
    Bake counterfactual scene state for all difficulty levels.

    Uses v2 data which has trajectory NPZs with scene_states.
    """
    all_difficulties = {}

    cf_dirs = {
        "easy": BATCH2_DIR / "metaworld_counterfactual_v2",
        "medium": BATCH2_DIR / "metaworld_counterfactual_v2_medium",
        "hard": BATCH2_DIR / "metaworld_counterfactual_v2_hard",
        "very_hard": BATCH2_DIR / "metaworld_counterfactual_v2_very_hard",
    }

    for diff_label, cf_dir in cf_dirs.items():
        meta_path = cf_dir / "metadata.jsonl"
        traj_dir = cf_dir / "trajectories"
        if not meta_path.exists():
            print(f"  [SKIP] No counterfactual metadata for {diff_label}")
            continue

        # Read metadata
        records = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # Group by task
        task_groups = defaultdict(lambda: defaultdict(list))
        for rec in records:
            task = rec.get("task", "unknown")
            category = rec.get("category", "unknown")
            task_groups[task][category].append(rec)

        tasks = []
        total_with_traj = 0

        for task_idx, task_name in enumerate(sorted(task_groups.keys())):
            categories = task_groups[task_name]
            task_conditions = {}

            for cat_name, cat_records in sorted(categories.items()):
                n_eps = len(cat_records)
                n_success = sum(1 for r in cat_records if r.get("success"))

                # Load trajectory data for this category
                cat_trials = []
                for rec in cat_records[:3]:  # limit to 3 trials per condition
                    trial = {
                        "trial": len(cat_trials),
                        "success": rec.get("success", False),
                        "n_steps": rec.get("n_steps", 0),
                        "prompt": rec.get("prompt", ""),
                        "seed": rec.get("seed", 0),
                    }

                    # Try to load trajectory NPZ
                    key = rec.get("key", "")
                    seed = rec.get("seed", 42)
                    npz_name = f"{key}.npz"
                    npz_path = traj_dir / npz_name
                    if npz_path.exists():
                        try:
                            npz_data = np.load(npz_path, allow_pickle=True)
                            # agent_pos has EEF data
                            if "agent_pos" in npz_data:
                                agent_pos = npz_data["agent_pos"]
                                traj = [[float(p[0]), float(p[1]), float(p[2])]
                                        for p in agent_pos[:, :3]]
                                trial["robot_eef_trajectory"] = subsample_trajectory(traj)
                                total_with_traj += 1
                            elif "scene_states" in npz_data:
                                # scene_states stored as JSON string
                                ss_str = str(npz_data["scene_states"])
                                try:
                                    ss = json.loads(ss_str)
                                    trial["robot_eef_trajectory"] = extract_eef_from_scene_states(ss)
                                    total_with_traj += 1
                                except json.JSONDecodeError:
                                    pass
                        except Exception as e:
                            pass

                    cat_trials.append(trial)

                task_conditions[cat_name] = {
                    "success_rate": round(n_success / n_eps, 4) if n_eps > 0 else 0.0,
                    "n_episodes": n_eps,
                    "successes": n_success,
                    "trials": cat_trials,
                }

            # Get task description from first baseline record
            task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
            for cat_records in categories.values():
                for rec in cat_records:
                    if rec.get("prompt"):
                        task_desc = rec["prompt"]
                        break
                break

            tasks.append({
                "task_id": task_idx,
                "task_name": task_name,
                "task_description": task_desc,
                "n_categories": len(task_conditions),
                "conditions": task_conditions,
            })

        output = {
            "suite": "metaworld",
            "type": "counterfactual",
            "model": "smolvla",
            "difficulty": diff_label,
            "n_tasks": len(tasks),
            "total_records": len(records),
            "tasks": tasks,
        }
        all_difficulties[diff_label] = output

        print(f"  Counterfactual {diff_label}: {len(tasks)} tasks, {len(records)} records, {total_with_traj} with trajectories")

        if not dry_run:
            out_path = OUTPUT_DIR / f"metaworld_{diff_label}_counterfactual.json"
            with open(out_path, "w") as f:
                json.dump(output, f)
            sz = out_path.stat().st_size
            print(f"    Wrote {out_path} ({sz / 1024:.0f} KB)")

    if all_difficulties and not dry_run:
        combined = {
            "suite": "metaworld",
            "type": "counterfactual",
            "model": "smolvla",
            "difficulties": all_difficulties,
        }
        out_path = OUTPUT_DIR / "metaworld_counterfactual.json"
        with open(out_path, "w") as f:
            json.dump(combined, f)
        sz = out_path.stat().st_size
        print(f"  Combined counterfactual: {out_path} ({sz / 1024:.0f} KB)")
def main():
    parser = argparse.ArgumentParser(
        description="Bake MetaWorld scene state JSONs for Action Atlas"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing files")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        choices=["baseline", "grid_ablation", "cross_task", "vision_perturbation", "counterfactual"],
        help="Only process one experiment type",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bakers = {
        "baseline": bake_baseline,
        "grid_ablation": bake_grid_ablation,
        "cross_task": bake_cross_task,
        "vision_perturbation": bake_vision_perturbation,
        "counterfactual": bake_counterfactual,
    }

    if args.only:
        bakers = {args.only: bakers[args.only]}

    for name, baker_fn in bakers.items():
        print(f"\n{'='*60}")
        print(f"Baking {name}...")
        print(f"{'='*60}")
        baker_fn(dry_run=args.dry_run)

    print(f"\nDone! Output directory: {OUTPUT_DIR}")
    if not args.dry_run:
        # List all output files
        print("\nGenerated files:")
        for f in sorted(OUTPUT_DIR.glob("metaworld_*.json")):
            sz = f.stat().st_size
            print(f"  {f.name} ({sz / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
