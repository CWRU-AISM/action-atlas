#!/usr/bin/env python3
"""
Build scene state JSONs and video/ablation indices for X-VLA, SmolVLA, GR00T.

Outputs:
    action_atlas/data/smolvla_scene_state/{suite}_baseline.json
    action_atlas/data/smolvla_scene_state/{suite}_grid_ablation.json
    action_atlas/data/smolvla_scene_state/metaworld_baseline.json
    action_atlas/data/xvla_scene_state/{suite}_baseline.json
    action_atlas/data/xvla_scene_state/{suite}_grid_ablation.json
    action_atlas/data/xvla_scene_state/{suite}_cross_task.json
    action_atlas/data/xvla_scene_state/{suite}_counterfactual.json
    action_atlas/data/xvla_scene_state/simplerenv_{env}_baseline.json
    action_atlas/data/xvla_scene_state/simplerenv_{env}_grid_ablation.json
    action_atlas/data/groot_scene_state/{suite}_fraction_to_failure.json
    action_atlas/data/groot_scene_state/{suite}_steering.json
    action_atlas/data/groot_scene_state/{suite}_temporal_ablation.json
    action_atlas/data/groot_scene_state/{suite}_cross_suite_ablation.json
    action_atlas/data/smolvla_ablation_index.json
    action_atlas/data/groot_ablation_index.json
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "action_atlas" / "data"
MAX_TRAJ_POINTS = 100


def subsample(points: list, max_n: int = MAX_TRAJ_POINTS) -> list:
    # Subsample a list of points to at most max_n, preserving first and last
    if not points or len(points) <= max_n:
        return points
    n = len(points)
    indices = [round(i * (n - 1) / (max_n - 1)) for i in range(max_n)]
    return [points[i] for i in indices]


def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: Failed to load {path}: {e}")
        return None


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {path.name} ({size_mb:.1f} MB)")
# SmolVLA Scene State
def build_smolvla_baseline(suite: str, results_path: Path) -> Optional[dict]:
    # Build SmolVLA baseline scene state from baselines/results.json
    data = safe_load_json(results_path)
    if data is None:
        return None

    tasks_dict = data.get("tasks", {})
    tasks = []
    for tk in sorted(tasks_dict.keys(), key=int):
        tv = tasks_dict[tk]
        episodes = tv.get("episodes", [])
        trials = []
        for ep in episodes:
            trial = {
                "n_steps": ep.get("steps", 0),
                "success": ep.get("success", False),
                "trial": ep.get("episode", len(trials)),
            }
            eef = ep.get("eef_trajectory", [])
            if eef:
                trial["robot_eef_trajectory"] = subsample(eef)
            obj_disp = ep.get("object_displacements", {})
            if obj_disp:
                trial["object_displacements"] = obj_disp
            trials.append(trial)

        success_count = sum(1 for t in trials if t.get("success"))
        tasks.append({
            "task_id": int(tk),
            "task_description": tv.get("task_description", ""),
            "success_rate": success_count / len(trials) if trials else 0,
            "trials": trials,
        })

    return {
        "suite": suite,
        "type": "baseline",
        "model": "smolvla",
        "tasks": tasks,
    }


def build_smolvla_grid_ablation(suite: str, results_path: Path) -> Optional[dict]:
    data = safe_load_json(results_path)
    if data is None:
        return None

    grid = data.get("grid", {})
    conditions_list = data.get("conditions", [])
    tasks_raw = data.get("tasks", [])

    tasks = []
    # tasks_raw is a list of task indices
    for task_idx in (tasks_raw if isinstance(tasks_raw, list) else range(10)):
        task_id = int(task_idx)
        conditions = {}

        for cond_name in conditions_list:
            if cond_name not in grid:
                continue
            task_key = str(task_id)
            if task_key not in grid[cond_name]:
                continue
            task_data = grid[cond_name][task_key]
            episodes = task_data.get("episodes", [])
            successes = sum(1 for ep in episodes if ep.get("success"))
            conditions[cond_name] = {
                "success_rate": successes / len(episodes) if episodes else 0,
                "n_episodes": len(episodes),
                "successes": successes,
            }

        tasks.append({
            "task_id": task_id,
            "task_description": grid.get("baseline", {}).get(str(task_id), {}).get("task_description", ""),
            "conditions": conditions,
        })

    return {
        "suite": suite,
        "type": "grid_ablation",
        "model": "smolvla",
        "tasks": tasks,
    }


def build_smolvla_metaworld_baseline(results_path: Path) -> Optional[dict]:
    data = safe_load_json(results_path)
    if data is None:
        return None

    tasks_dict = data.get("tasks", {})
    tasks = []
    for i, (tk, tv) in enumerate(sorted(tasks_dict.items())):
        tasks.append({
            "task_id": i,
            "task_name": tk,
            "task_description": tv.get("task_description", tk),
            "success_rate": tv.get("success_rate", 0),
            "n_episodes": tv.get("n_episodes", 0),
            "successes": tv.get("successes", 0),
        })

    return {
        "suite": "metaworld",
        "type": "baseline",
        "model": "smolvla",
        "overall_success_rate": data.get("overall_success_rate", 0),
        "tasks": tasks,
    }


def build_smolvla_scene_state():
    print("\n=== Building SmolVLA Scene State ===")
    out_dir = DATA_DIR / "smolvla_scene_state"

    # LIBERO baselines
    libero_suites = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    for suite in libero_suites:
        results_path = Path(f"/data/smolvla_rollouts/smolvla/baselines/{suite}/results.json")
        if results_path.exists():
            result = build_smolvla_baseline(suite, results_path)
            if result:
                write_json(out_dir / f"{suite}_baseline.json", result)

    # LIBERO grid ablation
    for suite in libero_suites:
        results_path = Path(f"/data/smolvla_rollouts/smolvla/grid_ablation/{suite}/results.json")
        if results_path.exists():
            result = build_smolvla_grid_ablation(suite, results_path)
            if result:
                write_json(out_dir / f"{suite}_grid_ablation.json", result)

    # MetaWorld baseline
    mw_path = Path("/data/smolvla_rollouts/metaworld_baseline/results.json")
    if mw_path.exists():
        result = build_smolvla_metaworld_baseline(mw_path)
        if result:
            write_json(out_dir / "metaworld_baseline.json", result)
# SmolVLA Ablation Index
def build_smolvla_ablation_index():
    print("\n=== Building SmolVLA Ablation Index ===")
    entries = []

    # Batch 1
    batch1_dir = Path("/data/smolvla_rollouts/smolvla/concept_ablation/results")
    # Batch 2
    batch2_dir = Path("/data/smolvla_rollouts/concept_ablation/results")

    for batch_idx, results_dir in enumerate([batch1_dir, batch2_dir], 1):
        if not results_dir.exists():
            continue
        for json_file in sorted(results_dir.glob("*.json")):
            if json_file.name.startswith("partial"):
                continue
            data = safe_load_json(json_file)
            if data is None:
                continue

            layer = data.get("layer", -1)
            component = data.get("component", "expert")
            suite = data.get("suite", "")
            mode = data.get("mode", "all")

            # Count concepts in ablation
            ablation = data.get("ablation", {})
            n_concepts = len(ablation)

            entries.append({
                "file": str(json_file),
                "layer": layer,
                "component": component,
                "suite": suite,
                "mode": mode,
                "n_concepts": n_concepts,
                "batch": batch_idx,
                "baseline_overall": data.get("baseline_overall", None),
            })

    index = {
        "model": "smolvla",
        "total": len(entries),
        "entries": entries,
    }
    write_json(DATA_DIR / "smolvla_ablation_index.json", index)
    return index
# X-VLA Scene State
LIBERO_TASK_DESCRIPTIONS = {}


def get_libero_task_descriptions() -> Dict[str, Dict[int, str]]:
    # Load LIBERO task descriptions from SmolVLA baselines (shared across models)
    global LIBERO_TASK_DESCRIPTIONS
    if LIBERO_TASK_DESCRIPTIONS:
        return LIBERO_TASK_DESCRIPTIONS

    for suite in ["libero_10", "libero_goal", "libero_object", "libero_spatial"]:
        results_path = Path(f"/data/smolvla_rollouts/smolvla/baselines/{suite}/results.json")
        if results_path.exists():
            data = safe_load_json(results_path)
            if data:
                tasks = data.get("tasks", {})
                suite_short = suite.replace("libero_", "")
                LIBERO_TASK_DESCRIPTIONS[suite_short] = {
                    int(tk): tv.get("task_description", "")
                    for tk, tv in tasks.items()
                }
    return LIBERO_TASK_DESCRIPTIONS


def build_xvla_libero_baseline(suite_short: str) -> Optional[dict]:
    # Build X-VLA LIBERO baseline from grid_ablation/baseline per-episode JSONs
    suite_name = f"libero_{suite_short}" if not suite_short.startswith("libero_") else suite_short
    suite_key = suite_name.replace("libero_", "")

    baseline_dir = Path(f"/data/batch_1/xvla_libero/experiments/grid_ablation_{suite_name}/baseline")
    if not baseline_dir.exists():
        return None

    task_descs = get_libero_task_descriptions().get(suite_key, {})
    tasks_data: Dict[int, List[dict]] = {}

    for json_file in sorted(baseline_dir.glob("task*_ep*.json")):
        data = safe_load_json(json_file)
        if data is None:
            continue

        task_id = data.get("task", 0)
        ep_id = data.get("episode", 0)

        trial = {
            "n_steps": data.get("n_steps", 0),
            "success": data.get("success", False),
            "trial": ep_id,
        }

        scene_summary = data.get("scene_summary", {})
        eef = scene_summary.get("robot_eef_trajectory", [])
        if eef:
            trial["robot_eef_trajectory"] = subsample(eef)

        obj_disp = scene_summary.get("object_displacements", {})
        if obj_disp:
            trial["object_displacements"] = obj_disp

        # Include object_trajectories if available
        obj_traj = scene_summary.get("object_trajectories", {})
        if obj_traj:
            trial["object_trajectories"] = {
                name: subsample(traj) for name, traj in obj_traj.items()
            }

        tasks_data.setdefault(task_id, []).append(trial)

    tasks = []
    for task_id in sorted(tasks_data.keys()):
        trials = tasks_data[task_id]
        success_count = sum(1 for t in trials if t.get("success"))
        tasks.append({
            "task_id": task_id,
            "task_description": task_descs.get(task_id, f"Task {task_id}"),
            "success_rate": success_count / len(trials) if trials else 0,
            "trials": trials,
        })

    return {
        "suite": suite_name,
        "type": "baseline",
        "model": "xvla",
        "tasks": tasks,
    }


def build_xvla_libero_grid_ablation(suite_short: str) -> Optional[dict]:
    """
    Build X-VLA LIBERO grid ablation from grid_results.json.

    X-VLA grid format: grid[condition] = {"per_task": {task_id: {success_rate, ...}}, "mean_success_rate": float}
    """
    suite_name = f"libero_{suite_short}" if not suite_short.startswith("libero_") else suite_short

    grid_results_path = Path(f"/data/batch_1/xvla_libero/experiments/grid_ablation_{suite_name}/grid_results.json")
    if not grid_results_path.exists():
        return None

    data = safe_load_json(grid_results_path)
    if data is None:
        return None

    task_prompts = data.get("task_prompts", {})
    grid = data.get("grid", {})

    # grid is: {condition: {"per_task": {task_id: {success_rate, ...}}, "mean_success_rate": float}}
    tasks: Dict[int, dict] = {}
    for cond_name, cond_data in grid.items():
        if not isinstance(cond_data, dict):
            continue
        per_task = cond_data.get("per_task", {})
        if not isinstance(per_task, dict):
            continue
        for task_key, task_data in per_task.items():
            try:
                task_id = int(task_key)
            except ValueError:
                continue
            if task_id not in tasks:
                tasks[task_id] = {
                    "task_id": task_id,
                    "task_description": task_prompts.get(str(task_id), f"Task {task_id}"),
                    "conditions": {},
                }
            if isinstance(task_data, dict):
                tasks[task_id]["conditions"][cond_name] = {
                    "success_rate": task_data.get("success_rate", 0),
                    "n_episodes": task_data.get("n_episodes", 0),
                    "successes": task_data.get("successes", 0),
                }

    return {
        "suite": suite_name,
        "type": "grid_ablation",
        "model": "xvla",
        "ablation_mode": data.get("ablation_mode", "zero"),
        "tasks": [tasks[tid] for tid in sorted(tasks.keys())],
    }


def build_xvla_libero_cross_task(suite_short: str) -> Optional[dict]:
    # Build X-VLA LIBERO cross-task scene state from per-episode JSONs
    suite_name = f"libero_{suite_short}" if not suite_short.startswith("libero_") else suite_short
    cross_dir = Path(f"/data/batch_1/xvla_libero/experiments/cross_task_{suite_name}")
    if not cross_dir.exists():
        return None

    pairs = []
    for pair_dir in sorted(cross_dir.iterdir()):
        if not pair_dir.is_dir() or not pair_dir.name.startswith("pair_"):
            continue

        # Parse pair ID: pair_A_B
        match = re.match(r"pair_(\d+)_(\d+)", pair_dir.name)
        if not match:
            continue
        task_a, task_b = int(match.group(1)), int(match.group(2))

        task_descs = get_libero_task_descriptions().get(suite_short, {})
        conditions = {}

        for inject_dir in sorted(pair_dir.iterdir()):
            if not inject_dir.is_dir():
                continue
            for json_file in sorted(inject_dir.glob("*.json")):
                data = safe_load_json(json_file)
                if data is None:
                    continue

                condition_name = data.get("condition", json_file.stem)
                cond_key = f"{inject_dir.name}/{condition_name}"

                cond_entry = {
                    "n_steps": data.get("steps", 0),
                    "success": data.get("success", False),
                }

                scene_summary = data.get("scene_summary", {})
                eef = scene_summary.get("robot_eef_trajectory", [])
                if eef:
                    cond_entry["robot_eef_trajectory"] = subsample(eef)

                obj_traj = scene_summary.get("object_trajectories", {})
                if obj_traj:
                    cond_entry["object_trajectories"] = {
                        name: subsample(traj) for name, traj in obj_traj.items()
                    }

                obj_disp = scene_summary.get("object_displacements", {})
                if obj_disp:
                    cond_entry["object_displacements"] = obj_disp

                conditions[cond_key] = cond_entry

        if conditions:
            pairs.append({
                "key": pair_dir.name,
                "task_a": task_a,
                "task_b": task_b,
                "prompt_a": task_descs.get(task_a, f"Task {task_a}"),
                "prompt_b": task_descs.get(task_b, f"Task {task_b}"),
                "conditions": conditions,
            })

    if not pairs:
        return None

    return {
        "suite": suite_name,
        "type": "cross_task",
        "model": "xvla",
        "pairs": pairs,
    }


def build_xvla_libero_counterfactual(suite_short: str) -> Optional[dict]:
    # Build X-VLA LIBERO counterfactual scene state
    suite_name = f"libero_{suite_short}" if not suite_short.startswith("libero_") else suite_short

    # Try v2 first, fall back to v1
    for suffix in [f"_v2", ""]:
        cf_dir = Path(f"/data/batch_1/xvla_libero/experiments/counterfactual_{suite_name}{suffix}")
        if cf_dir.exists():
            break
    else:
        return None

    task_descs = get_libero_task_descriptions().get(suite_short, {})
    tasks_data: Dict[int, Dict[str, List[dict]]] = {}

    for task_dir in sorted(cf_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue
        task_id = int(task_dir.name.split("_")[1])

        for json_file in sorted(task_dir.glob("*.json")):
            data = safe_load_json(json_file)
            if data is None:
                continue

            condition = data.get("condition", json_file.stem.split("_ep")[0])
            trial = {
                "n_steps": data.get("n_steps", 0),
                "success": data.get("is_success", data.get("success", False)),
                "episode": data.get("episode", 0),
            }

            scene_summary = data.get("scene_summary", {})
            eef = scene_summary.get("robot_eef_trajectory", [])
            if eef:
                trial["robot_eef_trajectory"] = subsample(eef)
            obj_disp = scene_summary.get("object_displacements", {})
            if obj_disp:
                trial["object_displacements"] = obj_disp

            tasks_data.setdefault(task_id, {}).setdefault(condition, []).append(trial)

    tasks = []
    for task_id in sorted(tasks_data.keys()):
        conditions = {}
        for cond_name, trials in sorted(tasks_data[task_id].items()):
            successes = sum(1 for t in trials if t.get("success"))
            conditions[cond_name] = {
                "success_rate": successes / len(trials) if trials else 0,
                "n_episodes": len(trials),
                "trials": trials,
            }
        tasks.append({
            "task_id": task_id,
            "task_description": task_descs.get(task_id, f"Task {task_id}"),
            "conditions": conditions,
        })

    if not tasks:
        return None

    return {
        "suite": suite_name,
        "type": "counterfactual",
        "model": "xvla",
        "tasks": tasks,
    }


def build_xvla_simplerenv_baseline(env: str) -> Optional[dict]:
    # Build X-VLA SimplerEnv baseline from per-episode JSONs
    baseline_dir = Path(f"/data/batch_1/xvla_SIMPLERENV/baselines/{env}_baseline")
    if not baseline_dir.exists():
        return None

    tasks_data: Dict[str, List[dict]] = {}

    for task_dir in sorted(baseline_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for json_file in sorted(task_dir.glob("ep*.json")):
            data = safe_load_json(json_file)
            if data is None:
                continue
            trial = {
                "episode": data.get("episode", 0),
                "success": data.get("success", False),
                "n_steps": data.get("steps", 0),
                "instruction": data.get("instruction", ""),
            }
            # SimplerEnv has obs_states with tcp_pose, extract EEF trajectory
            obs_states = data.get("obs_states", [])
            if obs_states and isinstance(obs_states[0], dict):
                eef = []
                for s in obs_states:
                    tcp = s.get("tcp_pose")
                    if tcp is not None and isinstance(tcp, list) and len(tcp) >= 3:
                        eef.append(tcp[:3])
                if eef:
                    trial["robot_eef_trajectory"] = subsample(eef)

            tasks_data.setdefault(task_name, []).append(trial)

    tasks = []
    for i, (task_name, trials) in enumerate(sorted(tasks_data.items())):
        successes = sum(1 for t in trials if t.get("success"))
        instruction = trials[0].get("instruction", task_name) if trials else task_name
        tasks.append({
            "task_id": i,
            "task_name": task_name,
            "task_description": instruction,
            "success_rate": successes / len(trials) if trials else 0,
            "n_episodes": len(trials),
            "trials": trials,
        })

    return {
        "suite": f"simplerenv_{env}",
        "type": "baseline",
        "model": "xvla",
        "tasks": tasks,
    }


def build_xvla_simplerenv_grid(env: str) -> Optional[dict]:
    """
    Build X-VLA SimplerEnv grid ablation from grid_results.json.

    Same per_task format as LIBERO: grid[condition] = {"per_task": {...}, "mean_success_rate": float}
    """
    grid_path = Path(f"/data/batch_1/xvla_SIMPLERENV/experiments/grid_ablation_{env}/grid_results.json")
    if not grid_path.exists():
        return None

    data = safe_load_json(grid_path)
    if data is None:
        return None

    task_prompts = data.get("task_prompts", {})
    grid = data.get("grid", {})

    tasks: Dict[str, dict] = {}
    for cond_name, cond_data in grid.items():
        if not isinstance(cond_data, dict):
            continue
        per_task = cond_data.get("per_task", {})
        if not isinstance(per_task, dict):
            continue
        for task_key, task_data in per_task.items():
            if task_key not in tasks:
                tasks[task_key] = {
                    "task_id": task_key,
                    "task_description": task_prompts.get(task_key, task_key),
                    "conditions": {},
                }
            if isinstance(task_data, dict):
                tasks[task_key]["conditions"][cond_name] = {
                    "success_rate": task_data.get("success_rate", 0),
                    "n_episodes": task_data.get("n_episodes", 0),
                    "successes": task_data.get("successes", 0),
                }

    return {
        "suite": f"simplerenv_{env}",
        "type": "grid_ablation",
        "model": "xvla",
        "ablation_mode": data.get("ablation_mode", "zero"),
        "tasks": list(tasks.values()),
    }


def build_xvla_scene_state():
    # Build all X-VLA scene state files
    print("\n=== Building X-VLA Scene State ===")
    out_dir = DATA_DIR / "xvla_scene_state"

    # LIBERO suites
    for suite_short in ["10", "goal", "object", "spatial"]:
        suite_name = f"libero_{suite_short}"

        # Baselines (from grid_ablation/baseline episodes)
        result = build_xvla_libero_baseline(suite_name)
        if result:
            write_json(out_dir / f"{suite_name}_baseline.json", result)

        # Grid ablation
        result = build_xvla_libero_grid_ablation(suite_short)
        if result:
            write_json(out_dir / f"{suite_name}_grid_ablation.json", result)

        # Cross-task
        result = build_xvla_libero_cross_task(suite_short)
        if result:
            write_json(out_dir / f"{suite_name}_cross_task.json", result)

        # Counterfactual
        result = build_xvla_libero_counterfactual(suite_short)
        if result:
            write_json(out_dir / f"{suite_name}_counterfactual.json", result)

    # SimplerEnv
    for env in ["google_robot", "widowx"]:
        result = build_xvla_simplerenv_baseline(env)
        if result:
            write_json(out_dir / f"simplerenv_{env}_baseline.json", result)

        result = build_xvla_simplerenv_grid(env)
        if result:
            write_json(out_dir / f"simplerenv_{env}_grid_ablation.json", result)
# GR00T Scene State
GROOT_TASK_DESCRIPTIONS = {
    "libero_goal": {
        0: "open the middle drawer of the cabinet",
        1: "put the bowl on the stove",
        2: "put the wine bottle on top of the cabinet",
        3: "open the top drawer and put the bowl inside",
        4: "put the bowl on top of the cabinet",
        5: "push the plate to the front of the stove",
        6: "put the cream cheese in the bowl",
        7: "turn on the stove",
        8: "put the bowl on the plate",
        9: "put the wine bottle on the rack",
    },
    "libero_object": {
        0: "pick up the alphabet soup and place it in the basket",
        1: "pick up the cream cheese and place it in the basket",
        2: "pick up the salad dressing and place it in the basket",
        3: "pick up the bbq sauce and place it in the basket",
        4: "pick up the ketchup and place it in the basket",
        5: "pick up the tomato sauce and place it in the basket",
        6: "pick up the butter and place it in the basket",
        7: "pick up the milk and place it in the basket",
        8: "pick up the chocolate pudding and place it in the basket",
        9: "pick up the orange juice and place it in the basket",
    },
    "libero_long": {},  # Will be populated if available
}


def build_groot_fraction_to_failure(suite: str) -> Optional[dict]:
    # Build GR00T fraction-to-failure scene state with trajectory data
    layers_data = []

    for batch_root in [
        Path("/data/groot_rollouts/sae_fraction_to_failure"),
        Path("/data/groot_rollouts_batch2/sae_fraction_to_failure"),
    ]:
        suite_dir = batch_root / suite
        if not suite_dir.exists():
            continue

        for layer_dir in sorted(suite_dir.iterdir()):
            if not layer_dir.is_dir():
                continue
            layer_name = layer_dir.name

            # Read fraction result JSONs
            for json_file in sorted(layer_dir.glob("fraction_*.json")):
                data = safe_load_json(json_file)
                if data is None:
                    continue

                category = data.get("category", json_file.stem.replace("fraction_", ""))
                titration = data.get("titration", [])
                baseline = data.get("baseline", {})

                titration_summary = []
                for t in titration:
                    titration_summary.append({
                        "n_features": t.get("n_features", 0),
                        "success_rate": t.get("success_rate", 0),
                        "successes": t.get("successes", 0),
                        "total": t.get("total", 0),
                    })

                layers_data.append({
                    "layer": layer_name,
                    "category": category,
                    "baseline": baseline,
                    "titration": titration_summary,
                })

            # Collect scene trajectory data from trajectories dir
            traj_dir = layer_dir / "trajectories"
            if traj_dir.exists():
                for scene_file in sorted(traj_dir.glob("*_scene.json")):
                    scene_data = safe_load_json(scene_file)
                    if scene_data is None:
                        continue

                    # Parse filename: N{count}_{category}_task{id}_ep{id}_scene.json
                    fname = scene_file.stem.replace("_scene", "")
                    match = re.match(r"N(\d+)_(\w+)_task(\d+)_ep(\d+)", fname)
                    if not match:
                        continue

                    n_features = int(match.group(1))
                    category = match.group(2)
                    task_id = int(match.group(3))
                    ep_id = int(match.group(4))

                    # Already collected in fraction JSONs, just note we have trajectories
                    # Don't inline all trajectories - too large

    if not layers_data:
        return None

    return {
        "suite": suite,
        "type": "fraction_to_failure",
        "model": "groot",
        "layers": layers_data,
    }


def build_groot_steering(suite: str) -> Optional[dict]:
    suite_dir = Path(f"/data/groot_rollouts_batch2/sae_steering/{suite}")
    if not suite_dir.exists():
        return None

    layers_data = []
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name

        results_file = layer_dir / "steering_results.json"
        data = safe_load_json(results_file)
        if data is None:
            continue

        strengths = data.get("strengths", [])
        groups = data.get("groups", {})

        groups_summary = {}
        for group_name, group_data in groups.items():
            strengths_data = group_data.get("strengths", {})
            strength_results = {}
            for s_key, s_val in strengths_data.items():
                strength_results[s_key] = {
                    "avg_success_rate": s_val.get("avg_success_rate", 0),
                    "avg_delta": s_val.get("avg_delta", 0),
                    "per_task": s_val.get("tasks", {}),
                }
            groups_summary[group_name] = {
                "n_features": group_data.get("n_features", 0),
                "strengths": strength_results,
            }

        layers_data.append({
            "layer": layer_name,
            "strengths_tested": strengths,
            "baseline": data.get("baseline", {}),
            "groups": groups_summary,
        })

    if not layers_data:
        return None

    return {
        "suite": suite,
        "type": "steering",
        "model": "groot",
        "layers": layers_data,
    }


def build_groot_temporal_ablation(suite: str) -> Optional[dict]:
    # Build GR00T temporal ablation scene state
    suite_dir = Path(f"/data/groot_rollouts_batch2/sae_temporal_ablation/{suite}")
    if not suite_dir.exists():
        return None

    layers_data = []
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name

        for json_file in sorted(layer_dir.glob("temporal_*.json")):
            data = safe_load_json(json_file)
            if data is None:
                continue

            category = data.get("category", json_file.stem.replace("temporal_", ""))
            raw_windows = data.get("windows", {})
            baseline = data.get("baseline", {})

            windows_summary = []
            # windows can be a dict {name: {start, end, success_rate, ...}}
            # or a list [{window, start_frac, ...}]
            if isinstance(raw_windows, dict):
                for w_name, w_data in raw_windows.items():
                    if not isinstance(w_data, dict):
                        continue
                    window_entry = {
                        "window": w_name,
                        "start": w_data.get("start", 0),
                        "end": w_data.get("end", 0),
                        "success_rate": w_data.get("success_rate", 0),
                        "successes": w_data.get("successes", 0),
                        "total": w_data.get("total", 0),
                    }
                    if "per_task" in w_data:
                        window_entry["per_task"] = w_data["per_task"]
                    windows_summary.append(window_entry)
            elif isinstance(raw_windows, list):
                for w in raw_windows:
                    if not isinstance(w, dict):
                        continue
                    window_entry = {
                        "window": w.get("window", ""),
                        "start_frac": w.get("start_frac", 0),
                        "end_frac": w.get("end_frac", 1),
                    }
                    for field in ["success_rate", "successes", "total", "per_task"]:
                        if field in w:
                            window_entry[field] = w[field]
                    windows_summary.append(window_entry)

            layers_data.append({
                "layer": layer_name,
                "category": category,
                "n_features": data.get("n_features", 0),
                "baseline": baseline,
                "windows": windows_summary,
            })

    if not layers_data:
        return None

    return {
        "suite": suite,
        "type": "temporal_ablation",
        "model": "groot",
        "layers": layers_data,
    }


def build_groot_cross_suite_ablation(suite: str) -> Optional[dict]:
    # Build GR00T cross-suite ablation scene state
    suite_dir = Path(f"/data/groot_rollouts_batch2/sae_cross_suite_ablation/{suite}")
    if not suite_dir.exists():
        return None

    layers_data = []
    for layer_dir in sorted(suite_dir.iterdir()):
        if not layer_dir.is_dir():
            continue
        layer_name = layer_dir.name

        results_file = layer_dir / "ablation_results.json"
        data = safe_load_json(results_file)
        if data is None:
            continue

        layers_data.append({
            "layer": layer_name,
            "source_suite": data.get("source_suite", ""),
            "target_suite": data.get("target_suite", suite),
            "mode": data.get("mode", ""),
            "baseline": data.get("baseline", {}),
            "results": data.get("results", data.get("ablation_results", {})),
        })

    if not layers_data:
        return None

    return {
        "suite": suite,
        "type": "cross_suite_ablation",
        "model": "groot",
        "layers": layers_data,
    }


def build_groot_scene_state():
    # Build all GR00T scene state files
    print("\n=== Building GR00T Scene State ===")
    out_dir = DATA_DIR / "groot_scene_state"

    groot_suites = ["libero_goal", "libero_object", "libero_long"]

    for suite in groot_suites:
        # Fraction to failure
        result = build_groot_fraction_to_failure(suite)
        if result:
            write_json(out_dir / f"{suite}_fraction_to_failure.json", result)

        # Steering
        result = build_groot_steering(suite)
        if result:
            write_json(out_dir / f"{suite}_steering.json", result)

        # Temporal ablation
        result = build_groot_temporal_ablation(suite)
        if result:
            write_json(out_dir / f"{suite}_temporal_ablation.json", result)

        # Cross-suite ablation
        result = build_groot_cross_suite_ablation(suite)
        if result:
            write_json(out_dir / f"{suite}_cross_suite_ablation.json", result)
# GR00T Ablation Index (Video Index)
def build_groot_ablation_index():
    # Build GR00T video/ablation index across all experiment types
    print("\n=== Building GR00T Ablation Index ===")

    videos = []

    experiment_dirs = [
        # (base_path, experiment_type)
        (Path("/data/groot_rollouts/sae_fraction_to_failure"), "fraction_to_failure"),
        (Path("/data/groot_rollouts_batch2/sae_fraction_to_failure"), "fraction_to_failure"),
        (Path("/data/groot_rollouts_batch2/sae_steering"), "steering"),
        (Path("/data/groot_rollouts_batch2/sae_temporal_ablation"), "temporal_ablation"),
        (Path("/data/groot_rollouts_batch2/sae_cross_suite_ablation"), "cross_suite_ablation"),
    ]

    for base_path, exp_type in experiment_dirs:
        if not base_path.exists():
            print(f"  Skipping {base_path} (not found)")
            continue

        for suite_dir in sorted(base_path.iterdir()):
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name

            for layer_dir in sorted(suite_dir.iterdir()):
                if not layer_dir.is_dir():
                    continue
                layer = layer_dir.name

                videos_dir = layer_dir / "videos"
                if not videos_dir.exists():
                    continue

                for mp4 in sorted(videos_dir.glob("*.mp4")):
                    # Construct relative path for serving
                    rel_path = f"groot_{exp_type}/{suite}/{layer}/{mp4.name}"

                    entry = {
                        "path": rel_path,
                        "filename": mp4.name,
                        "experiment_type": exp_type,
                        "suite": suite,
                        "layer": layer,
                        "model": "groot",
                    }

                    # Parse video filename for metadata
                    fname = mp4.stem

                    # Fraction-to-failure: N{count}_{category}_task{id}_ep{id}
                    ftf_match = re.match(r"N(\d+)_(\w+)_task(\d+)_ep(\d+)", fname)
                    if ftf_match:
                        entry["n_features"] = int(ftf_match.group(1))
                        entry["category"] = ftf_match.group(2)
                        entry["task"] = int(ftf_match.group(3))
                        entry["episode"] = int(ftf_match.group(4))

                    # Steering: steer_{layer}_{category}_s{strength}_task{id}_ep{id}
                    steer_match = re.match(r"steer_\w+_(\w+)_s([-\d.]+)_task(\d+)_ep(\d+)", fname)
                    if steer_match:
                        entry["category"] = steer_match.group(1)
                        entry["strength"] = float(steer_match.group(2))
                        entry["task"] = int(steer_match.group(3))
                        entry["episode"] = int(steer_match.group(4))

                    # Temporal: {window}_{category}_task{id}_ep{id}
                    temp_match = re.match(r"(\w+)_(\w+)_task(\d+)_ep(\d+)", fname)
                    if temp_match and exp_type == "temporal_ablation":
                        entry["window"] = temp_match.group(1)
                        entry["category"] = temp_match.group(2)
                        entry["task"] = int(temp_match.group(3))
                        entry["episode"] = int(temp_match.group(4))

                    # Cross-suite: ablation_{layer}_disc_{suite}_task{src}_task{dst}_ep{id}
                    cross_match = re.match(r"ablation_\w+_(\w+)_(\w+)_task(\d+)_task(\d+)_ep(\d+)", fname)
                    if cross_match:
                        entry["disc_mode"] = cross_match.group(1)
                        entry["target_suite"] = cross_match.group(2)
                        entry["source_task"] = int(cross_match.group(3))
                        entry["target_task"] = int(cross_match.group(4))
                        entry["episode"] = int(cross_match.group(5))

                    videos.append(entry)

    index = {
        "model": "groot",
        "total": len(videos),
        "videos": videos,
    }

    write_json(DATA_DIR / "groot_ablation_index.json", index)
    print(f"  Total GR00T videos indexed: {len(videos)}")
    return index
# Main
def main():
    print("Building scene state JSONs and ablation indices for X-VLA, SmolVLA, GR00T")
    print(f"Output directory: {DATA_DIR}")

    # SmolVLA
    build_smolvla_scene_state()
    build_smolvla_ablation_index()

    # X-VLA
    build_xvla_scene_state()

    # GR00T
    build_groot_scene_state()
    build_groot_ablation_index()

    print("\nDone!")

    # Print summary
    print("\n=== Output Summary ===")
    for model_dir in ["smolvla_scene_state", "xvla_scene_state", "groot_scene_state"]:
        d = DATA_DIR / model_dir
        if d.exists():
            files = list(d.glob("*.json"))
            total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
            print(f"  {model_dir}/: {len(files)} files, {total_mb:.1f} MB total")
        else:
            print(f"  {model_dir}/: NOT CREATED")

    for idx_name in ["smolvla_ablation_index.json", "groot_ablation_index.json", "xvla_ablation_index.json"]:
        idx_path = DATA_DIR / idx_name
        if idx_path.exists():
            data = safe_load_json(idx_path)
            total = data.get("total", len(data.get("videos", data.get("entries", []))))
            size_mb = idx_path.stat().st_size / (1024 * 1024)
            print(f"  {idx_name}: {total} entries, {size_mb:.1f} MB")
        else:
            print(f"  {idx_name}: NOT FOUND")


if __name__ == "__main__":
    main()
