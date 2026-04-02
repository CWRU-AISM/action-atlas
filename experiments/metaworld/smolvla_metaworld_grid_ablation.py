#!/usr/bin/env python3
"""
SmolVLA MetaWorld grid ablation: layer-by-layer MLP zeroing across all tasks.

For each layer (expert_0..N, vlm_0..N), zeros the MLP output and measures
success rate across all tasks. Produces a grid showing which layers are
critical for which tasks.

Usage:
    python experiments/metaworld/smolvla_metaworld_grid_ablation.py --difficulty easy
    python experiments/metaworld/smolvla_metaworld_grid_ablation.py --tasks reach-v3,push-v3 --n-episodes 3
    python experiments/metaworld/smolvla_metaworld_grid_ablation.py --difficulty easy,medium --layers expert_0 expert_8 vlm_0 vlm_8
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

from common import (
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, MAX_STEPS, TASK_DESCRIPTIONS,
    MLPZeroHook, create_env, force_free_memory, get_layer_modules,
    get_tasks_from_args, load_smolvla_policy, run_episode, save_video_frames,
)


@dataclass
class GridAblationConfig:
    """SmolVLA MetaWorld grid ablation: layer-by-layer MLP zeroing."""

    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    difficulty: Optional[str] = None
    n_episodes: int = 3
    layers: Optional[List[str]] = None
    """Layer names to ablate (e.g., expert_0 vlm_8). Default: all."""

    resolution: int = DEFAULT_RESOLUTION
    output_dir: Optional[str] = None
    resume: bool = False
    save_video: bool = False
    save_trajectory: bool = False
    """Save actions, agent_pos, scene_states per episode"""


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = get_tasks_from_args(cfg)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"rollouts/smolvla/metaworld_grid_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {len(tasks)}, Episodes: {cfg.n_episodes}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device)

    expert_layers, vlm_layers = get_layer_modules(policy)
    n_expert = len(expert_layers)
    n_vlm = len(vlm_layers)

    conditions = ['baseline']
    if cfg.layers:
        conditions.extend(cfg.layers)
    else:
        conditions.extend([f"expert_{i}" for i in range(n_expert)])
        conditions.extend([f"vlm_{i}" for i in range(n_vlm)])

    print(f"Conditions: {len(conditions)} ({n_expert} expert + {n_vlm} VLM + baseline)")

    results_path = output_dir / "results.json"
    if cfg.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            'checkpoint': cfg.checkpoint,
            'env': 'metaworld',
            'n_episodes': cfg.n_episodes,
            'max_steps': MAX_STEPS,
            'conditions': conditions,
            'tasks': tasks,
            'grid': {},
        }

    start_time = time.time()

    for cond_idx, condition in enumerate(conditions):
        print(f"\nCondition [{cond_idx+1}/{len(conditions)}]: {condition}")

        hook = None
        handle = None
        if condition != 'baseline':
            parts = condition.split('_')
            layer_type = parts[0]
            layer_idx = int(parts[1])

            hook = MLPZeroHook()
            if layer_type == 'expert':
                handle = expert_layers[layer_idx].mlp.register_forward_hook(hook)
            else:
                handle = vlm_layers[layer_idx].mlp.register_forward_hook(hook)

        cond_results = all_results['grid'].get(condition, {})

        for task_name in tasks:
            if task_name in cond_results and len(cond_results[task_name].get('episodes', [])) >= cfg.n_episodes:
                print(f"  {task_name}: SKIP (exists)")
                continue

            task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
            successes = []
            steps_list = []

            for ep in range(cfg.n_episodes):
                env = create_env(task_name, cfg.resolution)

                if hook:
                    hook.call_count = 0

                result = run_episode(
                    policy, env, preprocessor, postprocessor, device,
                    save_video=cfg.save_video,
                )
                env.close()

                successes.append(result['success'])
                steps_list.append(result['n_steps'])

                if cfg.save_trajectory:
                    traj_dir = output_dir / "trajectories" / condition / task_name
                    traj_dir.mkdir(parents=True, exist_ok=True)
                    traj_data = {
                        'task': task_name,
                        'task_description': task_desc,
                        'condition': condition,
                        'episode': ep,
                        'success': result['success'],
                        'n_steps': result['n_steps'],
                        'actions': result['actions'].tolist(),
                        'agent_pos_trajectory': result.get('agent_pos_trajectory', []),
                        'scene_states': result.get('scene_states', []),
                    }
                    with open(traj_dir / f"ep{ep:02d}.json", 'w') as f:
                        json.dump(traj_data, f)

                if cfg.save_video and result.get('frames'):
                    vid_dir = output_dir / "videos" / condition / task_name
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    save_video_frames(vid_dir / f"ep{ep:02d}.mp4", result['frames'], fps=10)

            success_rate = sum(successes) / len(successes)
            print(f"  {task_name}: {success_rate*100:.0f}% ({sum(successes)}/{len(successes)})")

            cond_results[task_name] = {
                'task_description': task_desc,
                'success_rate': success_rate,
                'episodes': [{'success': s, 'steps': st} for s, st in zip(successes, steps_list)],
            }

        all_results['grid'][condition] = cond_results

        if handle:
            handle.remove()

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        force_free_memory()

    elapsed = time.time() - start_time
    all_results['duration_seconds'] = elapsed
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nGrid Summary (MetaWorld)")
    print(f"{'Condition':<20}", end="")
    for t in tasks[:10]:
        print(f"{t[:8]:>9}", end="")
    print(f"  {'Avg':>5}")

    for condition in conditions:
        cond_data = all_results['grid'].get(condition, {})
        print(f"{condition:<20}", end="")
        rates = []
        for t in tasks[:10]:
            rate = cond_data.get(t, {}).get('success_rate', 0)
            rates.append(rate)
            print(f"{rate*100:>8.0f}%", end="")
        avg = np.mean(rates) if rates else 0
        print(f"  {avg*100:>4.0f}%")

    print(f"\nDone in {elapsed/60:.1f} min. Results: {results_path}")


if __name__ == "__main__":
    cfg = tyro.cli(GridAblationConfig)
    main(cfg)
