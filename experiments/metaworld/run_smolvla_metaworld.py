#!/usr/bin/env python3
"""
SmolVLA MetaWorld baseline evaluation with optional activation collection.

Runs SmolVLA on MetaWorld MT50 tasks. Captures actions, agent_pos trajectory,
activations (mean-pooled or per-token), and videos.

Usage:
    python experiments/metaworld/run_smolvla_metaworld.py --difficulty easy --n-episodes 10
    python experiments/metaworld/run_smolvla_metaworld.py --tasks reach-v3,push-v3 --n-episodes 20 --save-activations
    python experiments/metaworld/run_smolvla_metaworld.py --difficulty easy,medium,hard,very_hard --n-episodes 20
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from common import (
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, DIFFICULTY_TO_TASKS,
    TASK_DESCRIPTIONS, MeanPoolCollector, PerTokenCollector,
    create_env, get_tasks_from_args, load_smolvla_policy, run_episode,
    save_video_frames,
)


@dataclass
class MetaWorldEvalConfig:
    # SmolVLA MetaWorld baseline evaluation

    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    # Comma-separated task names (e.g., reach-v3,push-v3)

    difficulty: Optional[str] = None
    # Difficulty group(s): easy,medium,hard,very_hard

    n_episodes: int = 10
    resolution: int = DEFAULT_RESOLUTION
    save_activations: bool = False
    mean_pool: bool = False
    # Mean-pool activations across tokens (much smaller files)

    action_horizon: int = 10
    # Reuse predicted actions for N steps (default 10, chunk_size=50)

    save_video: bool = True
    no_video: bool = False
    output_dir: Optional[str] = None


def main(cfg):
    if cfg.no_video:
        cfg.save_video = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = get_tasks_from_args(cfg)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"rollouts/smolvla_metaworld_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"MetaWorld eval: {len(tasks)} tasks | eps={cfg.n_episodes} | "
          f"horizon={cfg.action_horizon} | output={output_dir}")

    print("\nLoading model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device, cfg.action_horizon)

    collector = None
    if cfg.save_activations:
        if cfg.mean_pool:
            collector = MeanPoolCollector(policy)
        else:
            collector = PerTokenCollector(policy)
        collector.setup()

    all_results = {
        'checkpoint': cfg.checkpoint,
        'n_episodes': cfg.n_episodes,
        'resolution': cfg.resolution,
        'timestamp': datetime.now().isoformat(),
        'tasks': {},
    }

    total_success = 0
    total_episodes = 0
    start_time = time.time()

    for task_idx, task_name in enumerate(tasks):
        task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
        print(f"\n[{task_idx+1}/{len(tasks)}] {task_name}: {task_desc}")

        successes = []
        for ep in range(cfg.n_episodes):
            env = create_env(task_name, cfg.resolution)

            ep_collector = collector if (cfg.save_activations and ep < cfg.n_episodes) else None

            result = run_episode(
                policy, env, preprocessor, postprocessor, device,
                collector=ep_collector,
                save_video=cfg.save_video,
            )
            env.close()

            successes.append(result['success'])
            status = "OK" if result['success'] else f"FAIL({result['n_steps']})"
            print(f"  ep {ep}: {status}", end="" if (ep + 1) % 5 != 0 else "\n")

            traj_dir = output_dir / "trajectories" / task_name
            traj_dir.mkdir(parents=True, exist_ok=True)
            traj_data = {
                'task': task_name,
                'task_description': task_desc,
                'episode': ep,
                'success': result['success'],
                'n_steps': result['n_steps'],
                'actions': result['actions'].tolist(),
                'agent_pos_trajectory': result['agent_pos_trajectory'],
                'scene_states': result['scene_states'],
                'obj_displacement': result['obj_displacement'],
            }
            with open(traj_dir / f"ep{ep:02d}.json", 'w') as f:
                json.dump(traj_data, f, indent=2)
            np.savez_compressed(
                str(traj_dir / f"ep{ep:02d}.npz"),
                actions=result['actions'],
                agent_pos=np.array(result['agent_pos_trajectory']),
                success=np.array(result['success']),
            )

            if 'activations' in result:
                act_dir = output_dir / "activations"
                act_dir.mkdir(exist_ok=True)
                np.savez_compressed(
                    str(act_dir / f"{task_name}_ep{ep}.npz"),
                    **result['activations'],
                    actions=result['actions'],
                    success=np.array(result['success']),
                )

            if 'frames' in result and result['frames']:
                vid_dir = output_dir / "videos" / task_name
                vid_dir.mkdir(parents=True, exist_ok=True)
                save_video_frames(vid_dir / f"ep{ep:02d}.mp4", result['frames'], fps=10)

        print()
        sr = sum(successes) / len(successes) if successes else 0
        print(f"  Success rate: {sr*100:.0f}% ({sum(successes)}/{len(successes)})")

        all_results['tasks'][task_name] = {
            'task_description': task_desc,
            'success_rate': sr,
            'n_episodes': len(successes),
            'successes': sum(successes),
        }

        total_success += sum(successes)
        total_episodes += len(successes)

        all_results['overall_success_rate'] = total_success / total_episodes if total_episodes > 0 else 0
        all_results['total_episodes'] = total_episodes
        with open(output_dir / "results.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    elapsed = time.time() - start_time
    all_results['duration_seconds'] = elapsed

    if collector:
        collector.cleanup()

    print(f"\nMetaWorld Results: {len(tasks)} tasks, {total_episodes} episodes")
    print(f"Overall success: {total_success}/{total_episodes} ({total_success/total_episodes*100:.1f}%)")
    print(f"Duration: {elapsed/60:.1f} min")

    for diff_name, diff_tasks in DIFFICULTY_TO_TASKS.items():
        diff_results = [all_results['tasks'][t] for t in diff_tasks if t in all_results['tasks']]
        if diff_results:
            avg = sum(r['success_rate'] for r in diff_results) / len(diff_results)
            print(f"  {diff_name}: {avg*100:.1f}% ({len(diff_results)} tasks)")

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults: {output_dir / 'results.json'}")


if __name__ == "__main__":
    cfg = tyro.cli(MetaWorldEvalConfig)
    main(cfg)
