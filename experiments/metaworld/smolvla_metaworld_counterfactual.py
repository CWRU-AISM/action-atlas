#!/usr/bin/env python3
"""
SmolVLA MetaWorld counterfactual prompting.

Collects activations under various prompt perturbations to study how
language instructions affect SmolVLA's action generation on MetaWorld.

Prompt types:
- baseline: correct task prompt
- null: empty string
- negation: "do not <original task>"
- motor: simple motor commands ("open gripper", "move left")
- cross_prompt: task B's prompt in task A's environment

Usage:
    python experiments/metaworld/smolvla_metaworld_counterfactual.py --difficulty easy --output-dir rollouts/smolvla/metaworld_counterfactual
    python experiments/metaworld/smolvla_metaworld_counterfactual.py --tasks reach-v3,push-v3 --save-videos
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
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, TASK_DESCRIPTIONS,
    PerTokenCollector, create_env, get_tasks_from_args,
    load_smolvla_policy, run_episode, save_video_frames,
)


MOTOR_COMMANDS = [
    "open gripper", "close gripper", "move left",
    "move right", "move forward", "move up", "stay still", "go home",
]

NEGATION_TEMPLATES = ["do not {task}", "stop", "don't move"]


def generate_counterfactual_prompts(tasks, task_descriptions, seeds=(42, 123, 456)):
    """Generate prompt configurations for counterfactual experiments."""
    configs = []

    for task_name in tasks:
        real_prompt = task_descriptions.get(task_name, task_name)

        for seed in seeds:
            configs.append({
                'task': task_name, 'prompt': real_prompt,
                'category': 'baseline',
                'desc': f"{task_name}_baseline_s{seed}", 'seed': seed,
            })
            configs.append({
                'task': task_name, 'prompt': '',
                'category': 'null',
                'desc': f"{task_name}_null_s{seed}", 'seed': seed,
            })

            for neg_template in NEGATION_TEMPLATES[:2]:
                neg_prompt = neg_template.format(task=real_prompt) if '{task}' in neg_template else neg_template
                configs.append({
                    'task': task_name, 'prompt': neg_prompt,
                    'category': 'negation',
                    'desc': f"{task_name}_neg_{neg_template.split()[0]}_s{seed}", 'seed': seed,
                })

            for motor_cmd in MOTOR_COMMANDS[:3]:
                configs.append({
                    'task': task_name, 'prompt': motor_cmd,
                    'category': 'motor',
                    'desc': f"{task_name}_motor_{motor_cmd.replace(' ','_')}_s{seed}", 'seed': seed,
                })

            other_tasks = [t for t in tasks if t != task_name][:3]
            for other_name in other_tasks:
                configs.append({
                    'task': task_name, 'prompt': task_descriptions.get(other_name, other_name),
                    'category': 'cross_prompt',
                    'desc': f"{task_name}_cross_{other_name}_s{seed}", 'seed': seed,
                    'source_task': other_name,
                })

    return configs


@dataclass
class CounterfactualConfig:
    """SmolVLA MetaWorld counterfactual prompting."""

    output_dir: str
    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    difficulty: Optional[str] = None
    seeds: str = "42,123,456"
    resolution: int = DEFAULT_RESOLUTION
    save_videos: bool = False
    no_activations: bool = False
    start_idx: int = 0
    end_idx: Optional[int] = None


def main(cfg):

    seeds = tuple(int(x) for x in cfg.seeds.split(','))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = get_tasks_from_args(cfg)

    print(f"Tasks: {len(tasks)}, Device: {device}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device)

    collector = None
    if not cfg.no_activations:
        collector = PerTokenCollector(policy)
        collector.setup()

    configs = generate_counterfactual_prompts(tasks, TASK_DESCRIPTIONS, seeds=seeds)
    if cfg.end_idx:
        configs = configs[cfg.start_idx:cfg.end_idx]
    else:
        configs = configs[cfg.start_idx:]

    print(f"Total prompt configs: {len(configs)}")

    (output_dir / "trajectories").mkdir(exist_ok=True)
    if collector:
        for i in range(collector.n_expert):
            (output_dir / "expert_layers" / f"layer{i:02d}").mkdir(parents=True, exist_ok=True)
        for i in range(collector.n_vlm):
            (output_dir / "vlm_layers" / f"layer{i:02d}").mkdir(parents=True, exist_ok=True)
    if cfg.save_videos:
        (output_dir / "videos").mkdir(exist_ok=True)

    metadata_path = output_dir / "metadata.jsonl"

    start_time = time.time()
    for idx, cfg in enumerate(configs):
        key = cfg['desc']
        task_name = cfg['task']
        prompt = cfg['prompt']
        category = cfg['category']

        traj_path = output_dir / "trajectories" / f"{key}.npz"
        if traj_path.exists():
            print(f"  [{idx+1}/{len(configs)}] {key} -- SKIP")
            continue

        print(f"  [{idx+1}/{len(configs)}] {key} (task={task_name}, cat={category}, prompt='{prompt[:40]}')")

        env = create_env(task_name, cfg.resolution)

        result = run_episode(
            policy, env, preprocessor, postprocessor, device,
            collector=collector, save_video=cfg.save_videos,
            task_override=prompt,
        )
        env.close()

        status = "OK" if result['success'] else f"FAIL({result['n_steps']})"
        print(f"    -> {status}")

        np.savez_compressed(str(traj_path),
                           actions=result['actions'],
                           agent_pos=np.array(result.get('agent_pos_trajectory', [])),
                           scene_states=json.dumps(result.get('scene_states', [])),
                           success=result['success'])

        if 'activations' in result:
            for act_key, acts in result['activations'].items():
                if act_key.startswith('expert_L'):
                    layer_idx = int(act_key.split('L')[1])
                    act_path = output_dir / "expert_layers" / f"layer{layer_idx:02d}" / f"{key}.npz"
                elif act_key.startswith('vlm_L'):
                    layer_idx = int(act_key.split('L')[1])
                    act_path = output_dir / "vlm_layers" / f"layer{layer_idx:02d}" / f"{key}.npz"
                else:
                    continue
                np.savez_compressed(str(act_path), activations=acts)

        if cfg.save_videos and result.get('frames'):
            video_path = output_dir / "videos" / f"{key}.mp4"
            save_video_frames(video_path, result['frames'], fps=10)

        meta = {
            'key': key, 'task': task_name, 'prompt': prompt,
            'category': category, 'seed': cfg.get('seed', 42),
            'success': result['success'], 'n_steps': result['n_steps'],
            'timestamp': datetime.now().isoformat(),
        }
        if 'source_task' in cfg:
            meta['source_task'] = cfg['source_task']
        with open(metadata_path, 'a') as f:
            f.write(json.dumps(meta) + '\n')

    elapsed = time.time() - start_time
    print(f"\nDone! {len(configs)} configs in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")

    if collector:
        collector.cleanup()


if __name__ == "__main__":
    cfg = tyro.cli(CounterfactualConfig)
    main(cfg)
