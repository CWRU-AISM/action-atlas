#!/usr/bin/env python3
"""
SmolVLA MetaWorld enhanced counterfactual prompting.

Addresses reviewer concerns about limited counterfactual prompt diversity:
1. Compositional prompts: "press the button AND open the drawer"
2. Identical visuals, different goals: same env with wrong task instruction
3. Paraphrased prompts: same semantics, different wording
4. Specificity gradient: vague -> specific instructions

Collects per-token activations for all experiments.

Usage:
    python experiments/metaworld/smolvla_metaworld_enhanced_counterfactual.py \
        --difficulty easy,medium --output-dir rollouts/smolvla/metaworld_counterfactual_enhanced
"""

import json
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from common import (
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, TASK_DESCRIPTIONS,
    DIFFICULTY_TO_TASKS, PerTokenCollector, create_env,
    load_smolvla_policy, run_episode, save_video_frames,
)


PARAPHRASES = {
    "press": ["push down on", "tap", "hit", "activate"],
    "push": ["shove", "slide", "move forward", "nudge"],
    "pull": ["drag", "tug", "draw toward you", "yank"],
    "pick": ["grab", "grasp", "lift", "take"],
    "place": ["put", "set down", "position", "deposit"],
    "open": ["pull open", "swing open", "unlatch"],
    "close": ["shut", "push closed", "seal"],
    "reach": ["move to", "extend to", "go to", "touch"],
    "turn": ["rotate", "twist", "spin"],
    "insert": ["put in", "slide in", "push into"],
}

VAGUE_PROMPTS = [
    "do the task",
    "complete the objective",
    "perform the action",
    "do something with the object",
]

MEDIUM_SPECIFIC_PROMPTS = {
    "press": "interact with the button",
    "push": "move the object",
    "pull": "bring the object closer",
    "pick": "handle the object",
    "place": "put the object somewhere",
    "reach": "go near the target",
    "open": "access the container",
    "close": "seal the container",
    "turn": "manipulate the dial",
    "insert": "put it inside",
}


def generate_compositional_prompts(task_name, task_desc, all_tasks, all_descs):
    """Generate compositional prompts combining two task descriptions."""
    configs = []
    other_tasks = [t for t in all_tasks if t != task_name][:5]

    for other in other_tasks:
        other_desc = all_descs.get(other, other)
        comp_prompt = f"{task_desc} and then {other_desc.lower()}"
        configs.append({
            'task': task_name, 'prompt': comp_prompt,
            'category': 'compositional',
            'desc': f"{task_name}_comp_{other}",
            'composed_with': other,
        })

    for other in other_tasks[:2]:
        other_desc = all_descs.get(other, other)
        contra_prompt = f"{task_desc} but do not {other_desc.lower()}"
        configs.append({
            'task': task_name, 'prompt': contra_prompt,
            'category': 'contrastive_compositional',
            'desc': f"{task_name}_contra_{other}",
            'composed_with': other,
        })

    return configs


def generate_paraphrase_prompts(task_name, task_desc):
    """Generate paraphrased versions of the task description."""
    configs = []
    desc_lower = task_desc.lower()

    for verb, replacements in PARAPHRASES.items():
        if verb in desc_lower:
            for i, replacement in enumerate(replacements[:2]):
                para_desc = desc_lower.replace(verb, replacement, 1)
                para_desc = para_desc[0].upper() + para_desc[1:]
                configs.append({
                    'task': task_name, 'prompt': para_desc,
                    'category': 'paraphrase',
                    'desc': f"{task_name}_para_{i}",
                    'original_verb': verb, 'replacement': replacement,
                })
            break
    return configs


def generate_specificity_prompts(task_name, task_desc):
    """Generate prompts at different specificity levels."""
    configs = []
    desc_lower = task_desc.lower()

    configs.append({
        'task': task_name, 'prompt': VAGUE_PROMPTS[0],
        'category': 'specificity_vague',
        'desc': f"{task_name}_vague",
    })

    for verb, medium in MEDIUM_SPECIFIC_PROMPTS.items():
        if verb in desc_lower:
            configs.append({
                'task': task_name, 'prompt': medium,
                'category': 'specificity_medium',
                'desc': f"{task_name}_medium_specific",
            })
            break

    over_specific = f"{task_desc} carefully and precisely using the robot gripper"
    configs.append({
        'task': task_name, 'prompt': over_specific,
        'category': 'specificity_over',
        'desc': f"{task_name}_over_specific",
    })

    return configs


def generate_same_visual_different_goal(task_name, task_desc, all_tasks, all_descs):
    """Same visual scene (env), different goal instruction."""
    configs = []
    other_tasks = [t for t in all_tasks if t != task_name]
    selected = other_tasks[:6]

    for other in selected:
        other_desc = all_descs.get(other, other)
        configs.append({
            'task': task_name, 'prompt': other_desc,
            'category': 'same_visual_diff_goal',
            'desc': f"{task_name}_svdg_{other}",
            'intended_task': other,
        })

    return configs


def generate_all_enhanced_configs(tasks, task_descriptions, seeds=(42,)):
    all_configs = []

    for task_name in tasks:
        task_desc = task_descriptions.get(task_name, task_name)

        for seed in seeds:
            all_configs.append({
                'task': task_name, 'prompt': task_desc,
                'category': 'baseline', 'seed': seed,
                'desc': f"{task_name}_baseline_s{seed}",
            })

            comps = generate_compositional_prompts(
                task_name, task_desc, tasks, task_descriptions)
            for c in comps:
                c['seed'] = seed
                c['desc'] += f"_s{seed}"
            all_configs.extend(comps)

            paras = generate_paraphrase_prompts(task_name, task_desc)
            for p in paras:
                p['seed'] = seed
                p['desc'] += f"_s{seed}"
            all_configs.extend(paras)

            specs = generate_specificity_prompts(task_name, task_desc)
            for s in specs:
                s['seed'] = seed
                s['desc'] += f"_s{seed}"
            all_configs.extend(specs)

            svdg = generate_same_visual_different_goal(
                task_name, task_desc, tasks, task_descriptions)
            for s in svdg:
                s['seed'] = seed
                s['desc'] += f"_s{seed}"
            all_configs.extend(svdg)

    return all_configs


@dataclass
class EnhancedCounterfactualConfig:
    """SmolVLA MetaWorld enhanced counterfactual prompting."""

    output_dir: str
    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    difficulty: Optional[str] = None
    seeds: str = "42"
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

    tasks = []
    if cfg.tasks:
        tasks = [t.strip() for t in cfg.tasks.split(',')]
    elif cfg.difficulty:
        for diff in cfg.difficulty.split(','):
            diff = diff.strip()
            if diff in DIFFICULTY_TO_TASKS:
                tasks.extend(DIFFICULTY_TO_TASKS[diff])
    else:
        tasks = DIFFICULTY_TO_TASKS.get('easy', [])

    print(f"Tasks: {len(tasks)}, Device: {device}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device)

    collector = None
    if not cfg.no_activations:
        collector = PerTokenCollector(policy)
        collector.setup()

    configs = generate_all_enhanced_configs(tasks, TASK_DESCRIPTIONS, seeds=seeds)
    if cfg.end_idx:
        configs = configs[cfg.start_idx:cfg.end_idx]
    else:
        configs = configs[cfg.start_idx:]

    cat_counts = Counter(c['category'] for c in configs)
    print(f"\nTotal configs: {len(configs)}")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

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

        print(f"  [{idx+1}/{len(configs)}] {key} (cat={category}, prompt='{prompt[:50]}')")

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
        for extra_key in ['composed_with', 'intended_task', 'original_verb', 'replacement']:
            if extra_key in cfg:
                meta[extra_key] = cfg[extra_key]
        with open(metadata_path, 'a') as f:
            f.write(json.dumps(meta) + '\n')

    elapsed = time.time() - start_time
    print(f"\nDone! {len(configs)} configs in {elapsed/60:.1f} min")
    print(f"Output: {output_dir}")

    if collector:
        collector.cleanup()


if __name__ == "__main__":
    cfg = tyro.cli(EnhancedCounterfactualConfig)
    main(cfg)
