#!/usr/bin/env python3
"""
SmolVLA MetaWorld cross-task activation injection.

Optimized approach:
1. Run each task once with capture hooks, cache activations in memory (LRU, max 4 tasks)
2. For each pair (A, B), inject A's cached activations into B and vice versa
3. Only run injection episodes (no redundant baseline re-runs)

Injection groups (default 8):
  expert_all, vlm_all, expert_early, expert_mid, expert_late, vlm_early, vlm_mid, vlm_late
Optionally add per-layer groups with --per-layer.

Usage:
    python experiments/metaworld/smolvla_metaworld_cross_task_injection.py --difficulty easy --resume
    python experiments/metaworld/smolvla_metaworld_cross_task_injection.py --tasks reach-v3,push-v3,pick-place-v3 --per-layer
"""

import gc
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

from common import (
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, MAX_STEPS, TASK_DESCRIPTIONS,
    MLPCaptureHook, MLPInjectionHook, cosine_similarity, create_env,
    force_free_memory, get_layer_modules, get_scene_state,
    get_tasks_from_args, load_smolvla_policy, save_video_frames,
)
from common import preprocess_observation


def run_episode_with_capture(policy, env, preprocessor, postprocessor, device,
                             expert_layers, vlm_layers,
                             save_trajectory=False, save_frames=False):
    """Run one episode, capture all MLP activations. Returns (result, captured)."""
    capture_hooks = {}
    handles = []
    for i, layer in enumerate(expert_layers):
        hook = MLPCaptureHook()
        h = layer.mlp.register_forward_hook(hook)
        capture_hooks[f'expert_{i}'] = hook
        handles.append(h)
    for i, layer in enumerate(vlm_layers):
        hook = MLPCaptureHook()
        h = layer.mlp.register_forward_hook(hook)
        capture_hooks[f'vlm_{i}'] = hook
        handles.append(h)

    policy.reset()
    observation, info = env.reset()
    actions = []
    agent_pos_traj = []
    scene_states = []
    frames = []

    if save_trajectory:
        scene_states.append(get_scene_state(env))

    for step in range(MAX_STEPS):
        if save_frames:
            frames.append(observation['pixels'].copy())
        if save_trajectory and 'agent_pos' in observation:
            agent_pos_traj.append(observation['agent_pos'].tolist())

        obs_tensor = preprocess_observation(observation)
        obs_tensor['task'] = [env.task_description]
        obs_tensor = preprocessor(obs_tensor)
        with torch.inference_mode():
            action = policy.select_action(obs_tensor)
        action_out = postprocessor(action)
        action_np = action_out.cpu().numpy() if isinstance(action_out, torch.Tensor) else action_out
        if action_np.ndim == 2:
            action_np = action_np[0]
        actions.append(action_np[:4].copy())
        observation, reward, terminated, truncated, step_info = env.step(actions[-1])
        if save_trajectory:
            scene_states.append(get_scene_state(env))
        if terminated or truncated:
            break

    for h in handles:
        h.remove()

    success = step_info.get("is_success", False) if actions else False
    captured = {name: list(hook.activations) for name, hook in capture_hooks.items()}

    result = {
        'success': bool(success),
        'n_steps': len(actions),
        'actions': np.array(actions) if actions else np.zeros((0, 4)),
    }
    if save_trajectory:
        result['agent_pos'] = agent_pos_traj
        result['scene_states'] = scene_states
    if save_frames:
        result['frames'] = frames
    return result, captured


def run_injection_episode(policy, env, preprocessor, postprocessor, device,
                          injection_hooks,
                          save_trajectory=False, save_frames=False):
    """Run episode with activation injection. Returns result dict."""
    policy.reset()
    observation, info = env.reset()
    actions = []
    agent_pos_traj = []
    scene_states = []
    frames = []

    if save_trajectory:
        scene_states.append(get_scene_state(env))

    for step in range(MAX_STEPS):
        if save_frames:
            frames.append(observation['pixels'].copy())
        if save_trajectory and 'agent_pos' in observation:
            agent_pos_traj.append(observation['agent_pos'].tolist())

        obs_tensor = preprocess_observation(observation)
        obs_tensor['task'] = [env.task_description]
        obs_tensor = preprocessor(obs_tensor)
        with torch.inference_mode():
            action = policy.select_action(obs_tensor)
        action_out = postprocessor(action)
        action_np = action_out.cpu().numpy() if isinstance(action_out, torch.Tensor) else action_out
        if action_np.ndim == 2:
            action_np = action_np[0]
        actions.append(action_np[:4].copy())
        observation, reward, terminated, truncated, step_info = env.step(actions[-1])
        if save_trajectory:
            scene_states.append(get_scene_state(env))
        if terminated or truncated:
            break

    success = step_info.get("is_success", False) if actions else False
    total_injections = sum(h.injection_count for h in injection_hooks.values())
    result = {
        'success': bool(success),
        'n_steps': len(actions),
        'actions': np.array(actions) if actions else np.zeros((0, 4)),
        'total_injections': total_injections,
    }
    if save_trajectory:
        result['agent_pos'] = agent_pos_traj
        result['scene_states'] = scene_states
    if save_frames:
        result['frames'] = frames
    return result


@dataclass
class CrossTaskInjectionConfig:
    """SmolVLA MetaWorld cross-task activation injection."""

    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    difficulty: Optional[str] = None
    pairs: Optional[List[str]] = None
    per_layer: bool = False
    """Add per-layer injection groups (64 extra conditions per pair)"""

    resolution: int = DEFAULT_RESOLUTION
    save_video: bool = False
    save_trajectory: bool = False
    """Save actions, agent_pos, scene_states per episode"""

    resume: bool = False
    output_dir: Optional[str] = None


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = get_tasks_from_args(cfg)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"rollouts/smolvla/metaworld_cross_task_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {len(tasks)}, Device: {device}")
    print(f"Output: {output_dir}")

    print("\nLoading model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device)

    expert_layers, vlm_layers = get_layer_modules(policy)
    n_expert = len(expert_layers)
    n_vlm = len(vlm_layers)

    inject_layer_groups = {}
    inject_layer_groups['expert_all'] = list(range(n_expert))
    inject_layer_groups['vlm_all'] = list(range(n_vlm))
    third_e = n_expert // 3
    inject_layer_groups['expert_early'] = list(range(0, third_e))
    inject_layer_groups['expert_mid'] = list(range(third_e, 2 * third_e))
    inject_layer_groups['expert_late'] = list(range(2 * third_e, n_expert))
    third_v = n_vlm // 3
    inject_layer_groups['vlm_early'] = list(range(0, third_v))
    inject_layer_groups['vlm_mid'] = list(range(third_v, 2 * third_v))
    inject_layer_groups['vlm_late'] = list(range(2 * third_v, n_vlm))

    if cfg.per_layer:
        for i in range(n_expert):
            inject_layer_groups[f'expert_L{i}'] = [i]
        for i in range(n_vlm):
            inject_layer_groups[f'vlm_L{i}'] = [i]

    print(f"Injection groups: {len(inject_layer_groups)}")

    if cfg.pairs:
        pairs = []
        for p in cfg.pairs:
            parts = p.split(',')
            pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        pairs = list(combinations(tasks, 2))

    print(f"Total pairs: {len(pairs)}")

    CACHE_SIZE = 4
    task_cache = OrderedDict()

    def get_task_data(task_name):
        if task_name in task_cache:
            task_cache.move_to_end(task_name)
            return task_cache[task_name]
        print(f"    [capture] {task_name}...", end=" ", flush=True)
        env = create_env(task_name, cfg.resolution)
        result, captured = run_episode_with_capture(
            policy, env, preprocessor, postprocessor, device,
            expert_layers, vlm_layers,
            save_trajectory=cfg.save_trajectory,
            save_frames=cfg.save_video,
        )
        env.close()
        status = "OK" if result['success'] else f"FAIL({result['n_steps']})"
        print(status, flush=True)
        task_cache[task_name] = {'result': result, 'captured': captured}
        while len(task_cache) > CACHE_SIZE:
            evicted = task_cache.popitem(last=False)
            del evicted
            gc.collect()
        return task_cache[task_name]

    all_results = {
        'checkpoint': cfg.checkpoint, 'env': 'metaworld',
        'max_steps': MAX_STEPS, 'resolution': cfg.resolution,
        'n_injection_groups': len(inject_layer_groups),
        'injection_groups': list(inject_layer_groups.keys()),
        'timestamp': datetime.now().isoformat(), 'pairs': {},
    }

    print(f"\nRunning {len(pairs)} pairs x {len(inject_layer_groups)} groups x 2 directions...")

    for pair_idx, (task_a, task_b) in enumerate(pairs):
        pair_key = f"cross_task_{task_a}_{task_b}"
        pair_file = output_dir / f"{pair_key}.json"

        if cfg.resume and pair_file.exists():
            with open(pair_file) as f:
                all_results['pairs'][pair_key] = json.load(f)
            continue

        t1 = time.time()
        data_a = get_task_data(task_a)
        data_b = get_task_data(task_b)

        pair_result = {
            'task_a': task_a, 'task_b': task_b,
            'task_a_desc': TASK_DESCRIPTIONS.get(task_a, task_a),
            'task_b_desc': TASK_DESCRIPTIONS.get(task_b, task_b),
            'baseline_A': {
                'success': data_a['result']['success'],
                'n_steps': data_a['result']['n_steps'],
                'actions': data_a['result']['actions'].tolist(),
            },
            'baseline_B': {
                'success': data_b['result']['success'],
                'n_steps': data_b['result']['n_steps'],
                'actions': data_b['result']['actions'].tolist(),
            },
        }

        if cfg.save_trajectory:
            traj_dir = output_dir / "trajectories" / pair_key
            traj_dir.mkdir(parents=True, exist_ok=True)
            for label, task, data in [('A', task_a, data_a), ('B', task_b, data_b)]:
                traj_data = {
                    'task': task, 'condition': 'baseline',
                    'success': data['result']['success'],
                    'n_steps': data['result']['n_steps'],
                    'actions': data['result']['actions'].tolist(),
                    'agent_pos_trajectory': data['result'].get('agent_pos', []),
                    'scene_states': data['result'].get('scene_states', []),
                }
                with open(traj_dir / f"baseline_{label}.json", 'w') as f:
                    json.dump(traj_data, f)

        for direction in ['A_into_B', 'B_into_A']:
            if direction == 'A_into_B':
                source_task, target_task = task_a, task_b
                target_label = 'B'
            else:
                source_task, target_task = task_b, task_a
                target_label = 'A'

            captured = get_task_data(source_task)['captured']

            for group_name, layer_indices in inject_layer_groups.items():
                injection_hooks = {}
                handles = []

                for idx in layer_indices:
                    if group_name.startswith('expert'):
                        hook_key = f'expert_{idx}'
                    else:
                        hook_key = f'vlm_{idx}'

                    if hook_key in captured and captured[hook_key]:
                        hook = MLPInjectionHook(captured[hook_key], device=str(device))
                        if hook_key.startswith('expert'):
                            h = expert_layers[idx].mlp.register_forward_hook(hook)
                        else:
                            h = vlm_layers[idx].mlp.register_forward_hook(hook)
                        injection_hooks[hook_key] = hook
                        handles.append(h)

                env = create_env(target_task, cfg.resolution)
                result = run_injection_episode(
                    policy, env, preprocessor, postprocessor, device, injection_hooks,
                    save_trajectory=cfg.save_trajectory,
                    save_frames=cfg.save_video,
                )
                env.close()
                for h in handles:
                    h.remove()

                target_actions = pair_result[f'baseline_{target_label}']['actions']
                target_arr = np.array(target_actions) if isinstance(target_actions, list) else target_actions
                cos_sim = cosine_similarity(result['actions'], target_arr)

                key = f"inject_{direction}_{group_name}"
                pair_result[key] = {
                    'success': result['success'],
                    'n_steps': result['n_steps'],
                    'actions': result['actions'].tolist(),
                    'total_injections': result['total_injections'],
                    'cosine_sim_with_target_baseline': cos_sim,
                }

                if cfg.save_trajectory:
                    traj_dir = output_dir / "trajectories" / pair_key
                    traj_dir.mkdir(parents=True, exist_ok=True)
                    traj_data = {
                        'source_task': source_task, 'target_task': target_task,
                        'direction': direction, 'group': group_name,
                        'success': result['success'],
                        'n_steps': result['n_steps'],
                        'actions': result['actions'].tolist(),
                        'agent_pos_trajectory': result.get('agent_pos', []),
                        'scene_states': result.get('scene_states', []),
                    }
                    with open(traj_dir / f"{key}.json", 'w') as f:
                        json.dump(traj_data, f)

                if cfg.save_video and result.get('frames'):
                    vid_dir = output_dir / "videos" / pair_key
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    save_video_frames(vid_dir / f"{key}.mp4", result['frames'], fps=10)

        elapsed = time.time() - t1
        n_ok = sum(1 for k, v in pair_result.items()
                   if k.startswith('inject_') and v.get('success'))
        n_inj = sum(1 for k in pair_result if k.startswith('inject_'))
        print(f"  [{pair_idx+1}/{len(pairs)}] ({task_a}, {task_b}): "
              f"{n_ok}/{n_inj} success, {elapsed:.1f}s")

        with open(pair_file, 'w') as f:
            json.dump(pair_result, f, indent=2)
        all_results['pairs'][pair_key] = pair_result

        if (pair_idx + 1) % 10 == 0 or pair_idx == len(pairs) - 1:
            with open(output_dir / "results.json", 'w') as f:
                json.dump(all_results, f, indent=2, default=str)

        force_free_memory()

    with open(output_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone! Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CrossTaskInjectionConfig)
    main(cfg)
