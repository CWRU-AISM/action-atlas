#!/usr/bin/env python3
"""
X-VLA SimplerEnv grid ablation: layer-by-layer zeroing across all tasks.

For each transformer block (0..23), zero its output and measure success
rate across all SimplerEnv tasks. Produces a [24 x N tasks] grid.

Examples:
    conda activate simpler_env

    python experiments/simplerenv/xvla_simplerenv_grid_ablation.py --model widowx --n-episodes 3
    python experiments/simplerenv/xvla_simplerenv_grid_ablation.py --model google-robot --n-episodes 3
    python experiments/simplerenv/xvla_simplerenv_grid_ablation.py --model widowx --layers 0 6 12 18 23 --n-episodes 1
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS, N_LAYERS,
    ZeroAblationHook, MeanAblationHook,
    load_xvla_policy, patch_eval_noop, run_episode,
    force_free_memory, log_ram, get_base_env,
    simpler_env,
)


@dataclass
class GridAblationConfig:
    """X-VLA SimplerEnv grid ablation: layer-by-layer zeroing across all tasks."""

    model: str
    """Model name: widowx, google-robot"""

    layers: Optional[List[int]] = None
    """Specific layers to ablate (default: all 24)"""

    tasks: Optional[List[str]] = None
    n_episodes: int = 3
    ablation_mode: str = "zero"
    """Ablation mode: zero, mean"""

    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 42
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    domain_id: Optional[int] = None


def main(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = MODEL_CONFIGS[cfg.model]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = cfg.domain_id if cfg.domain_id is not None else config["domain_id"]
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"rollouts/xvla_simplerenv_grid_ablation_{cfg.model}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = cfg.tasks if cfg.tasks else config["tasks"]
    print(f"Grid ablation: {cfg.model} | mode={cfg.ablation_mode} | "
          f"eps={cfg.n_episodes} | output={output_dir}")

    print(f"Loading model from {checkpoint}...")
    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)
    patch_eval_noop(policy)

    transformer_blocks = policy.model.transformer.blocks
    n_blocks = len(transformer_blocks)
    ablate_layers = cfg.layers if cfg.layers else list(range(n_blocks))

    print(f"  {n_blocks} blocks, ablating {len(ablate_layers)} layers")
    print(f"  Tasks: {task_names}")
    print(f"  Grid: {len(ablate_layers)} layers x {len(task_names)} tasks x {cfg.n_episodes} episodes")

    print("Creating environments...")
    task_envs = {}
    task_instructions = {}
    for task_name in task_names:
        env = simpler_env.make(task_name, max_episode_steps=cfg.max_steps)
        task_envs[task_name] = env
        obs, _ = env.reset(seed=0)
        task_instructions[task_name] = get_base_env(env).get_language_instruction()
        print(f"  {task_name}: \"{task_instructions[task_name]}\"")

    grid = defaultdict(lambda: defaultdict(dict))

    # Baseline (no ablation)
    print("\nRunning baseline (no ablation)...")
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_json = baseline_dir / "results.json"

    if baseline_json.exists():
        print("  [SKIP] Baseline already completed")
        with open(baseline_json) as f:
            baseline_results = json.load(f)
        for task_name, data in baseline_results.items():
            grid["baseline"][task_name] = data
    else:
        baseline_results = {}
        for task_name in task_names:
            env = task_envs[task_name]
            successes = 0
            for ep in range(cfg.n_episodes):
                result = run_episode(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=cfg.max_steps, seed=cfg.seed + ep,
                    save_video=(ep == 0),
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=ep, robot_type=cfg.model,
                )
                if result["success"]:
                    successes += 1
                ep_data = {
                    "condition": "baseline", "task": task_name, "episode": ep,
                    "success": result["success"], "n_steps": result["steps"],
                    "actions": [a.tolist() for a in result["actions"]],
                    "rewards": result["rewards"],
                    "obs_states": result["obs_states"],
                    "step_infos": result.get("step_infos"),
                    "episode_stats": result.get("episode_stats"),
                }
                with open(baseline_dir / f"{task_name}_ep{ep}.json", "w") as f:
                    json.dump(ep_data, f, indent=2)
                if result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(baseline_dir / f"{task_name}_ep{ep}.mp4"),
                                        result["frames"], fps=5)
                    except Exception:
                        pass

            rate = successes / cfg.n_episodes
            print(f"  {task_name}: {successes}/{cfg.n_episodes} = {rate:.1%}")
            grid["baseline"][task_name] = {
                "success_rate": rate, "successes": successes,
                "n_episodes": cfg.n_episodes,
            }
            baseline_results[task_name] = grid["baseline"][task_name]

        with open(baseline_json, "w") as f:
            json.dump(baseline_results, f, indent=2)

    # Single-layer ablations
    print(f"\nRunning single-layer ablations ({len(ablate_layers)} layers)...")
    for layer_idx in ablate_layers:
        layer_label = f"{cfg.ablation_mode}_L{layer_idx}"
        layer_dir = output_dir / layer_label
        layer_dir.mkdir(exist_ok=True)

        layer_json = layer_dir / "results.json"
        if layer_json.exists():
            print(f"\n  [SKIP] {layer_label} already completed")
            with open(layer_json) as f:
                layer_results = json.load(f)
            for task_name, data in layer_results.items():
                grid[layer_label][task_name] = data
            continue

        print(f"\n  Ablating layer {layer_idx}")
        if cfg.ablation_mode == "zero":
            hook = ZeroAblationHook()
        else:
            hook = MeanAblationHook()
        handle = transformer_blocks[layer_idx].register_forward_hook(hook)

        layer_results = {}
        for task_name in task_names:
            env = task_envs[task_name]
            successes = 0
            for ep in range(cfg.n_episodes):
                hook.call_count = 0
                hook.enabled = True
                result = run_episode(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=cfg.max_steps, seed=cfg.seed + ep,
                    save_video=(ep == 0),
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=ep, robot_type=cfg.model,
                )
                if result["success"]:
                    successes += 1
                ep_data = {
                    "condition": layer_label, "layer_ablated": layer_idx,
                    "task": task_name, "episode": ep,
                    "success": result["success"], "n_steps": result["steps"],
                    "actions": [a.tolist() for a in result["actions"]],
                    "rewards": result["rewards"],
                    "obs_states": result["obs_states"],
                    "step_infos": result.get("step_infos"),
                    "episode_stats": result.get("episode_stats"),
                }
                with open(layer_dir / f"{task_name}_ep{ep}.json", "w") as f:
                    json.dump(ep_data, f, indent=2)
                if result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(layer_dir / f"{task_name}_ep{ep}.mp4"),
                                        result["frames"], fps=5)
                    except Exception:
                        pass

            rate = successes / cfg.n_episodes
            baseline_rate = grid["baseline"].get(task_name, {}).get("success_rate", 0)
            delta = rate - baseline_rate
            print(f"    {task_name}: {successes}/{cfg.n_episodes} = {rate:.1%} (delta={delta:+.1%})")
            grid[layer_label][task_name] = {
                "success_rate": rate, "successes": successes,
                "n_episodes": cfg.n_episodes, "delta_from_baseline": delta,
            }
            layer_results[task_name] = grid[layer_label][task_name]

        handle.remove()
        del hook
        torch.cuda.empty_cache()
        with open(layer_json, "w") as f:
            json.dump(layer_results, f, indent=2)
        log_ram(f"after {layer_label}")

    for env in task_envs.values():
        env.close()

    # Summary
    print("\nGRID ABLATION RESULTS")
    header = f"{'Condition':<16}" + "".join(f"{t[:12]:<14}" for t in task_names) + f"{'Mean':<8}"
    print(header)

    all_grid_data = {}
    conditions = ["baseline"] + [f"{cfg.ablation_mode}_L{l}" for l in ablate_layers]
    for condition in conditions:
        row = f"{condition:<16}"
        rates = []
        for task_name in task_names:
            cell = grid[condition].get(task_name, {})
            rate = cell.get("success_rate", 0)
            rates.append(rate)
            row += f"{rate:<14.1%}"
        mean_rate = np.mean(rates) if rates else 0
        row += f"{mean_rate:<8.1%}"
        print(row)
        all_grid_data[condition] = {
            "per_task": {t: grid[condition].get(t, {}) for t in task_names},
            "mean_success_rate": float(mean_rate),
        }

    print(f"\nMost critical layers (biggest success drop):")
    layer_impacts = []
    for layer_idx in ablate_layers:
        label = f"{cfg.ablation_mode}_L{layer_idx}"
        deltas = [grid[label].get(t, {}).get("delta_from_baseline", 0) for t in task_names]
        layer_impacts.append((layer_idx, np.mean(deltas)))
    layer_impacts.sort(key=lambda x: x[1])
    for layer_idx, delta in layer_impacts[:5]:
        print(f"  Layer {layer_idx}: mean delta = {delta:+.1%}")

    duration = datetime.now() - start_time
    final = {
        "model": cfg.model, "checkpoint": checkpoint, "domain_id": domain_id,
        "ablation_mode": cfg.ablation_mode, "n_episodes": cfg.n_episodes,
        "max_steps": cfg.max_steps, "n_blocks": n_blocks,
        "ablated_layers": ablate_layers, "task_names": task_names,
        "task_instructions": task_instructions, "grid": all_grid_data,
        "most_critical_layers": [{"layer": l, "mean_delta": float(d)} for l, d in layer_impacts[:5]],
        "duration": str(duration),
    }
    with open(output_dir / "grid_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'grid_results.json'}")
    print(f"Duration: {duration}")


if __name__ == "__main__":
    cfg = tyro.cli(GridAblationConfig)
    main(cfg)
