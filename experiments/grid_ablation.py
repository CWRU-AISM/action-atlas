#!/usr/bin/env python3
"""
Layer-by-layer ablation across all tasks for any supported VLA model.

Zero (or mean-replace) each layer's output one at a time and measure
success rate. Produces a [layers x tasks] grid showing which layers
are critical for which tasks.

Examples:
    # X-VLA on libero_object, all 24 transformer blocks
    python experiments/grid_ablation.py --model xvla --suite libero_object

    # SmolVLA, specific layers only
    python experiments/grid_ablation.py --model smolvla --suite libero_goal \\
        --layers expert_0 expert_4 vlm_0 vlm_4

    # GR00T Eagle layers
    python experiments/grid_ablation.py --model groot --suite libero_goal \\
        --layer-group eagle

    # Fewer episodes for quick test
    python experiments/grid_ablation.py --model xvla --suite libero_spatial \\
        --n-episodes 1 --tasks 0 1 2
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import gc
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.hooks import ZeroAblationHook, MeanAblationHook
from experiments.model_adapters import get_adapter, list_models
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video,
    get_scene_state, summarize_scene, SUITE_MAX_STEPS,
)


@dataclass
class GridAblationConfig:
    # Layer-by-layer ablation experiment

    model: str = "xvla"
    # Model name: xvla, smolvla, groot, pi05

    suite: str = "libero_object"
    # Task suite: libero_spatial, libero_object, libero_goal, libero_10

    checkpoint: Optional[str] = None
    # Model checkpoint path. Uses model default if not set

    n_episodes: int = 3
    # Episodes per (layer, task) cell

    tasks: Optional[List[int]] = None
    # Task indices to evaluate. Default: all tasks in suite

    layers: Optional[List[str]] = None
    """
    Layer labels to ablate (e.g. 'transformer_L0', 'expert_4').
    Default: all layers in model."""

    layer_group: Optional[str] = None
    """
    Ablate only layers in this group (e.g. 'eagle', 'dit', 'expert').
    Overridden by --layers if both are set."""

    ablation_mode: str = "zero"
    # Ablation method: 'zero' (replace with zeros) or 'mean' (running mean)

    max_steps: Optional[int] = None
    # Max episode steps. Default: suite-specific value

    seed: int = 42
    # Random seed for episode resets

    output_dir: Optional[str] = None
    # Output directory. Auto-generated if not set

    record_video: bool = True
    # Save video of first episode per condition

    gpu: int = 0
    # GPU device index

    n_action_steps: Optional[int] = None
    """
    Override action chunk size (default: use checkpoint config).
    Higher values = fewer model forward passes = faster, but may affect
    accuracy if the model was trained with n_action_steps=1."""


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    # Output directory
    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/grid_ablation_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")

    print(f"Grid Ablation: {cfg.model} on {cfg.suite}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  output: {output_dir}")
    adapter.load_model(checkpoint, device)

    # Override action chunk size if requested
    if cfg.n_action_steps and hasattr(adapter, 'policy') and hasattr(adapter.policy, 'config'):
        adapter.policy.config.n_action_steps = cfg.n_action_steps
        print(f"  n_action_steps: {cfg.n_action_steps}")

    # Determine layers to ablate
    all_layers = adapter.get_all_layers()  # [(label, module), ...]
    if cfg.layers:
        # Filter to requested labels
        label_set = set(cfg.layers)
        layers_to_test = [(l, m) for l, m in all_layers if l in label_set]
    elif cfg.layer_group:
        groups = adapter.get_layer_groups()
        if cfg.layer_group not in groups:
            raise ValueError(f"Unknown layer group '{cfg.layer_group}'. "
                             f"Available: {list(groups.keys())}")
        target = groups[cfg.layer_group]
        layers_to_test = [(l, m) for l, m in all_layers if m in target]
    else:
        layers_to_test = all_layers

    # Set up tasks
    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    print(f"  layers: {len(layers_to_test)}, tasks: {len(task_ids)}, "
          f"episodes/cell: {cfg.n_episodes}")
    print(f"  grid size: {len(layers_to_test)} x {len(task_ids)} x {cfg.n_episodes} "
          f"= {len(layers_to_test) * len(task_ids) * cfg.n_episodes} episodes")

    grid = defaultdict(dict)

    # 1. Baseline (no ablation)
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_json = baseline_dir / "results.json"
    baseline_results = load_results(baseline_json)

    if baseline_results:
        print("\n  [SKIP] Baseline already completed, loading...")
        for tid_str, data in baseline_results.items():
            grid["baseline"][int(tid_str)] = data
    else:
        print("\nRunning baseline (no ablation)...")
        baseline_results = {}
        for tid in task_ids:
            _, task_obj, task_desc = all_tasks[tid]
            env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
            successes = 0

            for ep in range(cfg.n_episodes):
                ep_result = adapter.run_episode(
                    env, desc, max_steps=max_steps,
                    save_video=(cfg.record_video and ep == 0),
                    seed=cfg.seed + ep,
                )
                if ep_result["success"]:
                    successes += 1
                if ep_result.get("frames"):
                    save_video(ep_result["frames"],
                               baseline_dir / f"task{tid}_ep{ep}.mp4")

            if hasattr(env, "close"):
                env.close()

            rate = successes / cfg.n_episodes
            print(f"  Task {tid}: {successes}/{cfg.n_episodes} = {rate:.0%} ({desc})")
            grid["baseline"][tid] = {
                "success_rate": rate, "successes": successes,
                "n_episodes": cfg.n_episodes,
            }
            baseline_results[str(tid)] = grid["baseline"][tid]

        save_results(baseline_results, baseline_json)

    # 2. Per-layer ablations
    print(f"\nRunning {len(layers_to_test)} layer ablations...")
    start_time = time.time()

    for layer_idx, (layer_label, layer_module) in enumerate(layers_to_test):
        layer_dir = output_dir / layer_label
        layer_dir.mkdir(exist_ok=True)
        layer_json = layer_dir / "results.json"

        existing = load_results(layer_json)
        if existing:
            print(f"  [SKIP] {layer_label} already completed")
            for tid_str, data in existing.items():
                grid[layer_label][int(tid_str)] = data
            continue

        print(f"\n  Ablating {layer_label} [{layer_idx+1}/{len(layers_to_test)}]")
        hook = ZeroAblationHook() if cfg.ablation_mode == "zero" else MeanAblationHook()
        handle = layer_module.register_forward_hook(hook)

        layer_results = {}
        for tid in task_ids:
            _, task_obj, task_desc = all_tasks[tid]
            env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
            successes = 0

            for ep in range(cfg.n_episodes):
                hook.call_count = 0
                hook.enabled = True

                ep_result = adapter.run_episode(
                    env, desc, max_steps=max_steps,
                    save_video=(cfg.record_video and ep == 0),
                    seed=cfg.seed + ep,
                )
                if ep_result["success"]:
                    successes += 1
                if ep_result.get("frames"):
                    save_video(ep_result["frames"],
                               layer_dir / f"task{tid}_ep{ep}.mp4")

            if hasattr(env, "close"):
                env.close()

            rate = successes / cfg.n_episodes
            bl_rate = grid["baseline"].get(tid, {}).get("success_rate", 0)
            delta = rate - bl_rate
            print(f"    Task {tid}: {rate:.0%} (delta {delta:+.0%}) ({desc})")

            grid[layer_label][tid] = {
                "success_rate": rate, "successes": successes,
                "n_episodes": cfg.n_episodes, "delta_from_baseline": delta,
            }
            layer_results[str(tid)] = grid[layer_label][tid]

        handle.remove()
        del hook
        force_free_memory()
        save_results(layer_results, layer_json)

    # 3. Summary
    elapsed = time.time() - start_time
    print(f"\nGrid Ablation Results ({cfg.model} / {cfg.suite})")

    header = f"{'Layer':<20}" + "".join(f"T{t:<5}" for t in task_ids) + f"{'Avg':>6}"
    print(header)
    print("-" * len(header))

    all_grid_data = {}
    conditions = ["baseline"] + [l for l, _ in layers_to_test]
    for condition in conditions:
        row = f"{condition:<20}"
        rates = []
        for tid in task_ids:
            rate = grid[condition].get(tid, {}).get("success_rate", 0)
            rates.append(rate)
            row += f"{rate:<6.0%}"
        avg = np.mean(rates) if rates else 0
        row += f"{avg:>5.0%}"
        print(row)
        all_grid_data[condition] = {
            "per_task": {str(tid): grid[condition].get(tid, {}) for tid in task_ids},
            "mean_success_rate": float(avg),
        }

    # Most critical layers
    layer_impacts = []
    for label, _ in layers_to_test:
        deltas = [grid[label].get(tid, {}).get("delta_from_baseline", 0) for tid in task_ids]
        layer_impacts.append((label, float(np.mean(deltas))))
    layer_impacts.sort(key=lambda x: x[1])

    print(f"\nMost critical layers (biggest drop):")
    for label, delta in layer_impacts[:5]:
        print(f"  {label}: mean delta = {delta:+.1%}")

    summary = {
        "model": cfg.model,
        "suite": cfg.suite,
        "checkpoint": checkpoint,
        "ablation_mode": cfg.ablation_mode,
        "n_episodes": cfg.n_episodes,
        "task_ids": task_ids,
        "grid": all_grid_data,
        "most_critical_layers": [{"layer": l, "mean_delta": d} for l, d in layer_impacts[:5]],
        "duration_seconds": elapsed,
    }
    save_results(summary, output_dir / "grid_results.json")
    print(f"\nResults saved to {output_dir / 'grid_results.json'}")
    print(f"Duration: {elapsed/60:.1f} min")


if __name__ == "__main__":
    cfg = tyro.cli(GridAblationConfig)
    main(cfg)
