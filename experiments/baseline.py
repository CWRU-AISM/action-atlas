#!/usr/bin/env python3
"""Baseline rollouts with optional activation collection for any VLA model.

Runs clean episodes (no interventions) and optionally captures per-layer
activations for downstream SAE training or analysis.

Examples:
    # Basic baseline
    python experiments/baseline.py --model xvla --suite libero_object

    # With activation collection
    python experiments/baseline.py --model groot --suite libero_goal --collect-activations

    # Quick test
    python experiments/baseline.py --model smolvla --suite libero_spatial \\
        --n-episodes 1 --tasks 0 1 2
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.hooks import ActivationCollector
from experiments.model_adapters import get_adapter
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video,
    SUITE_MAX_STEPS,
)


@dataclass
class BaselineConfig:
    """Baseline rollout experiment."""

    model: str = "xvla"
    """Model name: xvla, smolvla, groot, pi05"""

    suite: str = "libero_object"
    """Task suite."""

    checkpoint: Optional[str] = None
    n_episodes: int = 3
    tasks: Optional[List[int]] = None
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = True

    gpu: int = 0

    n_action_steps: Optional[int] = None
    """Override action chunk size for faster inference."""
    """GPU device index."""

    collect_activations: bool = False
    """Capture per-layer activations (for SAE training)."""

    per_token: bool = True
    """Store per-token activations (vs mean-pooled)."""

    subsample_every: int = 1
    """Collect activations every Nth step (saves memory)."""


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/baseline_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")

    print(f"Baseline: {cfg.model} on {cfg.suite}")
    adapter.load_model(checkpoint, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    # Optional activation collection
    collector = None
    if cfg.collect_activations:
        collector = ActivationCollector(per_token=cfg.per_token,
                                        subsample_every=cfg.subsample_every)
        for label, module in adapter.get_all_layers():
            gated = "dit" in label.lower()
            collector.register(module, label, gated=gated)
        print(f"  Collecting activations from {len(collector.handles)} layers")

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))
    print(f"  tasks: {len(task_ids)}, episodes: {cfg.n_episodes}")

    all_results = {}
    start_time = time.time()

    for tid in task_ids:
        _, task_obj, task_desc = all_tasks[tid]
        env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
        successes = 0

        for ep in range(cfg.n_episodes):
            if collector:
                collector.clear()

            ep_result = adapter.run_episode(
                env, desc, max_steps=max_steps,
                save_video=(cfg.record_video and ep == 0),
                seed=cfg.seed + ep,
                collector=collector,
            )
            if ep_result["success"]:
                successes += 1
            if ep_result.get("frames"):
                save_video(ep_result["frames"],
                           output_dir / f"task{tid}_ep{ep}.mp4")

            # Save activations
            if collector and collector.activations:
                act_dir = output_dir / "activations" / f"task{tid}" / f"ep{ep}"
                act_dir.mkdir(parents=True, exist_ok=True)
                acts = collector.get_activations()
                for layer_name, tensor in acts.items():
                    torch.save(tensor, act_dir / f"{layer_name}.pt")

        if hasattr(env, "close"):
            env.close()

        rate = successes / cfg.n_episodes
        print(f"  Task {tid}: {successes}/{cfg.n_episodes} = {rate:.0%} ({desc})")
        all_results[str(tid)] = {
            "task_description": desc,
            "success_rate": rate,
            "successes": successes,
            "n_episodes": cfg.n_episodes,
        }
        force_free_memory()

    elapsed = time.time() - start_time
    all_results["_meta"] = {
        "model": cfg.model,
        "suite": cfg.suite,
        "checkpoint": checkpoint,
        "duration_seconds": elapsed,
    }
    save_results(all_results, output_dir / "results.json")
    print(f"\nDone in {elapsed/60:.1f} min. Results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    cfg = tyro.cli(BaselineConfig)
    main(cfg)
