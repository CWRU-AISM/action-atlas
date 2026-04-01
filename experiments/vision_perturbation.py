#!/usr/bin/env python3
"""Visual robustness testing for any supported VLA model.

Applies 24 image perturbations (noise, blur, color, spatial, extreme)
and measures success rate degradation compared to clean baseline.

Examples:
    python experiments/vision_perturbation.py --model xvla --suite libero_object

    python experiments/vision_perturbation.py --model groot --suite libero_goal \\
        --perturbations baseline gaussian_noise_low blur_heavy black_image

    python experiments/vision_perturbation.py --model smolvla --suite libero_spatial \\
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

from experiments.model_adapters import get_adapter
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video,
    get_standard_perturbations, SUITE_MAX_STEPS,
)


@dataclass
class VisionPerturbationConfig:
    """Visual robustness testing experiment."""

    model: str = "xvla"
    """Model name: xvla, smolvla, groot, pi05"""

    suite: str = "libero_object"
    """Task suite."""

    checkpoint: Optional[str] = None
    """Model checkpoint. Uses default if not set."""

    n_episodes: int = 3
    """Episodes per (perturbation, task) cell."""

    tasks: Optional[List[int]] = None
    """Task indices. Default: all."""

    perturbations: Optional[List[str]] = None
    """Perturbation names to test. Default: all 24 standard perturbations."""

    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = True

    gpu: int = 0

    n_action_steps: Optional[int] = None
    """Override action chunk size for faster inference."""
    """GPU device index."""


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/vision_perturbation_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")

    print(f"Vision Perturbation: {cfg.model} on {cfg.suite}")
    adapter.load_model(checkpoint, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    all_perturbations = get_standard_perturbations()
    if cfg.perturbations:
        name_set = set(cfg.perturbations)
        perturbations = [(n, fn) for n, fn in all_perturbations if n in name_set]
    else:
        perturbations = all_perturbations

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    print(f"  perturbations: {len(perturbations)}, tasks: {len(task_ids)}, "
          f"episodes/cell: {cfg.n_episodes}")

    results = {}
    start_time = time.time()

    for p_idx, (p_name, p_fn) in enumerate(perturbations):
        print(f"\n[{p_idx+1}/{len(perturbations)}] {p_name}")

        p_dir = output_dir / p_name
        p_dir.mkdir(exist_ok=True)
        p_json = p_dir / "results.json"
        existing = load_results(p_json)

        if existing:
            print(f"  [SKIP] Already completed")
            results[p_name] = existing
            continue

        p_results = {}
        for tid in task_ids:
            _, task_obj, task_desc = all_tasks[tid]
            env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
            successes = 0

            perturbation_fn = None if p_name == "baseline" else p_fn

            for ep in range(cfg.n_episodes):
                ep_result = adapter.run_episode(
                    env, desc, max_steps=max_steps,
                    save_video=(cfg.record_video and ep == 0),
                    perturbation_fn=perturbation_fn,
                    seed=cfg.seed + ep,
                )
                if ep_result["success"]:
                    successes += 1
                if ep_result.get("frames"):
                    save_video(ep_result["frames"], p_dir / f"task{tid}_ep{ep}.mp4")

            if hasattr(env, "close"):
                env.close()

            rate = successes / cfg.n_episodes
            print(f"  Task {tid}: {rate:.0%} ({desc})")
            p_results[str(tid)] = {
                "task_description": desc,
                "success_rate": rate,
                "successes": successes,
                "n_episodes": cfg.n_episodes,
            }

        avg_rate = np.mean([v["success_rate"] for v in p_results.values()])
        p_results["_avg_success_rate"] = float(avg_rate)
        save_results(p_results, p_json)
        results[p_name] = p_results
        force_free_memory()

    elapsed = time.time() - start_time
    baseline_avg = results.get("baseline", {}).get("_avg_success_rate", 0)

    print(f"\nVision Perturbation Results ({cfg.model} / {cfg.suite})")
    print(f"{'Perturbation':<25} {'Avg SR':>8} {'Delta':>8}")
    print("-" * 42)
    for p_name, _ in perturbations:
        if p_name in results:
            avg = results[p_name].get("_avg_success_rate", 0)
            delta = avg - baseline_avg
            print(f"  {p_name:<23} {avg:>7.0%} {delta:>+7.0%}")

    summary = {
        "model": cfg.model,
        "suite": cfg.suite,
        "checkpoint": checkpoint,
        "n_episodes": cfg.n_episodes,
        "task_ids": task_ids,
        "perturbation_results": {
            p: {"avg_success_rate": results[p].get("_avg_success_rate", 0)}
            for p in results
        },
        "duration_seconds": elapsed,
    }
    save_results(summary, output_dir / "summary.json")
    print(f"\nResults saved to {output_dir}")
    print(f"Duration: {elapsed/60:.1f} min")


if __name__ == "__main__":
    cfg = tyro.cli(VisionPerturbationConfig)
    main(cfg)
