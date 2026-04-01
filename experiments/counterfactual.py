#!/usr/bin/env python3
"""Counterfactual prompting experiment for any supported VLA model.

Tests how models respond to modified language inputs: null prompts,
nonsense, negation, wrong objects, generic motor commands, etc.
Optionally captures activations for downstream concept analysis.

Examples:
    python experiments/counterfactual.py --model xvla --suite libero_object

    python experiments/counterfactual.py --model groot --suite libero_goal \\
        --conditions null_prompt negation generic

    python experiments/counterfactual.py --model smolvla --suite libero_spatial \\
        --collect-activations --tasks 0 1
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.hooks import ActivationCollector
from experiments.model_adapters import get_adapter
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video,
    SUITE_MAX_STEPS, COUNTERFACTUAL_PROMPTS,
)


@dataclass
class CounterfactualConfig:
    """Counterfactual prompting experiment."""

    model: str = "xvla"
    suite: str = "libero_object"
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

    conditions: Optional[List[str]] = None
    """Prompt conditions to test. Default: all (null_prompt, random, negation,
    opposite, generic). Format: 'negation' uses 'do not {task}'."""

    collect_activations: bool = False
    """Capture activations under each condition."""

    per_token: bool = True


def make_prompt(condition: str, task_desc: str) -> str:
    """Generate the counterfactual prompt for a condition."""
    template = COUNTERFACTUAL_PROMPTS.get(condition, condition)
    if "{task}" in template:
        return template.format(task=task_desc)
    return template


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/counterfactual_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")

    print(f"Counterfactual: {cfg.model} on {cfg.suite}")
    adapter.load_model(checkpoint, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    # Activation collector
    collector = None
    if cfg.collect_activations:
        collector = ActivationCollector(per_token=cfg.per_token)
        for label, module in adapter.get_all_layers():
            gated = "dit" in label.lower()
            collector.register(module, label, gated=gated)

    conditions = cfg.conditions or list(COUNTERFACTUAL_PROMPTS.keys())
    # Always include baseline (original prompt)
    if "baseline" not in conditions:
        conditions = ["baseline"] + conditions

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    print(f"  conditions: {conditions}")
    print(f"  tasks: {len(task_ids)}, episodes: {cfg.n_episodes}")

    results = {}
    start_time = time.time()

    for cond in conditions:
        print(f"\nCondition: {cond}")
        cond_dir = output_dir / cond
        cond_dir.mkdir(exist_ok=True)
        cond_json = cond_dir / "results.json"

        existing = load_results(cond_json)
        if existing:
            print(f"  [SKIP] Already completed")
            results[cond] = existing
            continue

        cond_results = {}
        for tid in task_ids:
            _, task_obj, task_desc = all_tasks[tid]
            env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)

            # Determine prompt
            if cond == "baseline":
                prompt = desc
            else:
                prompt = make_prompt(cond, desc)

            successes = 0
            for ep in range(cfg.n_episodes):
                if collector:
                    collector.clear()

                ep_result = adapter.run_episode(
                    env, prompt, max_steps=max_steps,
                    save_video=(cfg.record_video and ep == 0),
                    seed=cfg.seed + ep,
                    collector=collector,
                )
                if ep_result["success"]:
                    successes += 1
                if ep_result.get("frames"):
                    save_video(ep_result["frames"],
                               cond_dir / f"task{tid}_ep{ep}.mp4")

                if collector and collector.activations:
                    act_dir = cond_dir / "activations" / f"task{tid}" / f"ep{ep}"
                    act_dir.mkdir(parents=True, exist_ok=True)
                    for name, tensor in collector.get_activations().items():
                        torch.save(tensor, act_dir / f"{name}.pt")

            if hasattr(env, "close"):
                env.close()

            rate = successes / cfg.n_episodes
            print(f"  Task {tid}: {rate:.0%} (prompt: '{prompt[:50]}...')")
            cond_results[str(tid)] = {
                "task_description": desc,
                "prompt_used": prompt,
                "success_rate": rate,
                "successes": successes,
                "n_episodes": cfg.n_episodes,
            }

        avg = np.mean([v["success_rate"] for v in cond_results.values()])
        cond_results["_avg_success_rate"] = float(avg)
        save_results(cond_results, cond_json)
        results[cond] = cond_results
        force_free_memory()

    elapsed = time.time() - start_time
    baseline_avg = results.get("baseline", {}).get("_avg_success_rate", 0)

    print(f"\nCounterfactual Results ({cfg.model} / {cfg.suite})")
    print(f"{'Condition':<20} {'Avg SR':>8} {'Delta':>8}")
    print("-" * 38)
    for cond in conditions:
        if cond in results:
            avg = results[cond].get("_avg_success_rate", 0)
            delta = avg - baseline_avg
            print(f"  {cond:<18} {avg:>7.0%} {delta:>+7.0%}")

    summary = {
        "model": cfg.model, "suite": cfg.suite, "checkpoint": checkpoint,
        "conditions": conditions, "n_episodes": cfg.n_episodes,
        "task_ids": task_ids, "duration_seconds": elapsed,
    }
    save_results(summary, output_dir / "summary.json")
    print(f"\nDone in {elapsed/60:.1f} min. Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CounterfactualConfig)
    main(cfg)
