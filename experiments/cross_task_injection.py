#!/usr/bin/env python3
"""
Cross-task activation injection for any supported VLA model.

Two-phase experiment:
1. Capture: Run source tasks, record per-layer activations every forward pass.
2. Inject:  Run target task environment but replay source task activations.
            Does the robot follow the injection or the actual environment?

Tests both directions (A->B and B->A) for each task pair.

Examples:
    # Capture activations first
    python experiments/cross_task_injection.py --model xvla --suite libero_object \\
        --phase capture --tasks 0 1 2 3

    # Then inject
    python experiments/cross_task_injection.py --model xvla --suite libero_object \\
        --phase inject --pairs 0,1 2,3

    # Full pipeline (capture + inject)
    python experiments/cross_task_injection.py --model groot --suite libero_goal \\
        --phase both --pairs 0,1 0,2
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.hooks import ActivationCaptureHook, ActivationInjectionHook
from experiments.model_adapters import get_adapter
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video,
    get_scene_state, summarize_scene, SUITE_MAX_STEPS,
)


@dataclass
class CrossTaskInjectionConfig:
    # Cross-task activation injection experiment

    model: str = "xvla"
    suite: str = "libero_object"
    checkpoint: Optional[str] = None
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = True

    gpu: int = 0

    n_action_steps: Optional[int] = None
    # Override action chunk size for faster inference
    # GPU device index

    phase: str = "both"
    # Phase to run: 'capture', 'inject', or 'both'

    tasks: Optional[List[int]] = None
    # Tasks to capture activations for (capture phase)

    pairs: Optional[List[str]] = None
    """
    Task pairs for injection as 'A,B' strings (inject phase).
    Default: all unique pairs from captured tasks."""

    layers: Optional[List[str]] = None
    # Layer labels to capture/inject. Default: all


def parse_pairs(pairs_str: List[str]) -> List[Tuple[int, int]]:
    result = []
    for p in pairs_str:
        a, b = p.split(",")
        result.append((int(a), int(b)))
    return result


def capture_activations(adapter, suite, task_ids, layers, max_steps, seed, output_dir):
    # Phase 1: Run each task and capture all forward-pass activations
    capture_dir = output_dir / "captures"
    capture_dir.mkdir(exist_ok=True)

    task_suite, all_tasks = adapter.setup_suite(suite)

    for tid in task_ids:
        task_file = capture_dir / f"task{tid}.pt"
        if task_file.exists():
            print(f"  [SKIP] Task {tid} already captured")
            continue

        _, task_obj, task_desc = all_tasks[tid]
        env, desc, meta = adapter.create_env(tid, suite=suite, max_steps=max_steps)

        # Register capture hooks
        hooks = {}
        handles = []
        for label, module in layers:
            hook = ActivationCaptureHook()
            handle = module.register_forward_hook(hook)
            hooks[label] = hook
            handles.append(handle)

        ep_result = adapter.run_episode(
            env, desc, max_steps=max_steps, seed=seed,
            save_video=False,
        )

        # Save captured activations
        captured = {}
        for label, hook in hooks.items():
            if hook.activations:
                captured[label] = hook.activations
        torch.save(captured, task_file)
        print(f"  Task {tid}: {len(captured)} layers, "
              f"{sum(len(v) for v in captured.values())} total activations, "
              f"success={ep_result['success']}")

        for h in handles:
            h.remove()
        if hasattr(env, "close"):
            env.close()
        force_free_memory()

    return capture_dir


def inject_activations(adapter, suite, pairs, capture_dir, layers, max_steps, seed,
                       output_dir, record_video):
    # Phase 2: For each pair (A,B), inject A's activations into B's environment
    task_suite, all_tasks = adapter.setup_suite(suite)
    inject_dir = output_dir / "injections"
    inject_dir.mkdir(exist_ok=True)

    for task_a, task_b in pairs:
        for src, tgt in [(task_a, task_b), (task_b, task_a)]:
            pair_label = f"inject_{src}_into_{tgt}"
            pair_dir = inject_dir / pair_label
            pair_dir.mkdir(exist_ok=True)
            result_json = pair_dir / "results.json"

            if result_json.exists():
                print(f"  [SKIP] {pair_label}")
                continue

            # Load source activations
            src_file = capture_dir / f"task{src}.pt"
            if not src_file.exists():
                print(f"  [SKIP] {pair_label}: no capture for task {src}")
                continue

            src_activations = torch.load(src_file, map_location="cpu")

            # Create target environment
            _, task_obj, task_desc = all_tasks[tgt]
            env, desc, meta = adapter.create_env(tgt, suite=suite, max_steps=max_steps)

            # Register injection hooks
            handles = []
            injection_hooks = {}
            for label, module in layers:
                if label in src_activations:
                    hook = ActivationInjectionHook(
                        src_activations[label], device=adapter.device
                    )
                    handle = module.register_forward_hook(hook)
                    handles.append(handle)
                    injection_hooks[label] = hook

            ep_result = adapter.run_episode(
                env, desc, max_steps=max_steps, seed=seed,
                save_video=record_video,
            )

            if ep_result.get("frames"):
                save_video(ep_result["frames"], pair_dir / "video.mp4")

            inject_stats = {}
            for label, hook in injection_hooks.items():
                inject_stats[label] = {
                    "injections": hook.injection_count,
                    "shape_mismatches": hook.shape_mismatches,
                }

            result = {
                "source_task": src,
                "target_task": tgt,
                "source_desc": all_tasks[src][2],
                "target_desc": desc,
                "success": ep_result["success"],
                "steps": ep_result["steps"],
                "injection_stats": inject_stats,
            }
            if ep_result.get("scene_states"):
                result["scene_summary"] = summarize_scene(ep_result["scene_states"])

            save_results(result, result_json)
            print(f"  {pair_label}: success={ep_result['success']}, "
                  f"steps={ep_result['steps']}")

            for h in handles:
                h.remove()
            if hasattr(env, "close"):
                env.close()
            force_free_memory()


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/cross_task_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")

    print(f"Cross-Task Injection: {cfg.model} on {cfg.suite}, phase={cfg.phase}")
    adapter.load_model(checkpoint, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    # Determine layers
    all_layers = adapter.get_all_layers()
    if cfg.layers:
        label_set = set(cfg.layers)
        layers = [(l, m) for l, m in all_layers if l in label_set]
    else:
        layers = all_layers

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    start_time = time.time()

    # Phase 1: Capture
    if cfg.phase in ("capture", "both"):
        print("\nPhase 1: Capturing activations...")
        capture_dir = capture_activations(
            adapter, cfg.suite, task_ids, layers,
            max_steps, cfg.seed, output_dir,
        )
    else:
        capture_dir = output_dir / "captures"

    # Phase 2: Inject
    if cfg.phase in ("inject", "both"):
        if cfg.pairs:
            pairs = parse_pairs(cfg.pairs)
        else:
            # All unique pairs from task_ids
            pairs = [(a, b) for i, a in enumerate(task_ids)
                     for b in task_ids[i+1:]]

        print(f"\nPhase 2: Injecting across {len(pairs)} pairs...")
        inject_activations(
            adapter, cfg.suite, pairs, capture_dir, layers,
            max_steps, cfg.seed, output_dir, cfg.record_video,
        )

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/60:.1f} min. Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CrossTaskInjectionConfig)
    main(cfg)
