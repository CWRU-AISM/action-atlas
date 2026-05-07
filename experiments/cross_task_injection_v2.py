#!/usr/bin/env python3
"""
Cross-task activation injection v2 (with override metrics).

Patched from cross_task_injection.py:
  - Caches baseline activations + actions in an LRU
  - Injects per pair in both directions (A->B and B->A)
  - Saves baseline + injection action arrays
  - Computes cos_to_src and cos_to_dst override metrics per injection
  - Adds --layer-group shorthand: expert_all, vlm_all (and per-pathway thirds)

Result schema (per-pair JSON) mirrors
ActionAtlas/experiments/metaworld/smolvla_metaworld_cross_task_injection.py:

  {
    "task_a": int, "task_b": int,
    "task_a_desc": str, "task_b_desc": str,
    "baseline_A": {success, n_steps, actions: list[list[float]]},
    "baseline_B": {success, n_steps, actions: list[list[float]]},
    "inject_<dir>_<group>": {
        "source_task", "target_task",
        "success", "n_steps",
        "actions", "total_injections",
        "cos_to_src", "cos_to_dst",
        "override": bool   # cos_to_src > cos_to_dst
    },
    ...
  }

Usage:
    CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl TORCH_COMPILE_DISABLE=1 \\
        python experiments/cross_task_injection_v2.py \\
            --model smolvla --suite libero_object \\
            --layer-group expert_all \\
            --pairs 0,1 0,2  --resume
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import gc
import json
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path(__file__).parent.parent))
_LIBERO_PATH = os.environ.get("LIBERO_PATH", "")
if _LIBERO_PATH and os.path.isdir(_LIBERO_PATH):
    sys.path.insert(0, _LIBERO_PATH)
_LEROBOT_SRC = os.environ.get("LEROBOT_SRC", "")
if _LEROBOT_SRC and os.path.isdir(_LEROBOT_SRC):
    sys.path.insert(0, _LEROBOT_SRC)

from experiments.hooks import ActivationCaptureHook, ActivationInjectionHook
from experiments.model_adapters import get_adapter
from experiments.utils import (
    force_free_memory, save_results, save_video,
    summarize_scene, SUITE_MAX_STEPS,
)


def cosine_similarity(a, b):
    # Cosine similarity of two action arrays (flattened, length-aligned)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    af = a[:n].flatten()
    bf = b[:n].flatten()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(af, bf) / (na * nb))


@dataclass
class CrossTaskInjectionV2Config:
    # Cross-task activation injection v2 with override metrics

    model: str = "smolvla"
    suite: str = "libero_object"
    checkpoint: Optional[str] = None
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = False
    save_trajectory: bool = False

    n_action_steps: Optional[int] = None

    pairs: Optional[List[str]] = None
    # Task pairs as 'A,B' strings. Default: all C(n_tasks,2) pairs

    tasks: Optional[List[int]] = None
    # Restrict to a subset of task IDs (default: all)

    layer_group: Optional[str] = None
    # Shorthand: 'expert_all', 'vlm_all', 'expert_early|mid|late', 'vlm_early|mid|late', 'all'

    layers: Optional[List[str]] = None
    # Explicit layer labels (overrides --layer-group)

    cache_size: int = 4
    resume: bool = True


def expand_layer_group(group: str, all_layers: List[Tuple[str, object]]) -> List[str]:
    """
    Resolve a layer-group shorthand against an adapter's labelled layer list.

    SmolVLAAdapter.get_all_layers() yields labels:
        expert_L0, expert_L1, ..., vlm_L0, vlm_L1, ...
    """
    expert_labels = [lbl for lbl, _ in all_layers if lbl.startswith("expert_")]
    vlm_labels = [lbl for lbl, _ in all_layers if lbl.startswith("vlm_")]

    def thirds(labels, which):
        n = len(labels)
        t = max(1, n // 3)
        if which == "early":
            return labels[:t]
        if which == "mid":
            return labels[t:2 * t]
        return labels[2 * t:]

    if group == "expert_all":
        return expert_labels
    if group == "vlm_all":
        return vlm_labels
    if group == "all":
        return expert_labels + vlm_labels
    if group.startswith("expert_"):
        which = group.split("_", 1)[1]
        return thirds(expert_labels, which)
    if group.startswith("vlm_"):
        which = group.split("_", 1)[1]
        return thirds(vlm_labels, which)
    raise ValueError(f"Unknown layer group: {group}")


def parse_pairs(pairs_str: List[str]) -> List[Tuple[int, int]]:
    out = []
    for p in pairs_str:
        a, b = p.split(",")
        out.append((int(a), int(b)))
    return out


def run_baseline_with_capture(adapter, suite, task_id, all_layers,
                              max_steps, seed, save_video):
    """
    Run baseline episode for `task_id`, capturing every layer in all_layers.

    Returns (result_dict, captured_dict[label -> List[Tensor]]).
    """
    env, desc, _ = adapter.create_env(task_id, suite=suite, max_steps=max_steps)

    capture_hooks = {}
    handles = []
    for label, module in all_layers:
        hook = ActivationCaptureHook()
        h = module.register_forward_hook(hook)
        capture_hooks[label] = hook
        handles.append(h)

    ep = adapter.run_episode(env, desc, max_steps=max_steps, seed=seed,
                             save_video=save_video)

    for h in handles:
        h.remove()
    if hasattr(env, "close"):
        env.close()

    captured = {}
    for label, hook in capture_hooks.items():
        if hook.activations:
            captured[label] = hook.activations
    return ep, desc, captured


def run_injection(adapter, suite, target_task, captured, layers_to_inject,
                  max_steps, seed, save_video):
    env, desc, _ = adapter.create_env(target_task, suite=suite, max_steps=max_steps)

    injection_hooks = {}
    handles = []
    for label, module in layers_to_inject:
        if label in captured and captured[label]:
            hook = ActivationInjectionHook(captured[label], device=adapter.device)
            h = module.register_forward_hook(hook)
            injection_hooks[label] = hook
            handles.append(h)

    ep = adapter.run_episode(env, desc, max_steps=max_steps, seed=seed,
                             save_video=save_video)

    total_injections = sum(h.injection_count for h in injection_hooks.values())
    shape_mismatches = sum(h.shape_mismatches for h in injection_hooks.values())

    for h in handles:
        h.remove()
    if hasattr(env, "close"):
        env.close()
    return ep, desc, total_injections, shape_mismatches


def main(cfg: CrossTaskInjectionV2Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gname = cfg.layer_group or "custom"
        output_dir = Path(
            f"rollouts/{cfg.model}/cross_task_v2/{cfg.suite}_{gname}_{ts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v2] {cfg.model} | suite={cfg.suite} | layer_group={cfg.layer_group} "
          f"| device={device}")
    print(f"[v2] Output: {output_dir}")

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}.")

    adapter.load_model(checkpoint, device)
    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    all_layers = adapter.get_all_layers()
    if cfg.layers:
        keep = set(cfg.layers)
        layers_to_inject = [(l, m) for l, m in all_layers if l in keep]
    elif cfg.layer_group:
        keep = set(expand_layer_group(cfg.layer_group, all_layers))
        layers_to_inject = [(l, m) for l, m in all_layers if l in keep]
    else:
        layers_to_inject = all_layers

    if not layers_to_inject:
        raise ValueError("Empty layer set after resolving --layer-group/--layers.")

    print(f"[v2] Injecting {len(layers_to_inject)} layers: "
          f"{[l for l,_ in layers_to_inject[:3]]}...{[l for l,_ in layers_to_inject[-3:]]}")

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))
    if cfg.pairs:
        pairs = parse_pairs(cfg.pairs)
    else:
        pairs = list(combinations(task_ids, 2))

    print(f"[v2] {len(pairs)} pairs x 2 directions = {2 * len(pairs)} injections")

    # Always capture all layers (cheap), inject only the requested subset.
    capture_layers = all_layers
    cache: "OrderedDict[int, dict]" = OrderedDict()

    def get_baseline(task_id):
        if task_id in cache:
            cache.move_to_end(task_id)
            return cache[task_id]
        t0 = time.time()
        ep, desc, captured = run_baseline_with_capture(
            adapter, cfg.suite, task_id, capture_layers,
            max_steps, cfg.seed, save_video=False,
        )
        dt = time.time() - t0
        print(f"  [capture task {task_id}] success={ep['success']} "
              f"steps={ep['steps']} ({dt:.1f}s) desc='{desc[:60]}'")
        cache[task_id] = {"ep": ep, "desc": desc, "captured": captured}
        while len(cache) > cfg.cache_size:
            cache.popitem(last=False)
            gc.collect()
        return cache[task_id]

    aggregate = {
        "model": cfg.model, "suite": cfg.suite,
        "checkpoint": checkpoint,
        "layer_group": cfg.layer_group,
        "n_layers_injected": len(layers_to_inject),
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(),
        "pairs": {},
    }

    t_start = time.time()
    for pair_idx, (task_a, task_b) in enumerate(pairs):
        gname = cfg.layer_group or "custom"
        pair_key = f"pair_{task_a}_{task_b}_{gname}"
        pair_file = output_dir / f"{pair_key}.json"

        if cfg.resume and pair_file.exists():
            try:
                with open(pair_file) as f:
                    aggregate["pairs"][pair_key] = json.load(f)
                continue
            except Exception:
                pass

        t1 = time.time()
        data_a = get_baseline(task_a)
        data_b = get_baseline(task_b)

        a_actions = data_a["ep"]["actions"]
        b_actions = data_b["ep"]["actions"]

        pair_result = {
            "task_a": task_a, "task_b": task_b,
            "task_a_desc": data_a["desc"],
            "task_b_desc": data_b["desc"],
            "layer_group": cfg.layer_group,
            "baseline_A": {
                "success": bool(data_a["ep"]["success"]),
                "n_steps": int(data_a["ep"]["steps"]),
                "actions": (a_actions.tolist() if isinstance(a_actions, np.ndarray)
                            else list(a_actions)),
            },
            "baseline_B": {
                "success": bool(data_b["ep"]["success"]),
                "n_steps": int(data_b["ep"]["steps"]),
                "actions": (b_actions.tolist() if isinstance(b_actions, np.ndarray)
                            else list(b_actions)),
            },
        }

        for direction in ("A_into_B", "B_into_A"):
            if direction == "A_into_B":
                src_id, tgt_id = task_a, task_b
                src_actions, tgt_actions = a_actions, b_actions
            else:
                src_id, tgt_id = task_b, task_a
                src_actions, tgt_actions = b_actions, a_actions

            src_data = get_baseline(src_id)
            captured = src_data["captured"]

            ep, desc, ninj, mismatches = run_injection(
                adapter, cfg.suite, tgt_id, captured, layers_to_inject,
                max_steps, cfg.seed, save_video=cfg.record_video,
            )
            inj_actions = ep["actions"]

            cos_src = cosine_similarity(inj_actions, src_actions)
            cos_dst = cosine_similarity(inj_actions, tgt_actions)
            override = bool(cos_src > cos_dst)

            key = f"inject_{direction}_{cfg.layer_group or 'custom'}"
            pair_result[key] = {
                "source_task": src_id,
                "target_task": tgt_id,
                "success": bool(ep["success"]),
                "n_steps": int(ep["steps"]),
                "actions": (inj_actions.tolist() if isinstance(inj_actions, np.ndarray)
                            else list(inj_actions)),
                "total_injections": int(ninj),
                "shape_mismatches": int(mismatches),
                "cos_to_src": cos_src,
                "cos_to_dst": cos_dst,
                "override": override,
            }

        # Save per-pair atomically
        save_results(pair_result, pair_file)
        aggregate["pairs"][pair_key] = pair_result

        n_inj = sum(1 for k in pair_result if k.startswith("inject_"))
        n_override = sum(1 for k in pair_result
                         if k.startswith("inject_") and pair_result[k].get("override"))
        elapsed = time.time() - t1
        print(f"  [{pair_idx + 1}/{len(pairs)}] ({task_a},{task_b}) "
              f"override={n_override}/{n_inj} succ_a={pair_result['baseline_A']['success']} "
              f"succ_b={pair_result['baseline_B']['success']} ({elapsed:.1f}s)")

        if (pair_idx + 1) % 5 == 0 or pair_idx == len(pairs) - 1:
            with open(output_dir / "aggregate.json", "w") as f:
                json.dump(aggregate, f, indent=2, default=str)

        force_free_memory()

    with open(output_dir / "aggregate.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\n[v2] Done in {(time.time() - t_start) / 60:.1f} min. "
          f"{len(aggregate['pairs'])} pairs -> {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CrossTaskInjectionV2Config)
    main(cfg)
