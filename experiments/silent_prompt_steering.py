#!/usr/bin/env python3
"""
Silent-prompt goal steering in the VLM pathway.

The task instruction is replaced by a filler of function words with the same
token count. We record the VLM-pathway residual stream at the instruction span
under the true instruction and under the filler, then edit that span during a
filler rollout and score the simulator's success check for the true task.

Arms (``--arms``):
    floor_empty, floor_filler, ceiling   no edit ("" / filler / true instruction)
    patch                                span := mean state under the instruction
    patch_joint                          patch at every --layers at once
    gap                                  span += per-position mean(A) - mean(F)
    gap_mean                             span += gap averaged over positions
    direction                            span += rho * |h| * unit(mean gap)
    direction_pos                        span += rho * |h| * per-position unit gap
    contrast                             direction minus the other tasks' mean direction
    sae_topk                             direction from the top-n per-token SAE features
    sae_pt, sae_mp_pos, sae_mp_mean      gap reconstructed by the per-token SAE, the
                                         mean-pool SAE per position (a deliberate
                                         train/eval mismatch), or the mean-pool SAE
                                         on the pooled gap
    random:<edit>, wrong:<edit>          same edit with a norm-matched random vector,
                                         or with the vector of the next task in --tasks

Examples:
    python experiments/silent_prompt_steering.py --model pi05 --suite libero_goal \\
        --layers 2 --arms floor_filler ceiling patch direction random:direction wrong:direction

    python experiments/silent_prompt_steering.py --model xvla --suite libero_goal \\
        --layers 0 --arms floor_filler ceiling gap gap_mean sae_pt sae_mp_pos sae_mp_mean \\
        random:gap wrong:gap

    python experiments/silent_prompt_steering.py --model oft --suite libero_goal \\
        --layers 12 --arms floor_filler ceiling gap sae_pt random:gap wrong:gap
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.model_adapters import get_adapter
from experiments.prompt_spans import SPECS, differing_positions, match_filler
from experiments.sae_hooks import load_release_sae, sae_encode, sae_roundtrip
from experiments.utils import (
    OUTPUTS_DIR, SUITE_MAX_STEPS, force_free_memory, load_results, save_results, wilson,
)

RHO_EDITS = {"direction", "direction_pos", "contrast", "sae_topk"}
DELTA_EDITS = {"gap", "gap_mean", "sae_pt", "sae_mp_pos", "sae_mp_mean"}
REPLACE_EDITS = {"patch", "patch_joint"}
NO_EDIT = {"floor_empty", "floor_filler", "ceiling"}
SAE_POOLING = {"sae_topk": "per_token", "sae_pt": "per_token",
               "sae_mp_pos": "mean_pool", "sae_mp_mean": "mean_pool"}


@dataclass
class SteeringConfig:
    # Silent-prompt goal steering

    model: Literal["pi05", "smolvla", "xvla", "groot", "oft"] = "pi05"
    suite: str = "libero_goal"
    checkpoint: Optional[str] = None
    tasks: Optional[List[int]] = None

    layers: Tuple[int, ...] = (2,)
    """Layer indices within the model's VLM-pathway layer group."""

    arms: Tuple[str, ...] = ("floor_filler", "ceiling", "patch", "direction",
                             "random:direction", "wrong:direction")
    """Conditions to run; see the module docstring."""

    rhos: Tuple[float, ...] = (0.5, 1.0)
    """Steering strengths for direction-style arms, in units of the residual norm."""

    span: Optional[Literal["instruction", "tail", "vlm_block", "sequence"]] = None
    """
    Edited positions. tail runs to the end of the prompt, vlm_block is X-VLA only,
    sequence is OFT only. Default per model.
    """

    sae_topn: int = 32
    """SAE features combined by the sae_topk arm."""

    n_capture: int = 3
    """Rollouts per prompt used to estimate the span states."""

    n_episodes: int = 10
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    gpu: int = 0


@dataclass
class Task:
    idx: int
    env: Any
    prompt: str
    episode_kwargs: dict


@dataclass
class TaskCapture:
    # Per-layer span states under the instruction (A) and the filler (F): [calls, n_span, D]

    prompt: str
    filler: str
    span: List[int]
    seq_len: int
    states_a: Dict[int, torch.Tensor]
    states_f: Dict[int, torch.Tensor]
    capture_ok: Dict[str, List[bool]]

    def mean_a(self, layer):
        return self.states_a[layer].mean(0)

    def mean_f(self, layer):
        return self.states_f[layer].mean(0)

    def gap(self, layer):
        return self.mean_a(layer) - self.mean_f(layer)


def fit_rows(vec: torch.Tensor, n: int) -> torch.Tensor:
    # Truncate or repeat the last row so a per-position vector covers n positions
    if vec.dim() == 1 or vec.shape[0] == n:
        return vec
    if vec.shape[0] > n:
        return vec[:n]
    return torch.cat([vec, vec[-1:].expand(n - vec.shape[0], -1)])


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)


def random_like(vec: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    # Random directions with the same per-row norms
    r = unit(torch.randn(vec.shape, generator=gen))
    return r * vec.norm(dim=-1, keepdim=True)


def edit_span(h: torch.Tensor, span: torch.Tensor, kind: str, vec: torch.Tensor,
              rho: float = 0.0) -> torch.Tensor:
    # h: [B, S, D] residual. Returns a copy with the span positions edited
    out = h.clone()
    v = vec.to(device=h.device, dtype=h.dtype)
    seg = h[:, span]
    if kind == "replace":
        out[:, span] = v.expand_as(seg)
    elif kind == "delta":
        out[:, span] = seg + v
    elif kind == "rho":
        out[:, span] = seg + rho * seg.norm(dim=-1, keepdim=True) * v
    else:
        raise ValueError(kind)
    return out


def parse_arm(arm: str) -> Tuple[Optional[str], str]:
    control, _, edit = arm.rpartition(":")
    if control not in ("", "random", "wrong"):
        raise ValueError(f"unknown control {control!r} in arm {arm!r}")
    if edit not in RHO_EDITS | DELTA_EDITS | REPLACE_EDITS | NO_EDIT:
        raise ValueError(f"unknown edit {edit!r} in arm {arm!r}")
    if control == "random" and edit in REPLACE_EDITS:
        raise ValueError("random control is defined for additive edits only")
    return control or None, edit


def sae_poolings(arms) -> set:
    edits = (parse_arm(arm)[1] for arm in arms)
    return {SAE_POOLING[edit] for edit in edits if edit in SAE_POOLING}


@torch.no_grad()
def sae_vector(edit, cap: TaskCapture, layer, saes, topn):
    sae, mean, std = saes[(SAE_POOLING[edit], layer)]

    def code(x):
        return sae_encode(sae, x.flatten(0, 1).to(mean.device), mean, std).mean(0)

    def reconstruct(x):
        return sae_roundtrip(sae, x.to(mean.device), mean, std)[0].cpu()

    if edit == "sae_topk":
        dz = code(cap.states_a[layer]) - code(cap.states_f[layer])
        top = torch.topk(dz, topn).indices
        columns = sae.decoder.weight[:, top] * (std + 1e-8)[:, None]
        return unit((columns * dz[top]).sum(1)).cpu()
    ma, mf = cap.mean_a(layer), cap.mean_f(layer)
    if edit == "sae_mp_mean":
        pooled = reconstruct(ma.mean(0, keepdim=True)) - reconstruct(mf.mean(0, keepdim=True))
        return pooled.expand_as(ma).clone()
    return reconstruct(ma) - reconstruct(mf)


def base_vector(edit, cap: TaskCapture, layer, others, saes, topn):
    if edit in SAE_POOLING:
        return sae_vector(edit, cap, layer, saes, topn)
    if edit == "patch":
        return cap.mean_a(layer)
    gap = cap.gap(layer)
    if edit == "gap":
        return gap
    if edit == "gap_mean":
        return gap.mean(0, keepdim=True).expand_as(gap).clone()
    if edit == "direction":
        return unit(gap.mean(0))
    if edit == "direction_pos":
        return unit(gap)
    if edit == "contrast":
        rest = torch.stack([unit(o.gap(layer).mean(0)) for o in others]).mean(0)
        return unit(unit(gap.mean(0)) - rest)
    raise ValueError(f"unknown edit {edit!r}")


def arm_cells(arm, cap: TaskCapture, other: TaskCapture, captures, saes, cfg, gen):
    # (cell name, prompt, [(layer, kind, vector, rho)]) for each cell of one arm on one task
    control, edit = parse_arm(arm)
    if edit in NO_EDIT:
        text = {"floor_empty": "", "floor_filler": cap.filler, "ceiling": cap.prompt}[edit]
        return [(arm, text, [])]

    source = other if control == "wrong" else cap
    rest = [c for c in captures if c is not source]

    def vector(layer):
        base = "patch" if edit == "patch_joint" else edit
        vec = fit_rows(base_vector(base, source, layer, rest, saes, cfg.sae_topn), len(cap.span))
        return random_like(vec, gen) if control == "random" else vec

    if edit == "patch_joint":
        edits = [(layer, "replace", vector(layer), 0.0) for layer in cfg.layers]
        return [(f"{arm}_L{'-'.join(map(str, cfg.layers))}", cap.filler, edits)]

    kind = "replace" if edit in REPLACE_EDITS else "delta" if edit in DELTA_EDITS else "rho"
    cells = []
    for layer in cfg.layers:
        vec = vector(layer)
        if kind == "rho":
            cells += [(f"{arm}_L{layer}_rho{rho}", cap.filler, [(layer, kind, vec, rho)])
                      for rho in cfg.rhos]
        else:
            cells.append((f"{arm}_L{layer}", cap.filler, [(layer, kind, vec, 0.0)]))
    return cells


def rollout(adapter, task: Task, text, handles, seed, max_steps):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        return adapter.run_episode(task.env, text, max_steps=max_steps, seed=seed,
                                   **task.episode_kwargs)
    finally:
        for h in handles:
            h.remove()


def capture_task(adapter, spec, site, task: Task, span_mode, cfg) -> TaskCapture:
    count = spec.tokenize(task.env, cfg.seed)
    ids_a, n_a = count(task.prompt)
    filler_text = match_filler(lambda t: count(t)[1], n_a)
    ids_f, _ = count(filler_text)

    acts = {(who, layer): [] for who in "AF" for layer in cfg.layers}
    ok = {"A": [], "F": []}
    for ep in range(cfg.n_capture):
        for who, text in (("A", task.prompt), ("F", filler_text)):
            hooks, handles = {}, []
            for layer in cfg.layers:
                hooks[layer], layer_handles = site.capture(layer)
                handles += layer_handles
            result = rollout(adapter, task, text, handles, cfg.seed + 1000 + ep, cfg.max_steps)
            ok[who].append(bool(result["success"]))
            for layer in cfg.layers:
                acts[(who, layer)] += [a.squeeze(0) for a in hooks[layer].activations]

    seq_len = Counter(a.shape[0] for a in acts[("A", cfg.layers[0])]).most_common(1)[0][0]
    span = spec.span(span_mode, differing_positions(ids_a, ids_f), seq_len, len(ids_a))
    states = {key: torch.stack([a[span].float() for a in seq if a.shape[0] == seq_len])
              for key, seq in acts.items()}
    return TaskCapture(task.prompt, filler_text, span, seq_len,
                       {layer: states[("A", layer)] for layer in cfg.layers},
                       {layer: states[("F", layer)] for layer in cfg.layers}, ok)


def run_cell(adapter, site, task: Task, cap: TaskCapture, text, edits, cfg) -> List[bool]:
    span = torch.as_tensor(cap.span)
    outcomes = []
    for ep in range(cfg.n_episodes):
        handles = []
        for layer, kind, vec, rho in edits:
            fn = partial(edit_span, span=span, kind=kind, vec=vec, rho=rho)
            handles += site.edit(layer, cap.seq_len, fn)
        result = rollout(adapter, task, text, handles, cfg.seed + ep, cfg.max_steps)
        outcomes.append(bool(result["success"]))
    return outcomes


def summarize(cells: Dict[str, Dict[str, List[bool]]]) -> Dict[str, dict]:
    totals: Dict[str, List[int]] = {}
    for per_task in cells.values():
        for cell, outcomes in per_task.items():
            t = totals.setdefault(cell, [0, 0])
            t[0] += sum(outcomes)
            t[1] += len(outcomes)
    return {cell: {"successes": s, "n": n, "rate": s / n if n else 0.0, "wilson95": wilson(s, n)}
            for cell, (s, n) in totals.items()}


def main(cfg: SteeringConfig):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)
    output_dir = Path(cfg.output_dir or OUTPUTS_DIR / f"{cfg.model}_experiments"
                      / f"silent_prompt_steering_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for arm in cfg.arms:
        parse_arm(arm)

    adapter = get_adapter(cfg.model)
    checkpoint = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    if not checkpoint:
        raise ValueError(f"No default checkpoint for {cfg.model}/{cfg.suite}. Pass --checkpoint.")
    adapter.load_model(checkpoint, device)
    spec = SPECS[cfg.model](adapter)
    site = spec.site()
    span_mode = cfg.span or spec.default_span
    saes = {(pooling, layer): load_release_sae(cfg.model, pooling, layer, device)
            for pooling in sae_poolings(cfg.arms) for layer in cfg.layers}

    _, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks or list(range(len(all_tasks)))
    if len(task_ids) < 2 and any(a.startswith("wrong:") for a in cfg.arms):
        raise ValueError("wrong:<edit> needs at least two tasks; "
                         "with one task the partner is the task itself")
    tasks = []
    for t in task_ids:
        env, prompt, info = adapter.create_env(t, suite=cfg.suite, max_steps=cfg.max_steps)
        tasks.append(Task(t, env, prompt, spec.episode_kwargs(info)))
    print(f"Silent-prompt steering: {cfg.model} on {cfg.suite}, tasks {task_ids}, span {span_mode}")
    if spec.warmup:
        rollout(adapter, tasks[0], tasks[0].prompt, [], cfg.seed, cfg.max_steps)

    captures = []
    for task in tasks:
        c = capture_task(adapter, spec, site, task, span_mode, cfg)
        captures.append(c)
        print(f"task {task.idx}: {task.prompt!r} | filler {c.filler!r} | "
              f"span {c.span[0]}..{c.span[-1]} ({len(c.span)}) | "
              f"capture A {sum(c.capture_ok['A'])}/{cfg.n_capture} "
              f"F {sum(c.capture_ok['F'])}/{cfg.n_capture}")

    results_path = output_dir / "results.json"
    results = load_results(results_path) or {"config": vars(cfg) | {"span": span_mode}, "cells": {}}
    gen = torch.Generator().manual_seed(cfg.seed)
    start = time.time()

    for i, (task, cap) in enumerate(zip(tasks, captures)):
        if str(task.idx) in results["cells"]:
            continue
        other = captures[(i + 1) % len(captures)]
        cells = {}
        for arm in cfg.arms:
            for name, text, edits in arm_cells(arm, cap, other, captures, saes, cfg, gen):
                cells[name] = run_cell(adapter, site, task, cap, text, edits, cfg)
                print(f"  task {task.idx} {name}: {sum(cells[name])}/{cfg.n_episodes}")
        results["cells"][str(task.idx)] = cells
        save_results(results, results_path)
        force_free_memory()

    summary = summarize(results["cells"])
    save_results({"model": cfg.model, "suite": cfg.suite, "checkpoint": checkpoint,
                  "tasks": task_ids, "span": span_mode, "cells": summary,
                  "duration_seconds": time.time() - start},
                 output_dir / "summary.json")
    print(f"\nSilent-prompt steering ({cfg.model} / {cfg.suite})")
    for cell, s in summary.items():
        lo, hi = s["wilson95"]
        print(f"{cell:<32} {s['successes']:>4}/{s['n']:<4} {s['rate']:>6.1%}  [{lo:.1%}, {hi:.1%}]")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(SteeringConfig))
