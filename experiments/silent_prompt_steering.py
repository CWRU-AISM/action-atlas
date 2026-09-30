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
                                         mean-pool SAE per position, or the mean-pool
                                         SAE on the pooled gap
    random:<edit>, wrong:<edit>          same edit with a norm-matched random vector,
                                         or with the vector of the next task in --tasks

Examples:
    python experiments/silent_prompt_steering.py --model pi05 --suite libero_goal \\
        --layers 2 --arms floor_filler ceiling patch direction random:direction wrong:direction

    python experiments/silent_prompt_steering.py --model xvla --suite libero_goal \\
        --layers 0 --arms floor_filler ceiling gap gap_mean sae_pt sae_mp_pos sae_mp_mean random:gap wrong:gap
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import copy
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.hooks import ActivationCaptureHook
from experiments.model_adapters import _ensure_attended_language, get_adapter
from experiments.sae_hooks import TopKSAE
from experiments.utils import SUITE_MAX_STEPS, force_free_memory, load_results, save_results

FILLER_WORDS = ("the", "of", "and", "to", "in", "a", "is", "it", "on", "at",
                "by", "for", "with", "as", "or", "an", "be", "this", "that", "from")
LANG_TOKENS = "observation.language.tokens"
LANG_MASK = "observation.language.attention_mask"

RHO_EDITS = {"direction", "direction_pos", "contrast", "sae_topk"}
DELTA_EDITS = {"gap", "gap_mean", "sae_pt", "sae_mp_pos", "sae_mp_mean"}
REPLACE_EDITS = {"patch", "patch_joint"}
NO_EDIT = {"floor_empty", "floor_filler", "ceiling"}

# (HF repo, file pattern) of the released residual-stream SAEs used at the edit site
SAE_RELEASE = {
    "pi05": ("bag100/action-atlas-pi05", "saes/{pooling}/paligemma/sae_layer{layer}.safetensors"),
    "xvla": ("bag100/action-atlas-xvla", "saes/{pooling}/libero/sae_layer{layer}.safetensors"),
    "groot": ("bag100/action-atlas-groot", "saes/{pooling}/eagle/sae_layer{layer}.safetensors"),
}


@dataclass
class SteeringConfig:
    # Silent-prompt goal steering

    model: Literal["pi05", "smolvla", "xvla", "groot"] = "pi05"
    suite: str = "libero_goal"
    checkpoint: Optional[str] = None
    tasks: Optional[List[int]] = None
    layers: Tuple[int, ...] = (2,)
    arms: Tuple[str, ...] = ("floor_filler", "ceiling", "patch", "direction",
                             "random:direction", "wrong:direction")
    rhos: Tuple[float, ...] = (0.5, 1.0)
    span: Optional[Literal["instruction", "tail", "vlm_block"]] = None
    """Edited positions; tail runs to the end of the prompt, vlm_block is X-VLA only. Default per model."""
    sae_topn: int = 32
    n_capture: int = 3
    n_episodes: int = 10
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    gpu: int = 0


def filler(n_words: int) -> str:
    return " ".join(FILLER_WORDS[i % len(FILLER_WORDS)] for i in range(n_words))


def match_filler(count_tokens, target: int, max_words: int = 40) -> str:
    # Shortest filler whose attended-token count equals the instruction's
    for n in range(1, max_words + 1):
        text = filler(n)
        if count_tokens(text) == target:
            return text
    raise ValueError(f"no filler of up to {max_words} words has {target} tokens")


def differing_positions(ids_a: torch.Tensor, ids_f: torch.Tensor) -> List[int]:
    return torch.nonzero(ids_a != ids_f).flatten().tolist()


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


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


# Per-model sites and prompt tokenization

class Site:
    # Residual stream after a decoder layer, read and written by a forward hook

    def __init__(self, layers):
        self.layers = layers

    def capture(self, idx):
        hook = ActivationCaptureHook()
        return hook, [self.layers[idx].register_forward_hook(hook)]

    def edit(self, idx, seq_len, fn):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] != seq_len:
                return output
            new = fn(h)
            return (new,) + output[1:] if isinstance(output, tuple) else new
        return [self.layers[idx].register_forward_hook(hook)]


class SmolVLASite(Site):
    # SmolVLA's interleaved loop calls attention and MLP submodules directly, so the
    # residual after layer L is (post-attention residual) + mlp(...). Read it from the
    # post_attention_layernorm input plus the MLP output; write it through the MLP output

    def _pair(self, idx, on_residual):
        layer = self.layers[idx]
        state = {}

        def pre(module, args):
            state["r"] = args[0].detach()

        def post(module, inputs, output):
            r = state.pop("r", None)
            return output if r is None else on_residual(r, output)

        return [layer.post_attention_layernorm.register_forward_pre_hook(pre),
                layer.mlp.register_forward_hook(post)]

    def capture(self, idx):
        hook = ActivationCaptureHook()

        def on_residual(r, out):
            hook.activations.append((r + out).detach().cpu())
            return out

        return hook, self._pair(idx, on_residual)

    def edit(self, idx, seq_len, fn):
        def on_residual(r, out):
            if out.shape[1] != seq_len:
                return out
            return fn(r + out) - r

        return self._pair(idx, on_residual)


def _lerobot_batch(adapter, env, seed, batch_robot_state):
    from lerobot.envs.utils import preprocess_observation
    obs, _ = env.reset(seed=seed)
    batch = preprocess_observation(obs)
    if not batch_robot_state:
        return batch
    for group in batch.get("observation.robot_state", {}).values():
        for key, t in group.items():
            if isinstance(t, torch.Tensor) and t.ndim <= 2:
                group[key] = t.unsqueeze(0)
    return batch


class ModelSpec:
    # Tokenizes a prompt as the policy sees it and maps instruction tokens to residual positions

    default_span = "instruction"
    # SmolVLA's single (non-vector) LiberoEnv returns an unbatched robot state
    batch_robot_state = False

    def __init__(self, adapter):
        self.adapter = adapter

    def site(self) -> Site:
        raise NotImplementedError

    def tokenize(self, env, seed):
        # Returns prompt -> (ids, attended token count) on one fixed observation
        batch = _lerobot_batch(self.adapter, env, seed, self.batch_robot_state)

        def run(prompt):
            b = copy.deepcopy(batch)
            b["task"] = [prompt]
            b = self.adapter.preprocessor(self.adapter.env_preprocessor(b))
            _ensure_attended_language(b)
            ids = b[LANG_TOKENS][0].cpu()
            mask = b.get(LANG_MASK)
            count = int(mask[0].sum()) if mask is not None else int((ids != ids[-1]).sum())
            return ids, count

        return run

    def text_offset(self, seq_len: int, n_text: int) -> int:
        raise NotImplementedError

    def span(self, mode, diff, seq_len, n_text):
        offset = self.text_offset(seq_len, n_text)
        if mode == "instruction":
            return [offset + p for p in diff]
        if mode == "tail":
            return list(range(offset + diff[0], offset + n_text))
        raise ValueError(f"span mode {mode!r} not supported for {type(self).__name__}")


class Pi05Spec(ModelSpec):
    # Prefix = image tokens then the padded text block
    def site(self):
        return Site(self.adapter.get_layer_groups()["paligemma"])

    def text_offset(self, seq_len, n_text):
        return seq_len - n_text


class SmolVLASpec(ModelSpec):
    batch_robot_state = True
    # Prefix = image tokens, the language block, then one state token
    def site(self):
        return SmolVLASite(list(self.adapter.policy.model.vlm_with_expert.vlm.model.text_model.layers))

    def text_offset(self, seq_len, n_text):
        return seq_len - n_text - 1


class XVLASpec(ModelSpec):
    # Transformer sequence = [action tokens | Florence encoder output (image + text) | aux views | soft prompts]
    default_span = "vlm_block"

    def __init__(self, adapter):
        super().__init__(adapter)
        self.layout = {}

        def record(module, args, kwargs):
            self.layout["n_actions"] = kwargs["action_with_noise"].shape[1]
            self.layout["n_vlm"] = kwargs["vlm_features"].shape[1]

        adapter.policy.model.transformer.register_forward_pre_hook(record, with_kwargs=True)

    def site(self):
        return Site(self.adapter.get_layer_groups()["transformer"])

    def text_offset(self, seq_len, n_text):
        return self.layout["n_actions"] + self.layout["n_vlm"] - n_text

    def span(self, mode, diff, seq_len, n_text):
        if mode == "vlm_block":
            start = self.layout["n_actions"]
            return list(range(start, start + self.layout["n_vlm"]))
        return super().span(mode, diff, seq_len, n_text)


class GR00TSpec(ModelSpec):
    # Eagle chat template: image tokens, the instruction, then the assistant suffix
    default_span = "tail"

    def site(self):
        return Site(self.adapter.get_layer_groups()["eagle"])

    def tokenize(self, env, seed):
        from experiments.groot_common import build_groot_inputs, get_libero_state_groot
        from experiments.libero_utils import get_libero_images
        obs = env.reset()
        images = get_libero_images(obs, target_size=(256, 256))
        state = get_libero_state_groot(obs, env)

        def run(prompt):
            inputs = build_groot_inputs(images, state, prompt, self.adapter.stats,
                                        self.adapter.eagle_processor, torch.device("cpu"))
            ids = inputs["eagle_input_ids"][0]
            return ids, len(ids)

        return run

    def text_offset(self, seq_len, n_text):
        return 0


SPECS = {"pi05": Pi05Spec, "smolvla": SmolVLASpec, "xvla": XVLASpec, "groot": GR00TSpec}


def load_release_sae(model: str, pooling: str, layer: int, device: str):
    # Returns (encode, reconstruct, decoder columns in residual units) for a released SAE
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from safetensors.torch import load_file
    if model not in SAE_RELEASE:
        raise ValueError(f"no residual-stream SAE release for {model}")
    repo, pattern = SAE_RELEASE[model]
    path = hf_hub_download(repo, pattern.format(pooling=pooling, layer=layer), repo_type="dataset")
    state = load_file(path)
    with safe_open(path, "pt") as f:
        k = int((f.metadata() or {}).get("k", 64))
    d_in, d_sae = state["encoder.weight"].shape[1], state["encoder.weight"].shape[0]
    sae = TopKSAE(d_in, d_sae, k=k)
    sae.load_state_dict({key: state[key] for key in sae.state_dict()})
    sae.eval().to(device)
    mean = state.get("mean", torch.zeros(1, d_in)).reshape(-1).float().to(device)
    std = state.get("std", torch.ones(1, d_in)).reshape(-1).float().to(device)

    @torch.no_grad()
    def encode(x):
        return sae.encode((x.to(device).float() - mean) / (std + 1e-8))

    @torch.no_grad()
    def reconstruct(x):
        return (sae.decode(encode(x)) * (std + 1e-8) + mean).cpu()

    columns = (sae.decoder.weight * (std + 1e-8)[:, None]).detach()
    return encode, reconstruct, columns


# Steering vectors

class TaskCapture:
    # Per-layer span states under the instruction (A) and the filler (F): [calls, n_span, D]

    def __init__(self, prompt, filler_text, span, seq_len, states_a, states_f, capture_ok):
        self.prompt, self.filler, self.span, self.seq_len = prompt, filler_text, span, seq_len
        self.states_a, self.states_f, self.capture_ok = states_a, states_f, capture_ok

    def mean_a(self, layer):
        return self.states_a[layer].mean(0)

    def gap(self, layer):
        return self.states_a[layer].mean(0) - self.states_f[layer].mean(0)


def base_vector(edit, cap: TaskCapture, layer, others, saes, topn):
    gap = cap.gap(layer)
    if edit == "patch":
        return cap.mean_a(layer)
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
    if edit == "sae_topk":
        encode, _, columns = saes[("per_token", layer)]
        xa, xf = cap.states_a[layer], cap.states_f[layer]
        dz = encode(xa.reshape(-1, xa.shape[-1])).mean(0) - encode(xf.reshape(-1, xf.shape[-1])).mean(0)
        top = torch.topk(dz, topn).indices
        return unit((columns[:, top] * dz[top]).sum(1)).cpu()
    ma, mf = cap.mean_a(layer), cap.states_f[layer].mean(0)
    if edit == "sae_pt":
        _, rec, _ = saes[("per_token", layer)]
        return rec(ma) - rec(mf)
    if edit == "sae_mp_pos":
        _, rec, _ = saes[("mean_pool", layer)]
        return rec(ma) - rec(mf)
    if edit == "sae_mp_mean":
        _, rec, _ = saes[("mean_pool", layer)]
        return (rec(ma.mean(0, keepdim=True)) - rec(mf.mean(0, keepdim=True))).expand_as(ma).clone()
    raise ValueError(f"unknown edit {edit!r}")


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
    needed = set()
    for arm in arms:
        _, edit = parse_arm(arm)
        if edit in ("sae_topk", "sae_pt"):
            needed.add("per_token")
        if edit in ("sae_mp_pos", "sae_mp_mean"):
            needed.add("mean_pool")
    return needed


# Rollouts

def rollout(adapter, env, prompt, handles, max_steps, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        return adapter.run_episode(env, prompt, max_steps=max_steps, seed=seed)
    finally:
        for h in handles:
            h.remove()


def capture_task(adapter, spec, site, env, prompt, cfg, span_mode, max_steps) -> TaskCapture:
    count = spec.tokenize(env, cfg.seed)
    ids_a, n_a = count(prompt)
    filler_text = match_filler(lambda t: count(t)[1], n_a)
    ids_f, _ = count(filler_text)
    diff = differing_positions(ids_a, ids_f)

    raw = {("A", l): [] for l in cfg.layers}
    raw.update({("F", l): [] for l in cfg.layers})
    ok = {"A": [], "F": []}
    for ep in range(cfg.n_capture):
        for who, text in (("A", prompt), ("F", filler_text)):
            hooks, handles = {}, []
            for l in cfg.layers:
                hooks[l], hs = site.capture(l)
                handles += hs
            result = rollout(adapter, env, text, handles, max_steps, cfg.seed + 1000 + ep)
            ok[who].append(bool(result["success"]))
            for l in cfg.layers:
                raw[(who, l)] += [a.squeeze(0).float() for a in hooks[l].activations]

    seq_len = Counter(a.shape[0] for a in raw[("A", cfg.layers[0])]).most_common(1)[0][0]
    span = spec.span(span_mode, diff, seq_len, len(ids_a))
    states = {key: torch.stack([a[span] for a in acts if a.shape[0] == seq_len])
              for key, acts in raw.items()}
    return TaskCapture(prompt, filler_text, span, seq_len,
                       {l: states[("A", l)] for l in cfg.layers},
                       {l: states[("F", l)] for l in cfg.layers}, ok)


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
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)
    output_dir = Path(cfg.output_dir or f"outputs/{cfg.model}_experiments/silent_prompt_steering_{cfg.suite}")
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
    saes = {(p, l): load_release_sae(cfg.model, p, l, device)
            for p in sae_poolings(cfg.arms) for l in cfg.layers}

    _, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks or list(range(len(all_tasks)))
    if len(task_ids) < 2 and any(a.startswith("wrong:") for a in cfg.arms):
        raise ValueError("wrong:<edit> needs at least two tasks; with one task the partner is the task itself")
    envs = {t: adapter.create_env(t, suite=cfg.suite, max_steps=max_steps)[:2] for t in task_ids}
    print(f"Silent-prompt steering: {cfg.model} on {cfg.suite}, tasks {task_ids}, span {span_mode}")

    captures: Dict[int, TaskCapture] = {}
    for t in task_ids:
        env, prompt = envs[t]
        captures[t] = capture_task(adapter, spec, site, env, prompt, cfg, span_mode, max_steps)
        c = captures[t]
        print(f"task {t}: {prompt!r} | filler {c.filler!r} | span {c.span[0]}..{c.span[-1]} "
              f"({len(c.span)}) | capture A {sum(c.capture_ok['A'])}/{cfg.n_capture} "
              f"F {sum(c.capture_ok['F'])}/{cfg.n_capture}")

    results_path = output_dir / "results.json"
    results = load_results(results_path) or {"config": vars(cfg) | {"span": span_mode}, "cells": {}}
    gen = torch.Generator().manual_seed(cfg.seed)
    start = time.time()

    for i, t in enumerate(task_ids):
        key = str(t)
        if key in results["cells"]:
            continue
        env, prompt = envs[t]
        cap = captures[t]
        other = captures[task_ids[(i + 1) % len(task_ids)]]
        span = torch.as_tensor(cap.span)
        cells: Dict[str, List[bool]] = {}

        def run_cell(name, text, edits):
            outcomes = []
            for ep in range(cfg.n_episodes):
                handles = []
                for layer, kind, vec, rho in edits:
                    handles += site.edit(layer, cap.seq_len,
                                         lambda h, k=kind, v=vec, r=rho: edit_span(h, span, k, v, r))
                result = rollout(adapter, env, text, handles, max_steps, cfg.seed + ep)
                outcomes.append(bool(result["success"]))
            cells[name] = outcomes
            print(f"  task {t} {name}: {sum(outcomes)}/{len(outcomes)}")

        for arm in cfg.arms:
            control, edit = parse_arm(arm)
            if edit in NO_EDIT:
                text = {"floor_empty": "", "floor_filler": cap.filler, "ceiling": prompt}[edit]
                run_cell(arm, text, [])
                continue
            source = other if control == "wrong" else cap

            def vector(layer):
                rest = [c for c in captures.values() if c is not source]
                vec = base_vector("patch" if edit == "patch_joint" else edit, source, layer, rest, saes, cfg.sae_topn)
                vec = fit_rows(vec, len(cap.span))
                return random_like(vec, gen) if control == "random" else vec

            if edit == "patch_joint":
                run_cell(f"{arm}_L{'-'.join(map(str, cfg.layers))}", cap.filler,
                         [(l, "replace", vector(l), 0.0) for l in cfg.layers])
                continue
            kind = "replace" if edit in REPLACE_EDITS else "delta" if edit in DELTA_EDITS else "rho"
            for layer in cfg.layers:
                vec = vector(layer)
                if kind == "rho":
                    for rho in cfg.rhos:
                        run_cell(f"{arm}_L{layer}_rho{rho}", cap.filler, [(layer, kind, vec, rho)])
                else:
                    run_cell(f"{arm}_L{layer}", cap.filler, [(layer, kind, vec, 0.0)])

        results["cells"][key] = cells
        save_results(results, results_path)
        force_free_memory()

    summary = summarize(results["cells"])
    save_results({"model": cfg.model, "suite": cfg.suite, "checkpoint": checkpoint, "tasks": task_ids,
                  "span": span_mode, "cells": summary, "duration_seconds": time.time() - start},
                 output_dir / "summary.json")
    print(f"\nSilent-prompt steering ({cfg.model} / {cfg.suite})")
    for cell, s in summary.items():
        lo, hi = s["wilson95"]
        print(f"{cell:<32} {s['successes']:>4}/{s['n']:<4} {s['rate']:>6.1%}  [{lo:.1%}, {hi:.1%}]")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main(tyro.cli(SteeringConfig))
