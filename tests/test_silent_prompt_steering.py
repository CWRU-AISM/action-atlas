#!/usr/bin/env python3
"""
Unit tests for experiments/silent_prompt_steering.py and its helpers.

Filler matching, span offsets, span edits, the SmolVLA residual write-back, arm
parsing and cell construction, release-SAE loading, and the Wilson interval, on
CPU tensors. No model, simulator, or GPU.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from experiments.hooks import SmolVLAResidualSite
from experiments.prompt_spans import (
    GR00TSpec,
    OFTSpec,
    Pi05Spec,
    SmolVLASpec,
    differing_positions,
    filler,
    match_filler,
)
from experiments.sae_hooks import RELEASE_SAE_FILES, load_release_sae, sae_roundtrip
from experiments.silent_prompt_steering import (
    TaskCapture,
    arm_cells,
    edit_span,
    fit_rows,
    parse_arm,
    random_like,
)
from experiments.utils import wilson


def test_match_filler_hits_target_count():
    text = match_filler(lambda t: len(t.split()), target=7)
    assert text == filler(7)
    assert len(set(text.split())) == 7


def test_match_filler_raises_when_unreachable():
    with pytest.raises(ValueError):
        match_filler(lambda t: 2 * len(t.split()), target=5, max_words=10)


def test_differing_positions():
    a = torch.tensor([1, 2, 3, 4, 0, 0])
    f = torch.tensor([1, 9, 9, 4, 0, 0])
    assert differing_positions(a, f) == [1, 2]


def test_span_offsets():
    # pi0.5: 768 image tokens + 200 text tokens
    assert Pi05Spec(None).span("instruction", [3, 4], seq_len=968, n_text=200) == [771, 772]
    # SmolVLA: one state token after the language block
    assert SmolVLASpec(None).span("instruction", [0, 1], seq_len=141, n_text=12) == [128, 129]
    # GR00T: tail runs from the first instruction token to the end of the prompt
    tail = GR00TSpec(None).span("tail", [541, 547], seq_len=554, n_text=554)
    assert tail == list(range(541, 554))


def test_oft_span_covers_the_sequence():
    spec = OFTSpec(adapter=None)
    assert spec.span("sequence", [5], seq_len=12, n_text=4) == list(range(12))
    with pytest.raises(ValueError):
        spec.span("instruction", [5], seq_len=12, n_text=4)
    assert spec.episode_kwargs({"init_states": [1, 2]}) == {"init_states": [1, 2]}
    assert Pi05Spec(adapter=None).episode_kwargs({"task_suite": object()}) == {}


def test_edit_span_modes():
    torch.manual_seed(0)
    h = torch.randn(1, 6, 4)
    span = torch.tensor([2, 3])
    target = torch.randn(2, 4)
    out = edit_span(h, span, "replace", target)
    assert torch.equal(out[0, 2:4], target)
    assert torch.equal(out[0, :2], h[0, :2]) and torch.equal(out[0, 4:], h[0, 4:])

    delta = torch.randn(4)
    out = edit_span(h, span, "delta", delta)
    assert torch.allclose(out[0, 2:4], h[0, 2:4] + delta)

    d = torch.tensor([1.0, 0.0, 0.0, 0.0])
    out = edit_span(h, span, "rho", d, rho=0.5)
    expected = h[0, 2:4] + 0.5 * h[0, 2:4].norm(dim=-1, keepdim=True) * d
    assert torch.allclose(out[0, 2:4], expected)


def test_fit_rows_and_random_like():
    v = torch.arange(12.0).reshape(3, 4)
    assert torch.equal(fit_rows(v, 2), v[:2])
    assert torch.equal(fit_rows(v, 5)[3:], v[-1:].expand(2, 4))
    r = random_like(v, torch.Generator().manual_seed(0))
    assert torch.allclose(r.norm(dim=-1), v.norm(dim=-1), atol=1e-5)


class _SmolLayer(nn.Module):
    # Mirrors SmolVLA's hand-written loop: out = mlp(post_ln(r)) + r
    def __init__(self, dim):
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, r):
        return self.mlp(self.post_attention_layernorm(r)) + r


def test_smolvla_site_reads_and_writes_the_residual():
    torch.manual_seed(0)
    layer = _SmolLayer(4)
    r = torch.randn(1, 5, 4)
    site = SmolVLAResidualSite([layer])

    hook, handles = site.capture(0)
    expected = layer(r)
    for h in handles:
        h.remove()
    assert torch.allclose(hook.activations[0], expected.detach())

    target = torch.zeros(2, 4)
    handles = site.edit(0, seq_len=5,
                        fn=lambda h: edit_span(h, torch.tensor([1, 2]), "replace", target))
    out = layer(r)
    for h in handles:
        h.remove()
    assert torch.allclose(out[0, 1:3], target, atol=1e-6)
    assert torch.allclose(out[0, [0, 3, 4]], expected[0, [0, 3, 4]], atol=1e-6)


def test_parse_arm():
    assert parse_arm("direction") == (None, "direction")
    assert parse_arm("wrong:gap") == ("wrong", "gap")
    for bad in ("random:patch", "noise:gap", "steer"):
        with pytest.raises(ValueError):
            parse_arm(bad)


def _capture(seed, layers=(0, 1), n_span=3, dim=4):
    g = torch.Generator().manual_seed(seed)
    states = {layer: torch.randn(2, n_span, dim, generator=g) for layer in layers}
    return TaskCapture(f"task {seed}", "the of and", list(range(n_span)), 10,
                       states, {layer: torch.zeros(2, n_span, dim) for layer in layers},
                       {"A": [], "F": []})


def test_arm_cells():
    cfg = SimpleNamespace(layers=(0, 1), rhos=(0.5, 1.0), sae_topn=2)
    cap, other = _capture(0), _capture(1)
    captures = [cap, other]
    gen = torch.Generator().manual_seed(0)

    assert arm_cells("ceiling", cap, other, captures, {}, cfg, gen) == [("ceiling", "task 0", [])]

    cells = arm_cells("direction", cap, other, captures, {}, cfg, gen)
    assert [name for name, _, _ in cells] == [
        "direction_L0_rho0.5", "direction_L0_rho1.0", "direction_L1_rho0.5", "direction_L1_rho1.0"]
    assert all(text == cap.filler for _, text, _ in cells)

    (_, _, edits), _ = arm_cells("wrong:gap", cap, other, captures, {}, cfg, gen)
    assert torch.equal(edits[0][2], other.gap(0))

    (_, _, edits), _ = arm_cells("random:gap", cap, other, captures, {}, cfg, gen)
    assert edits[0][1] == "delta"
    assert torch.allclose(edits[0][2].norm(dim=-1), cap.gap(0).norm(dim=-1), atol=1e-5)

    [(name, _, edits)] = arm_cells("patch_joint", cap, other, captures, {}, cfg, gen)
    assert name == "patch_joint_L0-1"
    assert [(layer, kind) for layer, kind, _, _ in edits] == [(0, "replace"), (1, "replace")]


def test_load_release_sae_reads_the_local_bundle(tmp_path):
    torch.manual_seed(0)
    dim, hidden = 4, 8
    state = {"encoder.weight": torch.randn(hidden, dim), "encoder.bias": torch.randn(hidden),
             "decoder.weight": torch.randn(dim, hidden), "decoder.bias": torch.randn(dim),
             "mean": torch.randn(1, dim), "std": torch.rand(1, dim) + 0.5}
    path = tmp_path / "pi05" / RELEASE_SAE_FILES["pi05"].format(pooling="per_token", layer=2)
    path.parent.mkdir(parents=True)
    save_file(state, str(path), metadata={"k": "3"})

    sae, mean, std = load_release_sae("pi05", "per_token", 2, device="cpu", release_dir=tmp_path)
    assert sae.k == 3
    assert torch.equal(mean, state["mean"].reshape(-1))
    x = torch.randn(5, dim)
    recon, z = sae_roundtrip(sae, x, mean, std)
    assert recon.shape == x.shape and int((z != 0).sum(-1).max()) <= 3


def test_wilson():
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.07 < hi < 0.072
    lo, hi = wilson(113, 150)
    assert lo < 113 / 150 < hi
