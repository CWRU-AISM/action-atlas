#!/usr/bin/env python3
"""
Unit tests for experiments/silent_prompt_steering.py.

Filler matching, span offsets, span edits, the SmolVLA residual write-back, arm
parsing, and the Wilson interval, on CPU tensors. No model, simulator, or GPU.
"""

import pytest
import torch
import torch.nn as nn

from experiments.silent_prompt_steering import (
    GR00TSpec,
    OFTSpec,
    Pi05Spec,
    SmolVLASite,
    SmolVLASpec,
    differing_positions,
    edit_span,
    filler,
    fit_rows,
    match_filler,
    parse_arm,
    random_like,
    wilson,
)


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
    assert GR00TSpec(None).span("tail", [541, 547], seq_len=554, n_text=554) == list(range(541, 554))


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
    site = SmolVLASite([layer])

    hook, handles = site.capture(0)
    expected = layer(r)
    for h in handles:
        h.remove()
    assert torch.allclose(hook.activations[0], expected.detach())

    target = torch.zeros(2, 4)
    handles = site.edit(0, seq_len=5, fn=lambda h: edit_span(h, torch.tensor([1, 2]), "replace", target))
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


def test_wilson():
    lo, hi = wilson(0, 50)
    assert lo == 0.0 and 0.07 < hi < 0.072
    lo, hi = wilson(113, 150)
    assert lo < 113 / 150 < hi
