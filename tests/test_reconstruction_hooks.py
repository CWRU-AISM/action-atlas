#!/usr/bin/env python3
"""
Unit tests for experiments/reconstruction_hooks.py.

Synthetic [B, T, D] tensors through a tiny random TopK SAE on CPU; verifies
each protocol's arithmetic against a hand-computed reference, plus the
mean_pool-checkpoint x pertoken_replace guardrail. No GPU required.
"""

import pytest
import torch

from experiments.reconstruction_hooks import (
    MeanPoolDeltaHook,
    MeanPoolReplaceHook,
    PerTokenReplaceHook,
    infer_sae_pooling,
    make_reconstruction_hook,
)
from experiments.sae_hooks import TopKSAE

BATCH, SEQ, DIM = 2, 5, 8
EPS = 1e-8


@pytest.fixture
def sae():
    torch.manual_seed(0)
    model = TopKSAE(input_dim=DIM, hidden_dim=4 * DIM, k=4)
    model.eval()
    return model


@pytest.fixture
def stats():
    torch.manual_seed(1)
    act_mean = torch.randn(DIM)
    act_std = torch.rand(DIM) + 0.5
    return act_mean, act_std


@pytest.fixture
def act():
    torch.manual_seed(2)
    return torch.randn(BATCH, SEQ, DIM)


def roundtrip(sae, act_mean, act_std, vectors):
    # Reference standardize -> encode -> decode -> de-standardize.
    normed = (vectors - act_mean) / (act_std + EPS)
    recon = sae.decode(sae.encode(normed))
    return recon * (act_std + EPS) + act_mean


def make_hook(sae, stats, protocol, **kwargs):
    act_mean, act_std = stats
    return make_reconstruction_hook(sae, act_mean, act_std, protocol,
                                    device='cpu', **kwargs)


def test_factory_returns_protocol_hooks(sae, stats):
    assert isinstance(make_hook(sae, stats, "pertoken_replace"), PerTokenReplaceHook)
    assert isinstance(make_hook(sae, stats, "meanpool_replace"), MeanPoolReplaceHook)
    assert isinstance(make_hook(sae, stats, "meanpool_delta"), MeanPoolDeltaHook)
    with pytest.raises(ValueError, match="Unknown protocol"):
        make_hook(sae, stats, "meanpool")


def test_pertoken_replace_matches_per_position_decode(sae, stats, act):
    hook = make_hook(sae, stats, "pertoken_replace")
    out = hook(None, None, act)

    with torch.no_grad():
        expected = roundtrip(sae, *stats, act.view(-1, DIM)).view(BATCH, SEQ, DIM)

    assert out.shape == act.shape
    assert torch.allclose(out, expected, atol=1e-5)
    assert hook.n_calls == 1
    assert hook.get_avg_recon_error_ratio() > 0


def test_meanpool_replace_broadcasts_pooled_recon(sae, stats, act):
    hook = make_hook(sae, stats, "meanpool_replace")
    out = hook(None, None, act)

    with torch.no_grad():
        expected_mean = roundtrip(sae, *stats, act.mean(dim=1))

    assert out.shape == act.shape
    # Every position carries the same vector: the pooled reconstruction.
    assert torch.allclose(out, out[:, :1].expand(-1, SEQ, -1), atol=1e-6)
    assert torch.allclose(out, expected_mean.unsqueeze(1).expand(BATCH, SEQ, DIM), atol=1e-5)


def test_meanpool_delta_adds_broadcast_delta_and_preserves_variance(sae, stats, act):
    hook = make_hook(sae, stats, "meanpool_delta")
    out = hook(None, None, act)

    with torch.no_grad():
        pooled = act.mean(dim=1)
        delta = roundtrip(sae, *stats, pooled) - pooled

    assert out.shape == act.shape
    assert torch.allclose(out, act + delta.unsqueeze(1), atol=1e-5)
    # A broadcast shift leaves the per-token variation around the mean intact.
    assert torch.allclose(out - out.mean(dim=1, keepdim=True),
                          act - act.mean(dim=1, keepdim=True), atol=1e-5)


def test_meanpool_guardrail_raises_on_pertoken_protocol(sae, stats):
    for pooling in ("mean_pool", "meanpool"):
        with pytest.raises(ValueError, match="train/eval mismatch"):
            make_hook(sae, stats, "pertoken_replace", sae_pooling=pooling)

    # Explicit override for deliberate ablations.
    hook = make_hook(sae, stats, "pertoken_replace", sae_pooling="mean_pool",
                     allow_mismatched_protocol=True)
    assert isinstance(hook, PerTokenReplaceHook)

    # Matched pairings pass without the override.
    assert isinstance(make_hook(sae, stats, "meanpool_replace", sae_pooling="mean_pool"),
                      MeanPoolReplaceHook)
    assert isinstance(make_hook(sae, stats, "meanpool_delta", sae_pooling="mean_pool"),
                      MeanPoolDeltaHook)
    assert isinstance(make_hook(sae, stats, "pertoken_replace", sae_pooling="per_token"),
                      PerTokenReplaceHook)


def test_infer_sae_pooling():
    assert infer_sae_pooling({"config": {"pooling": "mean_pool"}}) == "mean_pool"
    assert infer_sae_pooling({"pooling": "per_token"}) == "per_token"
    assert infer_sae_pooling({"config": {"input_dim": DIM}}) is None
    assert infer_sae_pooling("not-a-dict") is None


def test_tuple_output_and_dtype_preserved(sae, stats, act):
    aux = torch.randn(BATCH, 3)
    for protocol in ("pertoken_replace", "meanpool_replace", "meanpool_delta"):
        hook = make_hook(sae, stats, protocol)
        out = hook(None, None, (act.to(torch.bfloat16), aux))
        assert isinstance(out, tuple) and len(out) == 2
        assert out[0].shape == act.shape
        assert out[0].dtype == torch.bfloat16
        assert out[1] is aux


def test_2d_input_handled(sae, stats):
    torch.manual_seed(3)
    act2d = torch.randn(BATCH, DIM)
    with torch.no_grad():
        expected = roundtrip(sae, *stats, act2d)

    for protocol in ("pertoken_replace", "meanpool_replace"):
        out = make_hook(sae, stats, protocol)(None, None, act2d)
        assert out.shape == act2d.shape
        assert torch.allclose(out, expected, atol=1e-5)

    out = make_hook(sae, stats, "meanpool_delta")(None, None, act2d)
    assert torch.allclose(out, act2d + (expected - act2d), atol=1e-5)


def test_disabled_hook_passes_through(sae, stats, act):
    hook = make_hook(sae, stats, "meanpool_replace")
    hook.enabled = False
    out = hook(None, None, act)
    assert out is act
    assert hook.n_calls == 0
    assert hook.current_step == 1
