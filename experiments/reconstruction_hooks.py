#!/usr/bin/env python3
"""
Unified SAE rollout-reconstruction hooks.

One factory covers the three reconstruction protocols used across the release,
matching how each SAE arm was trained (see setup/DATA.md for the per-model
protocol table):

  - pertoken_replace: encode/decode every token position independently and
    replace each activation with its SAE reconstruction. Correct for the
    per_token arm (the primary release) on every architecture.
  - meanpool_replace: mean-pool over the token axis, reconstruct the pooled
    vector, and broadcast-replace it at every position. Correct for the
    mean_pool arm, whose SAEs were trained on token-axis mean-pooled vectors
    (experiments/hooks.py ActivationCollector: h.mean(dim=1)).
  - meanpool_delta: mean-pool, reconstruct, and add the pooled reconstruction
    delta (recon_mean - mean) at every position. The mean_pool protocol used
    for X-VLA, whose heterogeneous token streams carry per-token structure
    that a broadcast replacement would destroy; delta-add reconstructs the
    pooled component while preserving per-token variation.

Pointing a per-token applier at a mean_pool checkpoint is a silent train/eval
mismatch (the SAE only ever saw pooled vectors), so make_reconstruction_hook
refuses that combination when the checkpoint declares pooling metadata; pass
allow_mismatched_protocol=True only for deliberate ablations.

These hooks are inference-only: the SAE pass runs under torch.no_grad and the
returned activations do not carry gradients.
"""

import torch

PROTOCOLS = ("pertoken_replace", "meanpool_replace", "meanpool_delta")


def _normalize_pooling(value):
    # Collapse the naming variants used across the repo and release metadata
    # ("mean_pool", "meanpool", "per_token", "pertoken", "temporal").
    if value is None:
        return None
    return str(value).lower().replace("_", "").replace("-", "")


def infer_sae_pooling(checkpoint):
    """
    Best-effort read of the pooling arm recorded with an SAE checkpoint.

    Accepts the torch.save training output (experiments/train_sae.py layout:
    a dict with a 'config' sub-dict) or a flat safetensors-style string
    metadata dict; looks for a 'pooling' entry in either. Returns the raw
    string (e.g. 'mean_pool') or None when the checkpoint does not record it.
    """
    if not isinstance(checkpoint, dict):
        return None
    pooling = checkpoint.get("pooling")
    if isinstance(pooling, str):
        return pooling
    config = checkpoint.get("config")
    if isinstance(config, dict) and isinstance(config.get("pooling"), str):
        return config["pooling"]
    return None


class ReconstructionHookBase:
    """
    Shared state, stats, and shape handling for reconstruction hooks.

    Mirrors the interface of the hooks in experiments/sae_hooks.py and the
    X-VLA reconstruction eval: register with module.register_forward_hook,
    then reset() per episode and reset_stats() per condition. Handles [B, T, D]
    and [B, D] activations, tuple outputs, and preserves the incoming dtype;
    any other shape passes through unmodified.
    """

    protocol = None  # set by subclasses

    def __init__(self, sae, act_mean, act_std, device='cuda'):
        self.sae = sae
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device
        self.enabled = True
        self.current_step = 0
        self._verified = False
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0

    def reset(self):
        self.current_step = 0

    def reset_stats(self):
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0
        self._verified = False

    def get_avg_recon_error_ratio(self):
        if self.n_calls == 0:
            return 0.0
        return (self.total_recon_error / self.n_calls) / max(self.total_act_norm / self.n_calls, 1e-8)

    def _record(self, recon_error, act_norm_val, z):
        # Accumulate per-call recon-error stats; print once per condition.
        self.total_recon_error += recon_error
        self.total_act_norm += act_norm_val
        self.n_calls += 1
        if not self._verified:
            ratio = recon_error / max(act_norm_val, 1e-8)
            n_active = (z.detach().abs() > 0).sum(dim=-1).float().mean().item()
            print(f"[RECON {self.protocol}] error={recon_error:.4f}, norm={act_norm_val:.4f}, "
                  f"ratio={ratio:.4f}, active={n_active:.0f}")
            self._verified = True

    def _sae_roundtrip(self, vectors):
        # Standardize -> encode -> decode -> de-standardize a [N, D] batch.
        normed = (vectors - self.act_mean) / (self.act_std + 1e-8)
        z = self.sae.encode(normed)
        recon_norm = self.sae.decode(z)
        recon = recon_norm * (self.act_std + 1e-8) + self.act_mean
        return recon, z

    def __call__(self, module, input, output):
        if not self.enabled:
            self.current_step += 1
            return output

        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        extra = output[1:] if is_tuple else None

        if act.dim() not in (2, 3):
            self.current_step += 1
            return output

        original_dtype = act.dtype
        with torch.no_grad():
            modified = self._reconstruct(act.float())
        modified = modified.to(original_dtype)
        self.current_step += 1
        return (modified,) + extra if is_tuple else modified

    def _reconstruct(self, act):
        raise NotImplementedError


class PerTokenReplaceHook(ReconstructionHookBase):
    """
    Reconstruct each token position independently and replace it.

    For per-token SAEs: flatten [B, T, D] -> [B*T, D], encode/decode every
    position, and substitute the reconstruction.
    """

    protocol = "pertoken_replace"

    def _reconstruct(self, act):
        original_shape = act.shape
        act_flat = act.view(-1, original_shape[-1]) if act.dim() == 3 else act

        reconstructed, z = self._sae_roundtrip(act_flat)
        self._record((reconstructed - act_flat).norm().item(), act_flat.norm().item(), z)

        return reconstructed.view(original_shape) if act.dim() == 3 else reconstructed


class MeanPoolReplaceHook(ReconstructionHookBase):
    """
    Mean-pool over the token axis, reconstruct the pooled vector, and
    broadcast-replace it at every position.

    The faithful protocol for mean-pool SAEs: the SAE was trained on
    h.mean(dim=1) vectors, so it is applied to exactly that statistic and its
    output is what every position receives. Recon error is measured in pooled
    space. 2D [B, D] inputs are treated as an already-pooled single position.
    """

    protocol = "meanpool_replace"

    def _reconstruct(self, act):
        pooled = act.mean(dim=-2) if act.dim() == 3 else act

        reconstructed_mean, z = self._sae_roundtrip(pooled)
        self._record((reconstructed_mean - pooled).norm().item(), pooled.norm().item(), z)

        if act.dim() == 3:
            return reconstructed_mean.unsqueeze(-2).expand(act.shape).contiguous()
        return reconstructed_mean


class MeanPoolDeltaHook(ReconstructionHookBase):
    """
    Mean-pool, reconstruct, and add the pooled reconstruction delta
    (recon_mean - mean) at every position.

    The X-VLA mean-pool protocol: reconstructs the pooled component while
    preserving per-token variation, which heterogeneous token streams need.
    Recon error is the delta norm in pooled space.
    """

    protocol = "meanpool_delta"

    def _reconstruct(self, act):
        pooled = act.mean(dim=-2) if act.dim() == 3 else act

        reconstructed_mean, z = self._sae_roundtrip(pooled)
        delta = reconstructed_mean - pooled
        self._record(delta.norm().item(), pooled.norm().item(), z)

        return act + delta.unsqueeze(-2) if act.dim() == 3 else act + delta


PROTOCOL_HOOKS = {
    "pertoken_replace": PerTokenReplaceHook,
    "meanpool_replace": MeanPoolReplaceHook,
    "meanpool_delta": MeanPoolDeltaHook,
}


def make_reconstruction_hook(sae, act_mean, act_std, protocol, device='cuda',
                             sae_pooling=None, allow_mismatched_protocol=False):
    """
    Build a rollout-reconstruction forward hook for the given protocol.

    Args:
        sae: TopKSAE (or any module with encode/decode).
        act_mean, act_std: activation statistics stored with the checkpoint.
        protocol: one of 'pertoken_replace', 'meanpool_replace', 'meanpool_delta'.
        device: where the activation statistics live.
        sae_pooling: the pooling arm the checkpoint declares ('per_token',
            'mean_pool', 'temporal'; see infer_sae_pooling). When provided, it
            guards against applying a mean-pool-trained SAE per token.
        allow_mismatched_protocol: set True to bypass the guard for deliberate
            train/eval-mismatch ablations.

    Returns:
        A hook instance suitable for module.register_forward_hook.

    Raises:
        ValueError: unknown protocol, or a mean_pool checkpoint paired with
            protocol='pertoken_replace' without allow_mismatched_protocol.
    """
    if protocol not in PROTOCOL_HOOKS:
        raise ValueError(f"Unknown protocol '{protocol}'; choose from {PROTOCOLS}")

    if (_normalize_pooling(sae_pooling) == "meanpool"
            and protocol == "pertoken_replace" and not allow_mismatched_protocol):
        raise ValueError(
            f"This SAE was trained on mean-pooled activations (pooling='{sae_pooling}') but "
            f"protocol='pertoken_replace' would apply it to every token position independently. "
            f"That is a train/eval mismatch: the SAE never saw per-token vectors, so rollouts "
            f"silently measure a corrupted reconstruction. Use protocol='meanpool_replace' "
            f"(broadcast the pooled reconstruction) or 'meanpool_delta' (add the pooled "
            f"reconstruction delta), or pass allow_mismatched_protocol=True for a deliberate "
            f"ablation.")

    return PROTOCOL_HOOKS[protocol](sae, act_mean, act_std, device=device)
