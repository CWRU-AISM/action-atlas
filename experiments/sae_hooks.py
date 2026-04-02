#!/usr/bin/env python3
"""
SAE hooks for per-token ablation and steering.

Processes each token position through the SAE independently
(no mean pooling across sequence dimension).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder with correct per-token processing."""

    def __init__(self, input_dim: int, hidden_dim: int, k: int = 64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z


class PerTokenAblationHook:
    """
    Ablation hook that processes each token position through SAE independently.

    Uses a residual approach: computes the delta between normal and ablated
    SAE reconstructions, then applies that delta to the original activations.
    """

    def __init__(self, sae: TopKSAE, act_mean: torch.Tensor, act_std: torch.Tensor,
                 device: str = 'cuda'):
        self.sae = sae
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device

        # Ablation configuration
        self.ablate_features: List[int] = []
        self.ablate_start: int = 0
        self.ablate_end: float = float('inf')

        # State
        self.current_step: int = 0
        self.enabled: bool = True

    def set_ablation(self, features: List[int], start: int = 0, end: float = float('inf')):
        """Configure which features to ablate and temporal window."""
        self.ablate_features = features
        self.ablate_start = start
        self.ablate_end = end

    def reset(self):
        self.current_step = 0

    def clear(self):
        self.ablate_features = []
        self.ablate_start = 0
        self.ablate_end = float('inf')
        self.current_step = 0

    def __call__(self, module, input, output):
        """Forward hook that ablates specified features."""
        # Skip if disabled or no features to ablate
        if not self.enabled or not self.ablate_features:
            self.current_step += 1
            return output

        # Check temporal window
        if not (self.ablate_start <= self.current_step < self.ablate_end):
            self.current_step += 1
            return output

        # Handle tuple outputs
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        extra = output[1:] if is_tuple else None

        original_dtype = act.dtype
        act = act.float()
        original_shape = act.shape

        # Process all positions independently
        if len(original_shape) == 3:
            batch, seq, dim = original_shape
            # Flatten to [batch*seq, dim] - each position processed independently
            act_flat = act.view(-1, dim)
        else:
            act_flat = act
            batch, seq = act.shape[0], 1

        # Normalize each position
        act_norm = (act_flat - self.act_mean) / (self.act_std + 1e-8)

        # Encode each position through SAE
        z = self.sae.encode(act_norm)  # [batch*seq, hidden_dim]
        z_ablated = z.clone()

        # Ablate specified features across ALL positions
        for f in self.ablate_features:
            if f < z.shape[-1]:
                z_ablated[..., f] = 0

        # Residual approach: add ablation delta to original
        # This preserves information not captured by SAE
        act_decoded = self.sae.decode(z)
        act_decoded_ablated = self.sae.decode(z_ablated)
        ablation_delta = (act_decoded_ablated - act_decoded) * (self.act_std + 1e-8)

        act_modified = act_flat + ablation_delta

        # Reshape back to original shape
        if len(original_shape) == 3:
            act_modified = act_modified.view(batch, seq, dim)

        act_modified = act_modified.to(original_dtype)
        self.current_step += 1

        return (act_modified,) + extra if is_tuple else act_modified


class PerTokenSteeringHook:
    """Steering hook that amplifies/suppresses SAE features per token position."""

    def __init__(self, sae: TopKSAE, act_mean: torch.Tensor, act_std: torch.Tensor,
                 device: str = 'cuda'):
        self.sae = sae
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device

        # Steering configuration
        self.steer_features: List[int] = []
        self.steer_strength: float = 0.0
        self.feature_means: Optional[torch.Tensor] = None  # Mean activation per feature
        self.steer_start: int = 0
        self.steer_end: float = float('inf')

        # State
        self.current_step: int = 0
        self.enabled: bool = True

    def set_steering(self, features: List[int], strength: float,
                     feature_means: Optional[torch.Tensor] = None,
                     start: int = 0, end: float = float('inf')):
        """Configure steering parameters."""
        self.steer_features = features
        self.steer_strength = strength
        self.feature_means = feature_means
        self.steer_start = start
        self.steer_end = end

    def reset(self):
        self.current_step = 0

    def clear(self):
        self.steer_features = []
        self.steer_strength = 0.0
        self.feature_means = None
        self.current_step = 0

    def __call__(self, module, input, output):
        """Forward hook that steers specified features."""
        if not self.enabled or not self.steer_features or self.steer_strength == 0:
            self.current_step += 1
            return output

        # Check temporal window
        if not (self.steer_start <= self.current_step < self.steer_end):
            self.current_step += 1
            return output

        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        extra = output[1:] if is_tuple else None

        original_dtype = act.dtype
        act = act.float()
        original_shape = act.shape

        # Process all positions independently
        if len(original_shape) == 3:
            batch, seq, dim = original_shape
            act_flat = act.view(-1, dim)
        else:
            act_flat = act
            batch, seq = act.shape[0], 1

        # Normalize
        act_norm = (act_flat - self.act_mean) / (self.act_std + 1e-8)

        # Encode
        z = self.sae.encode(act_norm)
        z_steered = z.clone()

        # Steer features
        for f in self.steer_features:
            if f < z.shape[-1]:
                if self.feature_means is not None:
                    # Steer relative to feature mean
                    z_steered[..., f] += self.steer_strength * self.feature_means[f]
                else:
                    # Steer by fixed amount
                    z_steered[..., f] *= (1.0 + self.steer_strength)

        # Residual approach
        act_decoded = self.sae.decode(z)
        act_decoded_steered = self.sae.decode(z_steered)
        steer_delta = (act_decoded_steered - act_decoded) * (self.act_std + 1e-8)

        act_modified = act_flat + steer_delta

        if len(original_shape) == 3:
            act_modified = act_modified.view(batch, seq, dim)

        act_modified = act_modified.to(original_dtype)
        self.current_step += 1

        return (act_modified,) + extra if is_tuple else act_modified


def load_sae(layer_name: str, sae_dir: str, device: str = 'cuda') -> tuple:
    """
    Load SAE model and activation statistics.

    Returns:
        sae: TopKSAE model
        act_mean: Activation mean tensor
        act_std: Activation std tensor
    """
    from pathlib import Path
    import glob

    sae_path = Path(sae_dir) / layer_name
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE directory not found: {sae_path}")

    # Find most recent SAE
    sae_dirs = sorted(glob.glob(str(sae_path / "sae_*")))
    if not sae_dirs:
        raise FileNotFoundError(f"No SAE found in {sae_path}")

    sae_checkpoint = Path(sae_dirs[-1]) / "sae_final.pt"
    if not sae_checkpoint.exists():
        sae_checkpoint = Path(sae_dirs[-1]) / "sae_best.pt"

    data = torch.load(sae_checkpoint, map_location='cpu')
    config = data['config']

    sae = TopKSAE(config['input_dim'], config['hidden_dim'], k=config.get('k', 64))
    sae.load_state_dict(data['sae_state_dict'])
    sae.eval().to(device)

    act_mean = data.get('activation_mean', torch.zeros(config['input_dim']))
    act_std = data.get('activation_std', torch.ones(config['input_dim']))

    return sae, act_mean.to(device), act_std.to(device)
