#!/usr/bin/env python3
"""Train TopK sparse autoencoders on collected VLA activations.

Model-agnostic: just needs a directory of activation .pt files.
Trains one SAE per layer with early stopping and saves checkpoints
compatible with sae_hooks.py for downstream ablation/steering.

Examples:
    # Train on pre-collected activations directory
    python experiments/train_sae.py \\
        --activations-dir outputs/xvla_experiments/baseline_libero_object/activations \\
        --output-dir outputs/xvla_saes/libero_object

    # Specific layers only
    python experiments/train_sae.py \\
        --activations-dir outputs/groot_experiments/baseline_libero_goal/activations \\
        --layers transformer_L00 transformer_L04 transformer_L08

    # Custom hyperparameters
    python experiments/train_sae.py \\
        --activations-dir outputs/smolvla_experiments/baseline_libero_spatial/activations \\
        --expansion 4.0 --k 32 --epochs 200
"""

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tyro


@dataclass
class TrainSAEConfig:
    """TopK SAE training configuration."""

    activations_dir: str = ""
    """Directory containing activation .pt files (one per layer).
    Expected structure: activations_dir/task{N}/ep{M}/layer_name.pt
    or activations_dir/layer_name.pt (pre-concatenated)."""

    output_dir: str = "outputs/saes"
    """Where to save trained SAE checkpoints."""

    layers: Optional[List[str]] = None
    """Layer names to train (e.g. 'transformer_L00'). Default: all found."""

    expansion: float = 8.0
    """SAE expansion ratio (hidden_dim = input_dim * expansion)."""

    k: int = 64
    """TopK sparsity: number of active features per input."""

    epochs: int = 100
    batch_size: int = 4096
    lr: float = 3e-4
    patience: int = 5
    """Early stopping patience (epochs without improvement)."""

    max_samples: int = 500_000
    """Max training samples per layer (subsampled if exceeded)."""

    device: str = "cuda"


class TopKSAE(nn.Module):
    """TopK sparse autoencoder."""

    def __init__(self, input_dim: int, expansion: float = 8.0, k: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = int(input_dim * expansion)
        self.k = k
        self.encoder = nn.Linear(input_dim, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, input_dim)
        with torch.no_grad():
            self.decoder.weight.data = self.encoder.weight.data.T.clone()

    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def load_activations(act_dir: Path, layer_name: str, max_samples: int) -> Optional[torch.Tensor]:
    """Load activations for a given layer, handling multiple directory layouts."""
    all_acts = []

    # Layout 1: Single concatenated file
    direct = act_dir / f"{layer_name}.pt"
    if direct.exists():
        t = torch.load(direct, map_location="cpu", weights_only=True).float()
        if t.dim() > 2:
            t = t.reshape(-1, t.shape[-1])
        all_acts.append(t)

    # Layout 2: Per-task/episode files
    for task_dir in sorted(act_dir.glob("task*")):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.glob("ep*")):
            pt_file = ep_dir / f"{layer_name}.pt"
            if pt_file.exists():
                try:
                    t = torch.load(pt_file, map_location="cpu", weights_only=True).float()
                    if t.dim() > 2:
                        t = t.reshape(-1, t.shape[-1])
                    all_acts.append(t)
                except Exception:
                    continue

    if not all_acts:
        return None

    combined = torch.cat(all_acts, dim=0)
    if combined.shape[0] > max_samples:
        indices = torch.randperm(combined.shape[0])[:max_samples]
        combined = combined[indices]
    return combined


def train_sae_on_activations(activations: torch.Tensor, cfg: TrainSAEConfig) -> dict:
    """Train a TopK SAE on activation vectors."""
    input_dim = activations.shape[1]
    act_mean = activations.mean(dim=0)
    act_std = activations.std(dim=0).clamp(min=1e-8)
    activations_norm = (activations - act_mean) / act_std

    n_samples = activations_norm.shape[0]
    hidden_dim = int(input_dim * cfg.expansion)
    print(f"  Training: {n_samples:,} samples, dim={input_dim}")
    print(f"  SAE: {input_dim} -> {hidden_dim} (k={cfg.k})")

    sae = TopKSAE(input_dim, cfg.expansion, cfg.k).to(cfg.device)
    optimizer = optim.Adam(sae.parameters(), lr=cfg.lr)
    dataset = TensorDataset(activations_norm)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, persistent_workers=True)

    history = []
    best_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(cfg.epochs):
        sae.train()
        total_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(cfg.device, non_blocking=True)
            x_hat, z = sae(batch)
            loss = nn.functional.mse_loss(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        with torch.no_grad():
            sample = activations_norm[:min(10000, n_samples)].to(cfg.device)
            x_hat_s, z_s = sae(sample)
            residual = sample - x_hat_s
            ev = (1.0 - residual.var() / sample.var()).item()
            dead_frac = (z_s.abs().sum(dim=0) == 0).float().mean().item()
            l0 = (z_s != 0).float().sum(dim=-1).mean().item()

        history.append({"epoch": epoch, "loss": avg_loss, "explained_var": ev,
                        "dead_features_frac": dead_frac, "l0": l0})

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == cfg.epochs - 1 or wait >= cfg.patience:
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.6f} EV={ev:.4f} "
                  f"L0={l0:.0f} dead={dead_frac:.3f} patience={wait}/{cfg.patience}")

        if wait >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    return {
        "sae_state_dict": best_state,
        "activation_mean": act_mean,
        "activation_std": act_std,
        "config": {"input_dim": input_dim, "hidden_dim": hidden_dim,
                    "expansion": cfg.expansion, "k": cfg.k, "n_samples": n_samples},
        "history": history,
        "best_loss": best_loss,
    }


def main(cfg: TrainSAEConfig):
    act_dir = Path(cfg.activations_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not act_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {act_dir}")

    # Discover available layers
    if cfg.layers:
        layer_names = cfg.layers
    else:
        # Auto-discover from .pt files
        found = set()
        for pt_file in act_dir.rglob("*.pt"):
            found.add(pt_file.stem)
        layer_names = sorted(found)
        if not layer_names:
            raise FileNotFoundError(f"No .pt files found under {act_dir}")

    print(f"SAE Training")
    print(f"  activations: {act_dir}")
    print(f"  layers: {len(layer_names)}")
    print(f"  SAE: expansion={cfg.expansion}, k={cfg.k}")
    print(f"  output: {output_dir}")

    results = {}
    for layer_name in layer_names:
        print(f"\nLayer: {layer_name}")

        # Check if already trained
        layer_dir = output_dir / layer_name
        if (layer_dir / "sae_best.pt").exists():
            print(f"  [SKIP] Already trained")
            continue

        activations = load_activations(act_dir, layer_name, cfg.max_samples)
        if activations is None or len(activations) == 0:
            print(f"  [SKIP] No activations found")
            continue

        result = train_sae_on_activations(activations, cfg)

        layer_dir.mkdir(exist_ok=True)
        torch.save(result, layer_dir / "sae_best.pt")

        final = result["history"][-1]
        layer_result = {
            "layer": layer_name,
            "n_samples": result["config"]["n_samples"],
            "best_loss": result["best_loss"],
            "final_ev": final["explained_var"],
            "final_l0": final["l0"],
            "final_dead": final["dead_features_frac"],
            "epochs_trained": final["epoch"] + 1,
        }
        results[layer_name] = layer_result
        print(f"  Saved: {layer_dir}/sae_best.pt "
              f"(loss={result['best_loss']:.6f}, EV={final['explained_var']:.4f})")

        del activations, result
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "expansion": cfg.expansion, "k": cfg.k,
        "epochs": cfg.epochs, "lr": cfg.lr,
        "per_layer": results,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Trained {len(results)}/{len(layer_names)} layers.")
    print(f"Results: {output_dir / 'training_results.json'}")


if __name__ == "__main__":
    cfg = tyro.cli(TrainSAEConfig)
    main(cfg)
