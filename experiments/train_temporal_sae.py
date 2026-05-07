#!/usr/bin/env python3
"""
Train Temporal Sparse Autoencoders (T-SAEs) on collected VLA activations.

Reimplements the contrastive arm of Bhalla et al. 2025 "Temporal Sparse
Autoencoders" (arXiv:2511.05541, ICLR 2026) on VLA action token activations.
The upstream reference implementation is vendored at ActionAtlas/temporal_saes/
as a git submodule.

The total loss is

    L_total = L_recon + alpha * L_contr

where L_recon is per anchor MSE reconstruction loss and L_contr is symmetric
InfoNCE on adjacent pair latent codes within an episode. With alpha=1.0 this
matches the paper default. Setting alpha=0 reduces to a standard per token
TopK SAE (equivalent to experiments/train_sae.py).

Examples:

    python experiments/train_temporal_sae.py \\
        --activations-dir outputs/groot_experiments/baseline_libero_goal/activations \\
        --output-dir outputs/groot_tsae/libero_goal \\
        --layers transformer_L00 transformer_L08

    python experiments/train_temporal_sae.py \\
        --activations-dir outputs/xvla_experiments/baseline_libero_object/activations \\
        --output-dir outputs/xvla_tsae/libero_object \\
        --alpha 0.5 --tau 1.0

    python experiments/train_temporal_sae.py \\
        --activations-dir outputs/smolvla_experiments/baseline_libero_spatial/activations \\
        --output-dir outputs/smolvla_tsae/libero_spatial \\
        --k 128 --epochs 100
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tyro

from train_sae import TopKSAE


@dataclass
class TrainTemporalSAEConfig:
    """
    Temporal SAE training configuration.

    Adjacency choices:
        cross_time   pair token i at step t with token i at step t+1
        intra_step   pair token i with token i+1 within the same step
    """

    # Directory of activation .pt files. Layout matches train_sae.py:
    #     activations_dir/task{N}/ep{M}/layer_name.pt
    activations_dir: str = ""

    # Where to save trained T-SAE checkpoints.
    output_dir: str = "outputs/temporal_saes"

    # Layer names to train. Default: every .pt file found under activations_dir.
    layers: Optional[List[str]] = None

    # SAE expansion ratio so that hidden_dim = input_dim * expansion.
    expansion: float = 8.0

    # TopK sparsity: number of active features per token.
    k: int = 64

    # Contrastive loss weight. alpha=0 reduces to per token TopK SAE.
    alpha: float = 1.0

    # InfoNCE temperature (cosine similarity scale).
    tau: float = 1.0

    # Adjacent pair construction strategy.
    adjacency: str = "cross_time"

    # Maximum positive pairs per layer.
    max_pairs: int = 500_000

    epochs: int = 100
    batch_size: int = 512
    lr: float = 1e-3

    # Early stopping patience (epochs without recon improvement).
    patience: int = 5

    device: str = "cuda"


def _load_episode_activations(
    act_dir: Path, layer_name: str
) -> List[torch.Tensor]:
    """
    Return one (T*S, D) tensor per episode, preserving episode boundaries.
    """
    episodes: List[torch.Tensor] = []

    direct = act_dir / f"{layer_name}.pt"
    if direct.exists():
        t = torch.load(direct, map_location="cpu", weights_only=True).float()
        if t.dim() == 3:
            for ep_idx in range(t.shape[0]):
                episodes.append(t[ep_idx].reshape(-1, t.shape[-1]))
        elif t.dim() == 2:
            episodes.append(t)
        return episodes

    for task_dir in sorted(act_dir.glob("task*")):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.glob("ep*")):
            pt_file = ep_dir / f"{layer_name}.pt"
            if not pt_file.exists():
                continue
            try:
                t = torch.load(pt_file, map_location="cpu", weights_only=True).float()
                if t.dim() > 2:
                    t = t.reshape(-1, t.shape[-1])
                episodes.append(t)
            except Exception:
                continue
    return episodes


def _build_adjacent_pairs(
    episodes: List[torch.Tensor],
    adjacency: str,
    max_pairs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (anchors, positives) tensors from per episode activation tensors.
    """
    anchors_list: List[torch.Tensor] = []
    pairs_list: List[torch.Tensor] = []

    for ep in episodes:
        if ep.shape[0] < 2:
            continue
        if adjacency in ("cross_time", "intra_step"):
            anchors_list.append(ep[:-1])
            pairs_list.append(ep[1:])
        else:
            raise ValueError(f"Unknown adjacency: {adjacency}")

    anchors = torch.cat(anchors_list, dim=0) if anchors_list else torch.empty(0)
    positives = torch.cat(pairs_list, dim=0) if pairs_list else torch.empty(0)

    if anchors.shape[0] > max_pairs:
        idx = torch.randperm(anchors.shape[0])[:max_pairs]
        anchors = anchors[idx]
        positives = positives[idx]
    return anchors, positives


def _info_nce_symmetric(
    z_a: torch.Tensor, z_p: torch.Tensor, tau: float
) -> torch.Tensor:
    """
    Symmetric InfoNCE on L2 normalized latent codes.
    """
    z_a = F.normalize(z_a, dim=-1)
    z_p = F.normalize(z_p, dim=-1)
    logits = (z_a @ z_p.t()) / tau
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def train_temporal_sae(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    cfg: TrainTemporalSAEConfig,
) -> dict:
    """
    Train a TopK SAE with adjacent pair contrastive regularization.
    """
    input_dim = anchors.shape[1]
    hidden_dim = int(input_dim * cfg.expansion)

    stack = torch.cat([anchors, positives], dim=0)
    act_mean = stack.mean(dim=0)
    act_std = stack.std(dim=0).clamp(min=1e-8)
    anchors_n = (anchors - act_mean) / act_std
    positives_n = (positives - act_mean) / act_std

    n_pairs = anchors_n.shape[0]
    print(f"  pairs={n_pairs:,} dim={input_dim} hidden={hidden_dim} k={cfg.k} alpha={cfg.alpha} tau={cfg.tau}")

    sae = TopKSAE(input_dim, cfg.expansion, cfg.k).to(cfg.device)
    optimizer = optim.Adam(sae.parameters(), lr=cfg.lr)
    dataset = TensorDataset(anchors_n, positives_n)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    history = []
    best_recon = float("inf")
    best_state = None
    wait = 0

    for epoch in range(cfg.epochs):
        sae.train()
        total_recon = 0.0
        total_contr = 0.0
        n_batches = 0
        for anchor_b, pair_b in loader:
            anchor_b = anchor_b.to(cfg.device, non_blocking=True)
            pair_b = pair_b.to(cfg.device, non_blocking=True)

            x_hat_a, z_a = sae(anchor_b)
            x_hat_p, z_p = sae(pair_b)

            recon = 0.5 * (
                F.mse_loss(x_hat_a, anchor_b) + F.mse_loss(x_hat_p, pair_b)
            )
            if cfg.alpha > 0:
                contr = _info_nce_symmetric(z_a, z_p, cfg.tau)
            else:
                contr = torch.tensor(0.0, device=cfg.device)

            loss = recon + cfg.alpha * contr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_recon += recon.item()
            total_contr += contr.item() if cfg.alpha > 0 else 0.0
            n_batches += 1

        avg_recon = total_recon / max(n_batches, 1)
        avg_contr = total_contr / max(n_batches, 1)

        with torch.no_grad():
            sample = anchors_n[: min(10000, n_pairs)].to(cfg.device)
            x_hat_s, z_s = sae(sample)
            residual = sample - x_hat_s
            ev = (1.0 - residual.var() / sample.var()).item()
            l0 = (z_s != 0).float().sum(dim=-1).mean().item()
            dead_frac = (z_s.abs().sum(dim=0) == 0).float().mean().item()
            cos_adj = None
            if cfg.alpha > 0:
                pair_sample = positives_n[: min(10000, n_pairs)].to(cfg.device)
                _, z_p_s = sae(pair_sample)
                cos_adj = F.cosine_similarity(z_s, z_p_s, dim=-1).mean().item()

        history.append({
            "epoch": epoch,
            "recon": avg_recon,
            "contr": avg_contr,
            "explained_var": ev,
            "l0": l0,
            "dead_features_frac": dead_frac,
            "cos_adj": cos_adj,
        })

        if avg_recon < best_recon:
            best_recon = avg_recon
            best_state = {k: v.cpu().clone() for k, v in sae.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == cfg.epochs - 1 or wait >= cfg.patience:
            cos_str = f" cos_adj={cos_adj:.4f}" if cos_adj is not None else ""
            print(
                f"  epoch={epoch:3d} recon={avg_recon:.6f} contr={avg_contr:.4f} "
                f"EV={ev:.4f} L0={l0:.0f} dead={dead_frac:.3f}{cos_str}"
            )

        if wait >= cfg.patience:
            break

    return {
        "sae_state_dict": best_state,
        "activation_mean": act_mean,
        "activation_std": act_std,
        "config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "expansion": cfg.expansion,
            "k": cfg.k,
            "alpha": cfg.alpha,
            "tau": cfg.tau,
            "adjacency": cfg.adjacency,
            "n_pairs": n_pairs,
        },
        "history": history,
        "best_recon": best_recon,
    }


def main(cfg: TrainTemporalSAEConfig):
    act_dir = Path(cfg.activations_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not act_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {act_dir}")

    if cfg.layers:
        layer_names = cfg.layers
    else:
        found = set()
        for pt_file in act_dir.rglob("*.pt"):
            found.add(pt_file.stem)
        layer_names = sorted(found)
        if not layer_names:
            raise FileNotFoundError(f"No .pt files found under {act_dir}")

    results = {}
    for layer_name in layer_names:
        layer_dir = output_dir / layer_name
        if (layer_dir / "sae_best.pt").exists():
            continue

        episodes = _load_episode_activations(act_dir, layer_name)
        if not episodes:
            continue

        anchors, positives = _build_adjacent_pairs(episodes, cfg.adjacency, cfg.max_pairs)
        if anchors.shape[0] == 0:
            continue

        print(f"layer={layer_name}")
        result = train_temporal_sae(anchors, positives, cfg)

        layer_dir.mkdir(exist_ok=True)
        torch.save(result, layer_dir / "sae_best.pt")

        final = result["history"][-1]
        results[layer_name] = {
            "layer": layer_name,
            "n_pairs": result["config"]["n_pairs"],
            "best_recon": result["best_recon"],
            "final_ev": final["explained_var"],
            "final_l0": final["l0"],
            "final_dead": final["dead_features_frac"],
            "final_cos_adj": final["cos_adj"],
            "epochs_trained": final["epoch"] + 1,
        }

        del episodes, anchors, positives, result
        gc.collect()
        torch.cuda.empty_cache()

    summary = {
        "expansion": cfg.expansion,
        "k": cfg.k,
        "alpha": cfg.alpha,
        "tau": cfg.tau,
        "adjacency": cfg.adjacency,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "per_layer": results,
    }
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    tyro.cli(main)
