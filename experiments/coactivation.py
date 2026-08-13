#!/usr/bin/env python3
"""
SAE feature coactivation analysis for VLA models.

Finds which SAE features consistently fire together. If concept_id already
told you *which* features matter for a concept, this tells you *what they
travel with* -- revealing feature circuits inside the VLA.

Works on the sparse codes produced by encoding activations through a trained
SAE. For each concept feature, computes Jaccard similarity against every other
feature across all task episodes, then ranks the top co-firing partners.

Run concept_id.py first to get concept_results, then point this at the same
SAE dir and activations.

Examples:
    python experiments/coactivation.py \
        --sae-dir outputs/xvla_saes/libero_object \
        --activations-dir outputs/xvla_experiments/baseline_libero_object/activations \
        --concept-results results/concept_id/libero_object/all_layers.json \
        --suite libero_object

    python experiments/coactivation.py \
        --sae-dir outputs/groot_saes/libero_goal \
        --activations-dir outputs/groot_experiments/baseline_libero_goal/activations \
        --concept-results results/concept_id/libero_goal/all_layers.json \
        --suite libero_goal --layers eagle_L04 eagle_L08 --top-k 15
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.sae_hooks import TopKSAE
from experiments.concept_identification import get_concept_task_mapping


@dataclass
class CoactivationConfig:

    sae_dir: str = ""
    activations_dir: str = ""

    concept_results: str = ""
    # Path to concept_id all_layers.json -- used to pull the concept features
    # we actually care about so we're not computing jaccard over 8k x 8k

    suite: str = "libero_object"
    layers: Optional[List[str]] = None
    output_dir: Optional[str] = None

    top_k: int = 10
    # top co-firing partners to report per concept feature

    top_concept_features: int = 5
    # how many features per concept to trace (ranked by concept_id score)

    max_samples: int = 200_000
    # cap total samples to keep memory sane


def load_sae_checkpoint(sae_path: Path, device: str):
    data = torch.load(sae_path, map_location="cpu", weights_only=True)
    cfg = data["config"]
    sae = TopKSAE(cfg["input_dim"], cfg["hidden_dim"], k=cfg.get("k", 64))
    sae.load_state_dict(data["sae_state_dict"])
    sae.eval().to(device)
    act_mean = data.get("activation_mean", data.get("act_mean", torch.zeros(cfg["input_dim"])))
    act_std  = data.get("activation_std",  data.get("act_std",  torch.ones(cfg["input_dim"])))
    return sae, act_mean.to(device), act_std.to(device)


def load_all_activations(act_dir: Path, layer_name: str, max_samples: int) -> Optional[torch.Tensor]:
    # grab everything, don't care about task splits here -- we just want the
    # full distribution of co-firing patterns across the whole suite
    all_acts = []
    total = 0

    for task_dir in sorted(act_dir.glob("task*")):
        if not task_dir.is_dir():
            continue
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir():
                continue
            pt = ep_dir / f"{layer_name}.pt"
            if not pt.exists():
                continue
            try:
                t = torch.load(pt, map_location="cpu", weights_only=True).float()
                if t.dim() > 2:
                    t = t.reshape(-1, t.shape[-1])
                all_acts.append(t)
                total += t.shape[0]
            except Exception:
                continue
        if total >= max_samples:
            break

    # flat file fallback (pre-concatenated layout)
    if not all_acts:
        for pt in sorted(act_dir.glob(f"{layer_name}_task*.pt")):
            try:
                t = torch.load(pt, map_location="cpu", weights_only=True).float()
                if t.dim() > 2:
                    t = t.reshape(-1, t.shape[-1])
                all_acts.append(t)
            except Exception:
                continue

    if not all_acts:
        return None

    combined = torch.cat(all_acts, dim=0)
    if combined.shape[0] > max_samples:
        idx = torch.randperm(combined.shape[0])[:max_samples]
        combined = combined[idx]
    return combined


def jaccard_for_feature(binary_codes: np.ndarray, feature_idx: int) -> np.ndarray:
    # binary_codes is [N, hidden_dim] bool array
    # returns jaccard similarity of feature_idx against every other feature
    a = binary_codes[:, feature_idx]  # [N]
    # intersection: both active
    inter = (binary_codes & a[:, None]).sum(axis=0).astype(np.float32)
    # union: either active
    union = (binary_codes | a[:, None]).sum(axis=0).astype(np.float32)
    jac = np.where(union > 0, inter / union, 0.0)
    jac[feature_idx] = 0.0  # don't report self
    return jac


def pull_concept_features(concept_results: dict, layer_name: str, top_n: int) -> Dict[str, List[int]]:
    # returns {concept_name: [feature_idx, ...]} for the layer
    layer_data = concept_results.get(layer_name, {})
    out = {}
    for cat, concepts in layer_data.items():
        for concept_name, info in concepts.items():
            feats = info.get("top_features", [])[:top_n]
            if feats:
                key = f"{cat}/{concept_name}"
                out[key] = [f["feature_idx"] for f in feats]
    return out


def analyze_layer(sae, act_mean, act_std, acts: torch.Tensor,
                  concept_features: Dict[str, List[int]], top_k: int, device: str) -> dict:

    # encode activations -> sparse codes
    print(f"  encoding {acts.shape[0]:,} samples through SAE...")
    batch_size = 4096
    codes_list = []
    with torch.no_grad():
        for i in range(0, acts.shape[0], batch_size):
            batch = acts[i:i+batch_size].to(device)
            batch_norm = (batch - act_mean) / (act_std + 1e-8)
            z = sae.encode(batch_norm)
            codes_list.append(z.cpu())
    codes = torch.cat(codes_list, dim=0)  # [N, hidden_dim]

    # binarize -- a feature is "active" if it's nonzero (TopK so it's clean)
    binary = (codes.abs() > 0).numpy()

    results = {}
    for concept_key, feat_indices in concept_features.items():
        concept_result = {}
        for feat_idx in feat_indices:
            jac = jaccard_for_feature(binary, feat_idx)
            top_idx = np.argsort(jac)[::-1][:top_k]
            partners = []
            for partner_idx in top_idx:
                if jac[partner_idx] < 1e-4:
                    break
                partners.append({
                    "feature_idx": int(partner_idx),
                    "jaccard": float(jac[partner_idx]),
                    "co_fire_rate": float(binary[:, partner_idx].mean()),
                })
            concept_result[str(feat_idx)] = {
                "fire_rate": float(binary[:, feat_idx].mean()),
                "top_coactive": partners,
            }
        results[concept_key] = concept_result

    return results


def main(cfg: CoactivationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_dir = Path(cfg.sae_dir)
    act_dir = Path(cfg.activations_dir)

    if not cfg.concept_results:
        raise ValueError("--concept-results is required (point at concept_id all_layers.json)")

    with open(cfg.concept_results) as f:
        concept_results = json.load(f)

    output_dir = Path(cfg.output_dir) if cfg.output_dir else Path(f"results/coactivation/{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.layers:
        layer_names = cfg.layers
    else:
        layer_names = sorted([d.name for d in sae_dir.iterdir()
                               if d.is_dir() and (d / "sae_best.pt").exists()])

    print(f"Coactivation analysis: {cfg.suite}")
    print(f"  {len(layer_names)} layers, top_k={cfg.top_k}, concept_features={cfg.top_concept_features}")

    all_results = {}

    for layer_name in layer_names:
        print(f"\n--- {layer_name} ---")

        sae_path = sae_dir / layer_name / "sae_best.pt"
        if not sae_path.exists():
            print("  [skip] no SAE")
            continue

        concept_feats = pull_concept_features(concept_results, layer_name, cfg.top_concept_features)
        if not concept_feats:
            print("  [skip] no concept features for this layer in concept_results")
            continue

        n_feats_total = sum(len(v) for v in concept_feats.values())
        print(f"  tracing {n_feats_total} concept features across {len(concept_feats)} concepts")

        acts = load_all_activations(act_dir, layer_name, cfg.max_samples)
        if acts is None:
            print("  [skip] no activations found")
            continue
        print(f"  loaded {acts.shape[0]:,} samples")

        sae, act_mean, act_std = load_sae_checkpoint(sae_path, device)

        layer_out = analyze_layer(sae, act_mean, act_std, acts, concept_feats,
                                  cfg.top_k, device)
        all_results[layer_name] = layer_out

        # quick summary print
        for concept_key, feat_data in layer_out.items():
            for feat_idx, info in feat_data.items():
                if info["top_coactive"]:
                    top = info["top_coactive"][0]
                    print(f"    {concept_key} feat {feat_idx}: "
                          f"top partner={top['feature_idx']} jaccard={top['jaccard']:.3f}")

        out_path = output_dir / f"{layer_name}.json"
        with open(out_path, "w") as f:
            json.dump(layer_out, f, indent=2)
        print(f"  saved: {out_path}")

        del sae, acts
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(output_dir / "all_layers.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\ndone. results -> {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CoactivationConfig)
    main(cfg)
