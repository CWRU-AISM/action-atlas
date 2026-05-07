#!/usr/bin/env python3
"""
SAE-based concept identification for any VLA model.

Loads trained SAEs and activation data, encodes activations through the SAE,
groups by task, and computes Cohen's d x frequency to identify task-selective
features for each concept category (motion, object, spatial).

Examples:
    python experiments/concept_id.py \\
        --sae-dir outputs/xvla_saes/libero_object \\
        --activations-dir outputs/xvla_experiments/baseline_libero_object/activations \\
        --suite libero_object

    python experiments/concept_id.py \\
        --sae-dir outputs/groot_saes/libero_goal \\
        --activations-dir outputs/groot_experiments/baseline_libero_goal/activations \\
        --suite libero_goal --layers eagle_L04 eagle_L08
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
class ConceptIDConfig:
    # SAE-based concept identification

    sae_dir: str = ""
    # Directory containing trained SAE checkpoints (layer_name/sae_best.pt)

    activations_dir: str = ""
    # Directory containing activation .pt files

    suite: str = "libero_object"
    """
    Task suite for concept definitions:
    libero_goal, libero_object, libero_spatial, libero_10,
    widowx, google_robot"""

    layers: Optional[List[str]] = None
    # Layer names to process. Default: all found in sae_dir

    output_dir: Optional[str] = None
    top_k_features: int = 20
    # Number of top features to report per concept

    max_samples_per_task: int = 50000


def load_sae_checkpoint(sae_path: Path, device: str = "cuda"):
    # Load SAE model and normalization stats from checkpoint
    data = torch.load(sae_path, map_location="cpu", weights_only=True)
    config = data["config"]
    sae = TopKSAE(config["input_dim"], config["hidden_dim"], k=config.get("k", 64))
    sae.load_state_dict(data["sae_state_dict"])
    sae.eval().to(device)
    act_mean = data.get("activation_mean", data.get("act_mean", torch.zeros(config["input_dim"])))
    act_std = data.get("activation_std", data.get("act_std", torch.ones(config["input_dim"])))
    return sae, act_mean.to(device), act_std.to(device)


def load_task_activations(act_dir: Path, layer_name: str, n_tasks: int,
                          max_per_task: int) -> Dict[int, torch.Tensor]:
    task_acts = {}
    for tid in range(n_tasks):
        all_acts = []
        for task_dir in sorted(act_dir.glob(f"task{tid}")):
            for ep_dir in sorted(task_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                pt_file = ep_dir / f"{layer_name}.pt"
                if pt_file.exists():
                    try:
                        t = torch.load(pt_file, map_location="cpu", weights_only=True).float()
                        if t.dim() > 2:
                            t = t.reshape(-1, t.shape[-1])
                        all_acts.append(t)
                    except Exception:
                        continue

        # Also check direct file pattern
        direct = act_dir / f"{layer_name}_task{tid}.pt"
        if direct.exists():
            try:
                t = torch.load(direct, map_location="cpu", weights_only=True).float()
                if t.dim() > 2:
                    t = t.reshape(-1, t.shape[-1])
                all_acts.append(t)
            except Exception:
                pass

        if all_acts:
            combined = torch.cat(all_acts, dim=0)
            if combined.shape[0] > max_per_task:
                indices = torch.randperm(combined.shape[0])[:max_per_task]
                combined = combined[indices]
            task_acts[tid] = combined

    return task_acts


def compute_concept_scores(sae, act_mean, act_std, task_acts, concept_mapping,
                           device="cuda", top_k=20):
    # Compute Cohen's d x frequency for each concept's features
    # Encode all task activations through SAE
    task_features = {}
    for tid, acts in task_acts.items():
        acts_norm = (acts.to(device) - act_mean) / (act_std + 1e-8)
        with torch.no_grad():
            z = sae.encode(acts_norm)  # [N, hidden_dim]
        task_features[tid] = z.cpu()

    # Pool all features for global stats
    all_features = torch.cat(list(task_features.values()), dim=0)
    global_mean = all_features.mean(dim=0)
    global_std = all_features.std(dim=0).clamp(min=1e-8)

    results = {}
    for concept_type, concepts in concept_mapping.items():
        concept_results = {}
        for concept_name, concept_info in concepts.items():
            concept_tasks = concept_info["tasks"]
            if not concept_tasks:
                continue

            # In-concept activations
            in_acts = [task_features[t] for t in concept_tasks if t in task_features]
            if not in_acts:
                continue
            in_features = torch.cat(in_acts, dim=0)

            # Out-of-concept activations
            out_tasks = [t for t in task_features if t not in concept_tasks]
            out_acts = [task_features[t] for t in out_tasks]
            if not out_acts:
                continue
            out_features = torch.cat(out_acts, dim=0)

            # Cohen's d per feature
            in_mean = in_features.mean(dim=0)
            out_mean = out_features.mean(dim=0)
            in_std = in_features.std(dim=0).clamp(min=1e-8)
            out_std = out_features.std(dim=0).clamp(min=1e-8)
            pooled_std = torch.sqrt((in_std**2 + out_std**2) / 2).clamp(min=1e-8)
            cohens_d = (in_mean - out_mean) / pooled_std

            # Frequency: fraction of in-concept samples where feature is active
            freq = (in_features.abs() > 0).float().mean(dim=0)

            # Score = Cohen's d x frequency
            score = cohens_d * freq

            # Top features
            top_vals, top_idx = torch.topk(score.abs(), min(top_k, len(score)))
            top_features = []
            for rank, (val, idx) in enumerate(zip(top_vals, top_idx)):
                top_features.append({
                    "rank": rank,
                    "feature_idx": int(idx),
                    "score": float(score[idx]),
                    "cohens_d": float(cohens_d[idx]),
                    "frequency": float(freq[idx]),
                })

            concept_results[concept_name] = {
                "tasks": concept_tasks,
                "n_in_samples": int(in_features.shape[0]),
                "n_out_samples": int(out_features.shape[0]),
                "top_features": top_features,
            }

        results[concept_type] = concept_results

    return results


def main(cfg: ConceptIDConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae_dir = Path(cfg.sae_dir)
    act_dir = Path(cfg.activations_dir)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"results/concept_id/{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get concept mappings
    concept_mapping = get_concept_task_mapping(cfg.suite)
    if not concept_mapping:
        raise ValueError(f"No concept mapping found for suite: {cfg.suite}")

    # Determine number of tasks from concept mapping
    all_task_ids = set()
    for concepts in concept_mapping.values():
        for info in concepts.values():
            all_task_ids.update(info["tasks"])
    n_tasks = max(all_task_ids) + 1

    # Discover layers
    if cfg.layers:
        layer_names = cfg.layers
    else:
        layer_names = sorted([d.name for d in sae_dir.iterdir()
                              if d.is_dir() and (d / "sae_best.pt").exists()])

    print(f"Concept Identification: {cfg.suite}")
    print(f"  SAE dir: {sae_dir}")
    print(f"  Activations: {act_dir}")
    print(f"  Layers: {len(layer_names)}")
    print(f"  Concepts: {sum(len(v) for v in concept_mapping.values())} "
          f"across {len(concept_mapping)} categories")

    all_results = {}
    for layer_name in layer_names:
        print(f"\nLayer: {layer_name}")

        sae_path = sae_dir / layer_name / "sae_best.pt"
        if not sae_path.exists():
            print(f"  [SKIP] No SAE checkpoint")
            continue

        sae, act_mean, act_std = load_sae_checkpoint(sae_path, device)
        task_acts = load_task_activations(act_dir, layer_name, n_tasks,
                                          cfg.max_samples_per_task)
        if not task_acts:
            print(f"  [SKIP] No activations found")
            continue

        print(f"  Loaded activations for {len(task_acts)} tasks "
              f"({sum(t.shape[0] for t in task_acts.values()):,} total samples)")

        results = compute_concept_scores(
            sae, act_mean, act_std, task_acts, concept_mapping,
            device=device, top_k=cfg.top_k_features,
        )
        all_results[layer_name] = results

        # Save per-layer results
        layer_out = output_dir / f"{layer_name}.json"
        with open(layer_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {layer_out}")

        # Print top concept per category
        for cat, concepts in results.items():
            if not concepts:
                continue
            best = max(concepts.items(),
                       key=lambda x: abs(x[1]["top_features"][0]["score"]) if x[1]["top_features"] else 0)
            if best[1]["top_features"]:
                top = best[1]["top_features"][0]
                print(f"    {cat}: best concept='{best[0]}' "
                      f"(feature {top['feature_idx']}, score={top['score']:.3f})")

        del sae, task_acts
        gc.collect()
        torch.cuda.empty_cache()

    # Save combined results
    with open(output_dir / "all_layers.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ConceptIDConfig)
    main(cfg)
