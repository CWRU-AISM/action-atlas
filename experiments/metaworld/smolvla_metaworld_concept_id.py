#!/usr/bin/env python3
"""
Contrastive concept identification for SmolVLA SAE features on MetaWorld.

Groups 50 MetaWorld tasks into semantic concepts (motion + object),
compares SAE feature activations between concept-present vs concept-absent tasks,
and scores features by |Cohen's d| * frequency.

Runs on CPU only (no GPU needed).

Data: mean-pooled activations at rollouts/smolvla/metaworld_activations_meanpool/activations/
      SAEs at rollouts/smolvla/sae_models/metaworld/

Usage:
    python experiments/metaworld/smolvla_metaworld_concept_id.py --component expert --all-layers
    python experiments/metaworld/smolvla_metaworld_concept_id.py --component vlm --layer 16
    python experiments/metaworld/smolvla_metaworld_concept_id.py --component expert --component vlm --all-layers
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import tyro

from common import SparseAutoencoder


METAWORLD_CONCEPTS = {
    "motion": {
        "reach": ["reach-v3", "reach-wall-v3"],
        "push": ["push-v3", "push-back-v3", "push-wall-v3"],
        "pull": ["coffee-pull-v3", "handle-pull-v3", "handle-pull-side-v3",
                 "lever-pull-v3", "stick-pull-v3"],
        "press": ["button-press-v3", "button-press-topdown-v3",
                  "button-press-topdown-wall-v3", "button-press-wall-v3",
                  "coffee-button-v3", "handle-press-v3", "handle-press-side-v3"],
        "open": ["door-open-v3", "drawer-open-v3", "faucet-open-v3",
                 "window-open-v3", "door-unlock-v3"],
        "close": ["door-close-v3", "drawer-close-v3", "faucet-close-v3",
                  "window-close-v3", "door-lock-v3", "box-close-v3"],
        "pick": ["pick-place-v3", "pick-place-wall-v3", "pick-out-of-hole-v3",
                 "bin-picking-v3", "shelf-place-v3", "hand-insert-v3",
                 "assembly-v3", "disassemble-v3"],
        "slide": ["plate-slide-v3", "plate-slide-back-v3",
                  "plate-slide-side-v3", "plate-slide-back-side-v3"],
        "turn": ["dial-turn-v3"],
        "sweep": ["sweep-v3", "sweep-into-v3"],
        "hit": ["hammer-v3", "basketball-v3", "soccer-v3"],
        "peg": ["peg-insert-side-v3", "peg-unplug-side-v3"],
        "stick": ["stick-pull-v3", "stick-push-v3"],
        "coffee": ["coffee-button-v3", "coffee-pull-v3", "coffee-push-v3"],
    },
    "object": {
        "door": ["door-open-v3", "door-close-v3", "door-lock-v3", "door-unlock-v3"],
        "drawer": ["drawer-open-v3", "drawer-close-v3"],
        "window": ["window-open-v3", "window-close-v3"],
        "faucet": ["faucet-open-v3", "faucet-close-v3"],
        "button": ["button-press-v3", "button-press-topdown-v3",
                   "button-press-topdown-wall-v3", "button-press-wall-v3",
                   "coffee-button-v3"],
        "plate": ["plate-slide-v3", "plate-slide-back-v3",
                  "plate-slide-side-v3", "plate-slide-back-side-v3"],
        "handle": ["handle-press-v3", "handle-press-side-v3",
                   "handle-pull-v3", "handle-pull-side-v3"],
    },
}

ALL_METAWORLD_TASKS = [
    "assembly-v3", "basketball-v3", "bin-picking-v3", "box-close-v3",
    "button-press-topdown-v3", "button-press-topdown-wall-v3",
    "button-press-v3", "button-press-wall-v3",
    "coffee-button-v3", "coffee-pull-v3", "coffee-push-v3",
    "dial-turn-v3", "disassemble-v3",
    "door-close-v3", "door-lock-v3", "door-open-v3", "door-unlock-v3",
    "drawer-close-v3", "drawer-open-v3",
    "faucet-close-v3", "faucet-open-v3",
    "hammer-v3", "hand-insert-v3",
    "handle-press-side-v3", "handle-press-v3",
    "handle-pull-side-v3", "handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3", "peg-unplug-side-v3",
    "pick-out-of-hole-v3", "pick-place-v3", "pick-place-wall-v3",
    "plate-slide-back-side-v3", "plate-slide-back-v3",
    "plate-slide-side-v3", "plate-slide-v3",
    "push-back-v3", "push-v3", "push-wall-v3",
    "reach-v3", "reach-wall-v3",
    "shelf-place-v3", "soccer-v3",
    "stick-pull-v3", "stick-push-v3",
    "sweep-into-v3", "sweep-v3",
    "window-close-v3", "window-open-v3",
]


def load_sae(path, device='cpu'):
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ckpt['config']
    sae = SparseAutoencoder(cfg['input_dim'], cfg['hidden_dim'], cfg['k'])
    sae.load_state_dict(ckpt['state_dict'])
    return sae.to(device).eval()


def load_activations_by_task(
    data_dir: Path,
    component: str,
    layer_idx: int,
    max_samples_per_task: int = 50_000,
) -> Dict[str, np.ndarray]:
    """
    Load mean-pooled activations grouped by task name.

    MetaWorld files are named {task_name}_ep{N}.npz.
    Returns dict mapping task_name -> (n_samples, hidden_dim).
    """
    key = f'{component}_L{layer_idx:02d}'
    task_data = defaultdict(list)
    task_counts = defaultdict(int)

    files = sorted(data_dir.glob('*.npz'))
    for f in files:
        name = f.stem
        parts = name.rsplit('_ep', 1)
        if len(parts) != 2:
            continue
        task_name = parts[0]

        try:
            int(parts[1])
        except ValueError:
            continue

        if task_counts[task_name] >= max_samples_per_task:
            continue

        try:
            d = np.load(str(f), allow_pickle=False)
            if key not in d:
                continue
            arr = np.array(d[key], dtype=np.float32)
        except Exception:
            continue

        if arr.ndim != 2:
            continue

        remaining = max_samples_per_task - task_counts[task_name]
        if arr.shape[0] > remaining:
            arr = arr[:remaining]

        task_data[task_name].append(arr)
        task_counts[task_name] += arr.shape[0]

    result = {}
    for tname, arrays in task_data.items():
        result[tname] = np.concatenate(arrays, axis=0)

    return result


def encode_all_tasks(
    sae: SparseAutoencoder,
    task_activations: Dict[str, np.ndarray],
    device: str = 'cpu',
    batch_size: int = 4096,
) -> Dict[str, np.ndarray]:
    """
    Encode all task activations through SAE once.

    Returns dict: task_name -> (n_samples, hidden_dim) sparse feature array.
    """
    encoded = {}
    for tname in sorted(task_activations.keys()):
        acts = task_activations[tname]
        n = acts.shape[0]
        chunks = []
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(acts[i:i+batch_size]).to(device)
            batch_centered = batch - batch.mean(dim=-1, keepdim=True)
            with torch.no_grad():
                sparse_h = sae.encode(batch_centered).cpu().numpy().astype(np.float32)
            chunks.append(sparse_h)
        encoded[tname] = np.concatenate(chunks, axis=0)
    return encoded


def compute_group_stats(encoded_tasks: Dict[str, np.ndarray], task_names: Set[str]):
    # Compute mean, std, freq for a group of tasks
    arrays = [encoded_tasks[t] for t in sorted(task_names) if t in encoded_tasks]
    if not arrays:
        return None
    all_h = np.concatenate(arrays, axis=0)
    n = all_h.shape[0]
    if n < 10:
        return None
    mean = all_h.mean(axis=0).astype(np.float64)
    std = all_h.std(axis=0, ddof=1).astype(np.float64)
    freq = (all_h != 0).mean(axis=0).astype(np.float64)
    return n, mean, std, freq


def compute_contrastive_scores(
    encoded_tasks: Dict[str, np.ndarray],
    concept_tasks: List[str],
    all_task_names: Set[str],
) -> Optional[Dict]:
    # Compute Cohen's d and frequency for each SAE feature
    concept_set = set(concept_tasks) & all_task_names
    non_concept_set = all_task_names - concept_set

    if not concept_set or not non_concept_set:
        return None

    result_c = compute_group_stats(encoded_tasks, concept_set)
    result_nc = compute_group_stats(encoded_tasks, non_concept_set)

    if result_c is None or result_nc is None:
        return None

    n_c, mean_c, std_c, freq_c = result_c
    n_nc, mean_nc, std_nc, _ = result_nc

    pooled_std = np.sqrt(
        ((n_c - 1) * std_c**2 + (n_nc - 1) * std_nc**2)
        / (n_c + n_nc - 2)
    )
    cohens_d = (mean_c - mean_nc) / (pooled_std + 1e-8)
    scores = np.abs(cohens_d) * freq_c

    top_idx = np.argsort(-scores)[:50]

    features = []
    for idx in top_idx:
        features.append({
            'feature_idx': int(idx),
            'score': float(scores[idx]),
            'cohens_d': float(cohens_d[idx]),
            'frequency': float(freq_c[idx]),
            'concept_mean_activation': float(mean_c[idx]),
            'non_concept_mean_activation': float(mean_nc[idx]),
        })

    return {
        'top_features': features,
        'n_concept_samples': int(n_c),
        'n_non_concept_samples': int(n_nc),
        'n_positive_tasks': len(concept_set),
        'n_negative_tasks': len(non_concept_set),
        'positive_tasks': sorted(concept_set),
        'negative_tasks': sorted(non_concept_set),
    }


def run_concept_id(component, layer_idx, sae_dir, data_dir, output_dir, device='cpu'):
    # Run contrastive concept ID for one component/layer on MetaWorld
    sae_path = sae_dir / f'{component}_L{layer_idx:02d}.pt'
    if not sae_path.exists():
        print(f"  No SAE at {sae_path}")
        return None

    sae = load_sae(sae_path, device)
    print(f"  SAE loaded: input_dim={sae.input_dim}, hidden_dim={sae.hidden_dim}, k={sae.k}")

    print(f"  Loading activations...")
    t0 = time.time()
    task_acts = load_activations_by_task(data_dir, component, layer_idx)
    t1 = time.time()

    if not task_acts:
        print(f"  No activation data found")
        return None

    total_samples = sum(v.shape[0] for v in task_acts.values())
    print(f"  {len(task_acts)} tasks, {total_samples:,} total samples ({t1-t0:.1f}s)")

    print(f"  Encoding through SAE...")
    t2 = time.time()
    encoded_tasks = encode_all_tasks(sae, task_acts, device=device)
    t3 = time.time()
    print(f"  Encoding done ({t3-t2:.1f}s)")

    all_task_names = set(encoded_tasks.keys())

    all_concepts = {}
    for concept_type, concept_dict in METAWORLD_CONCEPTS.items():
        for concept_name, concept_tasks in concept_dict.items():
            full_name = f"{concept_type}/{concept_name}"

            scores = compute_contrastive_scores(
                encoded_tasks, concept_tasks, all_task_names
            )
            if scores is None:
                print(f"    {full_name}: SKIPPED (insufficient data)")
                continue

            top = scores['top_features'][0] if scores['top_features'] else {}
            top_score = top.get('score', 0)
            top_d = top.get('cohens_d', 0)
            print(f"    {full_name}: top_score={top_score:.3f}, top_d={top_d:.3f}, "
                  f"+tasks={scores['n_positive_tasks']}, -tasks={scores['n_negative_tasks']}")

            all_concepts[full_name] = scores

    output = {
        'model': 'smolvla',
        'benchmark': 'metaworld',
        'component': component,
        'layer': layer_idx,
        'n_tasks': len(task_acts),
        'n_total_samples': total_samples,
        'n_concepts': len(all_concepts),
        'concepts': all_concepts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'smolvla_{component}_L{layer_idx:02d}_metaworld_concept_id.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path} ({len(all_concepts)} concepts)")

    return output


def print_summary(output_dir: Path):
    # Print summary of all concept ID results
    files = sorted(output_dir.glob('*.json'))
    if not files:
        print("\nNo results found.")
        return

    print(f"\nSUMMARY: {len(files)} result files in {output_dir}")

    component_summaries = defaultdict(lambda: defaultdict(list))

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        comp = data['component']
        layer = data['layer']
        concepts = data.get('concepts', {})

        for cname, cdata in concepts.items():
            top = cdata['top_features'][0] if cdata['top_features'] else None
            if top:
                component_summaries[comp][cname].append({
                    'layer': layer,
                    'score': top['score'],
                    'cohens_d': top['cohens_d'],
                    'feature_idx': top['feature_idx'],
                })

    for comp in sorted(component_summaries.keys()):
        print(f"\n--- {comp.upper()} ---")
        concept_best = {}
        for cname, entries in component_summaries[comp].items():
            best = max(entries, key=lambda x: x['score'])
            concept_best[cname] = best

        for cname, best in sorted(concept_best.items(), key=lambda x: -x[1]['score']):
            print(f"  {cname:30s}  best_score={best['score']:.3f}  "
                  f"d={best['cohens_d']:.3f}  "
                  f"layer={best['layer']:2d}  feat={best['feature_idx']}")


@dataclass
class ConceptIdConfig:
    # MetaWorld contrastive concept identification for SmolVLA

    component: List[str] = ("expert",)
    # Component types: expert, vlm

    layer: int = -1
    all_layers: bool = False
    sae_dir: str = "rollouts/smolvla/sae_models/metaworld"
    data_dir: str = "rollouts/smolvla/metaworld_activations_meanpool/activations"
    output_dir: str = "rollouts/smolvla/metaworld_concept_id"
    device: str = "cpu"
    summary_only: bool = False


def main(cfg):

    sae_dir = Path(cfg.sae_dir)
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)

    if cfg.summary_only:
        print_summary(output_dir)
        return

    if cfg.all_layers:
        layers = list(range(32))
    elif cfg.layer >= 0:
        layers = [cfg.layer]
    else:
        print("Specify --layer N or --all-layers")
        return

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return
    if not sae_dir.exists():
        print(f"ERROR: SAE directory not found: {sae_dir}")
        return

    all_concept_tasks = set()
    for concept_type, concept_dict in METAWORLD_CONCEPTS.items():
        for concept_name, tasks in concept_dict.items():
            all_concept_tasks.update(tasks)

    missing = all_concept_tasks - set(ALL_METAWORLD_TASKS)
    if missing:
        print(f"WARNING: Concept tasks not in master list: {missing}")

    total_start = time.time()

    for component in cfg.component:
        for layer_idx in layers:
            print(f"\n{component.upper()} L{layer_idx:02d} | MetaWorld concept ID")

            t0 = time.time()
            run_concept_id(
                component=component,
                layer_idx=layer_idx,
                sae_dir=sae_dir,
                data_dir=data_dir,
                output_dir=output_dir,
                device=cfg.device,
            )
            elapsed = time.time() - t0
            print(f"  Layer time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.1f}s")

    print_summary(output_dir)


if __name__ == '__main__':
    cfg = tyro.cli(ConceptIdConfig)
    main(cfg)
