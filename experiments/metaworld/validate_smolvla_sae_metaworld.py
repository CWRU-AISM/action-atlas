#!/usr/bin/env python3
"""
Validate MetaWorld SAE reconstruction by hooking SAEs into SmolVLA rollouts.

For each layer, replaces MLP output with SAE(output) and measures success rate.
Compares against baseline (no SAE).

Usage:
    python experiments/metaworld/validate_smolvla_sae_metaworld.py --component expert --layer 0 --n-episodes 3
    python experiments/metaworld/validate_smolvla_sae_metaworld.py --component expert --all-layers --n-episodes 3
    python experiments/metaworld/validate_smolvla_sae_metaworld.py --component expert --layers 0 5 15 31 --n-episodes 5
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

from common import (
    DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, MAX_STEPS, TASK_DESCRIPTIONS,
    SparseAutoencoder, create_env, get_layer_modules, load_smolvla_policy,
    run_episode,
)


VALIDATION_TASKS = [
    'reach-v3', 'push-v3', 'drawer-close-v3', 'drawer-open-v3',
    'button-press-v3', 'window-open-v3', 'window-close-v3',
    'door-open-v3', 'faucet-open-v3', 'faucet-close-v3',
]


def load_sae(path, device):
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ckpt['config']
    sae = SparseAutoencoder(cfg['input_dim'], cfg['hidden_dim'], cfg['k'])
    sae.load_state_dict(ckpt['state_dict'])
    return sae.to(device).eval()


def make_sae_hook(sae):
    """Forward hook that replaces MLP output with SAE reconstruction."""
    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        orig_dtype = h.dtype
        with torch.no_grad():
            h_reconstructed = sae(h.float()).to(orig_dtype)
        if isinstance(output, tuple):
            return (h_reconstructed,) + output[1:]
        return h_reconstructed
    return hook_fn


def run_validation_episode(policy, preprocessor, postprocessor, task_name,
                           resolution, device):
    """Run one MetaWorld episode, return (success, n_steps)."""
    env = create_env(task_name, resolution)
    result = run_episode(policy, env, preprocessor, postprocessor, device)
    env.close()
    return result['success'], result['n_steps']


@dataclass
class ValidateSAEConfig:
    """Validate SAE reconstruction on MetaWorld rollouts."""

    component: str
    """Component type: expert, vlm"""

    layer: int = -1
    layers: Optional[List[int]] = None
    all_layers: bool = False
    sae_dir: str = "rollouts/smolvla/sae_models/metaworld"
    tasks: Optional[str] = None
    """Comma-separated task names (default: VALIDATION_TASKS)"""

    n_episodes: int = 3
    checkpoint: str = DEFAULT_CHECKPOINT
    resolution: int = DEFAULT_RESOLUTION
    output_dir: str = "rollouts/smolvla/sae_validation/metaworld"
    action_horizon: int = 10
    skip_baseline: bool = False


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.tasks:
        tasks = cfg.tasks.split(',')
    else:
        tasks = VALIDATION_TASKS

    if cfg.all_layers:
        layers = list(range(32))
    elif cfg.layers:
        layers = cfg.layers
    elif cfg.layer >= 0:
        layers = [cfg.layer]
    else:
        print("Specify --layer N, --layers N N N, or --all-layers")
        return

    sae_dir = Path(cfg.sae_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{cfg.component}_results.json'

    results = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming from {out_path} ({len(results)} entries)")

    print("Loading SmolVLA MetaWorld model...")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device, cfg.action_horizon)

    expert_layers, vlm_layers = get_layer_modules(policy)
    if cfg.component == 'expert':
        target_layers = expert_layers
    else:
        target_layers = vlm_layers

    if not cfg.skip_baseline and 'baseline' not in results:
        print(f"\nBASELINE (no SAE) -- {len(tasks)} tasks x {cfg.n_episodes} eps")
        successes = 0
        total = 0
        per_task = {}
        for task_name in tasks:
            task_s = 0
            for ep in range(cfg.n_episodes):
                success, steps = run_validation_episode(
                    policy, preprocessor, postprocessor, task_name,
                    cfg.resolution, device)
                successes += int(success)
                task_s += int(success)
                total += 1
                status = "OK" if success else "FAIL"
                print(f"  {task_name} ep{ep}: {status} ({steps})")
            per_task[task_name] = task_s / cfg.n_episodes
        rate = successes / total if total > 0 else 0
        print(f"  Baseline: {successes}/{total} = {rate:.1%}")
        results['baseline'] = {'successes': successes, 'total': total, 'rate': rate, 'per_task': per_task}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    for layer_idx in layers:
        key = f'{cfg.component}_L{layer_idx:02d}'
        if key in results:
            print(f"\nSkipping {key} (already done)")
            continue

        sae_path = sae_dir / f'{key}.pt'
        if not sae_path.exists():
            print(f"\nSkipping {key}: no SAE at {sae_path}")
            continue

        print(f"\nValidating SAE: {key}")

        sae = load_sae(sae_path, device)
        hook = target_layers[layer_idx].mlp.register_forward_hook(make_sae_hook(sae))

        successes = 0
        total = 0
        per_task = {}
        for task_name in tasks:
            task_s = 0
            for ep in range(cfg.n_episodes):
                success, steps = run_validation_episode(
                    policy, preprocessor, postprocessor, task_name,
                    cfg.resolution, device)
                successes += int(success)
                task_s += int(success)
                total += 1
                status = "OK" if success else "FAIL"
                print(f"  {task_name} ep{ep}: {status} ({steps})")
            per_task[task_name] = task_s / cfg.n_episodes

        hook.remove()

        rate = successes / total if total > 0 else 0
        baseline_rate = results.get('baseline', {}).get('rate', 0)
        fidelity = rate / baseline_rate if baseline_rate > 0 else float('inf')
        print(f"  {key}: {successes}/{total} = {rate:.1%} (fidelity={fidelity:.2f})")

        results[key] = {
            'successes': successes, 'total': total, 'rate': rate,
            'fidelity': fidelity, 'per_task': per_task,
        }

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\nSUMMARY")
    baseline_rate = results.get('baseline', {}).get('rate', 0)
    print(f"  baseline: {baseline_rate:.1%}")
    for key in sorted(k for k in results if k != 'baseline'):
        r = results[key]
        print(f"  {key}: {r['rate']:.1%} (fidelity={r.get('fidelity', 'N/A')})")
    print(f"\nResults: {out_path}")


if __name__ == '__main__':
    cfg = tyro.cli(ValidateSAEConfig)
    main(cfg)
