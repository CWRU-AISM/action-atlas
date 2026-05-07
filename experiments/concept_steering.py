#!/usr/bin/env python3
"""
SAE-based concept steering for any supported VLA model.

Amplifies or suppresses concept-selective SAE features to steer model
behavior. Tests whether identified concepts causally influence actions.

Examples:
    python experiments/concept_steering.py --model xvla --suite libero_object \\
        --sae-dir outputs/xvla_saes/libero_object \\
        --concept-results results/concept_id/libero_object/all_layers.json \\
        --layer transformer_L12 --strengths -2.0 -1.0 1.0 2.0

    python experiments/concept_steering.py --model groot --suite libero_goal \\
        --sae-dir outputs/groot_saes/libero_goal \\
        --concept-results results/concept_id/libero_goal/all_layers.json \\
        --layer eagle_L08 --concepts motion/put object/bowl
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.model_adapters import get_adapter
from experiments.sae_hooks import PerTokenSteeringHook, TopKSAE
from experiments.utils import (
    force_free_memory, save_results, save_video, SUITE_MAX_STEPS,
)


@dataclass
class ConceptSteeringConfig:
    # SAE-based concept steering experiment

    model: str = "xvla"
    suite: str = "libero_object"
    checkpoint: Optional[str] = None

    sae_dir: str = ""
    concept_results: str = ""
    layer: str = ""

    n_features: int = 5
    # Number of top features to steer per concept

    strengths: Tuple[float, ...] = (-2.0, -1.0, 1.0, 2.0)
    # Steering strengths to test. Negative = suppress, positive = amplify

    concepts: Optional[List[str]] = None
    # Specific concepts to steer (e.g. 'motion/put'). Default: all

    n_episodes: int = 3
    tasks: Optional[List[int]] = None
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = True

    gpu: int = 0

    n_action_steps: Optional[int] = None
    # Override action chunk size for faster inference
    # GPU device index


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/concept_steering_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.concept_results) as f:
        all_concept_results = json.load(f)
    layer_concepts = all_concept_results[cfg.layer]

    # Load SAE
    sae_path = Path(cfg.sae_dir) / cfg.layer / "sae_best.pt"
    sae_data = torch.load(sae_path, map_location="cpu")
    sae_config = sae_data["config"]
    sae = TopKSAE(sae_config["input_dim"], sae_config["hidden_dim"],
                  k=sae_config.get("k", 64))
    sae.load_state_dict(sae_data["sae_state_dict"])
    sae.eval().to(device)
    act_mean = sae_data.get("activation_mean", sae_data.get("act_mean",
                torch.zeros(sae_config["input_dim"]))).to(device)
    act_std = sae_data.get("activation_std", sae_data.get("act_std",
                torch.ones(sae_config["input_dim"]))).to(device)

    # Load model
    adapter = get_adapter(cfg.model)
    ckpt = cfg.checkpoint or adapter.default_checkpoints.get(cfg.suite)
    print(f"Concept Steering: {cfg.model} on {cfg.suite}, layer={cfg.layer}")
    adapter.load_model(ckpt, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    layer_module = None
    for label, module in adapter.get_all_layers():
        if label == cfg.layer:
            layer_module = module
            break
    if layer_module is None:
        raise ValueError(f"Layer '{cfg.layer}' not found")

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    # Build concept -> feature mapping
    concept_features = {}
    for concept_type, concepts in layer_concepts.items():
        for concept_name, concept_data in concepts.items():
            key = f"{concept_type}/{concept_name}"
            if cfg.concepts and key not in cfg.concepts:
                continue
            features = concept_data.get("top_features", [])
            feat_ids = [f["feature_idx"] for f in features[:cfg.n_features]]
            if feat_ids:
                concept_features[key] = feat_ids

    print(f"  concepts: {len(concept_features)}")
    print(f"  strengths: {list(cfg.strengths)}")

    # Register hook
    hook = PerTokenSteeringHook(sae, act_mean, act_std, device=device)
    handle = layer_module.register_forward_hook(hook)

    results = {}
    start_time = time.time()

    # Run baseline first
    print("\nBaseline (no steering):")
    hook.enabled = False
    baseline_results = {}
    for tid in task_ids:
        env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
        successes = 0
        for ep in range(cfg.n_episodes):
            hook.reset()
            r = adapter.run_episode(env, desc, max_steps=max_steps, seed=cfg.seed + ep)
            if r["success"]:
                successes += 1
        if hasattr(env, "close"):
            env.close()
        rate = successes / cfg.n_episodes
        baseline_results[str(tid)] = {"success_rate": rate, "successes": successes}
        print(f"  Task {tid}: {rate:.0%}")
    results["baseline"] = baseline_results

    # Run steering conditions
    for concept_key, feat_ids in concept_features.items():
        for strength in cfg.strengths:
            cond_name = f"{concept_key}_s{strength:+.1f}"
            print(f"\n{cond_name} (features: {feat_ids[:5]})")

            cond_dir = output_dir / cond_name.replace("/", "_")
            cond_dir.mkdir(parents=True, exist_ok=True)

            cond_results = {}
            for tid in task_ids:
                env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
                successes = 0

                for ep in range(cfg.n_episodes):
                    hook.clear()
                    hook.set_steering(feat_ids, strength)
                    hook.enabled = True
                    hook.reset()

                    r = adapter.run_episode(
                        env, desc, max_steps=max_steps,
                        save_video=(cfg.record_video and ep == 0),
                        seed=cfg.seed + ep,
                    )
                    if r["success"]:
                        successes += 1
                    if r.get("frames"):
                        save_video(r["frames"], cond_dir / f"task{tid}_ep{ep}.mp4")

                if hasattr(env, "close"):
                    env.close()

                rate = successes / cfg.n_episodes
                bl_rate = baseline_results.get(str(tid), {}).get("success_rate", 0)
                print(f"  Task {tid}: {rate:.0%} (delta {rate - bl_rate:+.0%})")
                cond_results[str(tid)] = {
                    "success_rate": rate, "successes": successes,
                    "delta": rate - bl_rate,
                }

            results[cond_name] = cond_results
            save_results(cond_results, cond_dir / "results.json")
            force_free_memory()

    handle.remove()
    elapsed = time.time() - start_time

    save_results({
        "model": cfg.model, "suite": cfg.suite, "layer": cfg.layer,
        "strengths": list(cfg.strengths), "n_features": cfg.n_features,
        "results": results, "duration_seconds": elapsed,
    }, output_dir / "summary.json")
    print(f"\nDone in {elapsed/60:.1f} min. Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ConceptSteeringConfig)
    main(cfg)
