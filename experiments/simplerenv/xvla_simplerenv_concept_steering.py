#!/usr/bin/env python3
"""
X-VLA Concept Steering using SAE Features on SimplerEnv.

Runs rollout episodes with concept features steered (amplified/suppressed)
via PerTokenSteeringHook to measure causal impact on task success across
multiple steering strengths in SimplerEnv (WidowX and Google Robot tasks).

Adapted from xvla_concept_steering.py (LIBERO) + xvla_simplerenv_eval.py:
- Model: XVLAPolicy from lerobot/xvla-widowx or lerobot/xvla-google-robot
- Hook target: policy.model.transformer.blocks[layer_idx] (24 blocks)
- SAE dir: outputs/xvla_saes/simplerenv_all_pertoken/layer_NN/sae_best.pt
- Concept features from: results/xvla_concept_id/xvla_concept_id_layerNN_{widowx,google_robot}.json

IMPORTANT: Must run in `simpler_env` conda env.

Usage:
    conda activate simpler_env

    # Single layer with default strengths
    python experiments/xvla_simplerenv_concept_steering.py \
        --robot widowx --layer 12 --n-episodes 5

    # Custom strengths
    python experiments/xvla_simplerenv_concept_steering.py \
        --robot widowx --layer 12 --strengths 0.5 1.0 2.0 5.0 --n-episodes 5

    # All layers
    python experiments/xvla_simplerenv_concept_steering.py \
        --robot widowx --all-layers --n-episodes 5

    # Google Robot
    python experiments/xvla_simplerenv_concept_steering.py \
        --robot google-robot --all-layers --n-episodes 5 --skip-baseline
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, N_LAYERS, DEFAULT_MAX_STEPS,
    load_xvla_policy, load_xvla_sae, load_concept_features,
    get_hook_target, run_episode,
    simpler_env,
)
from experiments.sae_hooks import PerTokenSteeringHook

SAE_DIR = "outputs/xvla_saes/simplerenv_all_pertoken"
CONCEPT_ID_DIR = "results/xvla_concept_id"
OUTPUT_DIR = "results/xvla_simplerenv_concept_steering"


def run_steering_experiment(
    policy, tasks, robot_type, domain_id, device, tokenizer,
    tokenizer_max_length, sae, act_mean, act_std,
    layer_idx, robot, concepts, n_episodes, max_steps,
    top_n_features, strengths, skip_baseline=False,
):
    # Run steering experiments across multiple strength levels
    results = {
        "model": "xvla",
        "robot": robot,
        "layer": layer_idx,
        "mode": "steering",
        "strengths": strengths,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "top_n_features": top_n_features,
        "timestamp": str(datetime.now()),
        "tasks": {},
    }

    if skip_baseline:
        print(f"\nUSING KNOWN BASELINES (skipping rollouts)")
        baseline_results = {}
        for task_name in tasks:
            baseline_results[task_name] = {
                "success_rate": 0.8,
                "successes": [True] * int(0.8 * n_episodes) + [False] * (n_episodes - int(0.8 * n_episodes)),
            }
            print(f"  {task_name}: 80% (estimated)")
    else:
        print(f"\nBASELINE (no hook)")
        baseline_results = {}
        for task_name in tasks:
            env = simpler_env.make(task_name, max_episode_steps=max_steps)
            successes = []
            for ep in range(n_episodes):
                result = run_episode(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=max_steps, episode_id=ep, task_name=task_name,
                    robot_type=robot_type,
                    tokenizer_max_length=tokenizer_max_length,
                )
                successes.append(result["success"])
            env.close()
            rate = sum(successes) / len(successes)
            baseline_results[task_name] = {
                "success_rate": rate,
                "successes": successes,
            }
            print(f"  {task_name}: {rate*100:.0f}% ({sum(successes)}/{n_episodes})")

    results["baseline"] = baseline_results

    target_module = get_hook_target(policy, layer_idx)
    hook = PerTokenSteeringHook(sae, act_mean, act_std, device=device)
    handle = target_module.register_forward_hook(hook)

    try:
        for concept_name, concept_info in concepts.items():
            features = concept_info["features"][:top_n_features]
            print(f"\nSTEERING: {concept_name} (features: {features[:5]}...)")

            concept_results = {
                "features": features,
                "scores": concept_info["scores"][:top_n_features],
                "strengths": {},
            }

            for strength in strengths:
                print(f"  Strength: {strength}")
                hook.set_steering(features, strength)

                strength_results = {}
                for task_name in tasks:
                    env = simpler_env.make(task_name, max_episode_steps=max_steps)
                    successes = []
                    for ep in range(n_episodes):
                        hook.reset()
                        hook._verified = False
                        result = run_episode(
                            policy, env, domain_id, device, tokenizer,
                            max_steps=max_steps, episode_id=ep, task_name=task_name,
                            robot_type=robot_type,
                            tokenizer_max_length=tokenizer_max_length,
                        )
                        successes.append(result["success"])
                    env.close()

                    rate = sum(successes) / len(successes)
                    delta = rate - baseline_results[task_name]["success_rate"]
                    strength_results[task_name] = {
                        "success_rate": rate,
                        "delta": delta,
                        "successes": successes,
                    }

                hook.clear()
                concept_results["strengths"][str(strength)] = strength_results

                avg_rate = np.mean([v["success_rate"] for v in strength_results.values()])
                avg_delta = np.mean([v["delta"] for v in strength_results.values()])
                print(f"    Avg rate: {avg_rate*100:.1f}%, avg delta: {avg_delta*100:+.1f}pp")

            results["tasks"][concept_name] = concept_results

    finally:
        handle.remove()

    return results


@dataclass
class ConceptSteeringConfig:
    # X-VLA concept steering on SimplerEnv

    robot: str
    # Robot type: widowx, google-robot

    layer: Optional[int] = None
    # Single layer index

    layers: Optional[str] = None
    # Comma-separated layers

    all_layers: bool = False
    strengths: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
    n_episodes: int = 5
    max_steps: int = DEFAULT_MAX_STEPS
    top_n_features: int = 30
    skip_baseline: bool = False
    device: str = "cuda"
    output_dir: str = OUTPUT_DIR
    sae_dir: str = SAE_DIR
    checkpoint: Optional[str] = None
    tasks: Optional[List[str]] = None
    # Specific tasks (default: all for robot)


def main(cfg):

    if cfg.all_layers:
        layers = list(range(N_LAYERS))
    elif cfg.layers:
        layers = [int(x) for x in cfg.layers.split(",")]
    elif cfg.layer is not None:
        layers = [cfg.layer]
    else:
        layers = [0, 12, 23]

    config = MODEL_CONFIGS[cfg.robot]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = config["domain_id"]
    tasks = cfg.tasks or config["tasks"]

    print(f"Concept steering: {cfg.robot} | layers={layers} | "
          f"strengths={cfg.strengths} | eps={cfg.n_episodes} | top_n={cfg.top_n_features}")

    policy, tokenizer = load_xvla_policy(cfg.robot, checkpoint, cfg.device)

    tokenizer_max_length = policy.config.tokenizer_max_length

    os.makedirs(cfg.output_dir, exist_ok=True)

    for layer_idx in layers:
        print(f"LAYER {layer_idx}")

        out_path = os.path.join(
            cfg.output_dir,
            f"steering_L{layer_idx:02d}_{cfg.robot.replace('-', '_')}.json",
        )
        if os.path.exists(out_path):
            print(f"  SKIP: Output already exists: {out_path}")
            continue

        try:
            sae, act_mean, act_std = load_xvla_sae(cfg.sae_dir, layer_idx, cfg.device)
            print(f"  SAE: {sae.input_dim} -> {sae.hidden_dim}, k={sae.k}")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        concepts = load_concept_features(
            CONCEPT_ID_DIR, layer_idx, cfg.robot, top_n=cfg.top_n_features
        )
        if not concepts:
            print(f"  SKIP: No concept features found for layer {layer_idx}")
            continue
        print(f"  Concepts: {len(concepts)} loaded ({list(concepts.keys())[:5]}...)")

        t0 = time.time()

        steering_results = run_steering_experiment(
            policy, tasks, cfg.robot, domain_id, cfg.device, tokenizer,
            tokenizer_max_length, sae, act_mean, act_std,
            layer_idx, cfg.robot, concepts, cfg.n_episodes, cfg.max_steps,
            cfg.top_n_features, cfg.strengths,
            skip_baseline=cfg.skip_baseline,
        )

        with open(out_path, "w") as f:
            json.dump(steering_results, f, indent=2)
        print(f"\n  Saved: {out_path}")

        elapsed = time.time() - t0
        print(f"  Layer {layer_idx} done in {elapsed:.0f}s")

        del sae, act_mean, act_std
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nX-VLA SimplerEnv concept steering complete")
    print(f"Results in: {cfg.output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ConceptSteeringConfig)
    main(cfg)
