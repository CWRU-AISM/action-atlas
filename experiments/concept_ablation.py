#!/usr/bin/env python3
"""
SAE-based concept ablation for any supported VLA model.

Removes (zeros out) concept-selective SAE features identified by concept_id.py,
then measures the effect on task success rate.

Examples:
    python experiments/concept_ablation.py --model xvla --suite libero_object \\
        --sae-dir outputs/xvla_saes/libero_object \\
        --concept-results results/concept_id/libero_object/all_layers.json \\
        --layer transformer_L12

    python experiments/concept_ablation.py --model groot --suite libero_goal \\
        --sae-dir outputs/groot_saes/libero_goal \\
        --concept-results results/concept_id/libero_goal/all_layers.json \\
        --layer eagle_L08 --n-features 10
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.model_adapters import get_adapter
from experiments.sae_hooks import PerTokenAblationHook, TopKSAE, load_sae
from experiments.utils import (
    force_free_memory, save_results, load_results, save_video, SUITE_MAX_STEPS,
)


@dataclass
class ConceptAblationConfig:
    """SAE-based concept ablation experiment."""

    model: str = "xvla"
    suite: str = "libero_object"
    checkpoint: Optional[str] = None

    sae_dir: str = ""
    """Directory with trained SAE checkpoints."""

    concept_results: str = ""
    """Path to concept_id all_layers.json with identified features."""

    layer: str = ""
    """Layer to ablate (e.g. 'transformer_L12')."""

    n_features: int = 5
    """Number of top concept features to ablate."""

    n_episodes: int = 3
    tasks: Optional[List[int]] = None
    max_steps: Optional[int] = None
    seed: int = 42
    output_dir: Optional[str] = None
    record_video: bool = True

    gpu: int = 0

    n_action_steps: Optional[int] = None
    """Override action chunk size for faster inference."""
    """GPU device index."""


def main(cfg):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cfg.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_steps = cfg.max_steps or SUITE_MAX_STEPS.get(cfg.suite, 300)

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/{cfg.model}_experiments/concept_ablation_{cfg.suite}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load concept identification results
    with open(cfg.concept_results) as f:
        all_concept_results = json.load(f)

    if cfg.layer not in all_concept_results:
        available = list(all_concept_results.keys())
        raise ValueError(f"Layer '{cfg.layer}' not in concept results. Available: {available}")

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
    print(f"Concept Ablation: {cfg.model} on {cfg.suite}, layer={cfg.layer}")
    adapter.load_model(ckpt, device)

    if cfg.n_action_steps and hasattr(adapter, "policy") and hasattr(adapter.policy, "config"):
        adapter.policy.config.n_action_steps = cfg.n_action_steps

    # Find the target layer module
    all_layers = adapter.get_all_layers()
    layer_module = None
    for label, module in all_layers:
        if label == cfg.layer:
            layer_module = module
            break
    if layer_module is None:
        raise ValueError(f"Layer '{cfg.layer}' not found in model")

    task_suite, all_tasks = adapter.setup_suite(cfg.suite)
    task_ids = cfg.tasks if cfg.tasks else list(range(len(all_tasks)))

    # Build ablation conditions: baseline + one per concept
    conditions = [("baseline", [])]
    for concept_type, concepts in layer_concepts.items():
        for concept_name, concept_data in concepts.items():
            features = concept_data.get("top_features", [])
            feat_ids = [f["feature_idx"] for f in features[:cfg.n_features]]
            if feat_ids:
                conditions.append((f"{concept_type}/{concept_name}", feat_ids))

    print(f"  conditions: {len(conditions)} (baseline + {len(conditions)-1} concepts)")
    print(f"  features per concept: {cfg.n_features}")

    # Register hook
    hook = PerTokenAblationHook(sae, act_mean, act_std, device=device)
    handle = layer_module.register_forward_hook(hook)

    results = {}
    start_time = time.time()

    for cond_name, feat_ids in conditions:
        print(f"\nCondition: {cond_name} (features: {feat_ids[:5]}{'...' if len(feat_ids) > 5 else ''})")
        cond_dir = output_dir / cond_name.replace("/", "_")
        cond_dir.mkdir(parents=True, exist_ok=True)

        cond_results = {}
        for tid in task_ids:
            _, task_obj, task_desc = all_tasks[tid]
            env, desc, meta = adapter.create_env(tid, suite=cfg.suite, max_steps=max_steps)
            successes = 0

            for ep in range(cfg.n_episodes):
                hook.clear()
                if feat_ids:
                    hook.set_ablation(feat_ids)
                    hook.enabled = True
                else:
                    hook.enabled = False
                hook.reset()

                ep_result = adapter.run_episode(
                    env, desc, max_steps=max_steps,
                    save_video=(cfg.record_video and ep == 0),
                    seed=cfg.seed + ep,
                )
                if ep_result["success"]:
                    successes += 1
                if ep_result.get("frames"):
                    save_video(ep_result["frames"],
                               cond_dir / f"task{tid}_ep{ep}.mp4")

            if hasattr(env, "close"):
                env.close()

            rate = successes / cfg.n_episodes
            print(f"  Task {tid}: {rate:.0%} ({desc})")
            cond_results[str(tid)] = {
                "success_rate": rate, "successes": successes,
                "n_episodes": cfg.n_episodes,
            }

        avg = np.mean([v["success_rate"] for v in cond_results.values()])
        cond_results["_avg"] = float(avg)
        results[cond_name] = cond_results
        save_results(cond_results, cond_dir / "results.json")
        force_free_memory()

    handle.remove()
    elapsed = time.time() - start_time

    # Summary
    baseline_avg = results.get("baseline", {}).get("_avg", 0)
    print(f"\nConcept Ablation Results ({cfg.model} / {cfg.suite} / {cfg.layer})")
    print(f"{'Concept':<30} {'Avg SR':>8} {'Delta':>8}")
    print("-" * 48)
    for cond_name, _ in conditions:
        avg = results[cond_name].get("_avg", 0)
        delta = avg - baseline_avg
        print(f"  {cond_name:<28} {avg:>7.0%} {delta:>+7.0%}")

    save_results({
        "model": cfg.model, "suite": cfg.suite, "layer": cfg.layer,
        "n_features": cfg.n_features, "results": results,
        "duration_seconds": elapsed,
    }, output_dir / "summary.json")
    print(f"\nDone in {elapsed/60:.1f} min. Results: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(ConceptAblationConfig)
    main(cfg)
