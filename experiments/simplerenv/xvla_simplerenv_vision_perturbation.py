#!/usr/bin/env python3
"""
X-VLA SimplerEnv vision perturbation robustness test.

Applies image perturbations (brightness, blur, noise, etc.) and measures
success rate degradation across SimplerEnv tasks.

Examples:
    conda activate simpler_env

    python experiments/simplerenv/xvla_simplerenv_vision_perturbation.py --model widowx --all-tasks
    python experiments/simplerenv/xvla_simplerenv_vision_perturbation.py --model google-robot --all-tasks
"""

import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS,
    load_xvla_policy, run_episode,
    simpler_env,
)
from xvla_vision_perturbation import get_standard_perturbations


@dataclass
class VisionPerturbationConfig:
    """X-VLA SimplerEnv vision perturbation robustness test."""

    model: str
    """Model name: widowx, google-robot"""

    task: Optional[str] = None
    all_tasks: bool = False
    n_episodes: int = 3
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 42
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    n_action_steps: Optional[int] = None


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[cfg.model]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = config["domain_id"]

    if cfg.all_tasks:
        tasks = config["tasks"]
    elif cfg.task:
        matching = [t for t in config["tasks"] if cfg.task in t]
        tasks = matching if matching else [cfg.task]
    else:
        tasks = config["tasks"][:1]

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/xvla_simplerenv/vision_{cfg.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    perturbations = get_standard_perturbations()
    print(f"Vision perturbation: {cfg.model} | tasks={tasks} | "
          f"{len(perturbations)} perturbations | eps={cfg.n_episodes}")

    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)
    if cfg.n_action_steps is not None:
        policy.config.n_action_steps = cfg.n_action_steps

    all_results = {"model": cfg.model, "tasks": {}}

    for task_name in tasks:
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            print(f"\n[SKIP] {task_name}")
            with open(task_results_path) as f:
                all_results["tasks"][task_name] = json.load(f)
            continue

        print(f"\nTask: {task_name}")
        env = simpler_env.make(task_name, max_episode_steps=cfg.max_steps)
        task_results = {"task": task_name, "perturbations": {}}

        for pert in perturbations:
            pert_dir = task_dir / pert.name
            pert_dir.mkdir(exist_ok=True)

            pert_results_path = pert_dir / "results.json"
            if pert_results_path.exists():
                print(f"  [SKIP] {pert.name}")
                with open(pert_results_path) as f:
                    task_results["perturbations"][pert.name] = json.load(f)
                continue

            print(f"  {pert.name}...", end=" ", flush=True)
            pert_fn = lambda img, p=pert: p.perturbation_fn(img, **p.params)
            successes = []

            for ep in range(cfg.n_episodes):
                result = run_episode(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=cfg.max_steps, seed=cfg.seed + ep,
                    save_video=(ep == 0),
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=ep, robot_type=cfg.model,
                    image_transform_fn=pert_fn, collect_step_infos=False,
                )
                successes.append(result["success"])

                ep_data = {
                    "perturbation": pert.name, "episode": ep,
                    "success": result["success"], "steps": result["steps"],
                    "actions": result["actions"].tolist(),
                    "obs_states": result["obs_states"],
                    "rewards": result["rewards"],
                }
                with open(pert_dir / f"ep{ep}.json", "w") as f:
                    json.dump(ep_data, f, indent=2)

                if ep == 0 and result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(pert_dir / f"ep{ep}.mp4"), result["frames"], fps=5)
                    except Exception:
                        pass

            sr = sum(successes) / len(successes) if successes else 0
            print(f"{sr*100:.0f}% ({sum(successes)}/{len(successes)})")

            pert_data = {
                "perturbation": pert.name, "success_rate": sr,
                "success_count": sum(successes), "n_episodes": len(successes),
            }
            with open(pert_results_path, "w") as f:
                json.dump(pert_data, f, indent=2)
            task_results["perturbations"][pert.name] = pert_data

        env.close()
        with open(task_results_path, "w") as f:
            json.dump(task_results, f, indent=2)
        all_results["tasks"][task_name] = task_results

        gc.collect()
        torch.cuda.empty_cache()

    print("\nVision perturbation summary:")
    for pert in perturbations:
        rates = [all_results["tasks"].get(t, {}).get("perturbations", {}).get(pert.name, {}).get("success_rate", 0)
                 for t in tasks]
        if rates:
            print(f"  {pert.name:<25} {np.mean(rates)*100:6.1f}%")

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(VisionPerturbationConfig)
    main(cfg)
