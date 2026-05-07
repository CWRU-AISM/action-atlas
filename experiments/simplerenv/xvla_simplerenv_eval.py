#!/usr/bin/env python3
"""
X-VLA SimplerEnv cross-embodiment evaluation.

Runs X-VLA on SimplerEnv tasks (WidowX domain_id=0, Google Robot domain_id=1)
with optional activation collection.

Examples:
    conda activate simpler_env

    python experiments/simplerenv/xvla_simplerenv_eval.py --model widowx --task widowx_stack_cube --n_episodes 1
    python experiments/simplerenv/xvla_simplerenv_eval.py --model widowx --all-tasks --n_episodes 20
    python experiments/simplerenv/xvla_simplerenv_eval.py --model google-robot --all-tasks --n_episodes 20
"""

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS, ActivationCollector,
    load_xvla_policy, run_episode,
    simpler_env,
)


@dataclass
class EvalConfig:
    # X-VLA SimplerEnv cross-embodiment evaluation

    model: str
    # Model name: widowx, google-robot

    task: Optional[str] = None
    # Specific task name (e.g. widowx_stack_cube)

    all_tasks: bool = False
    # Evaluate all tasks for the selected model

    n_episodes: int = 20
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 42
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    domain_id: Optional[int] = None
    no_activations: bool = False
    n_action_steps: Optional[int] = None


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[cfg.model]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = cfg.domain_id if cfg.domain_id is not None else config["domain_id"]

    if cfg.all_tasks:
        tasks = config["tasks"]
    elif cfg.task:
        matching = [t for t in config["tasks"] if cfg.task in t]
        if not matching:
            if cfg.task in simpler_env.ENVIRONMENTS:
                matching = [cfg.task]
            else:
                print(f"Task '{cfg.task}' not found. Available:")
                for t in config["tasks"]:
                    print(f"  {t}")
                return
        tasks = matching
    else:
        tasks = config["tasks"][:1]

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"rollouts/xvla_simplerenv_{cfg.model}_baseline_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_activations = not cfg.no_activations
    print(f"Eval: {cfg.model} | tasks={tasks} | eps={cfg.n_episodes} | "
          f"activations={save_activations} | output={output_dir}")

    print("Loading X-VLA model...")
    t0 = time.time()
    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)
    if cfg.n_action_steps is not None:
        policy.config.n_action_steps = cfg.n_action_steps
    n_params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(f"Model loaded in {time.time()-t0:.1f}s ({n_params:.1f}M params)")

    collector = None
    if save_activations:
        collector = ActivationCollector()
        collector.register_hooks(policy)

    all_results = {
        "model": cfg.model,
        "checkpoint": checkpoint,
        "domain_id": domain_id,
        "n_episodes": cfg.n_episodes,
        "max_steps": cfg.max_steps,
        "seed": cfg.seed,
        "tasks": {},
    }

    for task_name in tasks:
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            print(f"\n[SKIP] {task_name} -- already completed")
            with open(task_results_path) as f:
                all_results["tasks"][task_name] = json.load(f)
            continue

        print(f"\nTask: {task_name}")
        env = simpler_env.make(task_name, max_episode_steps=cfg.max_steps)

        task_results = {"task": task_name, "n_episodes": cfg.n_episodes, "episodes": []}
        successes = []

        for ep in range(cfg.n_episodes):
            use_collector = collector if (ep == 0 and save_activations) else None

            result = run_episode(
                policy, env, domain_id, device, tokenizer,
                max_steps=cfg.max_steps, seed=cfg.seed + ep,
                collector=use_collector, save_video=(ep == 0),
                tokenizer_max_length=policy.config.tokenizer_max_length,
                task_name=task_name, episode_id=ep, robot_type=cfg.model,
            )

            successes.append(result["success"])
            status = "SUCCESS" if result["success"] else "FAIL"
            print(f"  Ep {ep+1:2d}/{cfg.n_episodes}: {status} ({result['steps']} steps)")

            ep_data = {
                "episode": ep, "episode_id": ep,
                "success": result["success"], "steps": result["steps"],
                "instruction": result["instruction"],
                "actions": result["actions"].tolist(),
                "rewards": result["rewards"],
                "obs_states": result["obs_states"],
                "step_infos": result["step_infos"],
                "episode_stats": result["episode_stats"],
            }
            tcp_traj = [s["tcp_pose"] for s in result["obs_states"] if s.get("tcp_pose")]
            if len(tcp_traj) >= 2:
                init_pos = np.array(tcp_traj[0][:3])
                final_pos = np.array(tcp_traj[-1][:3])
                ep_data["eef_displacement"] = float(np.linalg.norm(final_pos - init_pos))
                ep_data["eef_trajectory_length"] = float(sum(
                    np.linalg.norm(np.array(tcp_traj[i+1][:3]) - np.array(tcp_traj[i][:3]))
                    for i in range(len(tcp_traj) - 1)
                ))
            with open(task_dir / f"ep{ep}.json", "w") as f:
                json.dump(ep_data, f, indent=2)

            if ep == 0 and "frames" in result:
                try:
                    import imageio
                    imageio.mimsave(str(task_dir / f"ep{ep}.mp4"), result["frames"], fps=5)
                except Exception:
                    pass

            if use_collector and "activations" in result:
                act_dir = task_dir / "activations"
                act_dir.mkdir(exist_ok=True)
                for layer_name, acts in result["activations"].items():
                    torch.save(acts, act_dir / f"baseline_{layer_name}.pt")

        env.close()

        success_rate = sum(successes) / len(successes) if successes else 0.0
        task_results["success_rate"] = success_rate
        task_results["success_count"] = sum(successes)
        task_results["episodes"] = [
            {"episode": i, "episode_id": i, "success": s}
            for i, s in enumerate(successes)
        ]
        print(f"  Result: {success_rate*100:.1f}% ({sum(successes)}/{len(successes)})")

        with open(task_results_path, "w") as f:
            json.dump(task_results, f, indent=2)
        all_results["tasks"][task_name] = task_results

        gc.collect()
        torch.cuda.empty_cache()

    if collector:
        collector.remove_hooks()

    print("\nSummary:")
    total_success = 0
    total_episodes = 0
    for task_name, result in all_results["tasks"].items():
        sr = result["success_rate"]
        sc = result["success_count"]
        ne = result["n_episodes"]
        print(f"  {task_name}: {sr*100:.1f}% ({sc}/{ne})")
        total_success += sc
        total_episodes += ne

    overall = total_success / total_episodes if total_episodes > 0 else 0
    print(f"\nOverall: {overall*100:.1f}% ({total_success}/{total_episodes})")

    all_results["overall_success_rate"] = overall
    all_results["overall_success_count"] = total_success
    all_results["overall_episodes"] = total_episodes
    all_results["timestamp"] = datetime.now().isoformat()

    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(EvalConfig)
    main(cfg)
