#!/usr/bin/env python3
"""
X-VLA SimplerEnv counterfactual prompting.

Tests how X-VLA responds to prompt perturbations (null, negation, wrong
objects, motor commands, nonsense) and compares trajectories to baseline.

Examples:
    conda activate simpler_env

    python experiments/simplerenv/xvla_simplerenv_counterfactual.py --model widowx --all-tasks --n_episodes 3
    python experiments/simplerenv/xvla_simplerenv_counterfactual.py --model google-robot --all-tasks --n_episodes 3
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS, ActivationCollector,
    load_xvla_policy, run_episode, compare_trajectories,
    force_free_memory, log_ram, get_base_env,
    simpler_env,
)


def get_counterfactual_prompts(baseline_prompt):
    # Generate counterfactual prompts for a given task
    prompts = {
        'baseline': baseline_prompt,
        'null': "",
        'nonsense': "asdfghjkl qwerty",
        'numbers': "12345",
        'greeting': "hello world",
        'stop': "stop",
        'freeze': "freeze",
        'do_nothing': "do nothing",
        'negation': f"do not {baseline_prompt}",
        'opposite': f"don't {baseline_prompt}",
        'open_gripper': "open gripper",
        'close_gripper': "close gripper",
        'move_left': "move left",
        'move_right': "move right",
        'move_up': "move up",
        'move_down': "move down",
        'move_forward': "move forward",
        'move_backward': "move backward",
    }
    for obj in ['banana', 'phone', 'dinosaur', 'laptop', 'pizza', 'hammer']:
        prompts[f'wrong_{obj}'] = f"pick up the {obj}"
    return prompts


@dataclass
class CounterfactualConfig:
    # X-VLA SimplerEnv counterfactual prompting

    model: str
    # Model name: widowx, google-robot

    task: Optional[str] = None
    all_tasks: bool = False
    n_episodes: int = 5
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 42
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    domain_id: Optional[int] = None
    no_activations: bool = False


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[cfg.model]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = cfg.domain_id if cfg.domain_id is not None else config["domain_id"]
    start_time = datetime.now()

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
        output_dir = Path(f"outputs/xvla_simplerenv/counterfactual_{cfg.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_activations = not cfg.no_activations
    print(f"Counterfactual: {cfg.model} | tasks={tasks} | eps={cfg.n_episodes} | output={output_dir}")

    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)

    collector = None
    if save_activations:
        collector = ActivationCollector()
        collector.register_hooks(policy)

    all_results = {
        "model": cfg.model, "checkpoint": checkpoint,
        "domain_id": domain_id, "n_episodes": cfg.n_episodes, "tasks": {},
    }

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
        obs, _ = env.reset(seed=0)
        baseline_prompt = get_base_env(env).get_language_instruction()
        prompts = get_counterfactual_prompts(baseline_prompt)
        print(f"  Baseline: \"{baseline_prompt}\" | {len(prompts)} conditions")

        task_results = {
            "task": task_name, "baseline_prompt": baseline_prompt,
            "n_episodes": cfg.n_episodes, "conditions": {},
        }
        baseline_actions_ep0 = None

        for cond_name, prompt_text in prompts.items():
            cond_dir = task_dir / cond_name
            cond_dir.mkdir(exist_ok=True)

            cond_results_path = cond_dir / "results.json"
            if cond_results_path.exists():
                print(f"  [SKIP] {cond_name}")
                with open(cond_results_path) as f:
                    task_results["conditions"][cond_name] = json.load(f)
                continue

            disp = prompt_text[:50] + "..." if len(prompt_text) > 50 else prompt_text
            print(f"  {cond_name}: \"{disp}\"")

            successes = []
            cond_data = {"condition": cond_name, "prompt": prompt_text, "episodes": []}

            for ep in range(cfg.n_episodes):
                use_collector = collector if (ep == 0 and save_activations) else None

                result = run_episode(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=cfg.max_steps, seed=cfg.seed + ep,
                    collector=use_collector, save_video=(ep == 0),
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=ep, robot_type=cfg.model,
                    instruction_override=prompt_text,
                )

                successes.append(result["success"])
                status = "OK" if result["success"] else "FAIL"
                print(f"    Ep {ep+1}: {status} ({result['steps']} steps)")

                comp = {}
                if cond_name == "baseline" and ep == 0:
                    baseline_actions_ep0 = result["actions"]
                elif baseline_actions_ep0 is not None and ep == 0:
                    comp = compare_trajectories(baseline_actions_ep0, result["actions"])

                ep_data = {
                    "condition": cond_name, "prompt": prompt_text, "episode": ep,
                    "success": result["success"], "steps": result["steps"],
                    "actions": result["actions"].tolist(),
                    "rewards": result["rewards"],
                    "obs_states": result["obs_states"],
                    "step_infos": result.get("step_infos"),
                    "episode_stats": result.get("episode_stats"),
                }
                if comp:
                    ep_data["cosine_to_baseline"] = comp["cosine"]
                    ep_data["xyz_diff_to_baseline"] = comp["xyz_diff"]

                tcp_traj = [s["tcp_pose"] for s in result["obs_states"] if s.get("tcp_pose")]
                if len(tcp_traj) >= 2:
                    ep_data["eef_displacement"] = float(np.linalg.norm(
                        np.array(tcp_traj[-1][:3]) - np.array(tcp_traj[0][:3])))

                with open(cond_dir / f"ep{ep}.json", "w") as f:
                    json.dump(ep_data, f, indent=2)

                if ep == 0 and "frames" in result:
                    try:
                        import imageio
                        imageio.mimsave(str(cond_dir / f"ep{ep}.mp4"), result["frames"], fps=5)
                    except Exception:
                        pass

                if use_collector and "activations" in result:
                    act_dir = cond_dir / "activations"
                    act_dir.mkdir(exist_ok=True)
                    for layer_name, acts in result["activations"].items():
                        torch.save(acts, act_dir / f"{cond_name}_{layer_name}.pt")

            success_rate = sum(successes) / len(successes) if successes else 0.0
            cond_data["success_rate"] = success_rate
            cond_data["success_count"] = sum(successes)
            cond_data["episodes"] = [{"episode": i, "success": s} for i, s in enumerate(successes)]
            print(f"    Result: {success_rate*100:.1f}% ({sum(successes)}/{len(successes)})")

            with open(cond_results_path, "w") as f:
                json.dump(cond_data, f, indent=2)
            task_results["conditions"][cond_name] = cond_data

        env.close()
        with open(task_results_path, "w") as f:
            json.dump(task_results, f, indent=2)
        all_results["tasks"][task_name] = task_results
        force_free_memory()

    if collector:
        collector.remove_hooks()

    print("\nCounterfactual summary:")
    cond_summary = defaultdict(list)
    for task_res in all_results["tasks"].values():
        for cond_name, cond_data in task_res.get("conditions", {}).items():
            cond_summary[cond_name].append(cond_data.get("success_rate", 0))

    print(f"{'Condition':<25} | {'Success %':>10} | {'Tasks':>6}")
    for cond_name in sorted(cond_summary.keys()):
        rates = cond_summary[cond_name]
        print(f"{cond_name:<25} | {np.mean(rates)*100:>9.1f}% | {len(rates):>6}")

    duration = datetime.now() - start_time
    all_results["duration"] = str(duration)
    all_results["timestamp"] = datetime.now().isoformat()
    all_results["summary"] = {
        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for k, v in cond_summary.items()
    }
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_dir} | Duration: {duration}")


if __name__ == "__main__":
    cfg = tyro.cli(CounterfactualConfig)
    main(cfg)
