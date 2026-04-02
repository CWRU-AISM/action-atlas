#!/usr/bin/env python3
"""
X-VLA SimplerEnv Cross-Task Activation Injection

THE steering experiment: Run task A's environment but inject task B's
transformer activations. Does the robot follow the injection (task B's
behavior) or the actual environment (task A)?

Tests both directions (A->B and B->A) for each task pair.

Usage:
    conda activate simpler_env
    python experiments/xvla_simplerenv_cross_task_injection.py --model widowx --task_pairs 0,1 0,2
    python experiments/xvla_simplerenv_cross_task_injection.py --model widowx --all-pairs
    python experiments/xvla_simplerenv_cross_task_injection.py --model google-robot --all-pairs
"""

import json
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import tyro

warnings.filterwarnings("ignore")

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS,
    load_xvla_policy, run_episode, compare_trajectories,
    force_free_memory, log_ram,
    ActivationCaptureHook, ActivationInjectionHook,
    simpler_env, get_base_env,
)


@dataclass
class CrossTaskInjectionConfig:
    """X-VLA SimplerEnv cross-task activation injection."""

    model: str
    """Model name: widowx, google-robot"""

    task_pairs: Optional[List[str]] = None
    """Task pairs as 'A,B' (e.g., 0,1 0,2 1,3)"""

    all_pairs: bool = False
    seed: int = 42
    max_steps: int = DEFAULT_MAX_STEPS
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[cfg.model]
    checkpoint = cfg.checkpoint or config["checkpoint"]
    domain_id = config["domain_id"]
    task_names = config["tasks"]

    if cfg.task_pairs:
        task_pairs = [tuple(int(x) for x in p.split(",")) for p in cfg.task_pairs]
    elif cfg.all_pairs:
        task_pairs = list(combinations(range(len(task_names)), 2))
    else:
        n = len(task_names)
        task_pairs = [(0, 1)]
        if n > 2:
            task_pairs.append((0, 2))
        if n > 3:
            task_pairs.append((1, 3))

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = Path(f"outputs/xvla_simplerenv/cross_task_{cfg.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"X-VLA SimplerEnv cross-task injection")
    print(f"Model: {cfg.model}, Pairs: {task_pairs}")
    print(f"Tasks: {task_names}")
    print(f"Output: {output_dir}")

    print("\nLoading X-VLA model...")
    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)
    transformer_blocks = policy.model.transformer.blocks
    n_blocks = len(transformer_blocks)

    task_prompts = {}
    for i, task_name in enumerate(task_names):
        env = simpler_env.make(task_name, max_episode_steps=cfg.max_steps)
        obs, _ = env.reset(seed=0)
        task_prompts[i] = get_base_env(env).get_language_instruction()
        env.close()
    print("\nTask prompts:")
    for i, prompt in task_prompts.items():
        print(f"  {i}: {task_names[i]} -> \"{prompt}\"")

    for task_a, task_b in task_pairs:
        pair_key = f"pair_{task_a}_{task_b}"
        pair_dir = output_dir / pair_key
        pair_dir.mkdir(exist_ok=True)

        pair_json = pair_dir / "pair_result.json"
        if pair_json.exists():
            print(f"\n  [SKIP] {pair_key}")
            continue

        print(f"PAIR: {task_names[task_a]} <-> {task_names[task_b]}")

        pair_results = {
            "task_a": task_a, "task_b": task_b,
            "name_a": task_names[task_a], "name_b": task_names[task_b],
            "prompt_a": task_prompts[task_a], "prompt_b": task_prompts[task_b],
        }

        for src_tid, dst_tid in [(task_a, task_b), (task_b, task_a)]:
            direction = f"inject_{src_tid}_into_{dst_tid}"
            dir_dir = pair_dir / direction
            dir_dir.mkdir(exist_ok=True)

            src_name = task_names[src_tid]
            dst_name = task_names[dst_tid]
            src_prompt = task_prompts[src_tid]
            dst_prompt = task_prompts[dst_tid]

            print(f"\n  {direction}")
            print(f"    SRC: {src_name} ({src_prompt})")
            print(f"    DST: {dst_name} ({dst_prompt})")

            print(f"\n    Capturing source baseline...")
            src_env = simpler_env.make(src_name, max_episode_steps=cfg.max_steps)

            transformer_hooks = {}
            handles = []
            for i in range(n_blocks):
                hook = ActivationCaptureHook()
                transformer_hooks[i] = hook
                handles.append(transformer_blocks[i].register_forward_hook(hook))

            src_result = run_episode(
                policy, src_env, domain_id, device, tokenizer,
                instruction_override=src_prompt, max_steps=cfg.max_steps, seed=cfg.seed,
                tokenizer_max_length=policy.config.tokenizer_max_length,
                task_name=src_name, episode_id=0, robot_type=cfg.model,
                collect_step_infos=False,
            )

            src_acts = {i: list(h.activations) for i, h in transformer_hooks.items()}

            act_dir = dir_dir / "activations"
            act_dir.mkdir(parents=True, exist_ok=True)
            for layer_idx, acts in src_acts.items():
                if acts:
                    torch.save(torch.stack(acts), act_dir / f"src_layer{layer_idx}.pt")

            for h in handles:
                h.remove()
            for h in transformer_hooks.values():
                h.reset()
            del transformer_hooks, handles
            torch.cuda.empty_cache()

            print(f"      Source: {src_result['steps']} steps, success={src_result['success']}")
            src_env.close()

            print(f"    Capturing destination baseline...")
            dst_env = simpler_env.make(dst_name, max_episode_steps=cfg.max_steps)

            dst_result = run_episode(
                policy, dst_env, domain_id, device, tokenizer,
                instruction_override=dst_prompt, max_steps=cfg.max_steps, seed=cfg.seed,
                tokenizer_max_length=policy.config.tokenizer_max_length,
                task_name=dst_name, episode_id=0, robot_type=cfg.model,
                collect_step_infos=False,
            )
            print(f"      Dest: {dst_result['steps']} steps, success={dst_result['success']}")

            conditions = [
                ("own_prompt_no_inject", dst_prompt, None),
                ("cross_prompt_no_inject", src_prompt, None),
                ("cross_prompt_transformer_ALL", src_prompt, list(range(n_blocks))),
                ("cross_prompt_transformer_L0", src_prompt, [0]),
                ("cross_prompt_transformer_L12", src_prompt, [12]),
                ("cross_prompt_transformer_L23", src_prompt, [23]),
                ("cross_prompt_transformer_L12_L23", src_prompt, [12, 23]),
                ("own_prompt_transformer_ALL", dst_prompt, list(range(n_blocks))),
            ]

            direction_results = {}
            for cond_label, prompt_text, inject_layers in conditions:
                inj_hooks = []
                inj_handles = []

                if inject_layers:
                    for layer_idx in inject_layers:
                        stored = src_acts.get(layer_idx, [])
                        if stored:
                            hook = ActivationInjectionHook(stored, device=str(device))
                            handle = transformer_blocks[layer_idx].register_forward_hook(hook)
                            inj_hooks.append(hook)
                            inj_handles.append(handle)

                result = run_episode(
                    policy, dst_env, domain_id, device, tokenizer,
                    instruction_override=prompt_text, max_steps=cfg.max_steps, seed=cfg.seed,
                    save_video=True,
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=dst_name, episode_id=0, robot_type=cfg.model,
                    collect_step_infos=False,
                )

                comp_src = compare_trajectories(src_result["actions"], result["actions"])
                comp_dst = compare_trajectories(dst_result["actions"], result["actions"])

                total_inj = sum(h.injection_count for h in inj_hooks) if inj_hooks else 0
                total_mm = sum(h.shape_mismatches for h in inj_hooks) if inj_hooks else 0

                print(f"    {cond_label}: {result['steps']} steps, "
                      f"cos_src={comp_src['cosine']:.4f}, cos_dst={comp_dst['cosine']:.4f}, "
                      f"success={result['success']}")

                if result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(dir_dir / f"{cond_label}.mp4"), result["frames"], fps=5)
                    except Exception:
                        pass

                cond_data = {
                    "condition": cond_label, "prompt": prompt_text,
                    "success": result["success"], "steps": result["steps"],
                    "cos_to_src": comp_src["cosine"], "cos_to_dst": comp_dst["cosine"],
                    "xyz_to_src": comp_src["xyz_diff"], "xyz_to_dst": comp_dst["xyz_diff"],
                    "actions": result["actions"].tolist(),
                    "obs_states": result["obs_states"],
                    "inject_layers": inject_layers,
                    "injections": total_inj, "shape_mismatches": total_mm,
                }
                with open(dir_dir / f"{cond_label}.json", "w") as f:
                    json.dump(cond_data, f, indent=2)

                direction_results[cond_label] = {
                    "success": result["success"], "steps": result["steps"],
                    "cos_to_src": comp_src["cosine"], "cos_to_dst": comp_dst["cosine"],
                    "injections": total_inj,
                }

                for h in inj_handles:
                    h.remove()
                del inj_hooks
                torch.cuda.empty_cache()

            pair_results[direction] = direction_results
            dst_env.close()

            del src_acts, src_result, dst_result
            force_free_memory()
            log_ram(f"after {direction}")

        with open(pair_json, "w") as f:
            json.dump(pair_results, f, indent=2)
        print(f"    [SAVED] {pair_json}")

    print("CROSS-TASK INJECTION SUMMARY")
    for task_a, task_b in task_pairs:
        pair_json = output_dir / f"pair_{task_a}_{task_b}" / "pair_result.json"
        if not pair_json.exists():
            continue
        with open(pair_json) as f:
            pr = json.load(f)
        print(f"\n  {task_names[task_a]} <-> {task_names[task_b]}:")
        for dk in [f"inject_{task_a}_into_{task_b}", f"inject_{task_b}_into_{task_a}"]:
            dr = pr.get(dk, {})
            if not dr:
                continue
            print(f"    {dk}:")
            for cl, d in dr.items():
                print(f"      {cl:<35} cos_src={d['cos_to_src']:.4f} cos_dst={d['cos_to_dst']:.4f} suc={'Y' if d['success'] else 'N'}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(CrossTaskInjectionConfig)
    main(cfg)
