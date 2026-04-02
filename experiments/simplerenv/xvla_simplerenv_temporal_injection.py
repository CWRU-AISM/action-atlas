#!/usr/bin/env python3
"""
X-VLA SimplerEnv Temporal Injection Experiments

Tests temporal dynamics of X-VLA's action generation:
- Prompt switching mid-episode (correct prompt -> alternate prompt at timestep T)
- Temporal activation injection (inject activations during specific windows)

Usage:
    conda activate simpler_env
    python experiments/xvla_simplerenv_temporal_injection.py --model widowx --task widowx_spoon_on_towel
    python experiments/xvla_simplerenv_temporal_injection.py --model widowx --all-tasks
    python experiments/xvla_simplerenv_temporal_injection.py --model google-robot --all-tasks
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

warnings.filterwarnings("ignore")

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS,
    WIDOWX_GRIPPER_THRESHOLDS, DEFAULT_GRIPPER_THRESHOLD,
    load_xvla_policy, run_episode, force_free_memory,
    create_simplerenv_batch, convert_xvla_action_widowx,
    compute_initial_proprio, extract_obs_state, get_base_env, get_tcp_pose,
    rotate6d_to_euler_xyz,
    ActivationCaptureHook, TimedActivationInjectionHook,
    simpler_env,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from sapien.core import Pose


def run_episode_with_switch(policy, env, domain_id, device, tokenizer,
                             prompt_schedule, max_steps=1200, seed=0,
                             save_video=False, tokenizer_max_length=50,
                             task_name=None, episode_id=None, robot_type="widowx"):
    """
    Run episode with prompt that can change at specified timesteps.

    Args:
        prompt_schedule: dict mapping step -> prompt. Step 0 must be present.
                         Example: {0: "pick up spoon", 50: "stop", 100: "open gripper"}
    """
    is_google = (robot_type == "google-robot")
    base_env = get_base_env(env)

    if episode_id is not None:
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
    else:
        obs, _ = env.reset(seed=seed)

    instruction = prompt_schedule[0]

    if is_google:
        proprio = np.zeros(20, dtype=np.float32)
        tcp_pose = get_tcp_pose(obs, env=base_env)
        ee_wrt_base = Pose(
            p=obs["agent"]["base_pose"][:3], q=obs["agent"]["base_pose"][3:]
        ).inv() * Pose(
            p=tcp_pose[:3], q=tcp_pose[3:]
        )
        current_xyz = ee_wrt_base.p.astype(np.float32)
    else:
        proprio = compute_initial_proprio(obs, env=base_env)
        current_xyz = None

    gripper_threshold = WIDOWX_GRIPPER_THRESHOLDS.get(task_name, DEFAULT_GRIPPER_THRESHOLD)
    policy.reset()

    actions = []
    frames = []
    obs_states = []
    rewards = []
    step = 0
    success = False
    info = {}
    google_action_buffer = []

    while step < max_steps:
        if step in prompt_schedule:
            instruction = prompt_schedule[step]
            if not is_google:
                policy.reset()
            else:
                google_action_buffer = []

        obs_states.append(extract_obs_state(obs))
        image = get_image_from_maniskill2_obs_dict(base_env, obs)

        if save_video:
            frames.append(image.copy())

        if is_google:
            if not google_action_buffer:
                batch = create_simplerenv_batch(image, instruction, domain_id, device, tokenizer,
                                                tokenizer_max_length=tokenizer_max_length,
                                                proprio=proprio)
                policy.reset()
                raw_actions = []
                with torch.inference_mode():
                    for _ in range(20):
                        a = policy.select_action(batch)
                        if isinstance(a, torch.Tensor):
                            a = a.cpu().numpy()
                        if a.ndim == 2:
                            a = a[0]
                        raw_actions.append(a.copy())
                raw_actions = raw_actions[::2][:10]
                chunk_base_xyz = current_xyz.copy()
                for raw in raw_actions:
                    pos = raw[:3] + chunk_base_xyz
                    euler_xyz = rotate6d_to_euler_xyz(raw[3:9]).astype(np.float32)
                    gripper_val = 1.0 if raw[9] > 0.25 else -1.0
                    google_action_buffer.append(np.concatenate([pos, euler_xyz, [gripper_val]]))

            action_7d = google_action_buffer.pop(0)
            current_xyz = action_7d[:3].copy()
            proprio[:3] = action_7d[:3]
        else:
            batch = create_simplerenv_batch(image, instruction, domain_id, device, tokenizer,
                                            tokenizer_max_length=tokenizer_max_length,
                                            proprio=proprio)
            with torch.inference_mode():
                action = policy.select_action(batch)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim == 2:
                    action = action[0]
            proprio[:10] = action[:10]
            action_7d = convert_xvla_action_widowx(action, gripper_threshold=gripper_threshold)

        actions.append(action_7d)

        obs, reward, terminated, truncated, info = env.step(action_7d)
        rewards.append(float(reward))
        step += 1

        if terminated:
            success = True
            break
        if truncated:
            break

    obs_states.append(extract_obs_state(obs))

    result = {
        "success": bool(success),
        "steps": step,
        "actions": np.array(actions),
        "obs_states": obs_states,
        "rewards": rewards,
    }
    if save_video and frames:
        result["frames"] = frames
    return result


@dataclass
class TemporalInjectionConfig:
    """X-VLA SimplerEnv temporal injection experiments."""

    model: str
    """Model name: widowx, google-robot"""

    task: Optional[str] = None
    all_tasks: bool = False
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 42
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None


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
        output_dir = Path(f"outputs/xvla_simplerenv/temporal_{cfg.model}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"X-VLA SimplerEnv temporal injection")
    print(f"Model: {cfg.model}, Tasks: {tasks}")
    print(f"Output: {output_dir}")

    print("\nLoading X-VLA model...")
    policy, tokenizer = load_xvla_policy(cfg.model, checkpoint, device)
    transformer_blocks = policy.model.transformer.blocks
    n_blocks = len(transformer_blocks)

    switch_timesteps = [20, 50, 100, 200]
    alt_prompts = {
        "null": "",
        "stop": "stop",
        "open_gripper": "open gripper",
        "move_left": "move left",
    }

    injection_windows = {
        "first_quarter": (0.0, 0.25),
        "first_half": (0.0, 0.5),
        "second_half": (0.5, 1.0),
        "last_quarter": (0.75, 1.0),
        "middle": (0.25, 0.75),
    }

    injection_layers = [0, 12, 23]

    for task_name in tasks:
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            print(f"\n[SKIP] {task_name}")
            continue

        print(f"Task: {task_name}")

        env = simpler_env.make(task_name, max_episode_steps=cfg.max_steps)
        base_env = get_base_env(env)
        obs, _ = env.reset(seed=0)
        baseline_prompt = base_env.get_language_instruction()
        print(f"  Prompt: \"{baseline_prompt}\"")

        task_results = {"task": task_name, "prompt": baseline_prompt, "conditions": {}}

        print("\n  Running baseline (with activation capture)...")
        capture_hooks = {}
        handles = []
        for layer_idx in injection_layers:
            hook = ActivationCaptureHook()
            capture_hooks[layer_idx] = hook
            handles.append(transformer_blocks[layer_idx].register_forward_hook(hook))

        baseline_result = run_episode_with_switch(
            policy, env, domain_id, device, tokenizer,
            prompt_schedule={0: baseline_prompt},
            max_steps=cfg.max_steps, seed=cfg.seed,
            save_video=True,
            tokenizer_max_length=policy.config.tokenizer_max_length,
            task_name=task_name, episode_id=0, robot_type=cfg.model,
        )

        baseline_acts = {layer_idx: list(h.activations) for layer_idx, h in capture_hooks.items()}
        baseline_n_steps = baseline_result["steps"]

        act_dir = task_dir / "activations"
        act_dir.mkdir(exist_ok=True)
        for layer_idx, acts in baseline_acts.items():
            if acts:
                torch.save(torch.stack(acts), act_dir / f"baseline_L{layer_idx}.pt")

        for h in handles:
            h.remove()
        for h in capture_hooks.values():
            h.reset()
        del capture_hooks, handles
        torch.cuda.empty_cache()

        print(f"    Baseline: {baseline_n_steps} steps, success={baseline_result['success']}")

        bl_data = {
            "condition": "baseline", "prompt": baseline_prompt,
            "success": baseline_result["success"], "steps": baseline_n_steps,
            "actions": baseline_result["actions"].tolist(),
            "obs_states": baseline_result["obs_states"],
        }
        with open(task_dir / "baseline.json", "w") as f:
            json.dump(bl_data, f, indent=2)
        if baseline_result.get("frames"):
            try:
                import imageio
                imageio.mimsave(str(task_dir / "baseline.mp4"), baseline_result["frames"], fps=5)
            except Exception:
                pass

        task_results["conditions"]["baseline"] = {
            "success": baseline_result["success"], "steps": baseline_n_steps,
        }

        print("\n  Prompt switching experiments...")
        for switch_t in switch_timesteps:
            if switch_t >= baseline_n_steps:
                continue
            for alt_name, alt_text in alt_prompts.items():
                cond_name = f"switch_t{switch_t}_{alt_name}"
                cond_path = task_dir / f"{cond_name}.json"
                if cond_path.exists():
                    print(f"    [SKIP] {cond_name}")
                    with open(cond_path) as f:
                        task_results["conditions"][cond_name] = json.load(f)
                    continue

                schedule = {0: baseline_prompt, switch_t: alt_text}
                result = run_episode_with_switch(
                    policy, env, domain_id, device, tokenizer,
                    prompt_schedule=schedule,
                    max_steps=cfg.max_steps, seed=cfg.seed,
                    save_video=True,
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=0, robot_type=cfg.model,
                )

                cond_data = {
                    "condition": cond_name, "switch_timestep": switch_t,
                    "alt_prompt": alt_text, "success": result["success"],
                    "steps": result["steps"],
                    "actions": result["actions"].tolist(),
                    "obs_states": result["obs_states"],
                }
                with open(cond_path, "w") as f:
                    json.dump(cond_data, f, indent=2)

                if result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(task_dir / f"{cond_name}.mp4"), result["frames"], fps=5)
                    except Exception:
                        pass

                tag = "OK" if result["success"] else "FAIL"
                print(f"    {cond_name}: {result['steps']} steps, {tag}")
                task_results["conditions"][cond_name] = {
                    "success": result["success"], "steps": result["steps"],
                    "switch_timestep": switch_t,
                }

        print("\n  Temporal injection experiments...")
        for window_name, (frac_start, frac_end) in injection_windows.items():
            start_step = int(frac_start * baseline_n_steps)
            end_step = int(frac_end * baseline_n_steps)
            if end_step <= start_step:
                continue

            for layer_idx in injection_layers:
                cond_name = f"inject_{window_name}_L{layer_idx}"
                cond_path = task_dir / f"{cond_name}.json"
                if cond_path.exists():
                    print(f"    [SKIP] {cond_name}")
                    with open(cond_path) as f:
                        task_results["conditions"][cond_name] = json.load(f)
                    continue

                stored = baseline_acts.get(layer_idx, [])
                if not stored:
                    continue

                hook = TimedActivationInjectionHook(
                    stored, device=str(device),
                    start_step=start_step, end_step=end_step
                )
                handle = transformer_blocks[layer_idx].register_forward_hook(hook)

                result = run_episode_with_switch(
                    policy, env, domain_id, device, tokenizer,
                    prompt_schedule={0: baseline_prompt},
                    max_steps=cfg.max_steps, seed=cfg.seed,
                    save_video=True,
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=0, robot_type=cfg.model,
                )

                handle.remove()

                cond_data = {
                    "condition": cond_name, "layer": layer_idx,
                    "window": window_name, "start_step": start_step, "end_step": end_step,
                    "success": result["success"], "steps": result["steps"],
                    "injections": hook.injection_count,
                    "actions": result["actions"].tolist(),
                    "obs_states": result["obs_states"],
                }
                with open(cond_path, "w") as f:
                    json.dump(cond_data, f, indent=2)

                if result.get("frames"):
                    try:
                        import imageio
                        imageio.mimsave(str(task_dir / f"{cond_name}.mp4"), result["frames"], fps=5)
                    except Exception:
                        pass

                tag = "OK" if result["success"] else "FAIL"
                print(f"    {cond_name}: {result['steps']} steps, {tag} ({hook.injection_count} injections)")
                task_results["conditions"][cond_name] = {
                    "success": result["success"], "steps": result["steps"],
                    "injections": hook.injection_count,
                }

                del hook
                torch.cuda.empty_cache()

        env.close()
        del baseline_acts
        force_free_memory()

        with open(task_results_path, "w") as f:
            json.dump(task_results, f, indent=2)

    print("TEMPORAL INJECTION SUMMARY")
    for task_name in tasks:
        tp = output_dir / task_name / "results.json"
        if not tp.exists():
            continue
        with open(tp) as f:
            tr = json.load(f)
        print(f"\n  {task_name} (baseline: {'OK' if tr['conditions'].get('baseline', {}).get('success') else 'FAIL'}):")
        for cn, cd in tr["conditions"].items():
            if cn == "baseline":
                continue
            tag = "OK" if cd.get("success") else "FAIL"
            print(f"    {cn}: {tag} ({cd.get('steps', '?')} steps)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(TemporalInjectionConfig)
    main(cfg)
