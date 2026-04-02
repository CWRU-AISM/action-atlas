#!/usr/bin/env python3
"""
X-VLA SimplerEnv SAE Reconstruction Fidelity Evaluation

Hooks a trained SAE at each transformer layer during live SimplerEnv rollouts
and measures whether encode->decode preserves task behavior.

Two reconstruction modes:
  - pertoken: Process each token position individually through SAE (for per-token SAEs)
  - meanpool: Mean-pool activations, reconstruct mean, add delta to all tokens
              (for mean-pooled SAEs that were trained on mean-pooled data)

Fidelity = recon_success_rate / baseline_success_rate

Usage:
    conda activate simpler_env

    # WidowX mean-pooled, all layers, 5 episodes
    python experiments/xvla_simplerenv_reconstruction_eval.py \
        --model widowx --all-layers --pooling meanpool \
        --sae-dir outputs/xvla_saes/simplerenv_all_meanpool \
        --n-episodes 5

    # Google Robot per-token, specific layers
    python experiments/xvla_simplerenv_reconstruction_eval.py \
        --model google-robot --layers 0,12,23 --pooling pertoken \
        --sae-dir outputs/xvla_saes/simplerenv_all_pertoken \
        --n-episodes 3

    # Both models, all layers
    python experiments/xvla_simplerenv_reconstruction_eval.py \
        --model all --all-layers --pooling meanpool \
        --sae-dir outputs/xvla_saes/simplerenv_all_meanpool
"""

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import tyro

from common import (
    MODEL_CONFIGS, DEFAULT_MAX_STEPS,
    WIDOWX_GRIPPER_THRESHOLDS, DEFAULT_GRIPPER_THRESHOLD,
    load_xvla_policy, load_xvla_sae,
    create_simplerenv_batch, convert_xvla_action_widowx,
    compute_initial_proprio, get_base_env, get_tcp_pose,
    rotate6d_to_euler_xyz,
    simpler_env,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from sapien.core import Pose

N_XVLA_LAYERS = 24


class PerTokenReconstructionHook:
    """
    Reconstruct each token position individually through SAE.
    Replaces original activation with SAE reconstruction.
    """

    def __init__(self, sae, act_mean, act_std, device='cuda'):
        self.sae = sae
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device
        self.enabled = True
        self.current_step = 0
        self._verified = False
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0

    def reset(self):
        self.current_step = 0

    def reset_stats(self):
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0
        self._verified = False

    def get_avg_recon_error_ratio(self):
        if self.n_calls == 0:
            return 0.0
        return (self.total_recon_error / self.n_calls) / max(self.total_act_norm / self.n_calls, 1e-8)

    def __call__(self, module, input, output):
        if not self.enabled:
            self.current_step += 1
            return output

        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        extra = output[1:] if is_tuple else None

        original_dtype = act.dtype
        act = act.float()
        original_shape = act.shape

        if len(original_shape) == 3:
            batch, seq, dim = original_shape
            act_flat = act.view(-1, dim)
        elif len(original_shape) == 2:
            act_flat = act
            batch, seq, dim = act.shape[0], 1, act.shape[-1]
        else:
            self.current_step += 1
            return output

        act_norm = (act_flat - self.act_mean) / (self.act_std + 1e-8)
        z = self.sae.encode(act_norm)
        reconstructed_norm = self.sae.decode(z)
        reconstructed = reconstructed_norm * (self.act_std + 1e-8) + self.act_mean

        with torch.no_grad():
            recon_error = (reconstructed - act_flat).norm().item()
            act_norm_val = act_flat.norm().item()
            self.total_recon_error += recon_error
            self.total_act_norm += act_norm_val
            self.n_calls += 1

        if not self._verified:
            ratio = recon_error / max(act_norm_val, 1e-8)
            n_active = (z.detach().abs() > 0).sum(dim=-1).float().mean().item()
            print(f"    [RECON] error={recon_error:.4f}, norm={act_norm_val:.4f}, "
                  f"ratio={ratio:.4f}, active={n_active:.0f}")
            self._verified = True

        if len(original_shape) == 3:
            reconstructed = reconstructed.view(batch, seq, dim)

        reconstructed = reconstructed.to(original_dtype)
        self.current_step += 1
        return (reconstructed,) + extra if is_tuple else reconstructed


class MeanPooledReconstructionHook:
    """
    Mean-pool activations, reconstruct via SAE, add delta to all tokens.

    This is the correct approach for mean-pooled SAEs:
    1. Compute mean over seq_len: mean_act [batch, dim]
    2. Normalize, encode, decode through SAE
    3. Compute delta = reconstructed_mean - original_mean
    4. Add delta (broadcast) to all tokens

    This preserves per-token variation while reconstructing the mean component.
    """

    def __init__(self, sae, act_mean, act_std, device='cuda'):
        self.sae = sae
        self.act_mean = act_mean.to(device)
        self.act_std = act_std.to(device)
        self.device = device
        self.enabled = True
        self.current_step = 0
        self._verified = False
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0

    def reset(self):
        self.current_step = 0

    def reset_stats(self):
        self.total_recon_error = 0.0
        self.total_act_norm = 0.0
        self.n_calls = 0
        self._verified = False

    def get_avg_recon_error_ratio(self):
        if self.n_calls == 0:
            return 0.0
        return (self.total_recon_error / self.n_calls) / max(self.total_act_norm / self.n_calls, 1e-8)

    def __call__(self, module, input, output):
        if not self.enabled:
            self.current_step += 1
            return output

        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        extra = output[1:] if is_tuple else None

        original_dtype = act.dtype
        act = act.float()
        original_shape = act.shape

        if len(original_shape) == 3:
            batch, seq, dim = original_shape
        elif len(original_shape) == 2:
            batch, dim = original_shape
            seq = 1
            act = act.unsqueeze(1)
        else:
            self.current_step += 1
            return output

        mean_act = act.mean(dim=1)

        mean_norm = (mean_act - self.act_mean) / (self.act_std + 1e-8)
        z = self.sae.encode(mean_norm)
        reconstructed_norm = self.sae.decode(z)
        reconstructed_mean = reconstructed_norm * (self.act_std + 1e-8) + self.act_mean

        delta = reconstructed_mean - mean_act
        modified = act + delta.unsqueeze(1)

        with torch.no_grad():
            recon_error = delta.norm().item()
            act_norm_val = mean_act.norm().item()
            self.total_recon_error += recon_error
            self.total_act_norm += act_norm_val
            self.n_calls += 1

        if not self._verified:
            ratio = recon_error / max(act_norm_val, 1e-8)
            n_active = (z.detach().abs() > 0).sum(dim=-1).float().mean().item()
            print(f"    [RECON meanpool] delta={recon_error:.4f}, mean_norm={act_norm_val:.4f}, "
                  f"ratio={ratio:.4f}, active={n_active:.0f}")
            self._verified = True

        if len(original_shape) == 2:
            modified = modified.squeeze(1)

        modified = modified.to(original_dtype)
        self.current_step += 1
        return (modified,) + extra if is_tuple else modified


SIMPLERENV_BASELINES = {
    "widowx_spoon_on_towel": 1.0,
    "widowx_carrot_on_plate": 0.85,
    "widowx_stack_cube": 0.85,
    "widowx_put_eggplant_in_basket": 0.90,
    "google_robot_pick_coke_can": 0.95,
    "google_robot_move_near": 1.0,
    "google_robot_open_top_drawer": 0.75,
    "google_robot_close_top_drawer": 0.70,
    "google_robot_open_middle_drawer": 0.80,
    "google_robot_close_middle_drawer": 0.95,
}


def run_episode_with_hook(policy, env, domain_id, device, tokenizer,
                          max_steps=1200, seed=0,
                          tokenizer_max_length=50, task_name=None,
                          episode_id=None, robot_type="widowx",
                          log_dir=None):
    """
    Run a single SimplerEnv episode. Hook is already registered on model.

    If log_dir is provided, logs video frames, trajectory (actions + ee poses),
    and object displacements to log_dir.
    """
    is_google = (robot_type == "google-robot")
    base_env = get_base_env(env)

    do_log = log_dir is not None
    log_frames = []
    log_actions = []
    log_ee_poses = []
    log_obj_source = []
    log_obj_target = []

    if episode_id is not None:
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
    else:
        obs, _ = env.reset(seed=seed)
    instruction = base_env.get_language_instruction()

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

    step = 0
    success = False
    info = {}
    google_action_buffer = []

    while step < max_steps:
        image = get_image_from_maniskill2_obs_dict(base_env, obs)

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

        obs, reward, terminated, truncated, info = env.step(action_7d)
        step += 1

        if do_log:
            log_actions.append(action_7d.tolist())
            if 'extra' in obs and 'tcp_pose' in obs['extra']:
                log_ee_poses.append(obs['extra']['tcp_pose'].tolist())
            if hasattr(base_env, 'source_obj_pose'):
                sp = base_env.source_obj_pose
                log_obj_source.append(sp.p.tolist() if hasattr(sp, 'p') else [0, 0, 0])
            if hasattr(base_env, 'target_obj_pose'):
                tp = base_env.target_obj_pose
                log_obj_target.append(tp.p.tolist() if hasattr(tp, 'p') else [0, 0, 0])
            if step % 5 == 0 or step <= 3:
                img = get_image_from_maniskill2_obs_dict(base_env, obs)
                log_frames.append(img)

        new_instruction = base_env.get_language_instruction()
        if new_instruction != instruction:
            instruction = new_instruction

        if terminated:
            success = True
            break
        if truncated:
            break

    if do_log and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_frames:
            video_path = log_dir / "video.mp4"
            imageio.mimsave(str(video_path), log_frames, fps=10)

        traj_data = {
            "success": bool(success),
            "steps": step,
            "actions": log_actions,
            "ee_poses": log_ee_poses,
            "source_obj_positions": log_obj_source,
            "target_obj_positions": log_obj_target,
        }
        if log_obj_source and len(log_obj_source) > 1:
            s0 = np.array(log_obj_source[0])
            sf = np.array(log_obj_source[-1])
            traj_data["source_obj_displacement"] = np.linalg.norm(sf - s0).item()
        if log_obj_target and len(log_obj_target) > 1:
            t0 = np.array(log_obj_target[0])
            tf = np.array(log_obj_target[-1])
            traj_data["target_obj_displacement"] = np.linalg.norm(tf - t0).item()

        with open(log_dir / "trajectory.json", "w") as f:
            json.dump(traj_data, f, indent=2)

    return {"success": bool(success), "steps": step}


def evaluate_model(model_type, layers, sae_dir, pooling, n_episodes,
                   max_steps, device, output_dir, tasks_override=None):
    """Evaluate reconstruction fidelity for one model type."""
    config = MODEL_CONFIGS[model_type]
    checkpoint = config["checkpoint"]
    domain_id = config["domain_id"]
    tasks = tasks_override or config["tasks"]

    print(f"MODEL: {model_type} ({checkpoint})")
    print(f"Tasks: {tasks}")
    print(f"Layers: {layers}")
    print(f"Pooling: {pooling}")
    print(f"Episodes/task: {n_episodes}")
    print(f"SAE dir: {sae_dir}")

    print("\nLoading X-VLA policy...")
    t0 = time.time()
    policy, tokenizer = load_xvla_policy(model_type, checkpoint, device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    results_path = output_dir / f"{model_type}_reconstruction_results.json"
    if results_path.exists():
        print(f"  Loading existing results for resume...")
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            "metadata": {
                "model": model_type,
                "checkpoint": checkpoint,
                "sae_dir": sae_dir,
                "pooling": pooling,
                "n_episodes": n_episodes,
                "max_steps": max_steps,
                "timestamp": datetime.now().isoformat(),
            },
            "per_layer": {},
        }

    for layer_idx in layers:
        layer_key = str(layer_idx)

        if layer_key in all_results.get("per_layer", {}):
            existing = all_results["per_layer"][layer_key]
            fid = existing.get("overall_fidelity", "?")
            print(f"\n  Layer {layer_idx}: SKIP (already done, fidelity={fid})")
            continue

        sae_path = Path(sae_dir) / f"layer_{layer_idx:02d}" / "sae_best.pt"
        if not sae_path.exists():
            print(f"\n  Layer {layer_idx}: SKIP (no SAE at {sae_path})")
            continue

        print(f"  Layer {layer_idx}")
        t_layer = time.time()

        sae, act_mean, act_std = load_xvla_sae(sae_dir, layer_idx, device)
        print(f"  SAE: {sae.input_dim} -> {sae.hidden_dim}, k={sae.k}")

        if pooling == "meanpool":
            hook = MeanPooledReconstructionHook(sae, act_mean, act_std, device)
        else:
            hook = PerTokenReconstructionHook(sae, act_mean, act_std, device)

        target_module = policy.model.transformer.blocks[layer_idx]
        handle = target_module.register_forward_hook(hook)

        layer_results = {"per_task": {}}

        for task_name in tasks:
            baseline_rate = SIMPLERENV_BASELINES.get(task_name, 0.0)
            if baseline_rate == 0.0:
                print(f"    {task_name}: SKIP (no baseline)")
                continue

            print(f"    {task_name} (baseline={baseline_rate*100:.0f}%)")

            env = simpler_env.make(task_name, max_episode_steps=max_steps)
            hook.reset_stats()

            successes = []
            for ep in range(n_episodes):
                hook.reset()
                ep_log_dir = output_dir / "logs" / f"layer_{layer_idx:02d}" / task_name / f"ep_{ep:02d}"
                result = run_episode_with_hook(
                    policy, env, domain_id, device, tokenizer,
                    max_steps=max_steps, seed=42 + ep,
                    tokenizer_max_length=policy.config.tokenizer_max_length,
                    task_name=task_name, episode_id=ep,
                    robot_type=model_type,
                    log_dir=ep_log_dir,
                )
                successes.append(result["success"])
                status = "OK" if result["success"] else "FAIL"
                print(f"      Ep {ep+1}: {status} ({result['steps']} steps)")

            env.close()

            recon_rate = sum(successes) / len(successes)
            fidelity = recon_rate / max(baseline_rate, 1e-8)
            fidelity = min(fidelity, 1.0) if baseline_rate > 0 else (1.0 if recon_rate == 0 else 0.0)
            avg_err_ratio = hook.get_avg_recon_error_ratio()

            delta = recon_rate - baseline_rate
            sign = "+" if delta > 0 else ""
            print(f"      -> {recon_rate*100:.0f}% (fidelity={fidelity:.2f}, "
                  f"delta={sign}{delta*100:.0f}pp, err_ratio={avg_err_ratio:.4f})")

            layer_results["per_task"][task_name] = {
                "recon_success": recon_rate,
                "baseline_success": baseline_rate,
                "fidelity": round(fidelity, 4),
                "recon_successes": [bool(s) for s in successes],
                "avg_recon_error_ratio": round(avg_err_ratio, 6),
            }

        handle.remove()
        del sae, act_mean, act_std, hook
        torch.cuda.empty_cache()
        gc.collect()

        task_results = layer_results["per_task"]
        if task_results:
            total_recon = sum(r["recon_success"] for r in task_results.values())
            total_base = sum(r["baseline_success"] for r in task_results.values())
            n_tasks = len(task_results)
            overall_recon = total_recon / n_tasks
            overall_base = total_base / n_tasks
            overall_fidelity = overall_recon / max(overall_base, 1e-8)
            overall_fidelity = min(overall_fidelity, 1.0) if overall_base > 0 else 1.0

            layer_results["overall_recon_rate"] = round(overall_recon, 4)
            layer_results["overall_baseline_rate"] = round(overall_base, 4)
            layer_results["overall_fidelity"] = round(overall_fidelity, 4)

            elapsed = time.time() - t_layer
            print(f"\n  Layer {layer_idx} DONE in {elapsed:.0f}s -- fidelity={overall_fidelity:.3f}")
        else:
            layer_results["overall_fidelity"] = None
            print(f"\n  Layer {layer_idx} DONE -- no tasks evaluated")

        all_results["per_layer"][layer_key] = layer_results

        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    fidelities = [v["overall_fidelity"] for v in all_results["per_layer"].values()
                  if v.get("overall_fidelity") is not None]
    if fidelities:
        all_results["avg_fidelity"] = round(sum(fidelities) / len(fidelities), 4)
        all_results["min_fidelity"] = round(min(fidelities), 4)
        all_results["max_fidelity"] = round(max(fidelities), 4)
        all_results["n_layers_evaluated"] = len(fidelities)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    del policy, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return all_results


def print_summary(all_model_results, output_dir):
    """Print a summary table of reconstruction fidelity."""
    print("RECONSTRUCTION FIDELITY SUMMARY")

    for model_name, results in all_model_results.items():
        print(f"\n  {model_name}:")
        print(f"    Avg fidelity: {results.get('avg_fidelity', 'N/A')}")
        print(f"    Min fidelity: {results.get('min_fidelity', 'N/A')}")
        print(f"    Max fidelity: {results.get('max_fidelity', 'N/A')}")

        per_layer = results.get("per_layer", {})
        if per_layer:
            print(f"    Per-layer:")
            for layer_key in sorted(per_layer.keys(), key=int):
                layer_data = per_layer[layer_key]
                fid = layer_data.get("overall_fidelity", "?")
                recon = layer_data.get("overall_recon_rate", "?")
                base = layer_data.get("overall_baseline_rate", "?")
                print(f"      L{int(layer_key):>2d}: fidelity={fid}, recon={recon}, baseline={base}")

                for task_name, task_data in layer_data.get("per_task", {}).items():
                    short_name = task_name.split("_", 1)[-1] if "_" in task_name else task_name
                    t_fid = task_data.get("fidelity", "?")
                    t_recon = task_data.get("recon_success", "?")
                    t_base = task_data.get("baseline_success", "?")
                    print(f"            {short_name}: {t_recon*100:.0f}% / {t_base*100:.0f}% (fid={t_fid:.2f})")

    print(f"\nResults saved to: {output_dir}")


@dataclass
class ReconstructionEvalConfig:
    """X-VLA SimplerEnv SAE reconstruction fidelity evaluation."""

    model: str
    """Robot model to evaluate: widowx, google-robot, all"""

    pooling: str
    """Pooling mode: pertoken, meanpool"""

    sae_dir: str
    """Path to SAE model directory"""

    task: Optional[str] = None
    """Specific task to evaluate (overrides model defaults)"""

    layer: Optional[int] = None
    layers: Optional[str] = None
    """Comma-separated layer indices"""

    all_layers: bool = False
    n_episodes: int = 5
    max_steps: int = DEFAULT_MAX_STEPS
    device: str = "cuda"
    output_dir: Optional[str] = None


def main(cfg):

    if cfg.all_layers:
        layers = list(range(N_XVLA_LAYERS))
    elif cfg.layers:
        layers = [int(x.strip()) for x in cfg.layers.split(",")]
    elif cfg.layer is not None:
        layers = [cfg.layer]
    else:
        raise ValueError("Specify --layer, --layers, or --all-layers")

    if cfg.model == "all":
        models = ["widowx", "google-robot"]
    else:
        models = [cfg.model]

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        sae_dir_name = Path(cfg.sae_dir).name
        output_dir = Path(f"results/xvla_reconstruction/{sae_dir_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks_override = [cfg.task] if cfg.task else None

    print(f"X-VLA SimplerEnv SAE reconstruction evaluation")
    print(f"Models: {models}")
    print(f"Layers: {layers}")
    print(f"Pooling: {cfg.pooling}")
    print(f"SAE dir: {cfg.sae_dir}")
    print(f"Episodes/task: {cfg.n_episodes}")
    print(f"Output: {output_dir}")

    all_model_results = {}

    for model_type in models:
        results = evaluate_model(
            model_type=model_type,
            layers=layers,
            sae_dir=cfg.sae_dir,
            pooling=cfg.pooling,
            n_episodes=cfg.n_episodes,
            max_steps=cfg.max_steps,
            device=cfg.device,
            output_dir=output_dir,
            tasks_override=tasks_override,
        )
        all_model_results[model_type] = results

    print_summary(all_model_results, output_dir)
    print("\nDONE")


if __name__ == "__main__":
    cfg = tyro.cli(ReconstructionEvalConfig)
    main(cfg)
