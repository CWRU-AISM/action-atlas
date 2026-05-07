#!/usr/bin/env python3
"""
X-VLA Vision Perturbation Experiments

Tests how X-VLA responds to various image manipulations:
- Noise injection (Gaussian, salt-pepper)
- Blur (light to heavy)
- Color/brightness perturbations
- Spatial transforms (rotations, flips, crops)
- Region masking

Logs: success rate, object trajectories, EEF trajectory, videos

X-VLA Architecture:
- VLM: Florence-2 (DaViT + BART encoder)
- Action Head: SoftPromptedTransformer (24 TransformerBlocks, 1024-dim)
- Domain ID: 3 (LIBERO)
- Requires: control_mode=absolute

Usage:
    conda activate vla_interp
    python experiments/xvla_vision_perturbation.py --suite libero_object --task 0 --perturbation all
    python experiments/xvla_vision_perturbation.py --suite libero_object --perturbation baseline
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import argparse
import json
import warnings
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Tuple

import numpy as np
import torch
import cv2
import imageio

warnings.filterwarnings("ignore")



@dataclass
class PerturbationConfig:
    name: str
    perturbation_fn: Callable
    params: dict = field(default_factory=dict)


class ImagePerturbations:
    # Collection of image perturbation functions

    @staticmethod
    def gaussian_noise(img: np.ndarray, std: float = 25.0) -> np.ndarray:
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_pepper_noise(img: np.ndarray, prob: float = 0.05) -> np.ndarray:
        output = img.copy()
        salt_mask = np.random.random(img.shape[:2]) < prob / 2
        output[salt_mask] = 255
        pepper_mask = np.random.random(img.shape[:2]) < prob / 2
        output[pepper_mask] = 0
        return output

    @staticmethod
    def blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @staticmethod
    def brightness(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
        adjusted = img.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def contrast(img: np.ndarray, factor: float = 1.5) -> np.ndarray:
        mean = img.mean()
        adjusted = (img.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    @staticmethod
    def color_jitter(img: np.ndarray, hue_shift: int = 20) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def grayscale(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def invert(img: np.ndarray) -> np.ndarray:
        return 255 - img

    @staticmethod
    def center_crop(img: np.ndarray, crop_frac: float = 0.8) -> np.ndarray:
        h, w = img.shape[:2]
        new_h, new_w = int(h * crop_frac), int(w * crop_frac)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = img[top:top+new_h, left:left+new_w]
        return cv2.resize(cropped, (w, h))

    @staticmethod
    def rotate(img: np.ndarray, angle: float = 15) -> np.ndarray:
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, matrix, (w, h))

    @staticmethod
    def horizontal_flip(img: np.ndarray) -> np.ndarray:
        return img[:, ::-1].copy()

    @staticmethod
    def vertical_flip(img: np.ndarray) -> np.ndarray:
        return img[::-1, :].copy()

    @staticmethod
    def edge_only(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    @staticmethod
    def posterize(img: np.ndarray, levels: int = 4) -> np.ndarray:
        factor = 256 // levels
        return (img // factor) * factor

    @staticmethod
    def mask_center(img: np.ndarray, fill_value: int = 128) -> np.ndarray:
        result = img.copy()
        h, w = img.shape[:2]
        y1, y2 = h//4, 3*h//4
        x1, x2 = w//4, 3*w//4
        result[y1:y2, x1:x2] = fill_value
        return result


def get_standard_perturbations() -> List[PerturbationConfig]:
    # Get list of standard perturbations to test
    return [
        # Baseline
        PerturbationConfig("baseline", lambda x: x),

        # Noise
        PerturbationConfig("gaussian_noise_low", ImagePerturbations.gaussian_noise, {"std": 15}),
        PerturbationConfig("gaussian_noise_high", ImagePerturbations.gaussian_noise, {"std": 50}),
        PerturbationConfig("salt_pepper", ImagePerturbations.salt_pepper_noise, {"prob": 0.05}),

        # Blur
        PerturbationConfig("blur_light", ImagePerturbations.blur, {"kernel_size": 5}),
        PerturbationConfig("blur_heavy", ImagePerturbations.blur, {"kernel_size": 15}),

        # Color/brightness
        PerturbationConfig("bright_up", ImagePerturbations.brightness, {"factor": 1.5}),
        PerturbationConfig("bright_down", ImagePerturbations.brightness, {"factor": 0.5}),
        PerturbationConfig("contrast_up", ImagePerturbations.contrast, {"factor": 1.5}),
        PerturbationConfig("contrast_down", ImagePerturbations.contrast, {"factor": 0.5}),
        PerturbationConfig("hue_shift", ImagePerturbations.color_jitter, {"hue_shift": 30}),
        PerturbationConfig("grayscale", ImagePerturbations.grayscale),
        PerturbationConfig("invert", ImagePerturbations.invert),

        # Spatial
        PerturbationConfig("center_crop_80", ImagePerturbations.center_crop, {"crop_frac": 0.8}),
        PerturbationConfig("center_crop_60", ImagePerturbations.center_crop, {"crop_frac": 0.6}),
        PerturbationConfig("rotate_15", ImagePerturbations.rotate, {"angle": 15}),
        PerturbationConfig("rotate_45", ImagePerturbations.rotate, {"angle": 45}),
        PerturbationConfig("h_flip", ImagePerturbations.horizontal_flip),
        PerturbationConfig("v_flip", ImagePerturbations.vertical_flip),

        # Extreme
        PerturbationConfig("edge_only", ImagePerturbations.edge_only),
        PerturbationConfig("posterize_4", ImagePerturbations.posterize, {"levels": 4}),
        PerturbationConfig("mask_center", ImagePerturbations.mask_center),

        # Half crops
        PerturbationConfig("crop_top_half", lambda x: cv2.resize(x[:x.shape[0]//2, :], (x.shape[1], x.shape[0])).astype(np.uint8)),
        PerturbationConfig("crop_bottom_half", lambda x: cv2.resize(x[x.shape[0]//2:, :], (x.shape[1], x.shape[0])).astype(np.uint8)),
        PerturbationConfig("crop_left_half", lambda x: cv2.resize(x[:, :x.shape[1]//2], (x.shape[1], x.shape[0])).astype(np.uint8)),
        PerturbationConfig("crop_right_half", lambda x: cv2.resize(x[:, x.shape[1]//2:], (x.shape[1], x.shape[0])).astype(np.uint8)),
    ]


def get_breaking_perturbations() -> List[PerturbationConfig]:
    # Get only perturbations that typically break models (for quick validation)
    return [
        PerturbationConfig("baseline", lambda x: x),
        PerturbationConfig("rotate_15", ImagePerturbations.rotate, {"angle": 15}),
        PerturbationConfig("h_flip", ImagePerturbations.horizontal_flip),
        PerturbationConfig("v_flip", ImagePerturbations.vertical_flip),
        PerturbationConfig("crop_top_half", lambda x: cv2.resize(x[:x.shape[0]//2, :], (x.shape[1], x.shape[0])).astype(np.uint8)),
        PerturbationConfig("invert", ImagePerturbations.invert),
    ]



def get_inner_env(vec_env):
    if hasattr(vec_env, 'envs'):
        libero_env = vec_env.envs[0]
    else:
        libero_env = vec_env
    control_env = libero_env._env
    return control_env.env


def get_scene_state(inner_env):
    state = {}
    try:
        eef_site_id = inner_env.robots[0].eef_site_id
        state['robot_eef'] = inner_env.sim.data.site_xpos[eef_site_id].copy().tolist()
    except Exception:
        state['robot_eef'] = None

    objects = {}
    if hasattr(inner_env, 'obj_body_id'):
        for obj_name, body_id in inner_env.obj_body_id.items():
            pos = inner_env.sim.data.body_xpos[body_id].copy().tolist()
            objects[obj_name] = {'pos': pos}
    state['objects'] = objects
    return state


def summarize_scene(scene_states):
    if not scene_states:
        return {}

    summary = {
        'n_steps': len(scene_states),
    }

    eef_traj = [s['robot_eef'] for s in scene_states if s.get('robot_eef')]
    if eef_traj:
        summary['robot_eef_trajectory'] = eef_traj

    if scene_states[0].get('objects') and scene_states[-1].get('objects'):
        displacements = {}
        for obj_name in scene_states[0]['objects']:
            if obj_name in scene_states[-1]['objects']:
                init_pos = np.array(scene_states[0]['objects'][obj_name]['pos'])
                final_pos = np.array(scene_states[-1]['objects'][obj_name]['pos'])
                displacements[obj_name] = float(np.linalg.norm(final_pos - init_pos))
        summary['object_displacements'] = displacements

    return summary



def run_perturbed_episode(policy, env, preprocessor, postprocessor,
                          env_preprocessor, env_postprocessor,
                          prompt, perturbation: PerturbationConfig,
                          max_steps=280, seed=42, save_video=False,
                          inner_env=None):
    from lerobot.envs.utils import preprocess_observation
    policy.reset()
    obs, _ = env.reset(seed=seed)
    actions = []
    frames = []
    perturbed_frames = []
    scene_states = []
    is_success = False

    for step in range(max_steps):
        if inner_env is not None:
            scene_states.append(get_scene_state(inner_env))

        # Get image
        if 'pixels' in obs and 'image' in obs['pixels']:
            img = obs['pixels']['image']
            if isinstance(img, np.ndarray):
                if img.ndim == 4:
                    img = img[0]
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        else:
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Apply perturbation
        perturbed_img = perturbation.perturbation_fn(img.copy(), **perturbation.params)

        # Create modified observation
        obs_modified = obs.copy()
        if 'pixels' in obs_modified and 'image' in obs_modified['pixels']:
            obs_modified['pixels'] = obs_modified['pixels'].copy()
            if perturbed_img.dtype != np.uint8:
                perturbed_img = (perturbed_img * 255).astype(np.uint8) if perturbed_img.max() <= 1 else perturbed_img.astype(np.uint8)
            if obs_modified['pixels']['image'].ndim == 4:
                obs_modified['pixels']['image'] = perturbed_img[np.newaxis, ...]
            else:
                obs_modified['pixels']['image'] = perturbed_img

        # Process through pipeline
        obs_proc = preprocess_observation(obs_modified)
        obs_proc["task"] = [prompt]
        obs_proc = env_preprocessor(obs_proc)
        obs_proc = preprocessor(obs_proc)

        with torch.inference_mode():
            action = policy.select_action(obs_proc)

        action_np = action.cpu().numpy().flatten()
        actions.append(action_np)

        action_proc = postprocessor(action)
        action_t = env_postprocessor({'action': action_proc})
        obs, reward, term, trunc, info = env.step(action_t['action'].cpu().numpy())

        success_val = info.get('is_success', False)
        if isinstance(success_val, (list, np.ndarray)):
            success_val = bool(success_val[0])
        if success_val:
            is_success = True

        # Save frames
        if save_video and step % 3 == 0:
            if 'pixels' in obs and 'image' in obs['pixels']:
                frame = obs['pixels']['image']
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 4:
                        frame = frame[0]
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                    frame = frame[::-1, ::-1]
                    frames.append(frame)
            perturbed_frames.append(cv2.resize(perturbed_img, (256, 256)))

        if term[0] or trunc[0]:
            break

    return {
        'actions': np.array(actions),
        'frames': frames,
        'perturbed_frames': perturbed_frames,
        'scene_states': scene_states,
        'is_success': is_success,
        'n_steps': len(actions),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task", type=int, default=None, help="Single task ID (legacy)")
    parser.add_argument("--tasks", type=int, nargs="+", default=None,
                        help="Task IDs to run (default: all 10)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--perturbation", default="all", help="Perturbation name or 'all' or 'breaking'")
    parser.add_argument("--max_steps", type=int, default=280)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="lerobot/xvla-libero")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    # Suite-specific max steps
    SUITE_MAX_STEPS = {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    if args.max_steps == 280:
        args.max_steps = SUITE_MAX_STEPS.get(args.suite, 280)

    device = 'cuda'
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"rollouts/xvla_vision_perturbation_{args.suite}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve task list
    if args.tasks is not None:
        task_ids = args.tasks
    elif args.task is not None:
        task_ids = [args.task]
    else:
        task_ids = list(range(10))

    print(f"{'='*70}")
    print(f"X-VLA VISION PERTURBATION EXPERIMENTS")
    print(f"Suite: {args.suite}, Tasks: {task_ids}, Seeds: {args.seeds}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}")

    # Load X-VLA model
    print(f"\n1. Loading X-VLA model from {args.checkpoint}...")
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    policy = XVLAPolicy.from_pretrained(args.checkpoint)
    policy.eval().to(device)
    print(f"   Model loaded: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params")

    # Create envs with control_mode=absolute
    print(f"2. Creating environments ({args.suite}) with control_mode=absolute...")
    import gymnasium as gym
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.factory import make_env_config, make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors

    # Use policy.config (XVLAConfig) for proper LIBERO preprocessing
    policy_cfg = policy.config

    envs_dict = create_libero_envs(
        task=args.suite,
        n_envs=1,
        control_mode="absolute",
        episode_length=args.max_steps,
        env_cls=gym.vector.SyncVectorEnv,
        gym_kwargs={'obs_type': 'pixels_agent_pos'},  # Required for X-VLA proprio
    )
    task_envs = envs_dict[args.suite]

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': device}}
    )
    env_cfg = make_env_config(
        env_type="libero",
        task=args.suite,
        control_mode="absolute",
        episode_length=args.max_steps,
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    # Get perturbations to run
    all_perturbations = get_standard_perturbations()
    if args.perturbation == "all":
        perturbations = all_perturbations
    elif args.perturbation == "breaking":
        perturbations = get_breaking_perturbations()
    else:
        perturbations = [p for p in all_perturbations if p.name == args.perturbation]
        if not perturbations:
            print(f"Unknown perturbation: {args.perturbation}")
            print(f"Available: {[p.name for p in all_perturbations]}")
            return

    save_video = not args.no_video
    all_results = []

    print(f"\n3. Running {len(task_ids)} tasks x {len(perturbations)} perturbations x {len(args.seeds)} seeds...")

    for tid in task_ids:
        env = task_envs[tid]
        inner_env = get_inner_env(env)

        if hasattr(env, 'envs'):
            task_description = env.envs[0].task_description
        else:
            task_description = getattr(env, 'task_description', f"task_{tid}")

        task_dir = output_dir / f"task_{tid}"
        task_dir.mkdir(exist_ok=True)

        # Resume: skip if task already has results
        task_results_path = task_dir / "results.json"
        if task_results_path.exists():
            print(f"\n  [SKIP] Task {tid} already done, loading...")
            with open(task_results_path) as f:
                task_data = json.load(f)
            all_results.extend(task_data.get("results", []))
            continue

        print(f"\n{'='*60}")
        print(f"TASK {tid}: {task_description}")
        print(f"{'='*60}")

        task_results = []

        for seed in args.seeds:
            for p in perturbations:
                result = run_perturbed_episode(
                    policy, env, preprocessor, postprocessor,
                    env_preprocessor, env_postprocessor,
                    prompt=task_description,
                    perturbation=p,
                    max_steps=args.max_steps,
                    seed=seed,
                    save_video=save_video,
                    inner_env=inner_env,
                )

                scene_summary = summarize_scene(result['scene_states'])

                result_entry = {
                    'perturbation': p.name,
                    'params': p.params,
                    'task': tid,
                    'task_description': task_description,
                    'seed': seed,
                    'success': result['is_success'],
                    'n_steps': result['n_steps'],
                    'scene': scene_summary,
                }
                task_results.append(result_entry)

                print(f"  t{tid} {p.name} s{seed}: {'OK' if result['is_success'] else 'FAIL'} ({result['n_steps']} steps)")

                if save_video and result['frames']:
                    video_path = task_dir / f"{p.name}_s{seed}.mp4"
                    imageio.mimsave(str(video_path), result['frames'], fps=10)

                traj_data = {
                    'perturbation': p.name,
                    'task': tid,
                    'seed': seed,
                    'is_success': result['is_success'],
                    'n_steps': result['n_steps'],
                    'actions': [a.tolist() if hasattr(a, 'tolist') else a for a in result['actions']],
                    'scene_states': result['scene_states'],
                    'scene_summary': scene_summary,
                }
                with open(task_dir / f"{p.name}_s{seed}.json", "w") as f:
                    json.dump(traj_data, f, indent=2, default=str)

                gc.collect()
                torch.cuda.empty_cache()

        # Save per-task results incrementally
        with open(task_results_path, "w") as f:
            json.dump({"task": tid, "task_description": task_description,
                        "results": task_results}, f, indent=2)
        all_results.extend(task_results)

    # ---- SUMMARY ----
    print(f"\n{'='*70}")
    print("VISION PERTURBATION SUMMARY")
    print(f"{'='*70}")

    by_perturbation = {}
    for r in all_results:
        name = r['perturbation']
        if name not in by_perturbation:
            by_perturbation[name] = []
        by_perturbation[name].append(r['success'])

    print(f"\n{'Perturbation':<25} | {'Success Rate':>12} | {'N':>4}")
    print("-" * 50)
    for name, successes in sorted(by_perturbation.items()):
        rate = sum(successes) / len(successes) * 100
        print(f"{name:<25} | {rate:>11.1f}% | {len(successes):>4}")

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "model": "xvla",
            "checkpoint": args.checkpoint,
            "suite": args.suite,
            "tasks": task_ids,
            "seeds": args.seeds,
            "max_steps": args.max_steps,
            "results": all_results,
            "summary": {k: {"success_rate": sum(v)/len(v), "n": len(v)}
                       for k, v in by_perturbation.items()},
            "duration": str(datetime.now() - start_time),
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print(f"Duration: {datetime.now() - start_time}")


if __name__ == "__main__":
    main()
