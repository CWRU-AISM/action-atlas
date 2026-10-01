#!/usr/bin/env python3
"""
SmolVLA MetaWorld vision perturbation experiments.

Tests how SmolVLA responds to various image manipulations on MetaWorld tasks.
Applies perturbations to camera input at every timestep and measures impact
on success rate, action trajectory, and scene state.

Perturbation categories:
- Noise: gaussian, salt & pepper
- Blur: light, heavy
- Color: brightness, contrast, hue shift, grayscale, invert
- Spatial: crop, rotate, flip
- Extreme: edge detection, posterize
- Region: mask top/bottom/center, keep only halves

Usage:
    python experiments/metaworld/smolvla_metaworld_vision_perturbation.py --difficulty easy
    python experiments/metaworld/smolvla_metaworld_vision_perturbation.py --tasks reach-v3,push-v3 --save-video
    python experiments/metaworld/smolvla_metaworld_vision_perturbation.py --difficulty easy --perturbations baseline,blur_heavy,h_flip
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import tyro

from common import (
    PROJECT_ROOT, DEFAULT_CHECKPOINT, DEFAULT_RESOLUTION, MAX_STEPS, TASK_DESCRIPTIONS,
    create_env, force_free_memory, get_tasks_from_args,
    load_smolvla_policy, run_episode, save_video_frames,
)
from experiments.utils import ImagePerturbations


def random_crop(img, crop_frac=0.8):
    h, w = img.shape[:2]
    nh, nw = int(h * crop_frac), int(w * crop_frac)
    top = np.random.randint(0, h - nh)
    left = np.random.randint(0, w - nw)
    return cv2.resize(img[top:top+nh, left:left+nw], (w, h))


def posterize(img, levels=4):
    factor = 256 // levels
    return (img // factor) * factor


def crop_top_half(img):
    return cv2.resize(img[:img.shape[0]//2, :], (img.shape[1], img.shape[0]))


def crop_bottom_half(img):
    return cv2.resize(img[img.shape[0]//2:, :], (img.shape[1], img.shape[0]))


def crop_left_half(img):
    return cv2.resize(img[:, :img.shape[1]//2], (img.shape[1], img.shape[0]))


def crop_right_half(img):
    return cv2.resize(img[:, img.shape[1]//2:], (img.shape[1], img.shape[0]))


def mask_top_quarter(img):
    out = img.copy()
    out[:img.shape[0]//4, :] = 128
    return out


def mask_bottom_quarter(img):
    out = img.copy()
    out[3*img.shape[0]//4:, :] = 128
    return out


def frozen_image(img, _cache={}):
    # Return the first frame for the entire episode (set externally)
    if 'frame' in _cache:
        return _cache['frame'].copy()
    return img


P = ImagePerturbations

PERTURBATIONS = {
    'baseline': lambda x: x,
    'gaussian_noise_low': lambda x: P.gaussian_noise(x, std=15),
    'gaussian_noise_high': lambda x: P.gaussian_noise(x, std=50),
    'salt_pepper': lambda x: P.salt_pepper_noise(x, prob=0.05),
    'blur_light': lambda x: P.blur(x, kernel_size=5),
    'blur_heavy': lambda x: P.blur(x, kernel_size=15),
    'bright_up': lambda x: P.brightness(x, factor=1.5),
    'bright_down': lambda x: P.brightness(x, factor=0.5),
    'contrast_up': lambda x: P.contrast(x, factor=1.5),
    'contrast_down': lambda x: P.contrast(x, factor=0.5),
    'hue_shift': lambda x: P.color_jitter(x, hue_shift=30),
    'grayscale': P.grayscale,
    'invert': P.invert,
    'center_crop_80': lambda x: P.center_crop(x, crop_frac=0.8),
    'center_crop_60': lambda x: P.center_crop(x, crop_frac=0.6),
    'random_crop': lambda x: random_crop(x, crop_frac=0.8),
    'rotate_15': lambda x: P.rotate(x, angle=15),
    'rotate_45': lambda x: P.rotate(x, angle=45),
    'h_flip': P.horizontal_flip,
    'v_flip': P.vertical_flip,
    'edge_only': P.edge_only,
    'posterize_4': lambda x: posterize(x, levels=4),
    'black_image': P.black_image,
    'frozen_first_frame': frozen_image,
    'crop_top_half': crop_top_half,
    'crop_bottom_half': crop_bottom_half,
    'crop_left_half': crop_left_half,
    'crop_right_half': crop_right_half,
    'center_crop_50': lambda x: P.center_crop(x, crop_frac=0.5),
    'mask_top_quarter': mask_top_quarter,
    'mask_bottom_quarter': mask_bottom_quarter,
    'mask_center': P.mask_center,
}


def make_perturbation_transform(pert_fn):
    # Wrap a perturbation function to handle frozen_first_frame state
    if pert_fn is frozen_image:
        first_frame_holder = {}

        def transform(pixels):
            if not first_frame_holder:
                first_frame_holder['frame'] = pixels.copy()
            return first_frame_holder['frame'].copy()
        return transform
    return pert_fn


@dataclass
class VisionPerturbationConfig:
    # SmolVLA MetaWorld vision perturbation experiments

    checkpoint: str = DEFAULT_CHECKPOINT
    tasks: Optional[str] = None
    difficulty: Optional[str] = None
    n_episodes: int = 3
    perturbations: Optional[str] = None
    """Comma-separated list of perturbation names. Default: all"""

    resolution: int = DEFAULT_RESOLUTION
    save_video: bool = False
    resume: bool = False
    output_dir: Optional[str] = None


def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = get_tasks_from_args(cfg)

    if cfg.perturbations:
        perturbation_names = [p.strip() for p in cfg.perturbations.split(',')]
    else:
        perturbation_names = list(PERTURBATIONS.keys())

    if cfg.output_dir:
        output_dir = Path(cfg.output_dir)
    else:
        output_dir = PROJECT_ROOT / "rollouts/smolvla/metaworld_vision_perturbation"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)
    if cfg.save_video:
        (output_dir / "videos").mkdir(exist_ok=True)

    print(f"Tasks: {len(tasks)}, Perturbations: {len(perturbation_names)}, Episodes: {cfg.n_episodes}")
    print(f"Device: {device}, Output: {output_dir}")
    print(f"Perturbations: {', '.join(perturbation_names)}")

    print("\nLoading model")
    policy, preprocessor, postprocessor = load_smolvla_policy(
        cfg.checkpoint, device)

    results_path = output_dir / "results.json"
    if cfg.resume and results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        all_results = {
            'checkpoint': cfg.checkpoint,
            'env': 'metaworld',
            'n_episodes': cfg.n_episodes,
            'max_steps': MAX_STEPS,
            'resolution': cfg.resolution,
            'perturbations': perturbation_names,
            'tasks': tasks,
            'timestamp': datetime.now().isoformat(),
            'grid': {},
        }

    total_configs = len(perturbation_names) * len(tasks)
    config_idx = 0
    start_time = time.time()

    for pert_name in perturbation_names:
        if pert_name not in PERTURBATIONS:
            print(f"WARNING: Unknown perturbation '{pert_name}', skipping")
            continue

        pert_fn = PERTURBATIONS[pert_name]
        print(f"\nPerturbation: {pert_name}")

        pert_results = all_results['grid'].get(pert_name, {})

        for task_name in tasks:
            config_idx += 1

            if task_name in pert_results and len(pert_results[task_name].get('episodes', [])) >= cfg.n_episodes:
                print(f"[{config_idx}/{total_configs}] {task_name}: SKIP (exists)")
                continue

            task_desc = TASK_DESCRIPTIONS.get(task_name, task_name)
            successes = []
            steps_list = []

            for ep in range(cfg.n_episodes):
                env = create_env(task_name, cfg.resolution)

                image_transform = make_perturbation_transform(pert_fn)

                result = run_episode(
                    policy, env, preprocessor, postprocessor, device,
                    save_video=cfg.save_video,
                    image_transform_fn=image_transform if pert_name != 'baseline' else None,
                )
                env.close()

                successes.append(result['success'])
                steps_list.append(result['n_steps'])

                traj_dir = output_dir / "trajectories" / pert_name / task_name
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_data = {
                    'task': task_name,
                    'task_description': task_desc,
                    'perturbation': pert_name,
                    'episode': ep,
                    'success': result['success'],
                    'n_steps': result['n_steps'],
                    'actions': result['actions'].tolist(),
                    'agent_pos_trajectory': result.get('agent_pos_trajectory', []),
                    'scene_states': result.get('scene_states', []),
                }
                with open(traj_dir / f"ep{ep:02d}.json", 'w') as f:
                    json.dump(traj_data, f)

                if cfg.save_video and result.get('frames'):
                    vid_dir = output_dir / "videos" / pert_name / task_name
                    vid_dir.mkdir(parents=True, exist_ok=True)
                    save_video_frames(vid_dir / f"ep{ep:02d}.mp4", result['frames'], fps=10)

            success_rate = sum(successes) / len(successes)
            print(f"[{config_idx}/{total_configs}] {task_name}: {success_rate*100:.0f}% "
                  f"({sum(successes)}/{len(successes)})")

            pert_results[task_name] = {
                'task_description': task_desc,
                'success_rate': success_rate,
                'episodes': [{'success': s, 'steps': st} for s, st in zip(successes, steps_list)],
            }

        all_results['grid'][pert_name] = pert_results

        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        force_free_memory()

    elapsed = time.time() - start_time
    all_results['duration_seconds'] = elapsed

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nVision Perturbation Summary")
    print(f"{'Perturbation':<25}", end="")
    for t in tasks[:8]:
        print(f"{t[:8]:>9}", end="")
    print(f"  {'Avg':>5}")

    for pert_name in perturbation_names:
        pdata = all_results['grid'].get(pert_name, {})
        print(f"{pert_name:<25}", end="")
        rates = []
        for t in tasks[:8]:
            rate = pdata.get(t, {}).get('success_rate', 0)
            rates.append(rate)
            print(f"{rate*100:>8.0f}%", end="")
        avg = np.mean(rates) if rates else 0
        print(f"  {avg*100:>4.0f}%")

    print(f"\nDone in {elapsed/60:.1f} min. Results: {results_path}")


if __name__ == "__main__":
    cfg = tyro.cli(VisionPerturbationConfig)
    main(cfg)
