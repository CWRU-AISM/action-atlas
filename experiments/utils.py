"""Shared utilities for VLA interpretability experiments."""

import ctypes
import gc
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def force_free_memory():
    """Free memory: GC + CUDA cache + libc malloc_trim."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass



def get_scene_state(env) -> dict:
    """Extract robot EEF + all object positions from a LIBERO env.

    Works with both raw env and wrapped envs (OffScreenRenderEnv, vec envs).
    """
    inner = env
    for attr in ("env", "_env", "envs"):
        if hasattr(inner, attr):
            child = getattr(inner, attr)
            if attr == "envs":
                inner = child[0] if isinstance(child, (list, tuple)) else child
            else:
                inner = child

    state = {}
    try:
        eef_site_id = inner.robots[0].eef_site_id
        state["robot_eef"] = inner.sim.data.site_xpos[eef_site_id].copy().tolist()
    except Exception:
        state["robot_eef"] = None
    try:
        state["gripper_qpos"] = inner.sim.data.qpos[
            inner.robots[0].gripper.joint_indexes
        ].copy().tolist()
    except Exception:
        state["gripper_qpos"] = None

    objects = {}
    obj_body_id = getattr(inner, "obj_body_id", None)
    sim = getattr(inner, "sim", None)
    if obj_body_id and sim:
        for obj_name, body_id in obj_body_id.items():
            objects[obj_name] = {
                "pos": sim.data.body_xpos[body_id].copy().tolist(),
                "quat": sim.data.body_xquat[body_id].copy().tolist(),
            }
    state["objects"] = objects
    return state


def summarize_scene(scene_states: list) -> dict:
    """Summarize a trajectory of scene states: EEF path + object displacements."""
    if not scene_states:
        return {}
    summary = {"n_steps": len(scene_states)}
    eef_traj = [s["robot_eef"] for s in scene_states if s.get("robot_eef")]
    if eef_traj:
        summary["robot_eef_trajectory"] = eef_traj
    first, last = scene_states[0], scene_states[-1]
    if first.get("objects") and last.get("objects"):
        displacements = {}
        for name in first["objects"]:
            if name in last["objects"]:
                init = np.array(first["objects"][name]["pos"])
                final = np.array(last["objects"][name]["pos"])
                displacements[name] = {
                    "distance": float(np.linalg.norm(final - init)),
                    "init_pos": init.tolist(),
                    "final_pos": final.tolist(),
                }
        summary["object_displacements"] = displacements
    return summary


def compare_trajectories(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two action trajectories (cosine similarity + xyz L2)."""
    n = min(len(a), len(b))
    if n == 0:
        return {"cos": 0.0, "xyz": 0.0}
    a, b = a[:n], b[:n]
    ma, mb = a.mean(0), b.mean(0)
    cos = float(np.dot(ma, mb) / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-10))
    xyz = float(np.mean(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)))
    return {"cos": cos, "xyz": xyz}


def top_moved_objects(displacements: dict, n: int = 3, threshold: float = 0.01):
    """Return top N most-displaced objects above threshold."""
    moved = {k: v for k, v in displacements.items() if v["distance"] > threshold}
    ranked = sorted(moved.items(), key=lambda x: -x[1]["distance"])
    return [(k, v["distance"]) for k, v in ranked[:n]]



def save_video(frames: list, path, fps: int = 10):
    """Save frames as MP4 video using imageio."""
    if not frames:
        return
    try:
        import imageio
        writer = imageio.get_writer(
            str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
            output_params=["-crf", "18", "-preset", "fast"],
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception as e:
        print(f"  Warning: Could not save video {path}: {e}")



def load_results(path: Path) -> Optional[dict]:
    """Load results.json if it exists, else return None."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def save_results(data: dict, path: Path):
    """Atomically write results JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)



class ImagePerturbations:
    """Standard image perturbations. All take uint8 HWC RGB and return same."""

    @staticmethod
    def gaussian_noise(img, std=25.0):
        noise = np.random.normal(0, std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def salt_pepper_noise(img, prob=0.05):
        out = img.copy()
        out[np.random.random(img.shape[:2]) < prob / 2] = 255
        out[np.random.random(img.shape[:2]) < prob / 2] = 0
        return out

    @staticmethod
    def blur(img, kernel_size=5):
        import cv2
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    @staticmethod
    def brightness(img, factor=1.5):
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def contrast(img, factor=1.5):
        mean = img.mean()
        return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    @staticmethod
    def color_jitter(img, hue_shift=20):
        import cv2
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def grayscale(img):
        import cv2
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

    @staticmethod
    def invert(img):
        return 255 - img

    @staticmethod
    def center_crop(img, crop_frac=0.8):
        import cv2
        h, w = img.shape[:2]
        nh, nw = int(h * crop_frac), int(w * crop_frac)
        top, left = (h - nh) // 2, (w - nw) // 2
        return cv2.resize(img[top:top + nh, left:left + nw], (w, h))

    @staticmethod
    def rotate(img, angle=15):
        import cv2
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    @staticmethod
    def horizontal_flip(img):
        return img[:, ::-1].copy()

    @staticmethod
    def vertical_flip(img):
        return img[::-1, :].copy()

    @staticmethod
    def edge_only(img):
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(cv2.Canny(gray, 50, 150), cv2.COLOR_GRAY2RGB)

    @staticmethod
    def mask_center(img, fill_value=128):
        out = img.copy()
        h, w = img.shape[:2]
        out[h // 4:3 * h // 4, w // 4:3 * w // 4] = fill_value
        return out

    @staticmethod
    def black_image(img):
        return np.zeros_like(img)

    @staticmethod
    def white_image(img):
        return np.full_like(img, 255)

    @staticmethod
    def random_image(img):
        return np.random.randint(0, 256, img.shape, dtype=np.uint8)


def get_standard_perturbations() -> list:
    """Return list of (name, fn) tuples for standard perturbation suite."""
    P = ImagePerturbations
    return [
        ("baseline", lambda x: x),
        ("gaussian_noise_low", lambda x: P.gaussian_noise(x, std=15)),
        ("gaussian_noise_high", lambda x: P.gaussian_noise(x, std=50)),
        ("salt_pepper", lambda x: P.salt_pepper_noise(x, prob=0.05)),
        ("blur_light", lambda x: P.blur(x, kernel_size=5)),
        ("blur_heavy", lambda x: P.blur(x, kernel_size=15)),
        ("bright_up", lambda x: P.brightness(x, factor=1.5)),
        ("bright_down", lambda x: P.brightness(x, factor=0.5)),
        ("contrast_up", lambda x: P.contrast(x, factor=1.5)),
        ("contrast_down", lambda x: P.contrast(x, factor=0.5)),
        ("hue_shift", lambda x: P.color_jitter(x, hue_shift=30)),
        ("grayscale", P.grayscale),
        ("invert", P.invert),
        ("center_crop_80", lambda x: P.center_crop(x, crop_frac=0.8)),
        ("center_crop_60", lambda x: P.center_crop(x, crop_frac=0.6)),
        ("rotate_15", lambda x: P.rotate(x, angle=15)),
        ("rotate_45", lambda x: P.rotate(x, angle=45)),
        ("h_flip", P.horizontal_flip),
        ("v_flip", P.vertical_flip),
        ("edge_only", P.edge_only),
        ("mask_center", P.mask_center),
        ("black_image", P.black_image),
        ("white_image", P.white_image),
        ("random_image", P.random_image),
    ]



SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_long": 520,
}

COUNTERFACTUAL_PROMPTS = {
    "null_prompt": "",
    "random": "blah foo bar baz",
    "negation": "do not {task}",
    "opposite": "undo the task",
    "generic": "move the robot arm",
}
