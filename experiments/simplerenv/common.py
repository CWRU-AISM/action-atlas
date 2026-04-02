#!/usr/bin/env python3
"""
Shared utilities for X-VLA SimplerEnv experiments.

Provides model loading, action conversion, batch creation, episode
execution, and hook classes for WidowX and Google Robot environments.
All SimplerEnv experiment scripts import from here.

Requires the simpler_env conda environment (Python 3.10, numpy<2).
"""

import os
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("DISPLAY", "")

import gc
import json
import math
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Stub out GR00T to prevent xformers/diffusers segfault in simpler_env conda env.
# lerobot.policies.__init__ imports GR00T, which triggers a xformers import chain
# that crashes due to an incompatible binary in user site-packages.
for _mod in [
    "lerobot.policies.groot",
    "lerobot.policies.groot.configuration_groot",
    "lerobot.policies.groot.modeling_groot",
    "lerobot.policies.groot.groot_n1",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
if not hasattr(sys.modules["lerobot.policies.groot.configuration_groot"], "GrootConfig"):
    sys.modules["lerobot.policies.groot.configuration_groot"].GrootConfig = type("GrootConfig", (), {})
if not hasattr(sys.modules["lerobot.policies.groot.modeling_groot"], "GrootPolicy"):
    sys.modules["lerobot.policies.groot.modeling_groot"].GrootPolicy = type("GrootPolicy", (), {})

import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

# Upstream simpler_env.make() doesn't accept max_episode_steps.
# Monkey-patch to forward it to gymnasium.make() which does.
_original_simpler_make = simpler_env.make
def _make_with_max_steps(task_name, max_episode_steps=None):
    import gymnasium as gym
    assert task_name in simpler_env.ENVIRONMENTS, f"Task {task_name} not supported"
    env_name, kwargs = simpler_env.ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps
    return gym.make(env_name, obs_mode="rgbd", **kwargs)
simpler_env.make = _make_with_max_steps
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from transformers import AutoTokenizer
from sapien.core import Pose


# Constants

WIDOWX_TASKS = [
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
]

GOOGLE_ROBOT_TASKS = [
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_top_drawer",
    "google_robot_close_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_close_middle_drawer",
]

MODEL_CONFIGS = {
    "widowx": {
        "checkpoint": "lerobot/xvla-widowx",
        "domain_id": 0,
        "tasks": WIDOWX_TASKS,
    },
    "google-robot": {
        "checkpoint": "lerobot/xvla-google-robot",
        "domain_id": 1,
        "tasks": GOOGLE_ROBOT_TASKS,
    },
}

DEFAULT_MAX_STEPS = 1200

WIDOWX_GRIPPER_THRESHOLDS = {
    "widowx_spoon_on_towel": 0.7,
    "widowx_carrot_on_plate": 0.95,
    "widowx_stack_cube": 0.91,
    "widowx_put_eggplant_in_basket": 0.8,
}
DEFAULT_GRIPPER_THRESHOLD = 0.8

N_LAYERS = 24

ROBOT_CONCEPT_MAP = {
    "widowx": "widowx",
    "google-robot": "google_robot",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

STEP_INFO_KEYS = [
    "moved_correct_obj", "moved_wrong_obj", "is_src_obj_grasped",
    "consecutive_grasp", "src_on_target", "success", "elapsed_steps",
]


# Memory helpers

def force_free_memory():
    gc.collect()
    torch.cuda.empty_cache()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log_ram(label=""):
    import psutil
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / (1024**3)
    vm = psutil.virtual_memory()
    print(f"  [RAM] {label}: process={rss_gb:.1f}GB, "
          f"system={vm.used/(1024**3):.1f}/{vm.total/(1024**3):.1f}GB ({vm.percent}%)")
    if vm.percent > 75:
        force_free_memory()


# Rotation / action conversion

def rotate6d_to_euler_xyz(v6):
    """Convert 6D rotation to Euler XYZ. Matches the official X-VLA code."""
    v6 = np.asarray(v6, dtype=np.float64)
    if v6.shape[-1] != 6:
        raise ValueError(f"Last dimension must be 6, got {v6.shape[-1]}")
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    return R.from_matrix(rot_mats).as_euler("xyz")

# Keep old name as alias for backward compat with scripts that import it
rotate6D_to_euler_xyz = rotate6d_to_euler_xyz


def convert_xvla_action_widowx(action_raw, gripper_threshold=0.8):
    """
    Convert X-VLA 20D action to SimplerEnv 7D for WidowX.

    6D rotation -> Euler XYZ + [0, pi/2, 0] pitch offset.
    Gripper opens if action[9] < threshold.
    """
    action_pred = np.asarray(action_raw, dtype=np.float32)
    pos = action_pred[:3]
    euler_xyz = (rotate6d_to_euler_xyz(action_pred[3:9])
                 + np.array([0, math.pi / 2, 0])).astype(np.float32)
    gripper_val = 1.0 if action_pred[9] < gripper_threshold else -1.0
    return np.concatenate([pos, euler_xyz, [gripper_val]])


def convert_xvla_action_google_robot(action_raw, current_xyz):
    """
    Convert X-VLA 20D action to SimplerEnv 7D for Google Robot.

    Position is relative from model, converted to absolute by adding current_xyz.
    No pi/2 offset. Gripper opens if action[9] > 0.25.
    Returns (action_7d, new_current_xyz).
    """
    action_pred = np.asarray(action_raw, dtype=np.float32)
    pos = action_pred[:3] + current_xyz
    euler_xyz = rotate6d_to_euler_xyz(action_pred[3:9]).astype(np.float32)
    gripper_val = 1.0 if action_pred[9] > 0.25 else -1.0
    action_7d = np.concatenate([pos, euler_xyz, [gripper_val]])
    return action_7d, pos.copy()


# Image / batch / observation helpers

def preprocess_image(img, target_size=256):
    """
    Convert HWC uint8 image to CHW ImageNet-normalized tensor.

    SimplerEnv cameras are in correct orientation (no flip, unlike LIBERO).
    """
    if img.shape[:2] != (target_size, target_size):
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        img = np.array(pil_img)
    img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    img_tensor = (img_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return img_tensor


def create_simplerenv_batch(image, instruction, domain_id, device, tokenizer,
                            tokenizer_max_length=50, proprio=None):
    """
    Create X-VLA input batch from a SimplerEnv observation.

    Only sends 1 image. Sending image2 (even zeros) causes _prepare_images
    to set mask=True, encoding zero features that corrupt aux_visual_inputs.
    """
    img_tensor = preprocess_image(image).unsqueeze(0).to(device)

    if proprio is not None:
        state = torch.from_numpy(proprio).float().unsqueeze(0).to(device)
    else:
        state = torch.zeros(1, 20, dtype=torch.float32, device=device)

    domain_id_tensor = torch.tensor([domain_id], dtype=torch.int64, device=device)

    tokens = tokenizer(
        instruction,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_max_length,
        return_tensors="pt",
    )

    return {
        "observation.images.image": img_tensor,
        "observation.state": state,
        "observation.language.tokens": tokens["input_ids"].to(device),
        "observation.language.attention_mask": tokens["attention_mask"].bool().to(device),
        "domain_id": domain_id_tensor,
    }


def get_tcp_pose(obs, env=None):
    """
    Get TCP pose [pos(3), quat(4)] from obs or env directly.

    Drawer tasks omit tcp_pose from obs["extra"], so we fall back to
    reading env.tcp.pose.
    """
    if "extra" in obs and "tcp_pose" in obs.get("extra", {}):
        return obs["extra"]["tcp_pose"]
    if env is not None:
        from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        return vectorize_pose(base.tcp.pose)
    raise KeyError("tcp_pose not in obs['extra'] and no env provided for fallback")


def compute_initial_proprio(obs, env=None):
    """
    Compute initial 20D proprioception from ManiSkill2 observation.

    EEF position relative to base (3D), identity 6D rotation (6D),
    gripper=0 (1D), zero padding (10D). The model updates proprio[:10]
    with predicted actions each step.
    """
    tcp_pose = get_tcp_pose(obs, env=env)
    ee_pose_wrt_base = Pose(
        p=obs["agent"]["base_pose"][:3],
        q=obs["agent"]["base_pose"][3:]
    ).inv() * Pose(
        p=tcp_pose[:3],
        q=tcp_pose[3:]
    )
    proprio_10d = np.concatenate([
        ee_pose_wrt_base.p,
        np.array([1, 0, 0, 1, 0, 0, 0], dtype=np.float32)
    ])
    proprio = np.zeros(20, dtype=np.float32)
    proprio[:10] = proprio_10d
    return proprio


def extract_obs_state(obs):
    state = {}
    for key, path in [
        ("tcp_pose", ("extra", "tcp_pose")),
        ("qpos", ("agent", "qpos")),
        ("qvel", ("agent", "qvel")),
        ("base_pose", ("agent", "base_pose")),
    ]:
        try:
            val = obs
            for p in path:
                val = val[p]
            state[key] = val.tolist()
        except (KeyError, AttributeError):
            state[key] = None
    return state


def extract_step_info(info):
    step_info = {}
    for k in STEP_INFO_KEYS:
        if k in info:
            v = info[k]
            step_info[k] = bool(v) if isinstance(v, (bool, np.bool_)) else v
    return step_info


def get_base_env(env):
    return env.unwrapped


def compare_trajectories(actions_a, actions_b):
    """Compare two action trajectories by cosine similarity and XYZ diff."""
    min_len = min(len(actions_a), len(actions_b))
    if min_len == 0:
        return {"cosine": 0.0, "xyz_diff": 0.0}
    a, b = actions_a[:min_len], actions_b[:min_len]
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    cos = float(np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-10))
    xyz_diff = float(np.mean(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)))
    return {"cosine": cos, "xyz_diff": xyz_diff}


# Model loading

def load_xvla_policy(model_type, checkpoint, device="cuda"):
    """
    Load X-VLA policy with config fixes for Google Robot.

    Google Robot checkpoints ship with tokenizer_max_length=1024 and
    empty_cameras=1, causing sequence length > max_len_seq. This patches
    the cached config.json before loading.

    Returns (policy, tokenizer).
    """
    if model_type == "google-robot":
        import huggingface_hub
        cache_dir = huggingface_hub.snapshot_download(checkpoint)
        config_path = Path(cache_dir) / "config.json"
        with open(config_path) as f:
            cfg_json = json.load(f)
        patched = False
        if cfg_json.get("tokenizer_max_length", 50) > 50:
            cfg_json["tokenizer_max_length"] = 50
            patched = True
        if cfg_json.get("empty_cameras", 0) > 0:
            cfg_json["empty_cameras"] = 0
            patched = True
        if patched:
            with open(config_path, "w") as f:
                json.dump(cfg_json, f, indent=2)
        policy = XVLAPolicy.from_pretrained(cache_dir)
    else:
        policy = XVLAPolicy.from_pretrained(checkpoint)

    policy = policy.to(device).eval()

    # 20 for Google Robot ([::2][:10] speedup), 10 for WidowX
    target_steps = 20 if model_type == "google-robot" else 10
    policy.config.n_action_steps = target_steps

    tokenizer = AutoTokenizer.from_pretrained(policy.config.tokenizer_name)
    return policy, tokenizer


def patch_eval_noop(policy):
    """
    Make policy.eval() a no-op when already in eval mode.

    select_action() calls self.eval() every invocation, traversing the
    full module tree. With hooks registered this triggers a PyTorch
    "super(): bad __class__ cell" crash.
    """
    _original = policy.eval
    def _safe_eval():
        if not policy.training:
            return policy
        return _original()
    policy.eval = _safe_eval


def get_hook_target(policy, layer_idx):
    return policy.model.transformer.blocks[layer_idx]


# SAE loading

def load_xvla_sae(sae_dir, layer_idx, device):
    """
    Load a trained SAE checkpoint for an X-VLA transformer layer.

    Returns (sae, act_mean, act_std).
    """
    from experiments.sae_hooks import TopKSAE

    path = os.path.join(sae_dir, f"layer_{layer_idx:02d}", "sae_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"SAE not found: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "sae_state_dict" in ckpt:
        sd = ckpt["sae_state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    input_dim = sd["encoder.weight"].shape[1]
    hidden_dim = sd["encoder.weight"].shape[0]
    k = ckpt.get("config", {}).get("k", 64)

    sae = TopKSAE(input_dim=input_dim, hidden_dim=hidden_dim, k=k)
    sae.load_state_dict(sd)
    sae = sae.to(device).eval()

    mean = ckpt.get("act_mean", ckpt.get("mean", torch.zeros(input_dim))).to(device)
    std = ckpt.get("act_std", ckpt.get("std", torch.ones(input_dim))).to(device)

    return sae, mean, std


def load_concept_features(concept_id_dir, layer_idx, robot, top_n=30):
    """
    Load top concept features from contrastive concept ID results.

    Returns dict of {concept_name: {"features": [...], "scores": [...]}}.
    """
    concept_key = ROBOT_CONCEPT_MAP[robot]
    path = os.path.join(concept_id_dir, f"xvla_concept_id_layer{layer_idx:02d}_{concept_key}.json")

    if not os.path.exists(path):
        print(f"  WARNING: No concept ID file found: {path}")
        return {}

    with open(path) as f:
        data = json.load(f)

    concepts = {}
    for name, info in data.get("concepts", {}).items():
        features = info.get("top_features", [])[:top_n]
        scores = info.get("top_scores", [])[:top_n]
        valid = [(f, s) for f, s in zip(features, scores) if s > 0]
        if valid:
            concepts[name] = {
                "features": [f for f, _ in valid],
                "scores": [s for _, s in valid],
            }
    return concepts


# Hook classes

class ActivationCollector:

    def __init__(self):
        self.activations = defaultdict(list)
        self.handles = []
        self.enabled = False

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if not self.enabled:
                return
            h = output[0] if isinstance(output, tuple) else output
            self.activations[name].append(h.detach().cpu().to(torch.bfloat16).squeeze(0))
        return hook_fn

    def register_hooks(self, policy):
        blocks = policy.model.transformer.blocks
        for i, block in enumerate(blocks):
            handle = block.register_forward_hook(self._make_hook(f"transformer_L{i:02d}"))
            self.handles.append(handle)

    def clear(self):
        self.activations = defaultdict(list)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get_activations(self):
        """Return {name: tensor[n_fwd_passes, seq_len, 1024]}."""
        return {name: torch.stack(acts, dim=0)
                for name, acts in self.activations.items() if acts}


class ZeroAblationHook:

    def __init__(self):
        self.enabled = True
        self.call_count = 0

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        self.call_count += 1
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)


class MeanAblationHook:
    """Replace a layer's output with its exponential running mean."""

    def __init__(self):
        self.enabled = True
        self.running_mean = None
        self.count = 0
        self.call_count = 0

    def _update_mean(self, output):
        h = output[0] if isinstance(output, tuple) else output
        h_mean = h.detach().mean(dim=0, keepdim=True) if h.dim() > 1 else h.detach()
        if self.running_mean is None:
            self.running_mean = h_mean
        else:
            self.running_mean = 0.9 * self.running_mean + 0.1 * h_mean
        self.count += 1

    def __call__(self, module, input, output):
        if not self.enabled:
            self._update_mean(output)
            return output
        self.call_count += 1
        if self.running_mean is not None:
            replacement = self.running_mean.expand_as(
                output[0] if isinstance(output, tuple) else output
            )
            if isinstance(output, tuple):
                return (replacement,) + output[1:]
            return replacement
        return output


class ActivationCaptureHook:

    def __init__(self):
        self.activations = []
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        h = output[0] if isinstance(output, tuple) else output
        self.activations.append(h.detach().clone().cpu())
        return output

    def reset(self):
        self.activations = []


class ActivationInjectionHook:

    def __init__(self, stored, device="cuda"):
        self.stored = stored
        self.device = device
        self.step = 0
        self.enabled = True
        self.injection_count = 0
        self.shape_mismatches = 0

    def __call__(self, module, input, output):
        if not self.enabled or self.step >= len(self.stored):
            return output
        actual = output[0] if isinstance(output, tuple) else output
        injected = self.stored[self.step].to(device=self.device, dtype=actual.dtype)
        self.step += 1
        if injected.shape != actual.shape:
            self.shape_mismatches += 1
            return output
        self.injection_count += 1
        if isinstance(output, tuple):
            return (injected,) + output[1:]
        return injected

    def reset(self):
        self.step = 0
        self.injection_count = 0
        self.shape_mismatches = 0


class TimedActivationInjectionHook:

    def __init__(self, stored, device="cuda", start_step=0, end_step=None):
        self.stored = stored
        self.device = device
        self.start_step = start_step
        self.end_step = end_step if end_step is not None else len(stored)
        self.step = 0
        self.enabled = True
        self.injection_count = 0

    def __call__(self, module, input, output):
        if not self.enabled:
            return output
        idx = self.step
        self.step += 1
        if idx < self.start_step or idx >= self.end_step or idx >= len(self.stored):
            return output
        actual = output[0] if isinstance(output, tuple) else output
        injected = self.stored[idx].to(device=self.device, dtype=actual.dtype)
        if injected.shape != actual.shape:
            return output
        self.injection_count += 1
        if isinstance(output, tuple):
            return (injected,) + output[1:]
        return injected

    def reset(self):
        self.step = 0
        self.injection_count = 0


# Episode runner

def run_episode(policy, env, domain_id, device, tokenizer,
                max_steps=1200, seed=0, collector=None, save_video=False,
                tokenizer_max_length=50, task_name=None, episode_id=None,
                robot_type="widowx", instruction_override=None,
                image_transform_fn=None, collect_step_infos=True):
    """
    Run a single SimplerEnv episode.

    Handles both WidowX and Google Robot with different action conversion.
    For Google Robot, uses manual action buffering with [::2][:10] speedup
    matching the official X-VLA fork.

    Args:
        policy: XVLAPolicy instance.
        env: SimplerEnv gymnasium environment.
        domain_id: 0 (WidowX/Bridge) or 1 (Google Robot/RT1).
        device: torch device.
        tokenizer: HuggingFace tokenizer.
        max_steps: maximum episode length.
        seed: random seed for env.reset().
        collector: ActivationCollector (optional).
        save_video: whether to record frames.
        tokenizer_max_length: from policy.config.tokenizer_max_length.
        task_name: task name for gripper threshold lookup.
        episode_id: episode ID for deterministic object placement.
        robot_type: "widowx" or "google-robot".
        instruction_override: fixed prompt (skips env instruction and updates).
        image_transform_fn: callable applied to image before inference.
        collect_step_infos: whether to extract per-step info dicts.

    Returns:
        dict with keys: success, steps, actions, instruction, obs_states,
        rewards. Optionally: step_infos, episode_stats, frames, activations.
    """
    is_google = (robot_type == "google-robot")
    base_env = get_base_env(env)

    if episode_id is not None:
        obs, _ = env.reset(options={"obj_init_options": {"episode_id": episode_id}})
    else:
        obs, _ = env.reset(seed=seed)

    if instruction_override is not None:
        instruction = instruction_override
    else:
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

    if collector:
        collector.clear()
        collector.enabled = True

    actions = []
    frames = []
    obs_states = []
    rewards = []
    step_infos = []
    step = 0
    success = False
    info = {}
    google_action_buffer = []

    while step < max_steps:
        obs_states.append(extract_obs_state(obs))
        image = get_image_from_maniskill2_obs_dict(base_env, obs)

        if image_transform_fn is not None:
            image = image_transform_fn(image)

        if save_video:
            frames.append(image.copy())

        if is_google:
            if not google_action_buffer:
                batch = create_simplerenv_batch(
                    image, instruction, domain_id, device, tokenizer,
                    tokenizer_max_length=tokenizer_max_length, proprio=proprio)
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
            batch = create_simplerenv_batch(
                image, instruction, domain_id, device, tokenizer,
                tokenizer_max_length=tokenizer_max_length, proprio=proprio)
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

        if collect_step_infos:
            step_infos.append(extract_step_info(info))

        step += 1

        # Update instruction for long-horizon tasks (only when not overridden)
        if instruction_override is None:
            new_instruction = base_env.get_language_instruction()
            if new_instruction != instruction:
                instruction = new_instruction

        if terminated:
            success = True
            break
        if truncated:
            break

    obs_states.append(extract_obs_state(obs))

    if collector:
        collector.enabled = False

    result = {
        "success": bool(success),
        "steps": step,
        "actions": np.array(actions),
        "instruction": instruction,
        "obs_states": obs_states,
        "rewards": rewards,
    }

    if collect_step_infos:
        result["step_infos"] = step_infos
        episode_stats = None
        if info and "episode_stats" in info:
            episode_stats = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v
                             for k, v in info["episode_stats"].items()}
        result["episode_stats"] = episode_stats

    if save_video and frames:
        result["frames"] = frames
    if collector:
        result["activations"] = collector.get_activations()

    return result
