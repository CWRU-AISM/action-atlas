#!/usr/bin/env python3
"""
Shared utilities for SmolVLA MetaWorld experiments.

Provides model loading, episode execution, activation hooks, scene state
extraction, and task management for MetaWorld MT50 experiments.

Uses the actionatlas/vla_interp conda environment (Python 3.12).
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lerobot.envs.metaworld import MetaworldEnv, TASK_DESCRIPTIONS, DIFFICULTY_TO_TASKS  # noqa: F401
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from experiments.hooks import (  # noqa: F401  re-exported for scripts
    ZeroAblationHook as MLPZeroHook,
    ActivationCaptureHook as MLPCaptureHook,
    ActivationInjectionHook as MLPInjectionHook,
)
from experiments.utils import force_free_memory  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = "jadechoghari/smolvla_metaworld"
DEFAULT_RESOLUTION = 480
MAX_STEPS = 400


def get_tasks_from_args(args):
    # Parse task list from CLI args (--tasks or --difficulty)
    if args.tasks:
        return [t.strip() for t in args.tasks.split(",")]
    if args.difficulty:
        tasks = []
        for diff in args.difficulty.split(","):
            diff = diff.strip()
            if diff in DIFFICULTY_TO_TASKS:
                tasks.extend(DIFFICULTY_TO_TASKS[diff])
        return tasks
    return DIFFICULTY_TO_TASKS.get("easy", [])


def get_scene_state(env):
    # Extract TCP, object, and goal positions from MetaWorld env
    uw = env._env.unwrapped if hasattr(env._env, "unwrapped") else env._env
    state = {}
    try:
        full_obs = uw._get_obs()
        state["full_obs"] = full_obs.tolist()
        if hasattr(uw, "tcp_center"):
            state["tcp_pos"] = uw.tcp_center.tolist()
        if len(full_obs) >= 7:
            state["obj_pos"] = full_obs[4:7].tolist()
        if hasattr(uw, "_target_pos"):
            state["goal_pos"] = np.array(uw._target_pos).tolist()
        elif hasattr(uw, "_get_pos_goal"):
            state["goal_pos"] = uw._get_pos_goal().tolist()
    except Exception:
        pass
    return state


def compute_obj_displacement(scene_states):
    # Compute object displacement from first to last scene state
    if (len(scene_states) >= 2
            and "obj_pos" in scene_states[0] and "obj_pos" in scene_states[-1]):
        init_obj = np.array(scene_states[0]["obj_pos"])
        final_obj = np.array(scene_states[-1]["obj_pos"])
        return (final_obj - init_obj).tolist()
    return None


def cosine_similarity(a, b):
    # Cosine similarity between two arrays (flattened)
    a_flat = np.array(a).flatten().astype(np.float64)
    b_flat = np.array(b).flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def load_smolvla_policy(checkpoint=DEFAULT_CHECKPOINT, device="cuda",
                        action_horizon=10):
    """
    Load SmolVLA policy with preprocessors.

    Returns (policy, preprocessor, postprocessor).
    """
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy = policy.to(device).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    policy.config.n_action_steps = min(action_horizon, policy.config.chunk_size)
    policy.reset()

    return policy, preprocessor, postprocessor


def get_layer_modules(policy):
    """
    Get expert and VLM layer module lists from SmolVLA policy.

    Returns (expert_layers, vlm_layers).
    """
    vlm_expert = policy.model.vlm_with_expert
    return vlm_expert.lm_expert.layers, vlm_expert.vlm.model.text_model.layers


class PerTokenCollector:
    """
    Collect per-token activations from SmolVLA layers.

    Expert layers fire multiple times per env step (denoising iterations).
    VLM layers fire once per env step. All iterations are stored.
    """

    def __init__(self, policy):
        self.handles = []
        expert_layers, vlm_layers = get_layer_modules(policy)
        self.expert_layers = expert_layers
        self.vlm_layers = vlm_layers
        self.n_expert = len(expert_layers)
        self.n_vlm = len(vlm_layers)
        self.step_data = {
            "expert": {i: [] for i in range(self.n_expert)},
            "vlm": {},
        }

    def _expert_hook(self, idx):
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.step_data["expert"][idx].append(
                h[0].detach().cpu().to(torch.float16).numpy())
        return fn

    def _vlm_hook(self, idx):
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.step_data["vlm"][idx] = h[0].detach().cpu().to(torch.float16).numpy()
        return fn

    def setup(self):
        for i, layer in enumerate(self.expert_layers):
            self.handles.append(layer.mlp.register_forward_hook(self._expert_hook(i)))
        for i, layer in enumerate(self.vlm_layers):
            self.handles.append(layer.mlp.register_forward_hook(self._vlm_hook(i)))

    def get_step(self):
        data = self.step_data
        self.step_data = {
            "expert": {i: [] for i in range(self.n_expert)},
            "vlm": {},
        }
        return data

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class MeanPoolCollector:
    # Collect mean-pooled activations: (hidden_dim,) per step per layer

    def __init__(self, policy):
        self.handles = []
        expert_layers, vlm_layers = get_layer_modules(policy)
        self.expert_layers = expert_layers
        self.vlm_layers = vlm_layers
        self.n_expert = len(expert_layers)
        self.n_vlm = len(vlm_layers)
        self.step_data = {
            "expert": {i: [] for i in range(self.n_expert)},
            "vlm": {},
        }

    def _expert_hook(self, idx):
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.step_data["expert"][idx].append(
                h[0].detach().cpu().mean(dim=0).to(torch.float16).numpy())
        return fn

    def _vlm_hook(self, idx):
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            self.step_data["vlm"][idx] = h[0].detach().cpu().mean(dim=0).to(torch.float16).numpy()
        return fn

    def setup(self):
        for i, layer in enumerate(self.expert_layers):
            self.handles.append(layer.mlp.register_forward_hook(self._expert_hook(i)))
        for i, layer in enumerate(self.vlm_layers):
            self.handles.append(layer.mlp.register_forward_hook(self._vlm_hook(i)))

    def get_step(self):
        data = self.step_data
        pooled = {"expert": {}, "vlm": data["vlm"]}
        for i, acts in data["expert"].items():
            if acts:
                pooled["expert"][i] = np.mean(acts, axis=0)
        self.step_data = {
            "expert": {i: [] for i in range(self.n_expert)},
            "vlm": {},
        }
        return pooled

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, k=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.k = k
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mean = nn.Parameter(torch.zeros(input_dim), requires_grad=False)

    def encode(self, x):
        z = self.encoder(x)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        sparse_z = torch.zeros_like(z)
        sparse_z.scatter_(-1, topk_idx, topk_vals)
        return sparse_z

    def forward(self, x):
        centered = x - self.mean
        z = self.encode(centered)
        return self.decoder(z) + self.mean


def load_sae(path, device="cpu"):
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ckpt['config']
    sae = SparseAutoencoder(cfg['input_dim'], cfg['hidden_dim'], cfg['k'])
    sae.load_state_dict(ckpt['state_dict'])
    return sae.to(device).eval()


def run_episode(policy, env, preprocessor, postprocessor, device,
                collector=None, save_video=False, task_override=None,
                image_transform_fn=None):
    """
    Run a single MetaWorld episode.

    Args:
        policy: SmolVLAPolicy instance.
        env: MetaworldEnv instance.
        preprocessor: lerobot preprocessor.
        postprocessor: lerobot postprocessor.
        device: torch device.
        collector: PerTokenCollector or MeanPoolCollector (optional).
        save_video: whether to record frames.
        task_override: override task description for counterfactual prompts.
        image_transform_fn: callable applied to pixel observation before preprocessing.

    Returns dict with: success, n_steps, actions, agent_pos_trajectory,
        scene_states, obj_displacement. Optionally: activations, frames.
    """
    policy.reset()
    observation, info = env.reset()

    task_desc = task_override if task_override is not None else env.task_description

    frames = []
    actions = []
    agent_pos_traj = []
    scene_states = [get_scene_state(env)]
    expert_acts = {i: [] for i in range(collector.n_expert)} if collector else {}
    vlm_acts = {i: [] for i in range(collector.n_vlm)} if collector else {}
    step_info = {}

    for step in range(MAX_STEPS):
        if save_video:
            frames.append(observation["pixels"].copy())

        if "agent_pos" in observation:
            agent_pos_traj.append(observation["agent_pos"].tolist())

        if image_transform_fn is not None:
            observation = dict(observation)
            observation["pixels"] = image_transform_fn(observation["pixels"])

        obs_tensor = preprocess_observation(observation)
        obs_tensor["task"] = [task_desc]
        obs_tensor = preprocessor(obs_tensor)

        with torch.inference_mode():
            action = policy.select_action(obs_tensor)

        if collector:
            sd = collector.get_step()
            has_data = any(
                (isinstance(v, list) and len(v) > 0)
                or (isinstance(v, np.ndarray) and v.size > 0)
                for v in sd["expert"].values()
            )
            if has_data:
                for i, act_data in sd["expert"].items():
                    if isinstance(act_data, list) and act_data:
                        expert_acts[i].append(np.stack(act_data, axis=0))
                    elif isinstance(act_data, np.ndarray) and act_data.size > 0:
                        expert_acts[i].append(act_data)
                for i, a in sd["vlm"].items():
                    vlm_acts[i].append(a)

        action_out = postprocessor(action)
        action_np = action_out.cpu().numpy() if isinstance(action_out, torch.Tensor) else action_out
        if action_np.ndim == 2:
            action_np = action_np[0]
        action_4d = action_np[:4].copy()
        actions.append(action_4d)

        observation, reward, terminated, truncated, step_info = env.step(action_4d)
        scene_states.append(get_scene_state(env))

        if terminated or truncated:
            break

    success = step_info.get("is_success", False) if actions else False

    result = {
        "success": bool(success),
        "n_steps": len(actions),
        "actions": np.array(actions),
        "agent_pos_trajectory": agent_pos_traj,
        "scene_states": scene_states,
        "obj_displacement": compute_obj_displacement(scene_states),
    }

    if collector and expert_acts:
        result["activations"] = {}
        for i in range(collector.n_expert):
            if expert_acts[i]:
                result["activations"][f"expert_L{i:02d}"] = np.stack(expert_acts[i], axis=0)
        for i in range(collector.n_vlm):
            if vlm_acts[i]:
                result["activations"][f"vlm_L{i:02d}"] = np.array(vlm_acts[i])

    if save_video:
        result["frames"] = frames

    return result


def create_env(task_name, resolution=DEFAULT_RESOLUTION):
    return MetaworldEnv(
        task=task_name,
        obs_type="pixels_agent_pos",
        observation_width=resolution,
        observation_height=resolution,
    )


def save_video_frames(filepath, frames, fps=5):
    try:
        import imageio
        imageio.mimsave(str(filepath), frames, fps=fps)
    except Exception:
        pass
