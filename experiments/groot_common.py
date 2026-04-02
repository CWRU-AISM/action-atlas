#!/usr/bin/env python3
"""
GR00T N1.5 model utilities: loading, hook paths, episode runner.

Works with both N1.5 (LIBERO) and N1.6 (RoboCasa) model versions.

Hook paths:
  N1.5 Eagle: model.backbone.eagle_model.language_model.model.layers[i]
  N1.5 DiT:   model.action_head.diffusion_model.transformer_blocks[i] (16 blocks)
  N1.6 Eagle: model.backbone.model.language_model.model.layers[i]
  N1.6 DiT:   model.action_head.model.transformer_blocks[i] (32 blocks)
"""

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "lerobot" / "src"))

# Re-export shared utilities so existing imports don't break
from experiments.utils import (
    force_free_memory, save_video, get_scene_state, summarize_scene,
    compare_trajectories as compare_traj, top_moved_objects,
    ImagePerturbations, get_standard_perturbations,
)
from experiments.hooks import (
    ZeroAblationHook, MeanAblationHook,
    ActivationCaptureHook, ActivationInjectionHook, NullInjectionHook,
    ActivationCollector,
)


# GR00T-specific ActivationCollector with register_hooks for eagle/dit/vlsa

class GR00TActivationCollector(ActivationCollector):
    """ActivationCollector with GR00T-specific hook registration."""

    def register_hooks(self, model, eagle_layers_idx=None, dit_layers_idx=None,
                       vl_sa_layers_idx=None):
        """Register hooks on GR00T layers (auto-detects N1.5 vs N1.6)."""
        eagle_layers = get_groot_eagle_layers(model)
        dit_blocks = get_groot_dit_blocks(model)
        vl_sa_blocks = get_groot_vl_self_attention_blocks(model)

        if eagle_layers_idx is None:
            eagle_layers_idx = list(range(len(eagle_layers)))
        if dit_layers_idx is None:
            dit_layers_idx = list(range(len(dit_blocks)))
        if vl_sa_layers_idx is None:
            vl_sa_layers_idx = list(range(len(vl_sa_blocks)))

        for i in eagle_layers_idx:
            if i < len(eagle_layers):
                self.register(eagle_layers[i], f"eagle_lm_L{i:02d}", gated=False)

        for i in vl_sa_layers_idx:
            if i < len(vl_sa_blocks):
                self.register(vl_sa_blocks[i], f"vl_sa_L{i:02d}", gated=False)

        for i in dit_layers_idx:
            if i < len(dit_blocks):
                self.register(dit_blocks[i], f"dit_L{i:02d}", gated=True)

        n_eagle = len([i for i in eagle_layers_idx if i < len(eagle_layers)])
        n_vl_sa = len([i for i in vl_sa_layers_idx if i < len(vl_sa_blocks)])
        n_dit = len([i for i in dit_layers_idx if i < len(dit_blocks)])
        print(f"Registered {len(self.handles)} hooks "
              f"(Eagle: {n_eagle}, VL-SA: {n_vl_sa}, DiT: {n_dit})")


# Layer access (version-aware)

def get_groot_eagle_layers(model):
    """Get Eagle VLM language layers (N1.5 and N1.6)."""
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        for path_attr in ['eagle_model', 'model']:
            if hasattr(backbone, path_attr):
                eagle = getattr(backbone, path_attr)
                if hasattr(eagle, 'language_model') and hasattr(eagle.language_model, 'model'):
                    lm = eagle.language_model.model
                    if hasattr(lm, 'layers'):
                        return list(lm.layers)
    return []


def get_groot_dit_blocks(model):
    """Get DiT action head transformer blocks (N1.5: 16, N1.6: 32)."""
    if hasattr(model, 'action_head'):
        ah = model.action_head
        if hasattr(ah, 'diffusion_model') and hasattr(ah.diffusion_model, 'transformer_blocks'):
            return list(ah.diffusion_model.transformer_blocks)
        if hasattr(ah, 'model') and hasattr(ah.model, 'transformer_blocks'):
            return list(ah.model.transformer_blocks)
    return []


def get_groot_vl_self_attention_blocks(model):
    """Get VL self-attention blocks bridging Eagle VLM to DiT (N1.5: 4 blocks)."""
    if hasattr(model, 'action_head'):
        ah = model.action_head
        if hasattr(ah, 'vl_self_attention') and hasattr(ah.vl_self_attention, 'transformer_blocks'):
            return list(ah.vl_self_attention.transformer_blocks)
    return []


# Model loading and configuration

SUITE_MAX_STEPS = {
    "libero_spatial": 220, "libero_object": 280, "libero_goal": 300,
    "libero_10": 520, "libero_long": 520,
}

SUITE_CHECKPOINTS = {
    "libero_spatial": "liorbenhorin-nv/groot-libero_spatial-128_20000",
    "libero_object": "liorbenhorin-nv/groot-libero_object-64_40000",
    "libero_goal": "aractingi/libero-groot-goal",
    "libero_10": "aractingi/groot-libero-10",
    "libero_long": "aractingi/groot-libero-10",
}

COUNTERFACTUAL_PROMPTS = {
    "null_prompt": "", "random": "blah foo bar baz",
    "negation": "do not {task}", "opposite": "undo the task",
    "generic": "move the robot arm",
}


def load_metadata_stats(checkpoint_path):
    """Load normalization statistics from checkpoint's metadata.json."""
    meta_path = Path(checkpoint_path) / "experiment_cfg" / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    emb_data = metadata.get("new_embodiment", {})
    stats = emb_data.get("statistics", {})
    result = {}

    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper", "extra"]
    if "state" in stats:
        result["state_min"] = np.array([stats["state"][k]["min"][0] for k in state_keys])
        result["state_max"] = np.array([stats["state"][k]["max"][0] for k in state_keys])

    action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    if "action" in stats:
        result["action_min"] = np.array([stats["action"][k]["min"][0] for k in action_keys])
        result["action_max"] = np.array([stats["action"][k]["max"][0] for k in action_keys])

    return result


def normalize_state(state_8d, stats):
    s_min, s_max = stats["state_min"], stats["state_max"]
    denom = s_max - s_min
    mask = denom != 0
    safe_denom = np.where(mask, denom, 1.0)
    normed = 2.0 * (state_8d - s_min) / safe_denom - 1.0
    return np.where(mask, normed, 0.0)


def denormalize_action(action_7d, stats):
    a_min, a_max = stats["action_min"], stats["action_max"]
    denom = a_max - a_min
    mask = denom != 0
    safe_denom = np.where(mask, denom, 1.0)
    raw = (action_7d + 1.0) * 0.5 * safe_denom + a_min
    return np.where(mask, raw, a_min)


def quat2axisangle(quat):
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(w)) / den


def get_libero_state_groot(obs, env=None):
    """Convert LIBERO observation to GR00T 8D state."""
    eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
    eef_quat = obs.get("robot0_eef_quat", np.array([0, 0, 0, 1]))
    axis_angle = quat2axisangle(eef_quat)
    gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))
    return np.concatenate([eef_pos, axis_angle, gripper_qpos[:1], gripper_qpos[1:2]])


def build_groot_inputs(images, state_8d, task_desc, stats, eagle_processor, device):
    """Build input dict for GR00T model.get_action()."""
    from einops import rearrange

    agentview = images["agentview"]
    wrist = images.get("wrist", agentview)

    video_frames = np.stack([agentview, wrist], axis=0)
    video = video_frames.transpose(0, 3, 1, 2)[np.newaxis, np.newaxis, ...]

    t, v, c, h, w = video[0].shape
    flat = rearrange(video[0], "t v c h w -> (t v) h w c")
    pil_images = [Image.fromarray(flat[i]) for i in range(t * v)]

    lang_formatted = str([task_desc])
    text_content = [{"type": "text", "text": lang_formatted}]
    image_content = [{"type": "image", "image": img} for img in pil_images]
    conv = [{"role": "user", "content": image_content + text_content}]

    text_list = [eagle_processor.apply_chat_template(
        conv, tokenize=False, add_generation_prompt=True
    )]
    img_inputs, _ = eagle_processor.process_vision_info(conv)

    eagle_inputs = eagle_processor(
        text=text_list, images=img_inputs,
        images_kwargs={"min_dynamic_tiles": 1, "max_dynamic_tiles": 1, "use_thumbnail": False},
        return_tensors="pt", padding=True,
    )

    normed_state = normalize_state(state_8d, stats)
    state_padded = np.zeros(64, dtype=np.float32)
    state_padded[:8] = normed_state
    state_tensor = torch.tensor(state_padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    state_mask = torch.zeros(1, 1, 64, dtype=torch.bool)
    state_mask[:, :, :8] = True
    embodiment_id = torch.tensor([31], dtype=torch.long)

    inputs = {
        "state": state_tensor.to(device),
        "state_mask": state_mask.to(device),
        "embodiment_id": embodiment_id.to(device),
    }
    for k, v in eagle_inputs.items():
        inputs[f"eagle_{k}"] = v.to(device) if isinstance(v, torch.Tensor) else v

    return inputs


def build_eagle_processor():
    """Build the Eagle processor for VLM tokenization."""
    from transformers import AutoProcessor
    from lerobot.policies.groot.utils import ensure_eagle_cache_ready
    from lerobot.utils.constants import HF_LEROBOT_HOME

    vendor_dir = str((
        Path(__file__).resolve().parent.parent
        / "lerobot" / "src" / "lerobot" / "policies" / "groot" / "eagle2_hg_model"
    ).resolve())
    tokenizer_repo = "lerobot/eagle2hg-processor-groot-n1p5"
    cache_dir = HF_LEROBOT_HOME / tokenizer_repo

    ensure_eagle_cache_ready(vendor_dir, cache_dir, tokenizer_repo)
    proc = AutoProcessor.from_pretrained(str(cache_dir), trust_remote_code=True, use_fast=True)
    proc.tokenizer.padding_side = "left"
    return proc


def load_groot_n15(checkpoint, device):
    """Load GR00T N1.5 model."""
    from lerobot.policies.groot.groot_n1 import GR00TN15

    print(f"Loading GR00T N1.5 from {checkpoint}...")
    start = time.time()
    model = GR00TN15.from_pretrained(
        checkpoint, tune_visual=False, tune_llm=False,
        tune_projector=False, tune_diffusion_model=False,
    )
    model = model.to(device)
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"
    params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {time.time() - start:.1f}s ({params/1e9:.2f}B params)")
    print(f"  DiT: {len(get_groot_dit_blocks(model))}, Eagle: {len(get_groot_eagle_layers(model))}")
    return model


# Environment setup

def setup_libero_envs(suite):
    """Set up LIBERO benchmark environments for a suite."""
    from libero.libero import benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    suite_key = "libero_10" if suite in ("libero_10", "libero_long") else suite
    task_suite = benchmark_dict[suite_key]()
    tasks = [(i, task_suite.get_task(i), task_suite.get_task(i).language)
             for i in range(task_suite.n_tasks)]
    return task_suite, tasks


def create_libero_env(task, resolution=256):
    """Create a single LIBERO OffScreenRenderEnv for a task."""
    from experiments.libero_utils import get_libero_env
    return get_libero_env(task, resolution=resolution)


# Episode runner

def run_groot_episode(model, env, task_desc, device, max_steps, eagle_processor, stats,
                      collector=None, save_video=False, action_horizon=16,
                      perturbation_fn=None, force_fresh_actions=False,
                      scene_state_fn=None):
    """Run a single episode with GR00T model."""
    from experiments.libero_utils import get_libero_images, set_control_mode

    obs = env.reset()
    set_control_mode(env, "relative")

    done = False
    step = 0
    frames = []
    actions_list = []
    eef_trajectory = []
    scene_states = []
    action_queue = deque()

    if collector:
        collector.clear()

    while not done and step < max_steps:
        images = get_libero_images(obs, target_size=(256, 256))
        if "wrist" in images:
            images["wrist"] = images["wrist"][::-1, ::-1].copy()

        state_8d = get_libero_state_groot(obs, env)

        if save_video:
            frames.append(images.get("agentview", np.zeros((256, 256, 3), dtype=np.uint8)))

        eef_trajectory.append(obs.get("robot0_eef_pos", np.zeros(3)).copy())

        if scene_state_fn is not None:
            scene_states.append(scene_state_fn(env))

        if collector:
            collector.new_step()

        if force_fresh_actions:
            action_queue.clear()

        if len(action_queue) == 0:
            model_images = images
            if perturbation_fn is not None:
                model_images = {k: perturbation_fn(v.copy()) for k, v in images.items()}

            inputs = build_groot_inputs(model_images, state_8d, task_desc, stats, eagle_processor, device)

            with torch.inference_mode():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    outputs = model.get_action(inputs)

            action_pred = outputs["action_pred"]
            for t in range(action_pred.shape[1]):
                a_normed = action_pred[0, t, :7].cpu().float().numpy()
                action_queue.append(denormalize_action(a_normed, stats))

        action_7d = action_queue.popleft()
        actions_list.append(action_7d.copy())
        obs, reward, done, info = env.step(action_7d)
        step += 1

    success = env.check_success()
    result = {
        "success": bool(success), "steps": step,
        "actions": np.array(actions_list), "eef_trajectory": np.array(eef_trajectory),
    }
    if save_video:
        result["frames"] = frames
    if scene_states:
        result["scene_states"] = scene_states
    if collector:
        result["activations"] = collector.get_activations()
    return result


# Backward compat alias
get_scene_state_libero = get_scene_state
