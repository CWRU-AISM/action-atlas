"""
Model adapters for VLA interpretability experiments.

Each adapter wraps model-specific loading, layer access, batch creation,
action conversion, and environment setup behind a uniform interface.
Use get_adapter(model_name) to obtain an adapter instance.

Supported models:
  - xvla:    X-VLA (24 transformer blocks, LIBERO + SimplerEnv)
  - smolvla: SmolVLA (expert + VLM layers, LIBERO + MetaWorld)
  - groot:   GR00T N1.5 (Eagle LM + VL-SA + DiT, LIBERO + RoboCasa)
  - pi05:    Pi0.5 (PaliGemma + Gemma Expert, LIBERO)
  - openvla: OpenVLA-OFT (7B Llama, LIBERO)
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "lerobot" / "src"))


# Base adapter

def _resolve_checkpoint(checkpoint: str) -> str:
    """
    Resolve checkpoint path: if it's a local path that exists, make it absolute.
    Otherwise return as-is (treated as HuggingFace repo ID)."""
    p = Path(checkpoint)
    if p.exists():
        return str(p.resolve())
    return checkpoint


class ModelAdapter(ABC):
    # Base interface for VLA model adapters
    ...

    name: str
    model: Any = None
    device: str = "cuda"

    @abstractmethod
    def load_model(self, checkpoint: str, device: str = "cuda") -> Any:
        # Load model from checkpoint, return the model object
        ...

    @abstractmethod
    def get_layer_groups(self) -> Dict[str, List]:
        """
        Return named groups of hookable layers.

        Example: {"transformer": [block0, block1, ...], "vlm": [layer0, ...]}
        """

    def get_all_layers(self) -> List[Tuple[str, Any]]:
        # Flat list of (label, module) for all hookable layers
        ...
        result = []
        for group_name, layers in self.get_layer_groups().items():
            for i, layer in enumerate(layers):
                result.append((f"{group_name}_L{i}", layer))
        return result

    @abstractmethod
    def create_env(self, task, suite: str, resolution: int = 256, **kwargs):
        """
        Create an environment for the given task.

        Returns: (env, task_description, env_meta)
        """

    @abstractmethod
    def setup_suite(self, suite: str, **kwargs):
        """
        Set up a LIBERO task suite.

        Returns: (task_suite, tasks_list) where tasks_list is
                 [(task_idx, task_obj, task_desc), ...]
        """

    @abstractmethod
    def run_episode(self, env, task_desc: str, max_steps: int,
                    save_video: bool = False, perturbation_fn=None, **kwargs) -> dict:
        """
        Run a single episode. Returns dict with at least:
        - success: bool
        - steps: int
        - actions: np.ndarray
        - frames: list (if save_video)
        - scene_states: list (if available)
        """

    @property
    def default_checkpoints(self) -> Dict[str, str]:
        # Map suite name -> default checkpoint path
        ...
        return {}

    @property
    def suite_max_steps(self) -> Dict[str, int]:
        return {
            "libero_spatial": 220,
            "libero_object": 280,
            "libero_goal": 300,
            "libero_10": 520,
            "libero_long": 520,
        }


# X-VLA

class XVLAAdapter(ModelAdapter):
    name = "xvla"

    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.env_preprocessor = None
        self.env_postprocessor = None
        self._env_cache = {}  # (suite, max_steps) -> {task_id: env}

    @property
    def default_checkpoints(self):
        return {
            "libero_spatial": "lerobot/xvla-libero",
            "libero_object": "lerobot/xvla-libero",
            "libero_goal": "lerobot/xvla-libero",
            "libero_10": "lerobot/xvla-libero",
        }

    def load_model(self, checkpoint="lerobot/xvla-libero", device="cuda"):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        self.device = device
        checkpoint = _resolve_checkpoint(checkpoint)
        self.policy = XVLAPolicy.from_pretrained(checkpoint)
        self.policy.eval().to(device)
        self.model = self.policy
        self._checkpoint = checkpoint
        return self.policy

    def get_layer_groups(self):
        blocks = list(self.policy.model.transformer.blocks)
        return {"transformer": blocks}

    def _ensure_processors(self, suite="libero_object", max_steps=280):
        if self.preprocessor is not None:
            return
        import gymnasium as gym
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.envs.factory import make_env_config, make_env_pre_post_processors

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=self._checkpoint,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )
        env_cfg = make_env_config(
            env_type="libero", task=suite,
            control_mode="absolute", episode_length=max_steps,
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy.config,
        )

    def setup_suite(self, suite="libero_object", **kwargs):
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()
        suite_key = "libero_10" if suite == "libero_long" else suite
        task_suite = bench[suite_key]()
        tasks = [(i, task_suite.get_task(i), task_suite.get_task(i).language)
                 for i in range(task_suite.n_tasks)]
        return task_suite, tasks

    def create_env(self, task, suite="libero_object", resolution=256, max_steps=280, **kwargs):
        import gymnasium as gym
        from lerobot.envs.libero import create_libero_envs
        self._ensure_processors(suite, max_steps)

        cache_key = (suite, max_steps)
        if cache_key not in self._env_cache:
            envs_dict = create_libero_envs(
                task=suite, n_envs=1, control_mode="absolute",
                episode_length=max_steps,
                env_cls=gym.vector.SyncVectorEnv,
                gym_kwargs={"obs_type": "pixels_agent_pos"},
            )
            self._env_cache[cache_key] = envs_dict[suite]

        env = self._env_cache[cache_key][task]
        if hasattr(env, "envs"):
            desc = env.envs[0].task_description
        else:
            desc = getattr(env, "task_description", f"task_{task}")
        return env, desc, {}

    def run_episode(self, env, task_desc, max_steps=280,
                    save_video=False, perturbation_fn=None, **kwargs):
        from lerobot.envs.utils import preprocess_observation
        seed = kwargs.get("seed", 42)
        inner_env = kwargs.get("inner_env", None)
        from experiments.utils import get_scene_state

        self.policy.reset()
        obs, _ = env.reset(seed=seed)
        actions, frames, scene_states = [], [], []
        is_success = False

        for step in range(max_steps):
            if inner_env is not None:
                scene_states.append(get_scene_state(inner_env))

            if save_video and step % 3 == 0 and "pixels" in obs and "image" in obs["pixels"]:
                frame = obs["pixels"]["image"]
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 4:
                        frame = frame[0]
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
                    frame = frame[::-1, ::-1]
                    frames.append(frame)

            obs_proc = preprocess_observation(obs)
            obs_proc["task"] = [task_desc]
            obs_proc = self.env_preprocessor(obs_proc)
            obs_proc = self.preprocessor(obs_proc)

            with torch.inference_mode():
                action = self.policy.select_action(obs_proc)

            actions.append(action.cpu().numpy().flatten())
            action_proc = self.postprocessor(action)
            action_t = self.env_postprocessor({"action": action_proc})
            obs, reward, term, trunc, info = env.step(action_t["action"].cpu().numpy())

            success_val = info.get("is_success", False)
            if isinstance(success_val, (list, np.ndarray)):
                success_val = bool(success_val[0])
            if success_val:
                is_success = True
            if term[0] or trunc[0]:
                break

        result = {"success": is_success, "steps": len(actions), "actions": np.array(actions)}
        if save_video:
            result["frames"] = frames
        if scene_states:
            result["scene_states"] = scene_states
        return result


# SmolVLA

class SmolVLAAdapter(ModelAdapter):
    name = "smolvla"

    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self._suite_cache = {}  # suite -> task_suite
        self.postprocessor = None
        self.env_preprocessor = None

    @property
    def default_checkpoints(self):
        return {
            "libero_spatial": "HuggingFaceVLA/smolvla_libero",
            "libero_object": "HuggingFaceVLA/smolvla_libero",
            "libero_goal": "HuggingFaceVLA/smolvla_libero",
            "libero_10": "HuggingFaceVLA/smolvla_libero",
            "metaworld": "jadechoghari/smolvla_metaworld",
        }

    def load_model(self, checkpoint="HuggingFaceVLA/smolvla_libero", device="cuda"):
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor.env_processor import LiberoProcessorStep
        from lerobot.processor.pipeline import PolicyProcessorPipeline

        self.device = device
        checkpoint = _resolve_checkpoint(checkpoint)
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy = self.policy.to(device).eval()
        self.model = self.policy
        self._checkpoint = checkpoint

        self.env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return self.policy

    def get_layer_groups(self):
        vlm_expert = self.policy.model.vlm_with_expert
        # Hook .mlp within each layer, not the layer itself.
        # SmolVLA interleaves VLM/expert layers internally, so the
        # top-level DecoderLayer forward is never called directly.
        expert = [layer.mlp for layer in vlm_expert.lm_expert.layers]
        vlm = [layer.mlp for layer in vlm_expert.vlm.model.text_model.layers]
        return {"expert": expert, "vlm": vlm}

    def setup_suite(self, suite="libero_object", **kwargs):
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()
        suite_key = "libero_10" if suite == "libero_long" else suite
        task_suite = bench[suite_key]()
        tasks = [(i, task_suite.get_task(i), task_suite.get_task(i).language)
                 for i in range(task_suite.n_tasks)]
        return task_suite, tasks

    def create_env(self, task, suite="libero_object", resolution=256, **kwargs):
        from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS

        if suite not in self._suite_cache:
            from libero.libero import benchmark
            bench = benchmark.get_benchmark_dict()
            suite_key = "libero_10" if suite == "libero_long" else suite
            self._suite_cache[suite] = bench[suite_key]()

        task_suite = self._suite_cache[suite]
        task_idx = task if isinstance(task, int) else 0
        ep_idx = kwargs.get("episode_index", 0)

        libero_env = LiberoEnv(
            task_suite=task_suite, task_id=task_idx, task_suite_name=suite,
            obs_type="pixels_agent_pos", observation_width=resolution,
            observation_height=resolution, init_states=True, episode_index=ep_idx,
        )
        return libero_env, task_suite.get_task(task_idx).language, {"task_suite": task_suite}

    def run_episode(self, env, task_desc, max_steps=280,
                    save_video=False, perturbation_fn=None, **kwargs):
        from lerobot.envs.utils import preprocess_observation

        self.policy.reset()
        observation, info = env.reset()
        step_info = {}
        actions = []

        for step in range(max_steps):
            obs_tensor = preprocess_observation(observation)
            # Add batch dim to robot state if needed
            if "observation.robot_state" in obs_tensor:
                rs = obs_tensor["observation.robot_state"]
                for gk in rs:
                    for sk in rs[gk]:
                        t = rs[gk][sk]
                        if isinstance(t, torch.Tensor) and t.ndim <= 2:
                            rs[gk][sk] = t.unsqueeze(0)

            obs_tensor["task"] = [env.task_description]
            obs_tensor = self.env_preprocessor(obs_tensor)
            obs_tensor = self.preprocessor(obs_tensor)

            with torch.inference_mode():
                action = self.policy.select_action(obs_tensor)

            action_out = self.postprocessor(action)
            action_np = action_out.cpu().numpy() if isinstance(action_out, torch.Tensor) else action_out
            if action_np.ndim == 2:
                action_np = action_np[0]
            actions.append(action_np[:7].copy())

            observation, reward, terminated, truncated, step_info = env.step(action_np[:7])
            if terminated or truncated:
                break

        success = step_info.get("is_success", False)
        return {
            "success": bool(success),
            "steps": len(actions),
            "actions": np.array(actions) if actions else np.zeros((0, 7)),
        }


# GR00T N1.5

class GR00TAdapter(ModelAdapter):
    name = "groot"

    def __init__(self):
        self.model = None
        self.eagle_processor = None
        self._env_cache = {}  # (suite, task_idx) -> (env, desc)
        self.stats = None

    @property
    def default_checkpoints(self):
        return {
            "libero_spatial": "liorbenhorin-nv/groot-libero_spatial-128_20000",
            "libero_object": "liorbenhorin-nv/groot-libero_object-64_40000",
            "libero_goal": "aractingi/libero-groot-goal",
            "libero_10": "aractingi/groot-libero-10",
            "libero_long": "aractingi/groot-libero-10",
        }

    def load_model(self, checkpoint="aractingi/libero-groot-goal", device="cuda"):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from experiments.groot_common import build_eagle_processor, load_metadata_stats
        self.device = device
        checkpoint = _resolve_checkpoint(checkpoint)

        # Try lerobot GrootPolicy first (HF checkpoints), fall back to GR00TN15
        try:
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            policy = GrootPolicy.from_pretrained(checkpoint, strict=False)
            policy = policy.to(device).eval()
            self.model = policy._groot_model
            self._policy = policy
        except Exception:
            from experiments.groot_common import load_groot_n15
            self.model = load_groot_n15(checkpoint, device)

        self.eagle_processor = build_eagle_processor()
        self.stats = load_metadata_stats(checkpoint)
        return self.model

    def get_layer_groups(self):
        from experiments.groot_common import (
            get_groot_eagle_layers, get_groot_dit_blocks,
            get_groot_vl_self_attention_blocks,
        )
        return {
            "eagle": get_groot_eagle_layers(self.model),
            "vlsa": get_groot_vl_self_attention_blocks(self.model),
            "dit": get_groot_dit_blocks(self.model),
        }

    def setup_suite(self, suite="libero_goal", **kwargs):
        from experiments.groot_common import setup_libero_envs
        task_suite, tasks = setup_libero_envs(suite)
        return task_suite, tasks

    def create_env(self, task, suite="libero_goal", resolution=256, **kwargs):
        # GR00T envs are OffScreenRenderEnv - cache per (suite, task)
        task_key = id(task) if not isinstance(task, int) else task
        cache_key = (suite, task_key)
        if cache_key not in self._env_cache:
            from experiments.groot_common import create_libero_env
            env, task_desc = create_libero_env(task, resolution=resolution)
            self._env_cache[cache_key] = (env, task_desc)
        env, task_desc = self._env_cache[cache_key]
        return env, task_desc, {}

    def run_episode(self, env, task_desc, max_steps=300,
                    save_video=False, perturbation_fn=None, **kwargs):
        from experiments.groot_common import run_groot_episode, get_scene_state_libero

        collector = kwargs.get("collector", None)
        action_horizon = kwargs.get("action_horizon", 16)
        force_fresh = kwargs.get("force_fresh_actions", False)

        result = run_groot_episode(
            self.model, env, task_desc,
            torch.device(self.device), max_steps,
            self.eagle_processor, self.stats,
            collector=collector,
            save_video=save_video,
            action_horizon=action_horizon,
            perturbation_fn=perturbation_fn,
            force_fresh_actions=force_fresh,
            scene_state_fn=get_scene_state_libero,
        )
        return result


# Pi0.5

class Pi05Adapter(ModelAdapter):
    name = "pi05"

    def __init__(self):
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.env_preprocessor = None
        self._env_cache = {}  # (suite, max_steps) -> {task_id: env}
        self.env_postprocessor = None

    @property
    def default_checkpoints(self):
        return {
            "libero_spatial": "checkpoints/pi05_libero_finetuned",
            "libero_object": "checkpoints/pi05_libero_finetuned",
            "libero_goal": "checkpoints/pi05_libero_finetuned",
            "libero_10": "checkpoints/pi05_libero_finetuned",
        }

    def load_model(self, checkpoint="checkpoints/pi05_libero_finetuned", device="cuda"):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("PYTORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors

        self.device = device
        checkpoint = _resolve_checkpoint(checkpoint)
        self.policy = PI05Policy.from_pretrained(checkpoint)
        self.policy.eval().to(device)
        self.policy.config.device = device
        self.model = self.policy
        self._checkpoint = checkpoint

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return self.policy

    def _ensure_env_processors(self, suite, max_steps):
        if self.env_preprocessor is not None:
            return
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        env_cfg = LiberoEnvConfig(task=suite, observation_height=256, observation_width=256)
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=env_cfg, policy_cfg=self.policy.config,
        )

    def get_layer_groups(self):
        # Pi0.5: paligemma_with_expert.gemma_expert.model.layers (18 Gemma layers)
        # The PaliGemma VLM backbone also has layers but the expert pathway
        # is the primary target for interpretability (action generation).
        pwe = self.policy.model.paligemma_with_expert
        expert_layers = list(pwe.gemma_expert.model.layers)
        pali_layers = list(pwe.paligemma.model.language_model.layers)
        return {"expert": expert_layers, "paligemma": pali_layers}

    def setup_suite(self, suite="libero_object", **kwargs):
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()
        suite_key = "libero_10" if suite == "libero_long" else suite
        task_suite = bench[suite_key]()
        tasks = [(i, task_suite.get_task(i), task_suite.get_task(i).language)
                 for i in range(task_suite.n_tasks)]
        return task_suite, tasks

    def create_env(self, task, suite="libero_object", resolution=256, max_steps=300, **kwargs):
        cache_key = (suite, max_steps)
        if cache_key not in self._env_cache:
            from lerobot.envs.factory import make_env
            from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig

            self._ensure_env_processors(suite, max_steps)
            env_cfg = LiberoEnvConfig(task=suite, observation_height=256, observation_width=256)
            envs_dict = make_env(cfg=env_cfg, n_envs=1, use_async_envs=False)
            if isinstance(envs_dict, dict):
                self._env_cache[cache_key] = envs_dict[suite]
            else:
                self._env_cache[cache_key] = {0: envs_dict}

        env = self._env_cache[cache_key][task]
        desc = ""
        if hasattr(env, "envs"):
            desc = getattr(env.envs[0], "task_description", f"task_{task}")
        return env, desc, {}

    def run_episode(self, env, task_desc, max_steps=300,
                    save_video=False, perturbation_fn=None, **kwargs):
        from lerobot.envs.utils import preprocess_observation, add_envs_task
        seed = kwargs.get("seed", 42)

        self.policy.reset()
        obs, _ = env.reset(seed=seed)
        actions, frames = [], []
        success = False

        for step in range(max_steps):
            if save_video and "pixels" in obs and "image" in obs["pixels"]:
                frame = obs["pixels"]["image"]
                if hasattr(frame, "cpu"):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4:
                    frame = frame[0]
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame[::-1, ::-1].copy())

            obs_proc = preprocess_observation(obs)
            obs_proc = add_envs_task(env, obs_proc)
            obs_proc = self.env_preprocessor(obs_proc)
            obs_proc = self.preprocessor(obs_proc)

            with torch.inference_mode():
                action = self.policy.select_action(obs_proc)

            action = self.postprocessor(action)
            action_t = self.env_postprocessor({"action": action})
            action_np = action_t["action"].cpu().numpy()
            action_flat = action_np.squeeze(0) if action_np.ndim > 1 else action_np
            actions.append(action_flat.tolist())

            obs, _, term, trunc, _ = env.step(action_np)
            if term[0]:
                success = True
                break

        result = {"success": success, "steps": len(actions),
                  "actions": np.array(actions)}
        if save_video:
            result["frames"] = frames
        return result


# OpenVLA-OFT (requires openvla-oft conda env)

class OpenVLAOFTAdapter(ModelAdapter):
    name = "oft"

    def __init__(self):
        self.components = None
        self._env_cache = {}  # (suite, task_id) -> (env, desc, init_states)

    @property
    def default_checkpoints(self):
        return {
            "libero_spatial": "/data/checkpoints/openvla-oft-spatial",
            "libero_object": "/data/checkpoints/openvla-oft-object",
            "libero_goal": "/data/checkpoints/openvla-oft-goal",
            "libero_10": "/data/checkpoints/openvla-oft-10",
        }

    def load_model(self, checkpoint="/data/checkpoints/openvla-oft-goal", device="cuda"):
        os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
        os.environ.setdefault("MUJOCO_GL", "egl")
        checkpoint = _resolve_checkpoint(checkpoint)
        self.device = device
        self.components = self._load_oft(checkpoint, "libero_goal", device)
        self.model = self.components["model"]
        return self.model

    @staticmethod
    def _load_oft(checkpoint_path, suite, device):
        from dataclasses import dataclass
        # These imports come from the openvla_oft submodule
        sys.path.insert(0, str(PROJECT_ROOT / "openvla_oft"))
        from experiments.robot.openvla_utils import (
            get_vla, get_processor, get_action_head, get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_image_resize_size

        @dataclass
        class _Cfg:
            model_family: str = "openvla"
            pretrained_checkpoint: str = checkpoint_path
            use_l1_regression: bool = True
            use_diffusion: bool = False
            use_film: bool = False
            num_images_in_input: int = 2
            use_proprio: bool = True
            center_crop: bool = True
            num_open_loop_steps: int = 8
            lora_rank: int = 32
            load_in_8bit: bool = False
            load_in_4bit: bool = False
            unnorm_key: str = f"{suite}_no_noops"

        cfg = _Cfg()
        vla = get_vla(cfg)
        return {
            "model": vla,
            "processor": get_processor(cfg),
            "action_head": get_action_head(cfg, vla.llm_dim),
            "proprio_projector": get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8),
            "resize_size": get_image_resize_size(cfg),
            "config": cfg,
        }

    def get_layer_groups(self):
        # OpenVLA-OFT is a 32-layer Llama model
        layers = list(self.model.language_model.model.layers)
        return {"llm": layers}

    def setup_suite(self, suite="libero_goal", **kwargs):
        from libero.libero import benchmark
        bench = benchmark.get_benchmark_dict()
        task_suite = bench[suite]()
        tasks = [(i, task_suite.get_task(i), task_suite.get_task(i).language)
                 for i in range(task_suite.n_tasks)]
        return task_suite, tasks

    def create_env(self, task, suite="libero_goal", resolution=256, **kwargs):
        task_idx = task if isinstance(task, int) else 0
        cache_key = (suite, task_idx)
        if cache_key not in self._env_cache:
            env, desc, init_states, _ = self._create_libero_env(suite, task_idx, resolution)
            self._env_cache[cache_key] = (env, desc, init_states)
        env, desc, init_states = self._env_cache[cache_key]
        return env, desc, {"init_states": init_states}

    @staticmethod
    def _create_libero_env(suite, task_id, resolution=256):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        task_suite = benchmark.get_benchmark_dict()[suite]()
        task = task_suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
        env.seed(0)
        return env, task.language, task_suite.get_task_init_states(task_id), task_suite

    def run_episode(self, env, task_desc, max_steps=300,
                    save_video=False, perturbation_fn=None, **kwargs):
        init_states = kwargs.get("init_states")
        seed = kwargs.get("seed", 42)

        obs = env.reset()
        if init_states is not None:
            obs = env.set_init_state(init_states[seed % len(init_states)])

        vla = self.components["model"]
        processor = self.components["processor"]
        action_head = self.components["action_head"]
        proprio_proj = self.components["proprio_projector"]
        resize_size = self.components["resize_size"]
        cfg = self.components["config"]

        actions_list = []
        frames = []
        success = False

        for step in range(max_steps):
            img = obs.get("agentview_image", obs.get("image"))[::-1, ::-1].copy()
            if save_video and step % 3 == 0:
                frames.append(img.copy())

            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(img).resize((resize_size, resize_size))

            inputs = processor(task_desc, pil_img).to(self.device, dtype=torch.bfloat16)

            with torch.inference_mode():
                output = vla(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                )
                hidden = output.last_hidden_state[:, -1, :]

                if proprio_proj is not None:
                    eef = obs["robot0_eef_pos"]
                    quat = obs["robot0_eef_quat"]
                    grip = obs["robot0_gripper_qpos"]
                    from experiments.libero_utils import quat2axisangle
                    proprio_vec = np.concatenate([eef, quat2axisangle(quat), grip])
                    proprio = torch.tensor(proprio_vec,
                                           dtype=torch.float32, device=self.device).unsqueeze(0)
                    hidden = hidden + proprio_proj(proprio)

                if action_head is not None:
                    action = action_head(hidden).squeeze(0).cpu().numpy()
                else:
                    action = hidden.squeeze(0).cpu().numpy()[:7]

            action_7d = action[:7]
            actions_list.append(action_7d.copy())
            obs, reward, done, info = env.step(action_7d)

            if done:
                success = env.check_success()
                break

        result = {"success": bool(success), "steps": len(actions_list),
                  "actions": np.array(actions_list)}
        if save_video:
            result["frames"] = frames
        return result


# Adapter registry

ADAPTERS = {
    "xvla": XVLAAdapter,
    "smolvla": SmolVLAAdapter,
    "groot": GR00TAdapter,
    "pi05": Pi05Adapter,
    "oft": OpenVLAOFTAdapter,
}


def get_adapter(model_name: str) -> ModelAdapter:
    # Get an adapter instance by model name
    ...
    if model_name not in ADAPTERS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(ADAPTERS.keys())}")
    return ADAPTERS[model_name]()


def list_models() -> List[str]:
    return list(ADAPTERS.keys())
