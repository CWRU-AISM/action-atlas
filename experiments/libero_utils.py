#!/usr/bin/env python3
"""
LIBERO utilities for steering experiments.

Functions for setting up LIBERO environments, getting images, and handling actions.
"""

import os
import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path  # noqa: F401  (benchmark: side-effect import)
from libero.libero.envs import OffScreenRenderEnv


def get_libero_env(task, model_family: str = "openvla", resolution: int = 256, video_resolution: int = 512, control_mode: str = "relative"):
    """
    Create a LIBERO environment for a given task.

    Args:
        task: LIBERO task object from benchmark
        model_family: Model type (affects camera settings)
        resolution: Image resolution for model input (default 256)
        video_resolution: Image resolution for video recording (default 512)
        control_mode: "relative" for delta actions (default), "absolute" for target positions
                      X-VLA requires "absolute" mode

    Returns:
        env: LIBERO environment
        task_description: Natural language task description
    """
    bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    # Use higher resolution for better video quality
    render_res = max(resolution, video_resolution)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": render_res,
        "camera_widths": render_res,
        # Use both cameras: agentview (third-person) and wrist camera
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "render_gpu_device_id": 0,
    }

    env = OffScreenRenderEnv(**env_args)

    # Set control mode after initial reset
    # This needs to be done after robots are initialized
    env._control_mode = control_mode

    task_description = task.language

    return env, task_description


def set_control_mode(env, control_mode: str):
    """
    Set the control mode for the LIBERO environment.

    Must be called after env.reset() to take effect.

    Args:
        env: LIBERO environment
        control_mode: "relative" for delta actions, "absolute" for target positions
    """
    if control_mode == "absolute":
        for robot in env.robots:
            robot.controller.use_delta = False
    elif control_mode == "relative":
        for robot in env.robots:
            robot.controller.use_delta = True
    else:
        raise ValueError(f"Invalid control mode: {control_mode}")


def get_libero_images(obs: dict, target_size: tuple = (256, 256)) -> dict:
    """
    Extract both camera images from LIBERO observation.

    Returns dict with:
        'agentview': RGB image from agentview camera (flipped 180 degrees)
        'wrist': RGB image from wrist camera (NOT flipped)

    Both images are resized to target_size and returned as uint8 arrays.
    """
    images = {}

    # Agentview image (flipped 180 degrees)
    if "agentview_image" in obs:
        img = obs["agentview_image"]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = img[::-1, ::-1].copy()  # Flip 180 degrees
        if img.shape[:2] != target_size:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.BILINEAR)
            img = np.array(pil_img)
        images['agentview'] = img

    # Wrist image (NOT flipped)
    if "robot0_eye_in_hand_image" in obs:
        img = obs["robot0_eye_in_hand_image"]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # NO flip for wrist camera
        if img.shape[:2] != target_size:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.BILINEAR)
            img = np.array(pil_img)
        images['wrist'] = img

    return images
