#!/usr/bin/env python3
"""
LIBERO utilities for steering experiments.

Functions for setting up LIBERO environments, getting images, and handling actions.
"""

import os
import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
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


def get_libero_image(obs: dict, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extract and resize image from LIBERO observation.

    Args:
        obs: Observation dict from env.step()
        target_size: Target (H, W) for output image

    Returns:
        RGB image as numpy array with shape (H, W, 3), dtype uint8
    """
    # Get the agentview image
    if "agentview_image" in obs:
        img = obs["agentview_image"]
    elif "image" in obs:
        img = obs["image"]
    else:
        # Try to find any image key
        for key in obs:
            if "image" in key.lower() or "rgb" in key.lower():
                img = obs[key]
                break
        else:
            raise KeyError(f"No image found in observation. Keys: {obs.keys()}")

    # Ensure proper format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Rotate 180 degrees to match training preprocessing
    img = img[::-1, ::-1].copy()

    # Resize if needed
    if img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.BILINEAR)
        img = np.array(pil_img)

    return img


def get_libero_dummy_action(model_family: str = "openvla") -> list:
    """
    Get a dummy action (zeros) for waiting/settling.

    Returns 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    return [0.0] * 7


def get_libero_image_hires(obs: dict, target_size: tuple = (1080, 1080)) -> np.ndarray:
    """
    Extract and resize image from LIBERO observation at higher resolution for video.

    This is separate from get_libero_image() to allow saving high-quality video frames
    while still using 224x224 for model input.

    Args:
        obs: Observation dict from env.step()
        target_size: Target (H, W) for output image (default 1080x1080 for 1080p video)

    Returns:
        RGB image as numpy array with shape (H, W, 3), dtype uint8
    """
    # Get the agentview image
    if "agentview_image" in obs:
        img = obs["agentview_image"]
    elif "image" in obs:
        img = obs["image"]
    else:
        for key in obs:
            if "image" in key.lower() or "rgb" in key.lower():
                img = obs[key]
                break
        else:
            raise KeyError(f"No image found in observation. Keys: {obs.keys()}")

    # Ensure proper format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # IMPORTANT: Rotate 180 degrees to match OpenVLA training preprocessing
    img = img[::-1, ::-1].copy()

    # Resize to target size for high-quality video
    if img.shape[:2] != target_size:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize(target_size, Image.LANCZOS)  # High quality resize
        img = np.array(pil_img)

    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to axis-angle representation.

    Args:
        quat: Quaternion as [w, x, y, z]

    Returns:
        Axis-angle as [ax, ay, az] where magnitude is angle
    """
    # Normalize
    quat = np.array(quat)
    quat = quat / (np.linalg.norm(quat) + 1e-8)

    w, x, y, z = quat

    # Handle numerical issues near identity
    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))

    if angle < 1e-6:
        return np.zeros(3)

    axis = np.array([x, y, z]) / (np.sin(angle / 2) + 1e-8)
    return axis * angle


def save_rollout_video(
    frames: list,
    output_path: str,
    fps: int = 20,
):
    """
    Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of RGB numpy arrays (H, W, 3), uint8
        output_path: Path to save video
        fps: Frames per second
    """
    if len(frames) == 0:
        print("Warning: No frames to save")
        return

    # Try imageio first (better codec support, high quality)
    try:
        import imageio
        # Use very high quality settings (CRF 10 = near lossless)
        writer = imageio.get_writer(
            output_path,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            output_params=['-crf', '10', '-preset', 'slow']  # Very high quality
        )
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved to: {output_path} ({len(frames)} frames)")
        return
    except ImportError:
        pass
    except Exception as e:
        print(f"imageio failed: {e}, trying OpenCV...")

    # Fallback to OpenCV with X264 codec
    try:
        import cv2
    except ImportError:
        print("Warning: Neither imageio nor opencv-python installed, cannot save video")
        return

    h, w = frames[0].shape[:2]

    # Try different codecs in order of preference
    codecs = [
        ('avc1', '.mp4'),   # H.264 (most compatible)
        ('X264', '.mp4'),   # H.264 alternative
        ('XVID', '.avi'),   # XVID (fallback)
        ('mp4v', '.mp4'),   # MPEG-4 (last resort)
    ]

    out = None
    for codec, ext in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Adjust extension if needed
        if not output_path.endswith(ext):
            test_path = output_path.rsplit('.', 1)[0] + ext
        else:
            test_path = output_path
        out = cv2.VideoWriter(test_path, fourcc, fps, (w, h))
        if out.isOpened():
            output_path = test_path
            break
        out.release()
        out = None

    if out is None:
        print("Warning: Could not open video writer with any codec")
        return

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr)

    out.release()
    print(f"Video saved to: {output_path} ({len(frames)} frames)")


def get_libero_state(obs: dict) -> np.ndarray:
    """
    Extract robot state from LIBERO observation.

    Returns 8D state: [ee_pos (3), ee_ori_axisangle (3), gripper_qpos (2)]
    """
    ee_pos = obs.get("robot0_eef_pos", np.zeros(3))
    ee_quat = obs.get("robot0_eef_quat", np.array([1, 0, 0, 0]))
    gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))

    ee_ori = quat2axisangle(ee_quat)

    return np.concatenate([ee_pos, ee_ori, gripper_qpos])
