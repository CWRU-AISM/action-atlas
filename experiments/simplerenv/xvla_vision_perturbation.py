#!/usr/bin/env python3
"""
Standard image perturbation suite for X-VLA vision perturbation experiments.

Provides ImagePerturbations and get_standard_perturbations(), used by
xvla_simplerenv_vision_perturbation.py.
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import warnings
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import cv2

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
