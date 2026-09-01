"""
Preprocessing — Image cropping utilities for inference.

Provides downsampling and cropping utilities to prepare images for super-resolution inference.
"""

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("preprocessing")


# Mapping from string names to OpenCV interpolation flags
INTERPOLATION_MAP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


# ---------------------------------------------------------------------------
# Cropping utilities
# ---------------------------------------------------------------------------

def center_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Center-crop an image to (crop_size, crop_size)."""
    h, w = img.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return img[top : top + crop_size, left : left + crop_size]


def mod_crop(img: np.ndarray, scale_factor: int) -> np.ndarray:
    """Crop image so dimensions are divisible by scale_factor."""
    h, w = img.shape[:2]
    h = h - (h % scale_factor)
    w = w - (w % scale_factor)
    return img[:h, :w]
