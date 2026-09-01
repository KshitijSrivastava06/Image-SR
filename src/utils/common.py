"""
Common utility functions for the super-resolution platform.

Provides device detection, reproducibility seeding, image I/O,
and tensor/numpy conversion helpers used throughout the project.
"""

import os
import random
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Union[str, Path], color_mode: str = "rgb") -> np.ndarray:
    """
    Load an image from disk as a numpy array (H, W, C) in [0, 255] uint8.

    Args:
        path: Path to the image file.
        color_mode: 'rgb' (default) or 'bgr'.

    Returns:
        Numpy array of shape (H, W, C) in uint8.
    """
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    if color_mode == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(
    img: Union[np.ndarray, torch.Tensor],
    path: Union[str, Path],
    quality: int = 95,
) -> None:
    """
    Save an image (numpy array or tensor) to disk.

    Handles both uint8 [0, 255] and float [0, 1] images.
    Creates parent directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)

    # Convert float [0,1] → uint8
    if img.dtype in (np.float32, np.float64):
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # RGB → BGR for OpenCV
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Tensor ↔ Numpy conversions
# ---------------------------------------------------------------------------

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch image tensor to a numpy array.

    Input:  (C, H, W) or (B, C, H, W) in [0, 1] float
    Output: (H, W, C) numpy array in [0, 1] float32

    If batch dimension is present, only the first image is returned.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)


def numpy_to_tensor(
    img: np.ndarray,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Convert a numpy image array to a PyTorch tensor.

    Input:  (H, W, C) numpy array in uint8 [0, 255] or float [0, 1]
    Output: (1, C, H, W) tensor in [0, 1] float32
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_model_size(model: torch.nn.Module) -> dict:
    """Return the number of parameters and estimated size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024 ** 2)  # float32 = 4 bytes
    return {
        "total_params": total_params,
        "size_mb": round(size_mb, 2),
    }


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist, return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
