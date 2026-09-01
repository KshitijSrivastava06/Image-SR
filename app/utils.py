"""
App Utilities — Image conversion and display helpers for Streamlit.
"""

import io
from typing import Optional

import numpy as np
import torch
from PIL import Image


def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array (H, W, C) uint8 RGB."""
    return np.array(pil_image.convert("RGB"))


def numpy_to_pil(img: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if img.dtype in (np.float32, np.float64):
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert (C, H, W) or (1, C, H, W) tensor to PIL Image."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return numpy_to_pil(img)


def pil_to_tensor(
    pil_image: Image.Image,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert PIL Image to (1, C, H, W) tensor in [0, 1]."""
    img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def pil_to_bytes(pil_image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for download."""
    buf = io.BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()


def format_metric(name: str, value: float) -> str:
    """Format a metric value for display."""
    if name.lower() == "psnr":
        return f"{value:.2f} dB"
    elif name.lower() == "ssim":
        return f"{value:.4f}"
    elif name.lower() == "lpips":
        return f"{value:.4f}"
    elif "time" in name.lower():
        return f"{value*1000:.1f} ms"
    else:
        return f"{value:.4f}"


def get_image_info(pil_image: Image.Image) -> dict:
    """Get image metadata for display."""
    return {
        "width": pil_image.width,
        "height": pil_image.height,
        "mode": pil_image.mode,
        "format": pil_image.format or "N/A",
        "size_kb": len(pil_to_bytes(pil_image)) / 1024,
    }
