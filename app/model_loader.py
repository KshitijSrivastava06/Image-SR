"""
Model Loader — Centralized model loading with caching for Streamlit app.

Handles weight file resolution, CPU/GPU detection, cascaded inference for 8×, and graceful fallbacks.
"""

import sys
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model
from src.utils.common import get_device, get_model_size
from src.utils.logger import get_logger

logger = get_logger("model_loader")


# Model names mapping
MODEL_NAMES = ["bicubic", "srcnn", "srresnet", "srgan", "esrgan"]

def load_sr_model(
    model_name: str,
    scale_factor: int = 4,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Load a super-resolution model with optional pretrained weights.

    Args:
        model_name:   One of 'bicubic', 'srcnn', 'srresnet', 'srgan', 'esrgan'.
        scale_factor: Upscaling factor (2, 4, or 8).
        weights_path: Optional explicit path to weight file.
        device:       Compute device (auto-detected if None).

    Returns:
        Loaded model in eval mode on the specified device.
    """
    device = device or get_device()
    name_lower = model_name.lower()
    
    if name_lower == "srgan":
        name_lower = "srgan_generator"
    elif name_lower == "esrgan":
        name_lower = "esrgan_generator"

    if name_lower == "bicubic":
        from models import get_model
        model = get_model("bicubic", scale_factor=scale_factor)
        model = model.to(device).eval()
    else:
        from models import get_pretrained
        model = get_pretrained(name_lower, scale_factor=scale_factor, device=device)

    size = get_model_size(model)
    logger.info(
        f"Loaded {model_name} on {device}: "
        f"{size['total_params']:,} params, {size['size_mb']:.2f} MB"
    )

    return model


def list_available_models() -> list[str]:
    """Return list of available model names."""
    return MODEL_NAMES


def get_weight_status() -> dict[str, bool]:
    """Check which models have pretrained weights available."""
    from models.pretrained_weights import PRETRAINED_REGISTRY
    status = {}
    for name in MODEL_NAMES:
        if name == "bicubic":
            status[name] = True
            continue
            
        reg_name = name
        if name in ("srgan", "esrgan"):
            reg_name = f"{name}_generator"
            
        key = f"{reg_name}_x4"
            
        if key in PRETRAINED_REGISTRY and PRETRAINED_REGISTRY[key].get("url"):
            status[name] = True
        else:
            # Check local fallback
            local_path = PROJECT_ROOT / f"models/{name}/final.pth"
            status[name] = local_path.exists()
            
    return status
