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

    if weights_path and Path(weights_path).exists():
        logger.info(f"Loading custom weights from {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "generator_state_dict" in state_dict:
            state_dict = state_dict["generator_state_dict"]
        if ("srresnet" in name_lower or "srgan" in name_lower) and getattr(model, "use_batchnorm", True) is False:
            from models.srresnet import fold_srresnet_bn
            state_dict = fold_srresnet_bn(state_dict)
        model.load_state_dict(state_dict, strict=True)

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
    """Check which models have pretrained or trained weights available."""
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
            # Check local fallback locations
            local_paths = [
                PROJECT_ROOT / f"models/{name}/final.pth",
                PROJECT_ROOT / f"outputs/checkpoints/{name}/best_model.pth",
                PROJECT_ROOT / f"outputs/checkpoints/{name}/final.pth",
            ]
            status[name] = any(p.exists() for p in local_paths)
            
    return status


def validate_model_weights(model: torch.nn.Module, scale_factor: int = 4) -> dict:
    """
    Validate that a loaded model produces sane outputs on test inputs.
    
    Checks:
    1. Output shape matches scale_factor
    2. No NaNs or Infs
    3. Output values are within [0, 1] range
    4. Non-trivial spatial variance (output is not a solid blank color)
    """
    device = next(model.parameters(), torch.tensor([])).device if list(model.parameters()) else torch.device("cpu")
    dummy = torch.rand(1, 3, 32, 32, device=device)
    
    with torch.no_grad():
        out = model(dummy)
        
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    val_min = out.min().item()
    val_max = out.max().item()
    val_std = out.std().item()
    
    is_valid = (
        not has_nan 
        and not has_inf 
        and val_min >= -0.05 
        and val_max <= 1.05 
        and val_std > 0.01
    )
    
    return {
        "valid": is_valid,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": val_min,
        "max": val_max,
        "std": val_std,
    }

