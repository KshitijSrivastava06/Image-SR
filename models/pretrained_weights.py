"""
Pre-Trained Weights Manager — Centralized system for downloading and loading pre-trained checkpoints.
"""

import pickle
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from models import get_model
from src.utils.logger import get_logger

logger = get_logger("pretrained_weights")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


PRETRAINED_REGISTRY = {
    "srcnn_x4": {
        "url": None,
        "sha256": None,
        "filename": "srcnn_x4.pth",
        "source": "none (lightweight model, fast to retrain)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "srresnet_x4": {
        "url": None,  # Original S3 URL is dead (403). No compatible HF mirror available.
        "sha256": None,
        "filename": "srresnet_x4.pth",
        "source": "none (identical arch to SRGAN generator — use SRGAN instead)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "srgan_generator_x4": {
        "url": None,  # Same architecture as SRResNet. No public checkpoint with matching BN layers.
        "sha256": None,
        "filename": "srgan_generator_x4.pth",
        "source": "none (architecture includes BN — incompatible with BasicSR MSRResNet weights)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "esrgan_generator_x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": None,
        "filename": "RealESRGAN_x4plus.pth",
        "source": "xinntao/Real-ESRGAN (Official GitHub Release v0.1.0)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "esrgan_generator_x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "sha256": None,
        "filename": "RealESRGAN_x2plus.pth",
        "source": "xinntao/Real-ESRGAN (Official GitHub Release v0.2.1)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
}


def download_pretrained(
    model_name: str, scale_factor: int, cache_dir: str = "models/pretrained",
) -> Path | None:
    """Resolve a registered checkpoint, downloading it atomically if needed."""
    info = PRETRAINED_REGISTRY.get(f"{model_name}_x{scale_factor}", {})
    if not info.get("url"):
        return None
    destination = PROJECT_ROOT / cache_dir / info["filename"]
    if destination.is_file():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {info['url']} to {destination}")
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
        pending = Path(output.name)
        try:
            with output, urllib.request.urlopen(info["url"], timeout=30) as response:
                shutil.copyfileobj(response, output)
            pending.replace(destination)
        finally:
            pending.unlink(missing_ok=True)
    return destination


def find_local_weights(model_name: str, scale_factor: int) -> Path | None:
    """Find a checkpoint for the requested scale, including legacy 4× paths."""
    name = model_name.removesuffix("_generator")
    if scale_factor != 4:
        path = PROJECT_ROOT / "models" / name / f"x{scale_factor}" / "final.pth"
        return path if path.is_file() else None
    candidates = (
        PROJECT_ROOT / "models" / name / "final.pth",
        PROJECT_ROOT / "outputs" / "checkpoints" / name / "best_model.pth",
        PROJECT_ROOT / "outputs" / "checkpoints" / name / "final.pth",
    )
    return next((path for path in candidates if path.is_file()), None)


def remap_esrgan_keys(state_dict):
    """Remap xinntao ESRGAN keys to our ESRGANGenerator keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        # Standard ESRGAN to our architecture
        if k.startswith("RRDB_trunk."):
            new_k = k.replace("RRDB_trunk.", "trunk.")
            new_k = new_k.replace("RDB", "db")
            
        # Real-ESRGAN to our architecture
        elif k.startswith("body."):
            new_k = k.replace("body.", "trunk.")
            new_k = new_k.replace(".rdb", ".db")
        elif k.startswith("trunk_conv."):
            new_k = k.replace("trunk_conv.", "conv_trunk.")
        elif k.startswith("conv_body."):
            new_k = k.replace("conv_body.", "conv_trunk.")
        elif k.startswith("upconv1."):
            new_k = k.replace("upconv1.", "upsample.1.")
        elif k.startswith("conv_up1."):
            new_k = k.replace("conv_up1.", "upsample.1.")
        elif k.startswith("upconv2."):
            new_k = k.replace("upconv2.", "upsample.4.")
        elif k.startswith("conv_up2."):
            new_k = k.replace("conv_up2.", "upsample.4.")
        elif k.startswith("HRconv."):
            new_k = k.replace("HRconv.", "conv_last.0.")
        elif k.startswith("conv_hr."):
            new_k = k.replace("conv_hr.", "conv_last.0.")
        elif k.startswith("conv_last."):
            new_k = k.replace("conv_last.", "conv_last.2.")
        else:
            new_k = k
            
        new_state_dict[new_k] = v
        
    return new_state_dict


def load_pretrained(
    model_name: str,
    scale_factor: int,
    device: Optional[torch.device] = None,
    cache_dir: str = "models/pretrained",
    *,
    weights_path: Path | None = None,
) -> nn.Module:
    """Require a complete, architecture-compatible checkpoint for inference."""
    if weights_path is None:
        weights_path = (
            download_pretrained(model_name, scale_factor, cache_dir)
            or find_local_weights(model_name, scale_factor)
        )
    if weights_path is None:
        raise FileNotFoundError(f"No trained checkpoint for {model_name} at {scale_factor}×.")

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights_path}")
    logger.info(f"Loading {model_name} from {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not read the checkpoint at {weights_path}.") from exc
    if not isinstance(checkpoint, dict):
        raise ValueError("The checkpoint must contain model weights.")
    # Restore architecture options saved with the local checkpoint, including its training scale.
    model_kwargs = checkpoint.get("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        raise ValueError("The checkpoint has invalid model options.")
    model_kwargs = model_kwargs.copy()
    if model_kwargs.pop("scale_factor", scale_factor) != scale_factor:
        raise ValueError(f"The checkpoint was trained for a different scale than {scale_factor}×.")
    try:
        model = get_model(model_name, scale_factor=scale_factor, **model_kwargs)
    except TypeError as exc:
        raise ValueError(f"The checkpoint has unsupported options for {model_name}.") from exc
    state_dict = next(
        (checkpoint[key] for key in (
            "params_ema", "params", "state_dict", "model_state_dict", "generator_state_dict"
        ) if key in checkpoint),
        checkpoint,
    )
    if not isinstance(state_dict, dict):
        raise ValueError("The checkpoint has invalid model weights.")
    if model_name == "esrgan_generator" and any(
        key.startswith(("body.", "RRDB_trunk.")) for key in state_dict
    ):
        state_dict = remap_esrgan_keys(state_dict)

    if model_name in ("srresnet", "srgan_generator") and not model.use_batchnorm:
        from models.srresnet import fold_srresnet_bn
        state_dict = fold_srresnet_bn(state_dict, num_blocks=model.num_residual_blocks)

    # Require matching parameter names and shapes so a bad checkpoint cannot leave random weights.
    model.load_state_dict(state_dict, strict=True)
    # Evaluation uses inference behavior and freezes the checkpoint parameters.
    model.eval().requires_grad_(False)
    return model.to(device) if device is not None else model
