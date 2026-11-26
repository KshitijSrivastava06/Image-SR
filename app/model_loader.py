"""Load inference models and report which checkpoints are available."""

from pathlib import Path

import torch

from models import get_model
from models.pretrained_weights import (
    PRETRAINED_REGISTRY,
    find_local_weights,
    load_pretrained,
)
from src.utils.common import get_device

MODEL_NAMES = ["bicubic", "srcnn", "srresnet", "srgan", "esrgan"]
MODEL_LABELS = {
    "bicubic": "Bicubic",
    "srcnn": "SRCNN",
    "srresnet": "SRResNet",
    "srgan": "SRGAN",
    "esrgan": "Real-ESRGAN",
}


def model_identifier(model_name: str) -> str:
    name = model_name.lower()
    # The UI names SRGAN and Real-ESRGAN by model. Inference instantiates their generators.
    return f"{name}_generator" if name in ("srgan", "esrgan") else name


def load_sr_model(
    model_name: str,
    scale_factor: int = 4,
    weights_path: str | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load an eval-mode model using a compatible checkpoint."""
    device = device or get_device()
    name = model_identifier(model_name)
    # Bicubic is a fixed interpolation operation and has no learned weights.
    if name == "bicubic":
        return get_model(name, scale_factor=scale_factor).to(device).eval()

    return load_pretrained(
        name, scale_factor, device, weights_path=Path(weights_path) if weights_path else None,
    )


def list_available_models() -> list[str]:
    return MODEL_NAMES.copy()


def get_weight_status(scale_factor: int = 4) -> dict[str, bool]:
    """Report whether each model can be used at the selected scale."""
    status = {"bicubic": True}
    for name in MODEL_NAMES[1:]:
        identifier = model_identifier(name)
        entry = PRETRAINED_REGISTRY.get(f"{identifier}_x{scale_factor}", {})
        status[name] = bool(entry.get("url") or find_local_weights(identifier, scale_factor))
    return status
