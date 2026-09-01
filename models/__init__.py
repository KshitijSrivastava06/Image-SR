"""
Model registry — factory function to instantiate SR models by name.

Provides a unified `get_model(name, config)` interface for all architectures.
"""

from typing import Any

import torch.nn as nn

from models.bicubic import BicubicUpsampler
from models.srcnn import SRCNN
from models.srresnet import SRResNet
from models.srgan import SRGANGenerator
from models.esrgan import (
    ESRGANGenerator,
    RRDB,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, type] = {
    "bicubic": BicubicUpsampler,
    "srcnn": SRCNN,
    "srresnet": SRResNet,
    "srgan_generator": SRGANGenerator,
    "esrgan_generator": ESRGANGenerator,
}


def get_model(name: str, **kwargs: Any) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name:     Model identifier (bicubic, srcnn, srresnet, srgan, esrgan).
        **kwargs: Model-specific constructor arguments.

    Returns:
        Instantiated nn.Module.

    Example::

        model = get_model('srcnn', scale_factor=4)
        model = get_model('esrgan', scale_factor=4, num_rrdb_blocks=23)
    """
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def get_pretrained(name: str, scale_factor: int, device=None, cache_dir: str = "models/pretrained") -> nn.Module:
    """
    Instantiate a model and load its pre-trained weights.
    
    Args:
        name: Model identifier (srcnn, srresnet, srgan_generator, esrgan_generator).
        scale_factor: Upscaling factor.
        device: Compute device.
        cache_dir: Directory to cache downloaded weights.
        
    Returns:
        Loaded model in eval mode.
    """
    from models.pretrained_weights import load_pretrained
    return load_pretrained(name, scale_factor, device, cache_dir)
