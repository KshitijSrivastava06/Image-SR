"""Apply learned brightness detail while preserving the interpolated input chroma."""

import torch
from torch.nn import functional as F


def add_luminance_detail(source: torch.Tensor, base: torch.Tensor, detail: torch.Tensor) -> torch.Tensor:
    base = base.clamp(0, 1)
    # Constrain brightness corrections by local contrast so flat areas keep their tone.
    brightness = source.mean(dim=1, keepdim=True)
    contrast = F.max_pool2d(brightness, 3, stride=1, padding=1)
    contrast += F.max_pool2d(-brightness, 3, stride=1, padding=1)
    limit = F.interpolate(contrast, size=base.shape[-2:], mode="bilinear", align_corners=False)

    # Bound the shared RGB offset before adding it, avoiding per-channel clipping.
    lower = torch.maximum(-limit, -base.amin(dim=1, keepdim=True))
    upper = torch.minimum(limit, 1 - base.amax(dim=1, keepdim=True))
    return base + detail.clamp(min=lower, max=upper)
