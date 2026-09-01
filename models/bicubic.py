"""
Bicubic Upsampler — Classical baseline for super-resolution benchmarking.

Wraps OpenCV bicubic interpolation in a consistent interface matching
the deep learning models, so it can be used interchangeably in evaluation.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn


class BicubicUpsampler(nn.Module):
    """
    Bicubic interpolation upsampler (non-learnable baseline).

    Accepts either numpy arrays or PyTorch tensors.
    When used as an nn.Module, operates on tensors using F.interpolate.
    """

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upscale a tensor using bicubic interpolation.

        Args:
            x: (B, C, H, W) tensor in [0, 1].

        Returns:
            Upscaled tensor of shape (B, C, H*s, W*s).
        """
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)

    def upscale_numpy(self, img: np.ndarray) -> np.ndarray:
        """
        Upscale a numpy image using OpenCV bicubic interpolation.

        Args:
            img: (H, W, C) uint8 or float image.

        Returns:
            Upscaled image of shape (H*s, W*s, C).
        """
        h, w = img.shape[:2]
        new_h, new_w = h * self.scale_factor, w * self.scale_factor
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}"
