"""
SRCNN — Super-Resolution Convolutional Neural Network.

Reference: Dong et al., "Image Super-Resolution Using Deep Convolutional Networks"
           TPAMI 2016 (originally ECCV 2014).

Architecture:
    Input (bicubic-interpolated LR) → Conv(64, 9×9, ReLU) →
    Conv(32, 1×1, ReLU) → Conv(C, 5×5) → Output

The input image is first upscaled to the target resolution using bicubic
interpolation, then refined by the 3-layer CNN.  ~57K parameters.
"""

import torch
import torch.nn as nn

from models.luminance import add_luminance_detail


class SRCNN(nn.Module):
    """
    SRCNN model for image super-resolution.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        n1:          Number of feature extraction filters (default: 64).
        n2:          Number of mapping filters (default: 32).
        f1:          Feature extraction kernel size (default: 9).
        f2:          Mapping kernel size (default: 1).
        f3:          Reconstruction kernel size (default: 5).
        scale_factor: Upscaling factor (for the internal bicubic pre-upscale).
        color_preserving: Predict brightness detail over interpolated RGB.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n1: int = 64,
        n2: int = 32,
        f1: int = 9,
        f2: int = 1,
        f3: int = 5,
        scale_factor: int = 4,
        color_preserving: bool = False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.color_preserving = color_preserving

        # Patch extraction & representation
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2),
            nn.ReLU(inplace=True),
        )

        # Non-linear mapping
        self.mapping = nn.Sequential(
            nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2),
            nn.ReLU(inplace=True),
        )

        # Reconstruction
        self.reconstruction = nn.Conv2d(
            n2, 1 if color_preserving else in_channels, kernel_size=f3, padding=f3 // 2
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) LR image tensor in [0, 1].
               The image is first upscaled internally using bicubic interpolation.

        Returns:
            SR image tensor of shape (B, C, H*s, W*s) in [0, 1].
        """
        # Pre-upscale LR input using bicubic interpolation
        base = nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )

        # 3-layer CNN refinement
        # The three learned convolutions extract patches, map features, and reconstruct RGB.
        out = self.feature_extraction(base)
        out = self.mapping(out)
        out = self.reconstruction(out)

        if self.color_preserving:
            return add_luminance_detail(x, base, out)

        # Keep training gradients outside the image range and clamp only the inference output.
        return out if self.training else out.clamp(0, 1)
