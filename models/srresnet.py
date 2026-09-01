"""
SRResNet — Super-Resolution Residual Network.

Reference: Ledig et al., "Photo-Realistic Single Image Super-Resolution
           Using a Generative Adversarial Network" CVPR 2017.

Architecture:
    Input → Conv(64, 9×9, PReLU) →
    16 × ResBlock(Conv(64, 3×3, BN, PReLU) → Conv(64, 3×3, BN) + skip) →
    Conv(64, 3×3, BN) + global skip →
    2 × UpsampleBlock(Conv, PixelShuffle, PReLU) →
    Conv(3, 9×9) → Output

~1.5M parameters.  Serves as both a standalone model and as the generator
backbone for SRGAN.
"""

import torch
import torch.nn as nn
import math


class ResidualBlock(nn.Module):
    """
    Residual block: Conv → BN → PReLU → Conv → BN + skip.

    Args:
        channels: Number of feature channels.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    Upsample block using sub-pixel convolution (PixelShuffle, ×2).

    Conv(channels → channels*4, 3×3) → PixelShuffle(2) → PReLU
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(num_parameters=channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRResNet(nn.Module):
    """
    SRResNet generator for super-resolution.

    Args:
        in_channels:          Input image channels (default: 3).
        num_channels:         Feature map channels (default: 64).
        num_residual_blocks:  Number of residual blocks (default: 16).
        scale_factor:         Upscaling factor, must be 2, 4, or 8.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_channels: int = 64,
        num_residual_blocks: int = 16,
        scale_factor: int = 4,
    ):
        super().__init__()

        if scale_factor not in (2, 4, 8):
            raise ValueError(f"scale_factor must be 2, 4, or 8, got {scale_factor}")

        self.scale_factor = scale_factor

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=num_channels),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution + BN (before global skip)
        self.post_residual = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels),
        )

        # Upsample blocks: log2(scale_factor) blocks of ×2 upsampling
        num_upsample = int(math.log2(scale_factor))
        self.upsample = nn.Sequential(
            *[UpsampleBlock(num_channels) for _ in range(num_upsample)]
        )

        # Final reconstruction
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, padding=4)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) LR image tensor.

        Returns:
            SR image tensor of shape (B, C, H*s, W*s).
        """
        initial = self.initial(x)
        residual = self.residual_blocks(initial)
        residual = self.post_residual(residual)
        out = initial + residual  # global skip connection
        out = self.upsample(out)
        out = self.final(out)
        return out.clamp(0, 1)
