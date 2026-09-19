"""
SRResNet — Super-Resolution Residual Network.

Reference: Ledig et al., "Photo-Realistic Single Image Super-Resolution
           Using a Generative Adversarial Network" CVPR 2017.
ESRGAN reference: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative
                  Adversarial Networks" ECCV 2018 (demonstrated that removing
                  BatchNorm removes unpleasant artifacts and stabilizes training).

Architecture:
    Input → Conv(64, 9×9, PReLU) →
    16 × ResBlock(Conv(64, 3×3, [BN], PReLU) → Conv(64, 3×3, [BN]) + skip) →
    Conv(64, 3×3, [BN]) + global skip →
    2 × UpsampleBlock(Conv, PixelShuffle, PReLU) →
    Conv(3, 9×9) → Output

~1.5M parameters.  Serves as both a standalone model and as the generator
backbone for SRGAN. By default, use_batchnorm is False to eliminate BN-induced
tile artifacts and color distortions.
"""

import math
from typing import Optional, Dict
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual block: Conv → [BN] → PReLU → Conv → [BN] + skip.

    Args:
        channels: Number of feature channels.
        use_batchnorm: If True, include BatchNorm layers. Defaults to False.
    """

    def __init__(self, channels: int = 64, use_batchnorm: bool = False, residual_clamp: Optional[float] = 4.0):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.residual_clamp = residual_clamp
        layers = [
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))
        layers.append(nn.PReLU(num_parameters=channels))
        layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.block(x)
        if self.residual_clamp is not None:
            res = res.clamp(-self.residual_clamp, self.residual_clamp)
        return x + res


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


def fold_srresnet_bn(state_dict: Dict[str, torch.Tensor], num_blocks: int = 16, eps: float = 1e-5) -> Dict[str, torch.Tensor]:
    """
    Folds BatchNorm parameters into adjacent Conv2d weights and biases for SRResNet.
    
    This converts a checkpoint trained with BatchNorm into an identical BN-free
    model with 100% mathematical fidelity in eval mode, eliminating tile edge artifacts.
    """
    # Check if the state dict actually contains BN keys
    has_bn = any("block.1.running_mean" in k for k in state_dict.keys()) or any("post_residual.1.running_mean" in k for k in state_dict.keys())
    if not has_bn:
        return state_dict

    def _fold(conv_w, conv_b, bn_rm, bn_rv, bn_w, bn_b):
        scale = bn_w / torch.sqrt(bn_rv + eps)
        w_new = conv_w * scale.view(-1, 1, 1, 1)
        if conv_b is None:
            conv_b = torch.zeros_like(bn_rm)
        b_new = (conv_b - bn_rm) * scale + bn_b
        return w_new, b_new

    new_sd = {}
    
    # Copy initial
    for k in ["initial.0.weight", "initial.0.bias", "initial.1.weight"]:
        if k in state_dict:
            new_sd[k] = state_dict[k]

    # Fold residual blocks
    for i in range(num_blocks):
        # conv1 + bn1
        w1, b1 = _fold(
            state_dict[f"residual_blocks.{i}.block.0.weight"],
            state_dict.get(f"residual_blocks.{i}.block.0.bias"),
            state_dict[f"residual_blocks.{i}.block.1.running_mean"],
            state_dict[f"residual_blocks.{i}.block.1.running_var"],
            state_dict[f"residual_blocks.{i}.block.1.weight"],
            state_dict[f"residual_blocks.{i}.block.1.bias"],
        )
        new_sd[f"residual_blocks.{i}.block.0.weight"] = w1
        new_sd[f"residual_blocks.{i}.block.0.bias"] = b1
        # prelu (was block.2 with BN, becomes block.1 without BN)
        new_sd[f"residual_blocks.{i}.block.1.weight"] = state_dict[f"residual_blocks.{i}.block.2.weight"]
        # conv2 + bn2
        w2, b2 = _fold(
            state_dict[f"residual_blocks.{i}.block.3.weight"],
            state_dict.get(f"residual_blocks.{i}.block.3.bias"),
            state_dict[f"residual_blocks.{i}.block.4.running_mean"],
            state_dict[f"residual_blocks.{i}.block.4.running_var"],
            state_dict[f"residual_blocks.{i}.block.4.weight"],
            state_dict[f"residual_blocks.{i}.block.4.bias"],
        )
        new_sd[f"residual_blocks.{i}.block.2.weight"] = w2
        new_sd[f"residual_blocks.{i}.block.2.bias"] = b2

    # Fold post_residual
    if "post_residual.1.running_mean" in state_dict:
        w_pr, b_pr = _fold(
            state_dict["post_residual.0.weight"],
            state_dict.get("post_residual.0.bias"),
            state_dict["post_residual.1.running_mean"],
            state_dict["post_residual.1.running_var"],
            state_dict["post_residual.1.weight"],
            state_dict["post_residual.1.bias"],
        )
        new_sd["post_residual.0.weight"] = w_pr
        new_sd["post_residual.0.bias"] = b_pr
    elif "post_residual.0.weight" in state_dict:
        new_sd["post_residual.0.weight"] = state_dict["post_residual.0.weight"]
        if "post_residual.0.bias" in state_dict:
            new_sd["post_residual.0.bias"] = state_dict["post_residual.0.bias"]

    # Copy upsample and final
    for k, v in state_dict.items():
        if k.startswith("upsample.") or k.startswith("final."):
            new_sd[k] = v

    return new_sd


class SRResNet(nn.Module):
    """
    SRResNet generator for super-resolution.

    Args:
        in_channels:          Input image channels (default: 3).
        num_channels:         Feature map channels (default: 64).
        num_residual_blocks:  Number of residual blocks (default: 16).
        scale_factor:         Upscaling factor, must be 2, 4, or 8.
        use_batchnorm:        Whether to use BatchNorm layers (default: False).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_channels: int = 64,
        num_residual_blocks: int = 16,
        scale_factor: int = 4,
        use_batchnorm: bool = False,
        residual_clamp: Optional[float] = 4.0,
    ):
        super().__init__()

        if scale_factor not in (2, 4, 8):
            raise ValueError(f"scale_factor must be 2, 4, or 8, got {scale_factor}")

        self.scale_factor = scale_factor
        self.use_batchnorm = use_batchnorm
        self.num_residual_blocks = num_residual_blocks
        self.residual_clamp = residual_clamp

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=num_channels),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_channels, use_batchnorm=use_batchnorm, residual_clamp=residual_clamp) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution (before global skip)
        post_residual_layers = [
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        ]
        if use_batchnorm:
            post_residual_layers.append(nn.BatchNorm2d(num_channels))
        self.post_residual = nn.Sequential(*post_residual_layers)

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
        if self.residual_clamp is not None:
            residual = residual.clamp(-self.residual_clamp, self.residual_clamp)
        out = initial + residual  # global skip connection
        out = self.upsample(out)
        out = self.final(out)
        return out.clamp(0, 1)
