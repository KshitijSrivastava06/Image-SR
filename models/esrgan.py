"""
ESRGAN — Enhanced Super-Resolution Generative Adversarial Network.

Reference: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative
           Adversarial Networks" ECCVW 2018.

Key improvements over SRGAN:
    1. RRDB (Residual-in-Residual Dense Block) replaces ResBlock
    2. No Batch Normalization (avoids artifacts)
    3. Residual scaling (β=0.2) for training stability
    4. Relativistic average GAN discriminator (RaGAN)
    5. VGG features extracted BEFORE activation for perceptual loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """
    Dense block with 5 convolutions and local residual learning.

    Each conv receives concatenated features from all prior convolutions
    within the block (dense connections).

    Args:
        channels:        Input/output feature channels (default: 64).
        growth_channels: Growth rate per conv (default: 32).
        residual_scaling: Residual scaling factor β (default: 0.2).
    """

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        super().__init__()
        self.beta = residual_scaling

        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4 * growth_channels, channels, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block.

    Three cascaded DenseBlocks with residual scaling.

    Args:
        channels:         Feature channels (default: 64).
        growth_channels:  Dense block growth rate (default: 32).
        residual_scaling: Scaling factor β (default: 0.2).
    """

    def __init__(
        self,
        channels: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
    ):
        super().__init__()
        self.beta = residual_scaling

        self.db1 = DenseBlock(channels, growth_channels, residual_scaling)
        self.db2 = DenseBlock(channels, growth_channels, residual_scaling)
        self.db3 = DenseBlock(channels, growth_channels, residual_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * self.beta + x


class ESRGANGenerator(nn.Module):
    """
    ESRGAN Generator with RRDB backbone.

    Architecture:
        [Optional PixelUnshuffle for 2×] →
        Input → Conv(64, 3×3) → N×RRDB → Conv(64, 3×3) + global_skip →
        Upsample(Nearest + Conv) → Conv(64, 3×3, LReLU) → Conv(3, 3×3) → Output

    For scale_factor=2, the Real-ESRGAN x2plus architecture uses PixelUnshuffle(2)
    at the input to rearrange (B, 3, H, W) → (B, 12, H/2, W/2), then upsamples 4×
    internally (2 blocks), achieving a net 2× upscale.

    Args:
        in_channels:      Input image channels (default: 3).
        num_channels:     Base feature channels (default: 64).
        num_rrdb_blocks:  Number of RRDB blocks (default: 23).
        growth_channels:  Dense block growth rate (default: 32).
        residual_scaling: Scaling factor β (default: 0.2).
        scale_factor:     Upscaling factor (2 or 4).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_channels: int = 64,
        num_rrdb_blocks: int = 23,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
        scale_factor: int = 4,
    ):
        super().__init__()

        if scale_factor not in (2, 4):
            raise ValueError(f"scale_factor must be 2 or 4, got {scale_factor}")

        self.scale_factor = scale_factor

        # For 2× upscaling: PixelUnshuffle reduces spatial by 2× at input,
        # then 2 upsample blocks (4×) give net 2× output.
        # This matches the Real-ESRGAN x2plus architecture.
        if scale_factor == 2:
            self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor=2)
            first_in_channels = in_channels * 4  # 3 * 2² = 12
        else:
            self.pixel_unshuffle = None
            first_in_channels = in_channels

        # First convolution (12 input channels for x2, 3 for x4)
        self.conv_first = nn.Conv2d(first_in_channels, num_channels, 3, 1, 1)

        # RRDB trunk
        self.trunk = nn.Sequential(
            *[RRDB(num_channels, growth_channels, residual_scaling)
              for _ in range(num_rrdb_blocks)]
        )

        # Post-trunk conv (before global skip)
        self.conv_trunk = nn.Conv2d(num_channels, num_channels, 3, 1, 1)

        # Upsampling layers — always 2 blocks (each 2×, totaling 4× internal upscale)
        # For x4: net 4×.  For x2: PixelUnshuffle(2) + 4× internal = net 2×.
        num_upsample = 2
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(num_channels, num_channels, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        self.upsample = nn.Sequential(*upsample_layers)

        # Final reconstruction
        self.conv_last = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_channels, in_channels, 3, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) LR image tensor in [0, 1].

        Returns:
            SR image tensor of shape (B, C, H*s, W*s).
        """
        # PixelUnshuffle for 2× mode: (B,3,H,W) → (B,12,H/2,W/2)
        if self.pixel_unshuffle is not None:
            x = self.pixel_unshuffle(x)

        feat = self.conv_first(x)
        trunk = self.conv_trunk(self.trunk(feat))
        feat = feat + trunk  # global residual connection
        feat = self.upsample(feat)
        out = self.conv_last(feat)
        return out.clamp(0, 1)


class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style discriminator with optional Relativistic average GAN.

    Architecture follows the SRGAN discriminator but removes BN from
    the first layer and supports the relativistic formulation.

    Args:
        in_channels:   Image channels (default: 3).
        features:      Channel progression (default: SRGAN-style).
        relativistic:  Use RaGAN formulation (default: True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        features: list[int] | None = None,
        relativistic: bool = True,
    ):
        super().__init__()
        self.relativistic = relativistic

        if features is None:
            features = [64, 64, 128, 128, 256, 256, 512, 512]

        layers = []
        prev_channels = in_channels

        for i, f in enumerate(features):
            stride = 1 if i % 2 == 0 else 2
            layers.append(nn.Conv2d(prev_channels, f, 3, stride, 1))
            if i > 0:
                layers.append(nn.BatchNorm2d(f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channels = f

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Flatten(),
            nn.Linear(prev_channels * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, 1)."""
        feat = self.features(x)
        return self.classifier(feat)


class ESRGAN(nn.Module):
    """
    Full ESRGAN model wrapper (Generator + Discriminator).

    Args:
        scale_factor:     Upscaling factor (2, 4, or 8).
        num_rrdb_blocks:  Number of RRDB blocks (default: 23).
        num_channels:     Feature channels (default: 64).
        growth_channels:  Dense block growth rate (default: 32).
        residual_scaling: Residual scaling β (default: 0.2).
        relativistic:     Use RaGAN discriminator (default: True).
    """

    def __init__(
        self,
        scale_factor: int = 4,
        num_rrdb_blocks: int = 23,
        num_channels: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2,
        in_channels: int = 3,
        relativistic: bool = True,
    ):
        super().__init__()

        self.generator = ESRGANGenerator(
            in_channels=in_channels,
            num_channels=num_channels,
            num_rrdb_blocks=num_rrdb_blocks,
            growth_channels=growth_channels,
            residual_scaling=residual_scaling,
            scale_factor=scale_factor,
        )
        self.discriminator = VGGStyleDiscriminator(
            in_channels=in_channels,
            relativistic=relativistic,
        )

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        """Inference: run generator only."""
        return self.generator(lr)

    @staticmethod
    def interpolate_models(
        psnr_model: ESRGANGenerator,
        gan_model: ESRGANGenerator,
        alpha: float = 0.8,
    ) -> ESRGANGenerator:
        """
        Network interpolation between a PSNR-oriented and GAN-oriented model.

        alpha=0 → pure PSNR model, alpha=1 → pure GAN model.
        Returns a new model with interpolated weights.
        """
        interp_model = ESRGANGenerator(
            in_channels=gan_model.conv_first.in_channels,
            num_channels=gan_model.conv_first.out_channels,
            scale_factor=gan_model.scale_factor,
        )

        psnr_state = psnr_model.state_dict()
        gan_state = gan_model.state_dict()
        interp_state = {}

        for key in gan_state:
            interp_state[key] = (1 - alpha) * psnr_state[key] + alpha * gan_state[key]

        interp_model.load_state_dict(interp_state)
        return interp_model
