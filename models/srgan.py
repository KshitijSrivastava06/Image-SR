"""
SRGAN — Super-Resolution Generative Adversarial Network.

Reference: Ledig et al., "Photo-Realistic Single Image Super-Resolution
           Using a Generative Adversarial Network" CVPR 2017.

Generator: SRResNet architecture (see srresnet.py).
Discriminator: VGG-style CNN that classifies real HR vs generated SR images.
"""

import torch
import torch.nn as nn

from models.srresnet import SRResNet


# ---------------------------------------------------------------------------
# Helper: a single conv block used inside the Discriminator
# ---------------------------------------------------------------------------

def _conv_block(in_ch, out_ch, stride, batch_norm=True):
    """Conv → [BatchNorm] → LeakyReLU."""
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class Discriminator(nn.Module):
    """
    SRGAN Discriminator — VGG-style binary classifier.

    Applies 8 conv blocks with alternating strides (s=1, s=2) to progressively
    downsample the image while increasing channels (64 → 512), then uses a small
    dense head to output a single real/fake logit.

    Args:
        in_channels: Number of input image channels (default: 3 for RGB).
        pool_size: Spatial size of the pooled classifier input.
        num_channels: Number of channels in the first convolution.
    """

    def __init__(self, in_channels: int = 3, pool_size: int = 6, num_channels: int = 64):
        super().__init__()

        # Stride alternates between 1 and 2 to halve the size every two blocks.
        self.features = nn.Sequential(
            *_conv_block(in_channels, num_channels, stride=1, batch_norm=False),
            *_conv_block(num_channels, num_channels, stride=2),
            *_conv_block(num_channels, 2 * num_channels, stride=1),
            *_conv_block(2 * num_channels, 2 * num_channels, stride=2),
            *_conv_block(2 * num_channels, 4 * num_channels, stride=1),
            *_conv_block(4 * num_channels, 4 * num_channels, stride=2),
            *_conv_block(4 * num_channels, 8 * num_channels, stride=1),
            *_conv_block(8 * num_channels, 8 * num_channels, stride=2),
        )

        # Dense head — AdaptiveAvgPool makes it input-size agnostic
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(8 * num_channels * pool_size**2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.
        Returns:
            (B, 1) raw logits — pass through sigmoid for probabilities.
        """
        # The discriminator outputs a real/fake logit used while training the generator.
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Generator (thin alias so the model registry can distinguish from SRResNet)
# ---------------------------------------------------------------------------

class SRGANGenerator(SRResNet):
    """
    SRGAN Generator — identical architecture to SRResNet.
    Exists as a separate class so the model registry can name it distinctly.
    """
    pass
