"""
Training Losses — Pixel, Perceptual, and Adversarial loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


class PixelLoss(nn.Module):
    """
    Pixel-wise reconstruction loss (L1 or MSE).
    
    Args:
        criterion: 'l1' or 'mse'
    """
    def __init__(self, criterion: str = 'l1'):
        super().__init__()
        if criterion.lower() == 'l1':
            self.loss_fn = nn.L1Loss()
        elif criterion.lower() == 'mse':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported pixel loss: {criterion}")
            
    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(sr, hr)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on VGG19 features.
    Extracts features before the activation (usually relu5_4 for SRGAN/ESRGAN).
    """
    def __init__(self, feature_layer: int = 35):
        super().__init__()
        # Load pre-trained VGG19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # We only need the features up to the requested layer
        # Layer 35 is the conv5_4 layer (before relu5_4)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer]).eval()
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        # VGG expects images normalized with ImageNet mean/std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.loss_fn = nn.MSELoss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Normalize to ImageNet stats (inputs are in [0, 1])
        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std
        
        # Extract features
        sr_features = self.features(sr_norm)
        hr_features = self.features(hr_norm).detach()
        
        return self.loss_fn(sr_features, hr_features)


class AdversarialLoss(nn.Module):
    """
    Adversarial loss for GANs.
    Standard BCE loss on discriminator output.
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, logits: torch.Tensor, is_real: bool, label_smoothing: bool = False) -> torch.Tensor:
        if is_real:
            if label_smoothing:
                # Smooth labels between 0.8 and 1.0
                target = torch.empty_like(logits).uniform_(0.8, 1.0)
            else:
                target = torch.ones_like(logits)
        else:
            target = torch.zeros_like(logits)
            
        return self.loss_fn(logits, target)


class TVLoss(nn.Module):
    """
    Total Variation regularization loss to encourage spatial smoothness.
    """
    def __init__(self, tv_weight: float = 1.0):
        super().__init__()
        self.tv_weight = tv_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        
        return self.tv_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        return t.size(1) * t.size(2) * t.size(3)
