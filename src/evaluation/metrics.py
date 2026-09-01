"""
Evaluation Metrics — PSNR, SSIM, and LPIPS for super-resolution quality assessment.

Provides both per-image and batch evaluation functions operating on
numpy arrays and/or PyTorch tensors.
"""

from typing import Optional, Union

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class MetricCalculator:
    """
    Compute PSNR, SSIM, and LPIPS metrics.

    LPIPS model is lazily loaded on first use to avoid import-time overhead.

    Usage::

        mc = MetricCalculator()
        psnr = mc.calculate_psnr(sr, hr)
        ssim = mc.calculate_ssim(sr, hr)
        lpips_val = mc.calculate_lpips(sr_tensor, hr_tensor)
        all_metrics = mc.calculate_all(sr, hr)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self._lpips_model = None
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # PSNR
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_psnr(
        sr: np.ndarray,
        hr: np.ndarray,
        crop_border: int = 0,
        data_range: float = 255.0,
    ) -> float:
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio).

        Args:
            sr: Super-resolved image (H, W, C) uint8 or float [0,1].
            hr: Ground truth image (H, W, C) same type as sr.
            crop_border: Pixels to crop from borders before comparison.
            data_range: Maximum value (255 for uint8, 1.0 for float).

        Returns:
            PSNR in dB (higher is better).
        """
        if sr.dtype in (np.float32, np.float64):
            data_range = 1.0

        if crop_border > 0:
            sr = sr[crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[crop_border:-crop_border, crop_border:-crop_border]

        return float(peak_signal_noise_ratio(hr, sr, data_range=data_range))

    # ------------------------------------------------------------------
    # SSIM
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_ssim(
        sr: np.ndarray,
        hr: np.ndarray,
        crop_border: int = 0,
        data_range: float = 255.0,
    ) -> float:
        """
        Calculate SSIM (Structural Similarity Index).

        Returns:
            SSIM score in [0, 1] (higher is better).
        """
        if sr.dtype in (np.float32, np.float64):
            data_range = 1.0

        if crop_border > 0:
            sr = sr[crop_border:-crop_border, crop_border:-crop_border]
            hr = hr[crop_border:-crop_border, crop_border:-crop_border]

        return float(
            structural_similarity(
                hr, sr,
                data_range=data_range,
                channel_axis=2 if sr.ndim == 3 else None,
            )
        )

    # ------------------------------------------------------------------
    # LPIPS
    # ------------------------------------------------------------------

    def calculate_lpips(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
    ) -> float:
        """
        Calculate LPIPS (Learned Perceptual Image Patch Similarity).

        Args:
            sr: (1, C, H, W) or (C, H, W) tensor in [0, 1].
            hr: Same shape as sr.

        Returns:
            LPIPS distance (lower is better).
        """
        if self._lpips_model is None:
            import lpips
            self._lpips_model = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_model.eval()

        if sr.dim() == 3:
            sr = sr.unsqueeze(0)
        if hr.dim() == 3:
            hr = hr.unsqueeze(0)

        # LPIPS expects [-1, 1]
        sr_norm = sr * 2.0 - 1.0
        hr_norm = hr * 2.0 - 1.0

        with torch.no_grad():
            dist = self._lpips_model(
                sr_norm.to(self.device),
                hr_norm.to(self.device),
            )
        return float(dist.item())

    # ------------------------------------------------------------------
    # All metrics
    # ------------------------------------------------------------------

    def calculate_all(
        self,
        sr: np.ndarray,
        hr: np.ndarray,
        crop_border: int = 0,
        include_lpips: bool = True,
    ) -> dict[str, float]:
        """
        Calculate all metrics at once.

        Args:
            sr: Super-resolved image (H, W, C).
            hr: Ground truth (H, W, C).
            crop_border: Border pixels to crop.
            include_lpips: Whether to compute LPIPS (requires torch tensors).

        Returns:
            Dict with 'psnr', 'ssim', and optionally 'lpips' keys.
        """
        metrics = {
            "psnr": self.calculate_psnr(sr, hr, crop_border),
            "ssim": self.calculate_ssim(sr, hr, crop_border),
        }

        if include_lpips:
            # Convert numpy to tensor for LPIPS
            if sr.dtype == np.uint8:
                sr_f = sr.astype(np.float32) / 255.0
                hr_f = hr.astype(np.float32) / 255.0
            else:
                sr_f, hr_f = sr, hr

            sr_t = torch.from_numpy(sr_f).permute(2, 0, 1).unsqueeze(0)
            hr_t = torch.from_numpy(hr_f).permute(2, 0, 1).unsqueeze(0)
            metrics["lpips"] = self.calculate_lpips(sr_t, hr_t)

        return metrics


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def calculate_psnr(sr: np.ndarray, hr: np.ndarray, **kwargs) -> float:
    """Shortcut for MetricCalculator.calculate_psnr."""
    return MetricCalculator.calculate_psnr(sr, hr, **kwargs)


def calculate_ssim(sr: np.ndarray, hr: np.ndarray, **kwargs) -> float:
    """Shortcut for MetricCalculator.calculate_ssim."""
    return MetricCalculator.calculate_ssim(sr, hr, **kwargs)
