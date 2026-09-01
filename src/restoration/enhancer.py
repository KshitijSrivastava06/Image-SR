"""
Enhancer — Image contrast and visibility improvement.

Provides CLAHE, gamma correction, and auto white balance.
"""

import cv2
import numpy as np


class ContrastEnhancer:
    """
    Image contrast and visibility enhancement.

    Args:
        method:   'clahe', 'gamma', or 'auto_wb'.
        strength: Method-specific strength parameter.
    """

    def __init__(self, method: str = "clahe", strength: float = 2.0):
        self.method = method.lower()
        self.strength = strength

    def enhance(self, img: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast/visibility.

        Args:
            img: (H, W, C) uint8 image in RGB.

        Returns:
            Enhanced image.
        """
        if self.method == "clahe":
            return self._clahe(img)
        elif self.method == "gamma":
            return self._gamma_correction(img)
        elif self.method == "auto_wb":
            return self._auto_white_balance(img)
        elif self.method == "histogram":
            return self._histogram_eq(img)
        else:
            raise ValueError(f"Unknown enhancement method: {self.method}")

    def _clahe(self, img: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Applied in LAB color space to preserve color balance.
        """
        # RGB → LAB
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.strength,
            tileGridSize=(8, 8),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # LAB → RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

    def _gamma_correction(self, img: np.ndarray) -> np.ndarray:
        """
        Gamma correction.

        gamma < 1: brighter (lighten shadows)
        gamma > 1: darker (darken highlights)
        """
        gamma = self.strength
        inv_gamma = 1.0 / gamma

        # Build lookup table
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)

        return cv2.LUT(img, table)

    def _auto_white_balance(self, img: np.ndarray) -> np.ndarray:
        """Simple gray-world white balance assumption."""
        result = img.copy().astype(np.float64)
        avg_r = np.mean(result[:, :, 0])
        avg_g = np.mean(result[:, :, 1])
        avg_b = np.mean(result[:, :, 2])
        avg = (avg_r + avg_g + avg_b) / 3

        result[:, :, 0] *= avg / max(avg_r, 1e-8)
        result[:, :, 1] *= avg / max(avg_g, 1e-8)
        result[:, :, 2] *= avg / max(avg_b, 1e-8)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _histogram_eq(self, img: np.ndarray) -> np.ndarray:
        """Histogram equalization in YCrCb space."""
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
