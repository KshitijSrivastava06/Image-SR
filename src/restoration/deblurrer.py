"""
Deblurrer — Restore blurred images using deconvolution and sharpening.

Supports Wiener deconvolution, unsharp masking, and Laplacian sharpening.
"""

import cv2
import numpy as np
from scipy import fft


class Deblurrer:
    """
    Image deblurring with multiple methods.

    Args:
        method:   'wiener', 'unsharp', or 'laplacian'.
        strength: Sharpening / deconvolution strength parameter.
    """

    def __init__(self, method: str = "unsharp", strength: float = 1.5):
        self.method = method.lower()
        self.strength = strength

    def deblur(self, img: np.ndarray) -> np.ndarray:
        """
        Restore a blurred image.

        Args:
            img: (H, W, C) uint8 image.

        Returns:
            Sharpened / deblurred image.
        """
        if self.method == "wiener":
            return self._wiener_deconvolution(img)
        elif self.method == "unsharp":
            return self._unsharp_mask(img)
        elif self.method == "laplacian":
            return self._laplacian_sharpen(img)
        else:
            raise ValueError(f"Unknown deblurring method: {self.method}")

    def _unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Unsharp masking: sharpen by subtracting a blurred version.

        sharp = original + strength * (original - blurred)
        """
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        # Unsharp masking adds a scaled high-frequency detail layer back to the original.
        sharpened = cv2.addWeighted(
            img, 1.0 + self.strength,
            blurred, -self.strength,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _laplacian_sharpen(self, img: np.ndarray) -> np.ndarray:
        """Sharpen using Laplacian edge detection."""
        if img.ndim == 3:
            # Process each channel
            channels = cv2.split(img)
            sharpened = []
            for ch in channels:
                lap = cv2.Laplacian(ch, cv2.CV_64F)
                sharp = ch.astype(np.float64) - self.strength * lap
                sharpened.append(np.clip(sharp, 0, 255).astype(np.uint8))
            return cv2.merge(sharpened)
        else:
            lap = cv2.Laplacian(img, cv2.CV_64F)
            sharp = img.astype(np.float64) - self.strength * lap
            return np.clip(sharp, 0, 255).astype(np.uint8)

    def _wiener_deconvolution(self, img: np.ndarray) -> np.ndarray:
        """
        Simplified Wiener deconvolution for motion blur.

        Uses a simple blur kernel and frequency-domain filtering.
        """
        radius = min(max(2, int(self.strength * 2.5)), img.shape[1] // 2)
        if radius == 0:
            return img.copy()
        # A centered horizontal blur kernel has a real frequency response.
        padded = np.pad(img.astype(np.float32), [(0, 0), (radius, radius)] + [(0, 0)] * (img.ndim - 2),
                        mode="reflect")
        kernel = np.zeros(padded.shape[1], dtype=np.float32)
        kernel[:radius + 1] = kernel[-radius:] = 1 / (2 * radius + 1)
        response = fft.rfft(kernel).real
        # Regularization limits noise amplification when the blur response is close to zero.
        wiener = response / (response**2 + 0.01)
        transfer = wiener[None, :, None] if img.ndim == 3 else wiener[None, :]
        restored = fft.irfft(
            fft.rfft(padded, axis=1) * transfer, n=padded.shape[1], axis=1,
        )
        return np.clip(np.rint(restored[:, radius:-radius]), 0, 255).astype(np.uint8)
