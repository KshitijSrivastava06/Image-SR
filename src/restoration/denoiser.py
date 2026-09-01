"""
Denoiser — Remove noise from images using classical and filter-based methods.

Supports Gaussian noise removal (Non-Local Means) and
Salt & Pepper noise removal (median filter).
"""

import cv2
import numpy as np


class Denoiser:
    """
    Image denoising with multiple methods.

    Args:
        method: 'nlm' (Non-Local Means) or 'median'.
        strength: Denoising strength / kernel size parameter.
    """

    def __init__(self, method: str = "nlm", strength: int = 10):
        self.method = method.lower()
        self.strength = strength

    def denoise(self, img: np.ndarray) -> np.ndarray:
        """
        Remove noise from an image.

        Args:
            img: (H, W, C) uint8 image.

        Returns:
            Denoised image (same shape and dtype).
        """
        if self.method == "nlm":
            return self._nlm_denoise(img)
        elif self.method == "median":
            return self._median_denoise(img)
        elif self.method == "bilateral":
            return self._bilateral_denoise(img)
        elif self.method == "gaussian":
            return self._gaussian_denoise(img)
        else:
            raise ValueError(f"Unknown denoising method: {self.method}")

    def _nlm_denoise(self, img: np.ndarray) -> np.ndarray:
        """Non-Local Means denoising (best for Gaussian noise)."""
        if img.ndim == 3 and img.shape[2] == 3:
            # Convert to BGR for OpenCV
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(
                bgr,
                None,
                self.strength,       # h
                self.strength,       # hForColorComponents
                7,                   # templateWindowSize
                21,                  # searchWindowSize
            )
            return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        else:
            return cv2.fastNlMeansDenoising(
                img, None, self.strength,
                7, 21,
            )

    def _median_denoise(self, img: np.ndarray) -> np.ndarray:
        """Median filter (best for salt & pepper noise)."""
        ksize = max(3, self.strength | 1)  # ensure odd
        return cv2.medianBlur(img, ksize)

    def _bilateral_denoise(self, img: np.ndarray) -> np.ndarray:
        """Bilateral filter (edge-preserving smoothing)."""
        return cv2.bilateralFilter(
            img, d=9,
            sigmaColor=self.strength * 7,
            sigmaSpace=self.strength * 7,
        )

    def _gaussian_denoise(self, img: np.ndarray) -> np.ndarray:
        """Gaussian blur (simple smoothing)."""
        ksize = max(3, self.strength | 1)
        return cv2.GaussianBlur(img, (ksize, ksize), 0)


# ---------------------------------------------------------------------------
# Noise generation (for testing / data augmentation)
# ---------------------------------------------------------------------------

def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Add Gaussian noise to an image."""
    noise = np.random.randn(*img.shape) * sigma
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(img: np.ndarray, prob: float = 0.02) -> np.ndarray:
    """Add salt & pepper noise to an image."""
    noisy = img.copy()
    # Salt
    salt_mask = np.random.random(img.shape[:2]) < prob / 2
    noisy[salt_mask] = 255
    # Pepper
    pepper_mask = np.random.random(img.shape[:2]) < prob / 2
    noisy[pepper_mask] = 0
    return noisy
