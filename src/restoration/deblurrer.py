"""
Deblurrer — Restore blurred images using deconvolution and sharpening.

Supports Wiener deconvolution, unsharp masking, and Laplacian sharpening.
"""

import cv2
import numpy as np


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
        result = img.copy().astype(np.float64)

        # Create motion blur kernel
        kernel_size = max(5, int(self.strength * 5))
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        # Noise-to-signal ratio (regularization)
        nsr = 0.01

        if img.ndim == 3:
            channels = cv2.split(result)
            deconv = []
            for ch in channels:
                deconv.append(self._wiener_channel(ch, kernel, nsr))
            result = cv2.merge(deconv)
        else:
            result = self._wiener_channel(result, kernel, nsr)

        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _wiener_channel(
        channel: np.ndarray,
        kernel: np.ndarray,
        nsr: float,
    ) -> np.ndarray:
        """Apply Wiener deconvolution to a single channel."""
        h, w = channel.shape
        kh, kw = kernel.shape

        # Pad kernel to image size
        padded_kernel = np.zeros((h, w))
        padded_kernel[:kh, :kw] = kernel

        # FFT
        F_img = np.fft.fft2(channel)
        F_kernel = np.fft.fft2(padded_kernel)

        # Wiener filter
        F_kernel_conj = np.conj(F_kernel)
        F_kernel_sq = np.abs(F_kernel) ** 2
        wiener = F_kernel_conj / (F_kernel_sq + nsr)

        result = np.real(np.fft.ifft2(F_img * wiener))
        return result
