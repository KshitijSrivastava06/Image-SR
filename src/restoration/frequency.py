"""Gaussian frequency filtering on the luminance channel."""

import cv2
import numpy as np
from scipy import fft


def gaussian_frequency_filter(
    image: np.ndarray, mode: str, cutoff: float = 0.12, gain: float = 0.7,
) -> np.ndarray:
    """Apply a low-pass filter or high-frequency boost with cutoff in cycles/pixel."""
    if mode not in ("lowpass", "highboost") or not 0 < cutoff <= 0.5 or gain < 0:
        raise ValueError("Choose a supported frequency filter and a cutoff in (0, 0.5].")

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0].astype(np.float32)
    pad = min(64, max(4, int(np.ceil(2 / cutoff))))
    # Reflect the luma boundary to reduce ringing from the FFT's periodic extension.
    padded = np.pad(y, pad, mode="reflect")
    height, width = padded.shape
    # Build a radial frequency grid in cycles per pixel.
    fy = fft.fftfreq(height).astype(np.float32)[:, None]
    fx = fft.rfftfreq(width).astype(np.float32)[None, :]
    lowpass = np.exp(-(fy**2 + fx**2) / (2 * cutoff**2))
    # Low-pass attenuates fine detail. High-boost keeps DC and amplifies higher frequencies.
    transfer = lowpass if mode == "lowpass" else 1 + gain * (1 - lowpass)
    # Multiplication in the frequency domain applies the filter. Inverse FFT returns to pixels.
    filtered = fft.irfft2(fft.rfft2(padded) * transfer, s=padded.shape)
    ycrcb[..., 0] = np.clip(np.rint(filtered[pad:-pad, pad:-pad]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
