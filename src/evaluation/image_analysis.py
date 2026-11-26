"""Image measurements shared by diagnostics and the reconstruction experiment."""

from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from scipy import fft

from src.evaluation.metrics import MetricCalculator


def sampling_pair(image: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Align the reference with the sampling grid and generate its bicubic LR image."""
    height, width = image.shape[:2]
    # Trim the HR reference so downsampling and exact-scale reconstruction align pixel for pixel.
    height, width = height - height % scale, width - width % scale
    if min(height, width) < max(7, scale):
        raise ValueError("The reference is too small for this scale. Use a larger image.")
    reference = image[:height, :width]
    resized = Image.fromarray(reference).resize(
        (width // scale, height // scale), Image.Resampling.BICUBIC,
    )
    return reference, np.asarray(resized)


def fidelity(reference: np.ndarray, result: np.ndarray) -> dict[str, float | None]:
    """Compute full-reference RGB PSNR and SSIM on an aligned reconstruction."""
    return {
        "PSNR (dB)": MetricCalculator.calculate_psnr(result, reference),
        "SSIM": MetricCalculator.calculate_ssim(result, reference)
        if min(reference.shape[:2]) >= 3 else None,
    }


def jpeg_roundtrip(image: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    """Return encoded JPEG bytes and the decoded RGB image at the chosen quality."""
    buffer = BytesIO()
    # The JPEG encoder uses quantization and 4:2:0 chroma subsampling.
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality, subsampling=2)
    with Image.open(buffer) as decoded:
        return buffer.getvalue(), np.asarray(decoded.convert("RGB"))


def color_channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the luminance and two chroma planes in Y, Cb, Cr order."""
    # OpenCV returns Y, Cr, Cb. Swap the last two planes for the UI's YCbCr order.
    y, cr, cb = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb))
    return y, cb, cr


def edge_map(luma: np.ndarray, method: str) -> np.ndarray:
    if method == "Canny":
        # Canny gives a thresholded edge map. Sobel below gives gradient magnitude instead.
        return cv2.Canny(luma, 60, 150, L2gradient=True)
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1)
    return np.clip(cv2.magnitude(gx, gy) / 4, 0, 255).astype(np.uint8)


def log_spectrum(luma: np.ndarray) -> np.ndarray:
    """Show centered log-magnitude after windowing to reduce border leakage."""
    height, width = luma.shape
    win_y = np.hanning(height) if height > 2 else np.ones(height)
    win_x = np.hanning(width) if width > 2 else np.ones(width)
    windowed = luma * win_y[:, None] * win_x[None, :]
    # Center zero frequency and compress magnitudes for display.
    spectrum = np.log1p(np.abs(fft.fftshift(fft.fft2(windowed))))
    peak = spectrum.max()
    return (spectrum * (255 / peak)).astype(np.uint8) if peak else np.zeros_like(luma)
