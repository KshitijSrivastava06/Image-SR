"""
SR Dataset — PyTorch Datasets for Super-Resolution inference and evaluation.

Provides InferenceDataset and SRTestDataset for loading evaluation images without requiring a training loop.
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.preprocessing import mod_crop

class InferenceDataset(Dataset):
    """
    Inference-only dataset for loading single images without HR/LR pairs.

    Args:
        image_dir:     Path to images.
        scale_factor:  Scale factor (2, 4, or 8).
        color_mode:    'rgb' or 'y' (luminance only).
    """

    def __init__(
        self,
        image_dir: str,
        scale_factor: int = 4,
        color_mode: str = "rgb",
    ):
        self.image_dir = Path(image_dir)
        self.scale_factor = scale_factor
        self.color_mode = color_mode

        self.files = sorted(
            f for f in self.image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        path = self.files[index]

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Ensure image is mod-croppable
        img = mod_crop(img, self.scale_factor)

        if self.color_mode == "y":
            img = self._rgb_to_y(img)

        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return img_tensor, path.stem

    @staticmethod
    def _rgb_to_y(img: np.ndarray) -> np.ndarray:
        """Convert RGB image to Y channel (ITU-R BT.601)."""
        y = 16.0 + (65.481 * img[:, :, 0] + 128.553 * img[:, :, 1] + 24.966 * img[:, :, 2]) / 255.0
        return np.expand_dims(y.astype(np.float64), axis=2).astype(np.uint8)


class SRTestDataset(Dataset):
    """
    Test dataset that loads single images and generates LR on-the-fly.

    Useful for evaluation when only HR images are available.
    """

    def __init__(
        self,
        hr_dir: str,
        scale_factor: int = 4,
        downscale_method: str = "bicubic",
    ):
        self.hr_dir = Path(hr_dir)
        self.scale_factor = scale_factor

        interp_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
        }
        self.interp = interp_map.get(downscale_method, cv2.INTER_CUBIC)

        self.files = sorted(
            f for f in self.hr_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        hr_path = self.files[index]

        hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = mod_crop(hr, self.scale_factor)

        h, w = hr.shape[:2]
        lr = cv2.resize(hr, (w // self.scale_factor, h // self.scale_factor),
                        interpolation=self.interp)

        hr_tensor = torch.from_numpy(hr.astype(np.float32) / 255.0).permute(2, 0, 1)
        lr_tensor = torch.from_numpy(lr.astype(np.float32) / 255.0).permute(2, 0, 1)

        return lr_tensor, hr_tensor, hr_path.stem
