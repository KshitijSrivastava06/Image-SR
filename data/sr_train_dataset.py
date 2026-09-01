"""
SR Train Dataset — PyTorch Dataset for training Super-Resolution models.

Supports patch-based training with random crops and augmentations.
"""

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.logger import get_logger

logger = get_logger("train_dataset")


class DIV2KTrainDataset(Dataset):
    """
    Training dataset for DIV2K.
    Loads full HR images into memory (if requested), crops random patches,
    applies augmentation, and generates LR patches on the fly via bicubic downsampling.

    Args:
        hr_dir:       Path to high-resolution images.
        patch_size:   Size of the HR crop (e.g., 96). LR patch will be patch_size // scale_factor.
        scale_factor: Scale factor (default: 4).
        cache_in_ram: If True, pre-load all images into RAM (~6-8 GB for DIV2K 800 images).
    """

    def __init__(
        self,
        hr_dir: str,
        patch_size: int = 96,
        scale_factor: int = 4,
        cache_in_ram: bool = True,
    ):
        self.hr_dir = Path(hr_dir)
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.cache_in_ram = cache_in_ram

        self.files = sorted(
            f for f in self.hr_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )

        if not self.files:
            raise RuntimeError(f"No images found in {self.hr_dir}")

        self.images = []
        if self.cache_in_ram:
            logger.info(f"Loading {len(self.files)} images into RAM...")
            for f in self.files:
                img = self._load_img(f)
                self.images.append(img)
            logger.info("Dataset cached in RAM.")

    def _load_img(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_in_ram:
            hr_img = self.images[index]
        else:
            hr_img = self._load_img(self.files[index])

        # 1. Random crop HR patch
        h, w = hr_img.shape[:2]
        
        # Ensure image is at least patch_size
        if h < self.patch_size or w < self.patch_size:
            hr_img = cv2.resize(hr_img, (max(w, self.patch_size), max(h, self.patch_size)), interpolation=cv2.INTER_CUBIC)
            h, w = hr_img.shape[:2]

        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        
        hr_patch = hr_img[top:top + self.patch_size, left:left + self.patch_size]

        # 2. Augmentation (flip, rotate)
        hr_patch = self._augment(hr_patch)

        # 3. Generate LR patch
        lr_h, lr_w = self.patch_size // self.scale_factor, self.patch_size // self.scale_factor
        lr_patch = cv2.resize(hr_patch, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # 4. To Tensor [0, 1]
        hr_tensor = torch.from_numpy(hr_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        lr_tensor = torch.from_numpy(lr_patch.astype(np.float32) / 255.0).permute(2, 0, 1)

        return lr_tensor, hr_tensor

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random flips and rotations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            
        # Random vertical flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            
        # Random 90-degree rotation
        k = random.randint(0, 3)
        if k != 0:
            img = np.rot90(img, k)
            
        return img.copy()  # Ensure it is contiguous in memory
