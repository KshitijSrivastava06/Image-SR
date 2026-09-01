"""
Dataset Splitter — Split image directories into train/validation/test sets.
"""

import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("dataset_splitter")


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy: bool = True,
    extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".tif"),
) -> dict[str, int]:
    """
    Split images from *source_dir* into train / val / test sub-directories.

    Args:
        source_dir:  Path to the source image directory.
        output_dir:  Root output directory (creates train/, val/, test/ inside).
        train_ratio: Fraction for training set.
        val_ratio:   Fraction for validation set.
        test_ratio:  Fraction for test set.
        seed:        Random seed for reproducibility.
        copy:        If True copies files; if False moves them.
        extensions:  Allowed file extensions.

    Returns:
        dict with counts: {'train': N, 'val': N, 'test': N}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    src = Path(source_dir)
    out = Path(output_dir)

    # Collect image files
    files = sorted(
        f for f in src.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not files:
        raise FileNotFoundError(f"No images found in {src}")

    n = len(files)
    np.random.seed(seed)
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remainder goes to test
    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    counts = {}
    for split_name, split_indices in splits.items():
        split_dir = out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in split_indices:
            src_file = files[idx]
            dst_file = split_dir / src_file.name
            if copy:
                shutil.copy2(src_file, dst_file)
            else:
                shutil.move(str(src_file), str(dst_file))
        counts[split_name] = len(split_indices)
        logger.info(f"  {split_name:5s}: {len(split_indices)} images → {split_dir}")

    logger.info(f"Split complete: {counts}")
    return counts
