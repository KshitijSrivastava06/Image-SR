"""
Dataset Manager — Download and manage super-resolution benchmark datasets.

Supports: DIV2K, Set5, Set14, BSD100, Urban100.
"""

import os
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger("dataset_manager")


# ---------------------------------------------------------------------------
# Dataset registry — name → download URLs and metadata
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "div2k": {
        "description": "DIV2K — 1000 2K resolution images for SR training",
        "urls": {
            "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
            "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
            "train_lr_x2": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip",
            "train_lr_x4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip",
            "valid_lr_x2": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip",
            "valid_lr_x4": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
        },
        "expected_counts": {"train": 800, "valid": 100},
    },
    "set5": {
        "description": "Set5 — 5 classic test images for SR evaluation",
        "urls": {
            "all": "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip",
        },
        "expected_counts": {"test": 5},
    },
    "set14": {
        "description": "Set14 — 14 classic test images for SR evaluation",
        "urls": {
            "all": "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk56du.zip",
        },
        "expected_counts": {"test": 14},
    },
    "bsd100": {
        "description": "BSD100 — 100 images from the Berkeley Segmentation Dataset",
        "urls": {
            "all": "https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip",
        },
        "expected_counts": {"test": 100},
    },
    "urban100": {
        "description": "Urban100 — 100 urban scene images with rich structures",
        "urls": {
            "all": "https://uofi.box.com/shared/static/65upber4dp4qlezz3ber4bjwgulaeo88.zip",
        },
        "expected_counts": {"test": 100},
    },
}


class DatasetDownloader:
    """
    Download and extract super-resolution benchmark datasets.

    Usage::

        dl = DatasetDownloader(root='data/datasets')
        dl.download('div2k')          # download DIV2K
        dl.download_all()             # download everything
        dl.list_available()           # show registered datasets
    """

    def __init__(self, root: str = "data/datasets"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_available(self) -> list[str]:
        """List all registered dataset names."""
        for name, info in DATASET_REGISTRY.items():
            logger.info(f"  {name:12s} — {info['description']}")
        return list(DATASET_REGISTRY.keys())

    def download(
        self,
        name: str,
        subset: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Download and extract a dataset.

        Args:
            name:   Dataset name (e.g., 'div2k', 'set5').
            subset: Optional subset key (e.g., 'train_hr'). None = all subsets.
            force:  Re-download even if already present.

        Returns:
            Path to the extracted dataset directory.
        """
        name = name.lower()
        if name not in DATASET_REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
            )

        info = DATASET_REGISTRY[name]
        dataset_dir = self.root / name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        urls = info["urls"]
        if subset:
            if subset not in urls:
                raise ValueError(f"Unknown subset '{subset}' for {name}. Available: {list(urls.keys())}")
            urls = {subset: urls[subset]}

        for key, url in urls.items():
            dest = dataset_dir / f"{key}.zip"
            if dest.exists() and not force:
                logger.info(f"[{name}/{key}] Archive already exists, skipping download.")
            else:
                logger.info(f"[{name}/{key}] Downloading from {url} ...")
                self._download_file(url, dest)

            # Extract
            logger.info(f"[{name}/{key}] Extracting ...")
            self._extract(dest, dataset_dir)

        logger.info(f"[{name}] Dataset ready at {dataset_dir}")
        return dataset_dir

    def download_all(self, force: bool = False) -> dict[str, Path]:
        """Download all registered datasets."""
        paths = {}
        for name in DATASET_REGISTRY:
            paths[name] = self.download(name, force=force)
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_file(url: str, dest: Path, chunk_size: int = 8192) -> None:
        """Download a file with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"Direct download failed: {e}. Trying gdown ...")
            try:
                import gdown
                gdown.download(url, str(dest), quiet=False)
                return
            except Exception:
                raise RuntimeError(f"Failed to download {url}: {e}")

        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))

    @staticmethod
    def _extract(archive: Path, dest_dir: Path) -> None:
        """Extract a zip or tar archive."""
        if archive.suffix == ".zip" or str(archive).endswith(".zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(dest_dir)
        elif archive.suffix in (".tar", ".gz", ".tgz"):
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(dest_dir)
        else:
            logger.warning(f"Unknown archive format: {archive}")
