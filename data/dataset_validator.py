"""
Dataset Validator — Verify integrity and quality of downloaded image datasets.

Checks for corrupt files, resolution anomalies, and produces dataset statistics.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.logger import get_logger

logger = get_logger("dataset_validator")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class DatasetValidator:
    """
    Validate an image dataset directory.

    Usage::

        v = DatasetValidator('data/datasets/div2k/DIV2K_train_HR')
        report = v.validate()
        v.print_report(report)
    """

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.dataset_dir}")

    def validate(self) -> dict:
        """
        Run full validation on the dataset directory.

        Returns:
            dict with keys: total, valid, corrupt, formats, resolutions, sizes, issues
        """
        image_files = self._find_images()
        report = {
            "directory": str(self.dataset_dir),
            "total": len(image_files),
            "valid": 0,
            "corrupt": [],
            "formats": {},
            "resolutions": [],
            "file_sizes": [],
            "issues": [],
        }

        if len(image_files) == 0:
            report["issues"].append("No image files found in directory.")
            return report

        for img_path in tqdm(image_files, desc="Validating", unit="img"):
            result = self._validate_image(img_path)

            if result["corrupt"]:
                report["corrupt"].append(str(img_path))
                continue

            report["valid"] += 1
            ext = img_path.suffix.lower()
            report["formats"][ext] = report["formats"].get(ext, 0) + 1
            report["resolutions"].append(result["resolution"])
            report["file_sizes"].append(result["file_size"])

        # Compute statistics
        if report["resolutions"]:
            heights = [r[0] for r in report["resolutions"]]
            widths = [r[1] for r in report["resolutions"]]
            report["stats"] = {
                "min_resolution": f"{min(heights)}x{min(widths)}",
                "max_resolution": f"{max(heights)}x{max(widths)}",
                "mean_height": round(np.mean(heights), 1),
                "mean_width": round(np.mean(widths), 1),
                "total_size_mb": round(sum(report["file_sizes"]) / (1024 ** 2), 2),
                "mean_size_kb": round(np.mean(report["file_sizes"]) / 1024, 2),
            }

        if report["corrupt"]:
            report["issues"].append(
                f"{len(report['corrupt'])} corrupt file(s) detected."
            )

        return report

    def print_report(self, report: dict) -> None:
        """Pretty-print a validation report."""
        logger.info(f"{'='*60}")
        logger.info(f"Dataset Validation Report: {report['directory']}")
        logger.info(f"{'='*60}")
        logger.info(f"  Total files:   {report['total']}")
        logger.info(f"  Valid images:  {report['valid']}")
        logger.info(f"  Corrupt:       {len(report['corrupt'])}")
        logger.info(f"  Formats:       {report['formats']}")

        if "stats" in report:
            s = report["stats"]
            logger.info(f"  Resolution:    {s['min_resolution']} → {s['max_resolution']}")
            logger.info(f"  Mean size:     {s['mean_height']} x {s['mean_width']}")
            logger.info(f"  Total size:    {s['total_size_mb']} MB")

        if report["issues"]:
            for issue in report["issues"]:
                logger.warning(f"  ⚠ {issue}")
        else:
            logger.info("  ✓ No issues found.")
        logger.info(f"{'='*60}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_images(self) -> list[Path]:
        """Recursively find all image files."""
        images = []
        for f in sorted(self.dataset_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
        return images

    @staticmethod
    def _validate_image(path: Path) -> dict:
        """Check if a single image can be loaded and read its metadata."""
        result = {
            "corrupt": False,
            "resolution": None,
            "file_size": path.stat().st_size,
        }
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                result["corrupt"] = True
            else:
                result["resolution"] = (img.shape[0], img.shape[1])  # (H, W)
        except Exception:
            result["corrupt"] = True
        return result
