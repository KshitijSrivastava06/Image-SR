"""
Model Evaluator — Run SR models against test datasets and collect metrics.

Handles inference, metric computation, and result aggregation.
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import MetricCalculator
from src.utils.common import get_device, tensor_to_numpy, save_image, ensure_dir
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """
    Evaluate a super-resolution model on a test dataset.

    Usage::

        evaluator = ModelEvaluator(model, test_loader)
        results = evaluator.evaluate()
        evaluator.save_results('outputs/metrics/srcnn_results.csv')
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        model_name: str = "model",
        device: Optional[torch.device] = None,
        save_images: bool = True,
        output_dir: str = "outputs/images",
    ):
        self.device = device or get_device()
        self.model = model.to(self.device).eval()
        self.test_loader = test_loader
        self.model_name = model_name
        self.save_images = save_images
        self.output_dir = ensure_dir(output_dir)
        self.metrics_calc = MetricCalculator(device=self.device)

    @torch.no_grad()
    def evaluate(self, include_lpips: bool = True) -> dict:
        """
        Run evaluation over the entire test set.

        Returns:
            dict with:
                - 'per_image': list of per-image metric dicts
                - 'mean': aggregated mean metrics
                - 'std': standard deviations
                - 'total_time': total inference time (seconds)
                - 'mean_time': mean per-image inference time
        """
        results = []
        total_time = 0.0

        logger.info(f"Evaluating {self.model_name} on {len(self.test_loader.dataset)} images ...")

        for batch in tqdm(self.test_loader, desc=f"Eval {self.model_name}"):
            # Handle both 2-tuple (lr, hr) and 3-tuple (lr, hr, name) datasets
            if len(batch) == 3:
                lr, hr, names = batch
            else:
                lr, hr = batch
                names = [None] * lr.shape[0]

            lr = lr.to(self.device)
            hr = hr.to(self.device)

            # Inference with timing
            start = time.perf_counter()
            sr = self.model(lr)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            total_time += elapsed

            # Process each image in batch
            for i in range(sr.shape[0]):
                sr_np = tensor_to_numpy(sr[i])
                hr_np = tensor_to_numpy(hr[i])

                metrics = self.metrics_calc.calculate_all(
                    sr_np, hr_np, include_lpips=include_lpips
                )
                metrics["inference_time"] = elapsed / sr.shape[0]

                name = names[i] if isinstance(names[i], str) else f"img_{len(results):04d}"
                metrics["image_name"] = name
                results.append(metrics)

                # Save SR image
                if self.save_images:
                    save_image(sr_np, self.output_dir / f"{name}_sr.png")

        # Aggregate
        psnr_vals = [r["psnr"] for r in results]
        ssim_vals = [r["ssim"] for r in results]
        times = [r["inference_time"] for r in results]

        summary = {
            "per_image": results,
            "mean": {
                "psnr": float(np.mean(psnr_vals)),
                "ssim": float(np.mean(ssim_vals)),
                "inference_time": float(np.mean(times)),
            },
            "std": {
                "psnr": float(np.std(psnr_vals)),
                "ssim": float(np.std(ssim_vals)),
            },
            "total_time": total_time,
            "num_images": len(results),
            "model_name": self.model_name,
        }

        if include_lpips and "lpips" in results[0]:
            lpips_vals = [r["lpips"] for r in results]
            summary["mean"]["lpips"] = float(np.mean(lpips_vals))
            summary["std"]["lpips"] = float(np.std(lpips_vals))

        logger.info(
            f"Results: PSNR={summary['mean']['psnr']:.2f}±{summary['std']['psnr']:.2f}  "
            f"SSIM={summary['mean']['ssim']:.4f}±{summary['std']['ssim']:.4f}"
        )

        self._results = summary
        return summary

    def save_results(self, path: str = "outputs/metrics/results.csv") -> None:
        """Save per-image results to CSV."""
        if not hasattr(self, "_results"):
            raise RuntimeError("No results to save. Run evaluate() first.")

        df = pd.DataFrame(self._results["per_image"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")

    def save_to_db(self, dataset_name: str, scale_factor: int = 4) -> None:
        """Save evaluation results to SQLite tracking database."""
        if not hasattr(self, "_results"):
            raise RuntimeError("No results to save. Run evaluate() first.")

        from src.utils.inference_db import InferenceTracker
        tracker = InferenceTracker()

        # Log individual image results
        for row in self._results["per_image"]:
            tracker.log_inference(
                model_arch=self.model_name,
                scale_factor=scale_factor,
                device=str(self.device),
                inference_time_ms=row["inference_time"] * 1000,
                input_filename=row.get("image_name"),
                dataset_name=dataset_name,
                psnr=row.get("psnr"),
                ssim=row.get("ssim")
            )
        logger.info(f"Saved {len(self._results['per_image'])} evaluation records to database.")
