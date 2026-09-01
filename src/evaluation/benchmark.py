"""
Benchmark Runner — Compare all models on benchmark datasets.

Runs Bicubic, SRCNN, SRResNet, SRGAN, ESRGAN on test datasets and
produces a comprehensive comparison table.
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
from src.utils.common import get_device, get_model_size, tensor_to_numpy, ensure_dir
from src.utils.logger import get_logger

logger = get_logger("benchmark")


class BenchmarkRunner:
    """
    Run all models against test datasets and collect comprehensive results.

    Usage::

        runner = BenchmarkRunner(
            models={'bicubic': model_b, 'srcnn': model_s, ...},
            test_loader=test_loader,
        )
        results = runner.run()
        runner.save_report('outputs/metrics/benchmark_results.csv')
    """

    def __init__(
        self,
        models: dict[str, torch.nn.Module],
        test_loader: DataLoader,
        device: Optional[torch.device] = None,
        output_dir: str = "outputs/metrics",
    ):
        self.device = device or get_device()
        self.models = {
            name: model.to(self.device).eval()
            for name, model in models.items()
        }
        self.test_loader = test_loader
        self.output_dir = ensure_dir(output_dir)
        self.metrics_calc = MetricCalculator(device=self.device)
        self.results = []

    @torch.no_grad()
    def run(self, include_lpips: bool = True) -> pd.DataFrame:
        """
        Benchmark all models.

        Returns:
            DataFrame with model comparison metrics.
        """
        records = []

        for model_name, model in self.models.items():
            logger.info(f"Benchmarking: {model_name}")

            # Model size
            size_info = get_model_size(model)

            psnr_list, ssim_list, lpips_list, times = [], [], [], []

            for batch in tqdm(self.test_loader, desc=model_name, leave=False):
                if len(batch) == 3:
                    lr, hr, _ = batch
                else:
                    lr, hr = batch

                lr = lr.to(self.device)
                hr = hr.to(self.device)

                start = time.perf_counter()
                sr = model(lr)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed / lr.shape[0])

                for i in range(sr.shape[0]):
                    sr_np = tensor_to_numpy(sr[i])
                    hr_np = tensor_to_numpy(hr[i])

                    psnr_list.append(
                        MetricCalculator.calculate_psnr(sr_np, hr_np)
                    )
                    ssim_list.append(
                        MetricCalculator.calculate_ssim(sr_np, hr_np)
                    )

                    if include_lpips:
                        lpips_list.append(
                            self.metrics_calc.calculate_lpips(sr[i], hr[i])
                        )

            record = {
                "model": model_name,
                "psnr_mean": np.mean(psnr_list),
                "psnr_std": np.std(psnr_list),
                "ssim_mean": np.mean(ssim_list),
                "ssim_std": np.std(ssim_list),
                "inference_time_ms": np.mean(times) * 1000,
                "total_params": size_info["total_params"],
                "model_size_mb": size_info["size_mb"],
            }

            if include_lpips and lpips_list:
                record["lpips_mean"] = np.mean(lpips_list)
                record["lpips_std"] = np.std(lpips_list)

            records.append(record)
            logger.info(
                f"  {model_name}: PSNR={record['psnr_mean']:.2f}  "
                f"SSIM={record['ssim_mean']:.4f}  "
                f"Time={record['inference_time_ms']:.1f}ms  "
                f"Params={record['total_params']:,}"
            )

        self.results = pd.DataFrame(records)
        return self.results

    def save_report(self, path: Optional[str] = None) -> Path:
        """Save benchmark results to CSV."""
        if self.results is None or (isinstance(self.results, pd.DataFrame) and self.results.empty):
            raise RuntimeError("No results. Run benchmark first.")

        path = Path(path or self.output_dir / "benchmark_results.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(path, index=False)
        logger.info(f"Benchmark report saved: {path}")
        return path

    def save_to_db(self, dataset_name: str, scale_factor: int = 4) -> None:
        """Save benchmark results to the SQLite inference tracking database."""
        if self.results is None or (isinstance(self.results, pd.DataFrame) and self.results.empty):
            raise RuntimeError("No results to save to database. Run benchmark first.")

        from src.utils.inference_db import InferenceTracker
        tracker = InferenceTracker()

        for _, row in self.results.iterrows():
            tracker.log_inference(
                model_arch=row["model"],
                scale_factor=scale_factor,
                device=str(self.device),
                inference_time_ms=row["inference_time_ms"],
                dataset_name=dataset_name,
                psnr=row.get("psnr_mean"),
                ssim=row.get("ssim_mean")
            )
        logger.info(f"Saved {len(self.results)} benchmark records to database.")

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table to console."""
        if self.results is None or (isinstance(self.results, pd.DataFrame) and self.results.empty):
            print("No results. Run benchmark first.")
            return

        print("\n" + "=" * 90)
        print("MODEL COMPARISON BENCHMARK")
        print("=" * 90)
        print(
            f"{'Model':<12} {'PSNR↑':>10} {'SSIM↑':>10} "
            f"{'LPIPS↓':>10} {'Time(ms)':>10} {'Params':>12} {'Size(MB)':>10}"
        )
        print("-" * 90)

        for _, row in self.results.iterrows():
            lpips_str = f"{row.get('lpips_mean', 0):.4f}" if "lpips_mean" in row else "N/A"
            print(
                f"{row['model']:<12} "
                f"{row['psnr_mean']:>8.2f}dB "
                f"{row['ssim_mean']:>10.4f} "
                f"{lpips_str:>10} "
                f"{row['inference_time_ms']:>8.1f}ms "
                f"{row['total_params']:>12,} "
                f"{row['model_size_mb']:>8.2f}MB"
            )
        print("=" * 90)
