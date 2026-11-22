"""
Super-Resolution Platform — Comprehensive Benchmark Suite.

Benchmarks all registered models (Bicubic, SRCNN, SRResNet, SRGAN, ESRGAN)
against validation images, measuring PSNR, SSIM, Delta-vs-Bicubic, and latency.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.model_loader import load_sr_model
from src.evaluation.metrics import MetricCalculator
from src.utils.common import get_device, get_model_size
from src.utils.logger import get_logger
from src.utils.preprocessing import mod_crop

logger = get_logger("benchmark")


def evaluate_models(
    image_dir: str,
    models_to_test: list[str],
    scale_factor: int = 4,
    num_images: int = 5,
    device: torch.device = None,
    output_dir: str = "outputs/metrics",
):
    device = device or get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img_dir_path = Path(image_dir)
    if not img_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = sorted([f for f in img_dir_path.iterdir() if f.suffix.lower() in valid_extensions])

    if not files:
        raise RuntimeError(f"No valid images found in {image_dir}")

    if num_images > 0:
        files = files[:num_images]

    logger.info(f"Loaded {len(files)} benchmark images from {image_dir}")
    logger.info(f"Benchmarking models: {models_to_test} on device: {device}")

    # Load all models
    loaded_models = {}
    for name in models_to_test:
        try:
            logger.info(f"Loading model: {name} (scale={scale_factor})...")
            m = load_sr_model(name, scale_factor=scale_factor, device=device)
            m.eval()
            loaded_models[name] = m
        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")

    mc = MetricCalculator()
    results = {name: {"psnr_hr": [], "ssim_hr": [], "psnr_bic": [], "times_ms": []} for name in loaded_models}

    print("\n" + "=" * 80)
    print(f"🔬 RUNNING BENCHMARK (Scale: {scale_factor}×, Images: {len(files)}, Device: {device})")
    print("=" * 80)

    for idx, fpath in enumerate(files, 1):
        raw_bgr = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if raw_bgr is None:
            continue
        hr_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        hr_rgb = mod_crop(hr_rgb, scale_factor)

        h, w = hr_rgb.shape[:2]
        lr_w, lr_h = max(w // scale_factor, 1), max(h // scale_factor, 1)
        lr_rgb = cv2.resize(hr_rgb, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        hr_t = torch.from_numpy(hr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        lr_t = torch.from_numpy(lr_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        # Baseline bicubic upscale
        bicubic_t = F.interpolate(lr_t, scale_factor=scale_factor, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        bicubic_np = (bicubic_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

        print(f"[{idx}/{len(files)}] Processing {fpath.name} ({lr_w}x{lr_h} -> {w}x{h})...")

        for name, model in loaded_models.items():
            start_t = time.perf_counter()
            with torch.no_grad():
                if name.lower() == "bicubic":
                    sr_t = bicubic_t
                else:
                    sr_t = model(lr_t).clamp(0.0, 1.0)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            sr_np = (sr_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)

            p_hr = mc.calculate_psnr(sr_np, hr_rgb)
            s_hr = mc.calculate_ssim(sr_np, hr_rgb)
            p_bic = mc.calculate_psnr(sr_np, bicubic_np)

            results[name]["psnr_hr"].append(p_hr)
            results[name]["ssim_hr"].append(s_hr)
            results[name]["psnr_bic"].append(p_bic)
            results[name]["times_ms"].append(elapsed_ms)

    # Compute summary averages
    summary = []
    bicubic_mean_psnr = float(np.mean(results.get("bicubic", {}).get("psnr_hr", [28.5])))

    for name in loaded_models:
        mean_p_hr = float(np.mean(results[name]["psnr_hr"]))
        mean_s_hr = float(np.mean(results[name]["ssim_hr"]))
        mean_p_bic = float(np.mean(results[name]["psnr_bic"]))
        mean_time = float(np.mean(results[name]["times_ms"]))
        delta_p = mean_p_hr - bicubic_mean_psnr

        # Health status
        if name.lower() == "bicubic":
            status = "Baseline"
        elif delta_p > 1.0:
            status = "Optimal (+{:.2f}dB)".format(delta_p)
        elif delta_p >= -0.5 and name.lower() in ("srgan", "esrgan"):
            status = "Perceptual Sharp (Expected PSNR drop)"
        elif delta_p < 0.0:
            status = "Undertrained ({:.2f}dB vs Bicubic)".format(delta_p)
        else:
            status = "Marginal (+{:.2f}dB)".format(delta_p)

        summary.append({
            "model": name.upper(),
            "psnr_hr": mean_p_hr,
            "ssim_hr": mean_s_hr,
            "delta_vs_bicubic": delta_p,
            "psnr_vs_bicubic": mean_p_bic,
            "latency_ms": mean_time,
            "status": status,
        })

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<12} {'PSNR (HR)':>10} {'SSIM (HR)':>10} {'Delta (dB)':>12} {'Latency':>10} {'Diagnosis':<30}")
    print("-" * 90)
    for row in summary:
        delta_str = f"{row['delta_vs_bicubic']:+.2f} dB" if row['model'] != "BICUBIC" else "0.00 dB"
        print(f"{row['model']:<12} {row['psnr_hr']:>9.2f}dB {row['ssim_hr']:>10.4f} {delta_str:>12} {row['latency_ms']:>8.1f}ms  {row['status']:<30}")
    print("=" * 90 + "\n")

    # Export markdown and json
    md_lines = [
        "# Model Benchmark Report",
        f"- **Scale Factor**: {scale_factor}×",
        f"- **Evaluation Dataset**: `{image_dir}` ({len(files)} images)",
        f"- **Compute Device**: `{device}`",
        f"- **Timestamp**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "| Model | PSNR vs HR (dB) | SSIM vs HR | Delta vs Bicubic | Latency (ms) | Quality Assessment |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for row in summary:
        delta_str = f"{row['delta_vs_bicubic']:+.2f} dB" if row['model'] != "BICUBIC" else "—"
        md_lines.append(
            f"| **{row['model']}** | {row['psnr_hr']:.2f} dB | {row['ssim_hr']:.4f} | {delta_str} | {row['latency_ms']:.1f} ms | {row['status']} |"
        )

    md_content = "\n".join(md_lines)
    report_file = output_path / "benchmark_report.md"
    report_file.write_text(md_content, encoding="utf-8")

    json_file = output_path / "benchmark_report.json"
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)

    logger.info(f"Saved benchmark report to {report_file} and {json_file}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Super-Resolution Platform Benchmark")
    parser.add_argument("--image_dir", type=str, default="data/datasets/div2k/DIV2K_valid_HR", help="Path to evaluation images")
    parser.add_argument("--models", nargs="+", default=["bicubic", "srcnn", "srresnet", "srgan", "esrgan"], help="Models to benchmark")
    parser.add_argument("--scale", type=int, default=4, help="Upscaling scale factor (2 or 4)")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to benchmark (0 for all)")
    parser.add_argument("--output_dir", type=str, default="outputs/metrics", help="Directory to save benchmark reports")
    args = parser.parse_args()

    evaluate_models(
        image_dir=args.image_dir,
        models_to_test=args.models,
        scale_factor=args.scale,
        num_images=args.num_images,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

