"""Measure reconstruction against folder-based high-resolution references."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_loader import MODEL_LABELS, get_weight_status, load_sr_model, model_identifier
from src.evaluation.image_analysis import fidelity, sampling_pair
from src.utils.common import get_device
from src.utils.image_inference import ImageProcessor, MAX_OUTPUT_PIXELS
from src.utils.inference_db import InferenceTracker

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_models(
    image_dir: str,
    models_to_test: list[str] | None,
    scale_factor: int = 4,
    num_images: int = 5,
    output_dir: str = "outputs/metrics",
) -> list[dict]:
    folder = Path(image_dir)
    if not folder.is_dir():
        raise ValueError(f"Reference folder does not exist: {folder}")
    if scale_factor not in (2, 4) or num_images < 0:
        raise ValueError("Choose scale 2 or 4 and a nonnegative image count.")

    paths = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    status = get_weight_status(scale_factor)
    names = list(dict.fromkeys(["bicubic", *(models_to_test or [n for n in status if status[n]])]))
    unavailable = [name for name in names if not status.get(name)]
    if unavailable:
        raise ValueError(f"Missing compatible checkpoints for {', '.join(unavailable)} at {scale_factor}×.")

    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    processors = {
        name: ImageProcessor(
            load_sr_model(name, scale_factor, device=device), model_identifier(name), scale_factor,
        )
        for name in names
    }
    measurements = {name: [] for name in names}
    for path in paths:
        try:
            with Image.open(path) as opened:
                if opened.width * opened.height > MAX_OUTPUT_PIXELS:
                    raise ValueError("Reference exceeds 16 megapixels.")
                image = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
            # Derive a single LR input from the aligned reference and share it across models.
            reference, low = sampling_pair(image, scale_factor)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            print(f"Skipping {path.name}. {exc}", file=sys.stderr)
            continue
        for name, processor in processors.items():
            result = processor.process(low)
            # Score raw model output against the same HR pixels for a fair comparison.
            scores = fidelity(reference, result.image)
            measurements[name].append((scores["PSNR (dB)"], scores["SSIM"], result.seconds * 1000))
        print(f"Measured {path.name}")
        if num_images and len(measurements["bicubic"]) >= num_images:
            break

    count = len(measurements["bicubic"])
    if count == 0:
        raise ValueError(f"No usable reference images in {folder}.")

    average_baseline = float(np.mean([sample[0] for sample in measurements["bicubic"]]))
    summary = []
    report = [
        "# Reconstruction benchmark",
        "",
        f"Dataset {folder.resolve()} · {count} images · {scale_factor}×",
        "",
        "| Model | Device | RGB PSNR (dB) | RGB SSIM | Δ PSNR vs Bicubic (dB) | Mean processing (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    tracker = InferenceTracker(ROOT / "outputs" / "inference_runs.db")
    for name, values in measurements.items():
        psnr, ssim, latency = map(float, np.mean(values, axis=0))
        delta = 0.0 if psnr == average_baseline else psnr - average_baseline
        model_device = str(processors[name].device)
        summary.append({
            "model": name, "scale": scale_factor, "device": model_device, "images": count,
            "psnr_hr": psnr, "ssim_hr": ssim, "delta_vs_bicubic": delta, "latency_ms": latency,
        })
        report.append(
            f"| {MODEL_LABELS[name]} | {model_device} | {psnr:.2f} | {ssim:.4f} | {delta:+.2f} | {latency:.1f} |",
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "benchmark_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    serializable = [
        {key: str(value) if isinstance(value, float) and math.isinf(value) else value
         for key, value in row.items()}
        for row in summary
    ]
    (target / "benchmark_report.json").write_text(
        json.dumps(serializable, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    for row in summary:
        tracker.log_inference(
            row["model"], scale_factor, row["device"], row["latency_ms"],
            dataset_name=f"{folder.resolve()} ({count} images)", psnr=row["psnr_hr"], ssim=row["ssim_hr"],
        )
    print("\n".join(report))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True, help="Folder containing high-resolution references")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_LABELS))
    parser.add_argument("--scale", type=int, choices=[2, 4], default=4)
    parser.add_argument("--num_images", type=int, default=5, help="Images to measure (0 for all)")
    parser.add_argument("--output_dir", default="outputs/metrics")
    args = parser.parse_args()
    try:
        evaluate_models(args.image_dir, args.models, args.scale, args.num_images, args.output_dir)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
