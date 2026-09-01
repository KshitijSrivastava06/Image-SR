"""
Visualizer — Comparison plots and side-by-side images for SR evaluation.

Generates matplotlib figures for model comparison and image quality analysis.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from src.utils.common import ensure_dir
from src.utils.logger import get_logger

logger = get_logger("visualizer")


def plot_metric_comparison(
    results: pd.DataFrame,
    save_path: str = "outputs/metrics/comparison_bar.png",
) -> None:
    """
    Create a grouped bar chart comparing all models on PSNR/SSIM/LPIPS.

    Args:
        results: DataFrame with columns: model, psnr_mean, ssim_mean, [lpips_mean].
        save_path: Output image path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    models = results["model"].tolist()
    colors = ["#4A90D9", "#67B26F", "#E8A838", "#E85D75", "#9B6FD1"][:len(models)]
    x = np.arange(len(models))

    # PSNR
    ax = axes[0]
    bars = ax.bar(x, results["psnr_mean"], color=colors, edgecolor="white", linewidth=1.5)
    if "psnr_std" in results.columns:
        ax.errorbar(x, results["psnr_mean"], yerr=results["psnr_std"],
                     fmt="none", capsize=4, color="black", alpha=0.6)
    ax.set_title("PSNR (dB) ↑", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("dB")
    for bar, val in zip(bars, results["psnr_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # SSIM
    ax = axes[1]
    bars = ax.bar(x, results["ssim_mean"], color=colors, edgecolor="white", linewidth=1.5)
    ax.set_title("SSIM ↑", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    for bar, val in zip(bars, results["ssim_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    # LPIPS or Inference Time
    ax = axes[2]
    if "lpips_mean" in results.columns:
        bars = ax.bar(x, results["lpips_mean"], color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title("LPIPS ↓", fontsize=13)
        for bar, val in zip(bars, results["lpips_mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    else:
        bars = ax.bar(x, results["inference_time_ms"], color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title("Inference Time (ms) ↓", fontsize=13)
        ax.set_ylabel("ms")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison chart saved: {save_path}")


def create_side_by_side(
    images: dict[str, np.ndarray],
    save_path: str = "outputs/images/comparison.png",
    title: str = "Super-Resolution Comparison",
) -> None:
    """
    Create a side-by-side comparison of multiple SR results.

    Args:
        images: Dict mapping method name → image array (H, W, C).
        save_path: Output path.
        title: Figure title.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n == 1:
        axes = [axes]

    for ax, (name, img) in zip(axes, images.items()):
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(name, fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Side-by-side comparison saved: {save_path}")


def create_zoom_comparison(
    images: dict[str, np.ndarray],
    crop_box: tuple[int, int, int, int] = (100, 100, 200, 200),
    save_path: str = "outputs/images/zoom_comparison.png",
) -> None:
    """
    Create zoomed-in patch comparisons for texture detail analysis.

    Args:
        images: Dict mapping method name → image (H, W, C).
        crop_box: (top, left, bottom, right) crop coordinates.
        save_path: Output path.
    """
    top, left, bottom, right = crop_box
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.suptitle("Zoom-In Comparison", fontsize=14, fontweight="bold")

    for i, (name, img) in enumerate(images.items()):
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0, 1)

        # Full image
        axes[0, i].imshow(img)
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis("off")

        # Draw crop box
        rect = plt.Rectangle(
            (left, top), right - left, bottom - top,
            linewidth=2, edgecolor="red", facecolor="none",
        )
        axes[0, i].add_patch(rect)

        # Zoomed patch
        patch = img[top:bottom, left:right]
        axes[1, i].imshow(patch)
        axes[1, i].set_title(f"{name} (zoomed)", fontsize=10)
        axes[1, i].axis("off")

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Zoom comparison saved: {save_path}")


def plot_training_history(
    history: dict,
    save_path: str = "outputs/metrics/training_history.png",
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss' and optionally 'val_loss' lists.
        save_path: Output path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)

    if history.get("val_loss"):
        val_epochs = np.linspace(1, len(history["train_loss"]),
                                  len(history["val_loss"]))
        ax.plot(val_epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training History", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ensure_dir(Path(save_path).parent)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training history saved: {save_path}")
