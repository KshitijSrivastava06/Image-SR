"""Evaluation module for super-resolution quality assessment."""

from src.evaluation.metrics import MetricCalculator, calculate_psnr, calculate_ssim
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.visualizer import (
    plot_metric_comparison,
    create_side_by_side,
    create_zoom_comparison,
    plot_training_history,
)
