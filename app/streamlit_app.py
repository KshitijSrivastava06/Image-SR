"""Interactive image and video super-resolution coursework app."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import time

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
from skimage import data as sample_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_loader import MODEL_LABELS, get_weight_status, load_sr_model, model_identifier
from src.evaluation.image_analysis import (
    color_channels, edge_map, fidelity, jpeg_roundtrip, log_spectrum, sampling_pair,
)
from src.evaluation.metrics import MetricCalculator
from src.utils.common import get_device
from src.utils.image_inference import ImageProcessor, MAX_OUTPUT_PIXELS
from src.utils.inference_db import InferenceTracker
from src.utils.video import VIDEO_FORMATS, process_video

DB_PATH = PROJECT_ROOT / "outputs" / "inference_runs.db"
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
STAGE_LABELS = {
    "input": "Input",
    "jpeg_deblock": "JPEG deblocking",
    "denoise": "Spatial smoothing",
    "deblur": "Sharpening",
    "enhance": "Intensity enhancement",
    "frequency_filter": "Frequency filtering",
}
PROCESS_ERRORS = (ValueError, OSError, RuntimeError, StopIteration, Image.DecompressionBombError, cv2.error)


@st.cache_resource(max_entries=4)
def cached_model(name: str, scale: int):
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    return load_sr_model(name, scale, device=device)


def make_processor(model: str, scale: int, settings: dict, refinement: float) -> ImageProcessor:
    return ImageProcessor(
        cached_model(model, scale), model_identifier(model), scale,
        restoration=settings, refinement=refinement,
    )


def read_image(payload: bytes, scale: int) -> np.ndarray:
    with Image.open(BytesIO(payload)) as opened:
        if opened.width * opened.height * scale**2 > MAX_OUTPUT_PIXELS:
            raise ValueError("Output would exceed 16 megapixels. Use a smaller image or scale.")
        return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))


def read_settings():
    scale = st.sidebar.radio("Scale", [2, 4], horizontal=True, format_func=lambda x: f"{x}×")
    status = get_weight_status(scale)
    model = st.sidebar.selectbox(
        "Upscaling method", [name for name, ready in status.items() if ready],
        format_func=MODEL_LABELS.__getitem__,
    )
    with st.sidebar.expander("Restoration", expanded=True):
        preset = st.selectbox(
            "Preset", ["None", "Photo Cleanup", "Old Photo Restore", "JPEG Fix", "Custom"],
        )
        settings = {
            "None": {},
            "Photo Cleanup": dict(denoise=True, denoise_method="nlm", denoise_strength=5),
            "Old Photo Restore": dict(
                denoise=True, denoise_method="nlm", denoise_strength=12,
                enhance=True, enhance_method="clahe", enhance_strength=2.5,
            ),
            "JPEG Fix": dict(jpeg_deblock=True, jpeg_quality=40),
            "Custom": {},
        }[preset]
        refinement = 0.0
        if preset == "Custom":
            refinement = custom_settings(settings)
    return model, scale, settings, refinement


def custom_settings(settings: dict) -> float:
    if st.checkbox("JPEG deblocking"):
        settings.update(
            jpeg_deblock=True, jpeg_quality=st.slider("Estimated source JPEG quality", 10, 95, 45),
        )
    spatial = st.selectbox(
        "Spatial smoothing", ["Off", "Gaussian", "Median", "Bilateral", "NLM"],
    )
    if spatial != "Off":
        strength = (
            st.slider("Kernel size", 3, 15, 5, step=2)
            if spatial in ("Gaussian", "Median") else st.slider("Smoothing strength", 1, 20, 5)
        )
        settings.update(denoise=True, denoise_method=spatial.lower(), denoise_strength=strength)

    intensity = st.selectbox(
        "Intensity transformation", ["Off", "Histogram equalization", "CLAHE", "Gamma", "White balance"],
    )
    if intensity != "Off":
        method = {
            "Histogram equalization": "histogram", "White balance": "auto_wb",
        }.get(intensity, intensity.lower())
        strength = {
            "gamma": lambda: st.slider("Gamma", 0.4, 3.0, 1.2, step=0.1),
            "clahe": lambda: st.slider("CLAHE clip limit", 0.5, 5.0, 2.0, step=0.1),
        }
        settings.update(
            enhance=True, enhance_method=method,
            enhance_strength=strength[method]() if method in strength else 1.0,
        )

    frequency = st.selectbox("Frequency filtering", ["Off", "Gaussian low-pass", "Gaussian high-boost"])
    if frequency != "Off":
        settings.update(
            frequency_mode="lowpass" if frequency == "Gaussian low-pass" else "highboost",
            frequency_cutoff=st.slider("Cutoff (cycles/pixel)", 0.03, 0.40, 0.12, step=0.01),
        )
        if settings["frequency_mode"] == "highboost":
            settings["frequency_gain"] = st.slider("High-frequency gain", 0.1, 2.0, 0.7)
    sharpening = st.selectbox("Deblurring and sharpening", ["Off", "Unsharp", "Laplacian", "Wiener"])
    if sharpening != "Off":
        settings.update(
            deblur=True, deblur_method=sharpening.lower(),
            deblur_strength=st.slider("Deblur strength", 0.1, 3.0, 0.6),
        )
    if st.checkbox("Post-SR refinement"):
        return st.slider("Refinement amount", 0.1, 1.0, 0.4)
    return 0.0


def show_images(parts):
    for column, (label, image) in zip(st.columns(len(parts)), parts):
        height, width = image.shape[:2]
        column.image(image, caption=f"{label} · {width} × {height}", width="stretch")


def detail_crop(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y = (image.shape[0] - height) // 2
    x = (image.shape[1] - width) // 2
    return image[y:y + height, x:x + width]


def show_diagnostics(baseline: np.ndarray, output: np.ndarray):
    with st.expander("Compare edges, color and frequency"):
        method = st.radio("Edge detector", ["Canny", "Sobel"], horizontal=True)
        height, width = (min(dimension, 640) for dimension in output.shape[:2])
        before = detail_crop(baseline, width, height)
        after = detail_crop(output, width, height)
        y_before, _, _ = color_channels(before)
        y_after, cb, cr = color_channels(after)
        st.caption("Diagnostics use the same center region of both outputs at native resolution.")
        show_images([
            (f"Bicubic · {method}", edge_map(y_before, method)),
            (f"Processed · {method}", edge_map(y_after, method)),
        ])
        show_images([("Y · Luma", y_after), ("Cb · Blue chroma", cb), ("Cr · Red chroma", cr)])
        st.line_chart(pd.DataFrame({
            "Bicubic": np.bincount(y_before.ravel(), minlength=256),
            "Processed": np.bincount(y_after.ravel(), minlength=256),
        }), x_label="Luma level", y_label="Pixels")
        show_images([
            ("Bicubic · DFT log magnitude", log_spectrum(y_before)),
            ("Processed · DFT log magnitude", log_spectrum(y_after)),
        ])
        st.caption("Each spectrum is windowed and scaled independently for display.")


def show_image_export(output: np.ndarray, name: str):
    format_name = st.radio("Export format", ["PNG", "JPEG"], horizontal=True)
    if format_name == "JPEG":
        quality = st.slider("JPEG export quality", 5, 95, 85)
        payload, decoded = jpeg_roundtrip(output, quality)
        psnr = format_value(MetricCalculator.calculate_psnr(decoded, output))
        st.caption(f"4:2:0 JPEG · {len(payload) / 1024:.1f} KiB · {psnr} dB vs the uncompressed output")
        suffix, mime = "jpg", "image/jpeg"
    else:
        buffer = BytesIO()
        Image.fromarray(output).save(buffer, format="PNG")
        payload = buffer.getvalue()
        suffix, mime = "png", "image/png"
    st.download_button(
        "Download enhanced image", payload, f"{name}-enhanced.{suffix}", mime,
        type="primary", on_click="ignore",
    )


def format_value(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return "∞" if np.isinf(value) else f"{value:.{decimals}f}"


def show_fidelity(scores):
    columns = st.columns(2)
    for column, (metric, decimals) in zip(columns, (("PSNR (dB)", 2), ("SSIM", 4))):
        processed = scores["Processed"][metric]
        bicubic = scores["Bicubic"][metric]
        column.metric(metric, format_value(processed, decimals))
        column.caption(f"Bicubic {format_value(bicubic, decimals)}")
    st.caption("Both scores use the aligned high-resolution reference on RGB pixels.")


def render_image_page(model, scale, settings, refinement, experiment=False):
    st.title("Reconstruction experiment" if experiment else "Image super-resolution")
    st.caption(
        "Create a low-resolution sample from a known reference and measure reconstruction fidelity."
        if experiment else "Restore a low-resolution image and compare the result with bicubic interpolation."
    )
    sample = st.checkbox("Use the sample photo", value=experiment)
    upload = None if sample else st.file_uploader(
        "High-resolution reference" if experiment else "Low-resolution image", type=IMAGE_TYPES,
    )
    if not sample and upload is None:
        return

    name = "sample" if sample else Path(upload.name).stem
    payload = b"" if sample else upload.getvalue()
    source_id = "sample" if sample else sha256(payload).hexdigest()
    jpeg_quality = None
    if experiment and st.checkbox("Simulate JPEG compression on the low-resolution input"):
        jpeg_quality = st.slider("Input JPEG quality", 5, 95, 30)

    try:
        image = sample_data.astronaut() if sample else read_image(payload, 1 if experiment else scale)
        if sample and not experiment:
            _, image = sampling_pair(image, 4)
        reference = None
        if experiment:
            reference, image = sampling_pair(image, scale)
        if jpeg_quality is not None:
            encoded, image = jpeg_roundtrip(image, jpeg_quality)
            st.caption(f"Low-resolution JPEG · {len(encoded) / 1024:.1f} KiB")
    except PROCESS_ERRORS as error:
        st.error(f"Could not prepare image. {error}")
        return

    height, width = image.shape[:2]
    st.caption(f"{width} × {height} → {width * scale} × {height * scale} · {MODEL_LABELS[model]}")
    if reference is not None and reference.shape != (height * scale, width * scale, 3):
        st.error("The reference could not be aligned with the sample.")
        return

    key = (source_id, experiment, model, scale, tuple(settings.items()), refinement, jpeg_quality)
    state = st.session_state.get("image_run")
    if state and state[0] != key:
        st.session_state.pop("image_run")
        state = None

    if st.button("Run experiment" if experiment else "Enhance image", type="primary"):
        try:
            with st.spinner("Processing image…"):
                processor = make_processor(model, scale, settings, refinement)
                result = processor.process(image, intermediates=True)
                baseline = result.image
                if model != "bicubic" or settings or refinement:
                    baseline = make_processor("bicubic", scale, {}, 0).process(image).image

                scores = {}
                if reference is not None:
                    scores["Processed"] = fidelity(reference, result.image)
                    scores["Bicubic"] = (
                        scores["Processed"] if baseline is result.image else fidelity(reference, baseline)
                    )
            state = key, result, baseline, scores
            st.session_state["image_run"] = state
            processed_scores = scores.get("Processed", {})
            try:
                InferenceTracker(DB_PATH).log_inference(
                    model, scale, str(processor.device), result.seconds * 1000,
                    input_filename=name, input_width=width, input_height=height,
                    psnr=processed_scores.get("PSNR (dB)"), ssim=processed_scores.get("SSIM"),
                )
            except (OSError, sqlite3.Error):
                st.warning("The result is ready. The local run history could not be saved.")
        except PROCESS_ERRORS as error:
            st.error(f"Could not process image. {error}")
            return

    if state is None:
        st.image(image, caption="Low-resolution input", width=min(width * 2, 440))
        return

    _, result, baseline, scores = state
    st.subheader("Results")
    st.caption(f"Processing time {result.seconds:.2f} s")
    if reference is not None:
        show_fidelity(scores)
        show_images([("Ground truth", reference), ("Bicubic", baseline), ("Processed", result.image)])
    else:
        show_images([("Input", image), ("Bicubic", baseline), ("Processed", result.image)])

    if result.intermediates:
        with st.expander("View restoration stages"):
            stage = st.selectbox(
                "Pipeline stage", list(result.intermediates), format_func=STAGE_LABELS.__getitem__,
            )
            st.image(result.intermediates[stage], width="stretch")
    show_diagnostics(baseline, result.image)
    show_image_export(result.image, name)


def write_demo_clip(path: Path):
    photo = sample_data.astronaut()[10:370:3, 60:444:3].copy()
    writer = imageio_ffmpeg.write_frames(str(path), (96, 72), fps=12, macro_block_size=1)
    try:
        writer.send(None)
        for offset in range(24):
            writer.send(np.ascontiguousarray(photo[12:84, offset:offset + 96]))
    finally:
        writer.close()


def render_video_page(model, scale, settings, refinement):
    st.title("Video super-resolution")
    st.caption("Enhance each frame and export an H.264 MP4 with the first audio track, when present.")
    sample = st.checkbox("Use the sample clip")
    upload = None if sample else st.file_uploader(
        "Video clip", type=[ext[1:] for ext in VIDEO_FORMATS], max_upload_size=50,
    )
    crf = st.slider("H.264 compression (CRF)", 15, 35, 20, help="Lower values retain more detail.")
    st.caption("Up to 15 seconds, 300 frames, and 2.1 megapixels per output frame.")
    if upload is None and not sample:
        return
    name = "sample" if sample else Path(upload.name).stem
    payload = b"" if sample else upload.getvalue()
    source_id = "sample" if sample else sha256(payload).hexdigest()
    key = (source_id, model, scale, tuple(settings.items()), refinement, crf)
    state = st.session_state.get("video_run")
    if state and state[0] != key:
        st.session_state.pop("video_run")
        state = None

    if st.button("Enhance video", type="primary"):
        progress = st.progress(0.0, "Preparing clip…")
        started = time.perf_counter()
        try:
            processor = make_processor(model, scale, settings, refinement)
            with TemporaryDirectory() as directory:
                source = Path(directory) / ("source.mp4" if sample else f"source{Path(upload.name).suffix}")
                output = Path(directory) / "output.mp4"
                if sample:
                    write_demo_clip(source)
                else:
                    source.write_bytes(payload)
                result = process_video(
                    source, output, processor, crf=crf,
                    progress=lambda done, total: progress.progress(
                        min(0.98, done / total), f"Processing frame {done}",
                    ),
                )
                state = key, result, output.read_bytes(), time.perf_counter() - started
                st.session_state["video_run"] = state
        except PROCESS_ERRORS as error:
            st.error(f"Could not process video. {error}")
            return
        finally:
            progress.empty()

    if state is not None:
        _, result, encoded, seconds = state
        st.subheader("Results")
        width, height = result.output_size
        st.caption(
            f"{width} × {height} · {result.frame_count} frames · {result.fps:g} fps · "
            f"{seconds:.2f} s processing · {len(encoded) / 1024:.1f} KiB"
        )
        st.video(encoded)
        show_images([("First input frame", result.input_frame), ("First processed frame", result.output_frame)])
        st.download_button(
            "Download enhanced video", encoded, f"{name}-enhanced.mp4", "video/mp4",
            type="primary", on_click="ignore",
        )


def render_models_page():
    st.title("Models and benchmarks")
    x2, x4 = get_weight_status(2), get_weight_status(4)
    st.dataframe([
        {"Model": MODEL_LABELS[name], "2×": "Available" if x2[name] else "Training required",
         "4×": "Available" if x4[name] else "Training required"}
        for name in MODEL_LABELS
    ], hide_index=True, width="stretch")
    st.subheader("Recorded dataset benchmarks")
    try:
        records = InferenceTracker(DB_PATH).get_benchmark_stats()
        if records.empty:
            st.caption("Run the dataset benchmark command in the README to populate this table.")
        else:
            st.dataframe(records, hide_index=True, width="stretch")
    except (OSError, sqlite3.Error, pd.errors.DatabaseError):
        st.warning("The local benchmark history could not be opened.")


def main():
    st.set_page_config(page_title="Image & Video Super-Resolution", layout="wide")
    st.sidebar.title("Super-resolution")
    page = st.sidebar.radio(
        "Workspace", ["Image", "Experiment", "Video", "Models and benchmarks"],
    )
    if page == "Models and benchmarks":
        render_models_page()
    else:
        options = read_settings()
        if page == "Video":
            render_video_page(*options)
        else:
            render_image_page(*options, experiment=page == "Experiment")


if __name__ == "__main__":
    main()
