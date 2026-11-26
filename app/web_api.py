"""Local FastAPI application for the standalone super-resolution studio."""

from base64 import b64encode
from functools import lru_cache
from io import BytesIO
import json
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Annotated, Literal

import torch
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.background import BackgroundTask

from app.model_loader import MODEL_LABELS, get_weight_status, load_sr_model, model_identifier
from models.pretrained_weights import PRETRAINED_REGISTRY, find_local_weights
from src.evaluation.image_analysis import (
    color_channels, edge_map, fidelity, jpeg_roundtrip, log_spectrum, sampling_pair,
)
from src.evaluation.metrics import MetricCalculator
from src.utils.common import get_device
from src.utils.image_inference import ImageProcessor, MAX_OUTPUT_PIXELS
from src.utils.video import VIDEO_FORMATS, process_video

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
ASSETS = WEB / "assets"
OUTPUT = ROOT / "outputs"
app = FastAPI(title="Image & Video Super-Resolution")
# Static routes serve the browser files and sample media from the same local origin.
app.mount("/assets", StaticFiles(directory=ASSETS), name="assets")
app.mount("/ui", StaticFiles(directory=WEB), name="ui")
# Serialize model execution so simultaneous requests do not compete for model memory.
processing = Lock()
DEVICE = get_device()
if DEVICE.type == "cpu":
    torch.set_num_threads(min(torch.get_num_threads(), 4))


class Options(BaseModel):
    # Reject unknown controls and validate values before selecting a model.
    model_config = ConfigDict(extra="forbid")

    model: Literal["bicubic", "srcnn", "srresnet", "srgan", "esrgan"] = "esrgan"
    scale: Literal[2, 4] = 4
    experiment: bool = False
    input_quality: int | None = Field(default=None, ge=5, le=95)
    denoise: Literal["off", "gaussian", "median", "bilateral", "nlm"] = "off"
    intensity: Literal["off", "histogram", "clahe", "gamma", "auto_wb"] = "off"
    sharpening: Literal["off", "unsharp", "laplacian", "wiener"] = "off"
    frequency: Literal["off", "lowpass", "highboost"] = "off"
    cutoff: float = Field(default=0.12, ge=0.03, le=0.4)
    deblock: bool = False
    refinement: bool = False
    crf: int = Field(default=20, ge=15, le=35)


def parse_options(raw: str) -> Options:
    try:
        options = Options.model_validate_json(raw)
    except ValidationError as exc:
        raise HTTPException(422, "The processing settings are invalid.") from exc
    # A trained checkpoint is tied to a supported model and scale.
    if not get_weight_status(options.scale)[options.model]:
        raise HTTPException(422, "This model has no checkpoint for the selected scale.")
    return options


def checkpoint_path(name: str, scale: int) -> Path | None:
    identifier = model_identifier(name)
    registered = PRETRAINED_REGISTRY.get(f"{identifier}_x{scale}", {})
    if registered.get("url"):
        return ROOT / "models" / "pretrained" / registered["filename"]
    return find_local_weights(identifier, scale)


@lru_cache(maxsize=4)
def cached_model(name: str, scale: int, signature: int):
    return load_sr_model(name, scale, device=DEVICE)


def make_processor(options: Options, *, baseline: bool = False) -> ImageProcessor:
    # The reference baseline uses Bicubic with no optional restoration.
    name = "bicubic" if baseline else options.model
    settings = {}
    if not baseline:
        if options.denoise != "off":
            settings.update(denoise=True, denoise_method=options.denoise, denoise_strength=5)
        if options.intensity != "off":
            strength = 1.4 if options.intensity == "gamma" else 2.0
            settings.update(enhance=True, enhance_method=options.intensity, enhance_strength=strength)
        if options.sharpening != "off":
            settings.update(deblur=True, deblur_method=options.sharpening, deblur_strength=0.6)
        if options.frequency != "off":
            settings.update(frequency_mode=options.frequency, frequency_cutoff=options.cutoff)
        if options.deblock:
            settings.update(jpeg_deblock=True, jpeg_quality=options.input_quality or 40)
    path = checkpoint_path(name, options.scale)
    # A changed checkpoint timestamp gives the model cache a new key.
    signature = path.stat().st_mtime_ns if path and path.is_file() else 0
    return ImageProcessor(
        cached_model(name, options.scale, signature), model_identifier(name), options.scale,
        restoration=settings, refinement=0.4 if options.refinement and not baseline else 0,
    )


@lru_cache(maxsize=16)
def read_provenance(path: Path, signature: int) -> dict:
    payload = torch.load(path, weights_only=True, map_location="cpu")
    return payload.get("provenance", {}) if isinstance(payload, dict) else {}


@app.get("/")
def homepage():
    return FileResponse(WEB / "index.html")


@app.get("/api/models")
def models():
    statuses = {scale: get_weight_status(scale) for scale in (2, 4)}
    entries = []
    for name, label in MODEL_LABELS.items():
        scales = [scale for scale, availability in statuses.items() if availability[name]]
        meta = {}
        path = checkpoint_path(name, 4) if name in ("srcnn", "srresnet", "srgan") else None
        if path:
            try:
                meta = read_provenance(path, path.stat().st_mtime_ns)
            except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
                meta = {}
        entries.append({
            "id": name, "label": label, "scales": scales, "source": meta.get("source"),
            "epochs": meta.get("epochs_completed"), "training_images": meta.get("training_images"),
            "selected_epoch": meta.get("selected_epoch"), "validation_images": meta.get("validation_images"),
        })

    # Prefer the locally generated benchmark and fall back to the bundled snapshot.
    report = OUTPUT / "presentation" / "metrics" / "benchmark_report.json"
    if not report.is_file():
        report = ASSETS / "div2k-benchmark.json"
    rows = []
    if report.is_file():
        try:
            rows = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rows = []
    return {"device": str(DEVICE).upper(), "models": entries, "benchmark": rows}


def read_image(data: bytes | None, options: Options) -> np.ndarray:
    if data is None:
        path = ASSETS / ("demo-reference.png" if options.experiment else "demo-input.png")
        data = path.read_bytes()
    # Pillow decodes the upload and applies camera orientation before returning RGB pixels.
    with Image.open(BytesIO(data)) as image:
        scale = 1 if options.experiment else options.scale
        if image.width * image.height * scale**2 > MAX_OUTPUT_PIXELS:
            raise ValueError("Output exceeds 16 megapixels. Use a smaller image or select 2×.")
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def encode(image: np.ndarray, *, preview: bool = False) -> str:
    bitmap = Image.fromarray(image)
    if preview:
        # Only visual previews are resized. The downloadable output keeps its full dimensions.
        bitmap.thumbnail((1000, 720), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    bitmap.save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def size(image: np.ndarray) -> list[int]:
    return [image.shape[1], image.shape[0]]


def comparison_metrics(reference: np.ndarray, result: np.ndarray) -> dict:
    return {key: value if value is None or np.isfinite(value) else "∞"
            for key, value in fidelity(reference, result).items()}


def diagnostics(baseline: np.ndarray, output: np.ndarray) -> dict:
    height, width = output.shape[:2]
    # Use identical crop coordinates on both outputs for the diagnostic comparison.
    y, x = max(0, (height - 384) // 2), max(0, (width - 512) // 2)
    before, _, _ = color_channels(baseline[y:y + 384, x:x + 512])
    after, cb, cr = color_channels(output[y:y + 384, x:x + 512])
    return {
        "edges": {method: [encode(edge_map(image, method), preview=True) for image in (before, after)]
                  for method in ("Canny", "Sobel")},
        "spectrum": [encode(log_spectrum(image), preview=True) for image in (before, after)],
        "channels": [encode(channel, preview=True) for channel in (after, cb, cr)],
        "histogram": [np.bincount(image.ravel(), minlength=256).tolist() for image in (before, after)],
    }


def limited_read(source: UploadFile | None, maximum: int) -> bytes | None:
    if source is None:
        return None
    # Read one extra byte to distinguish an exact-limit upload from an oversized upload.
    payload = source.file.read(maximum + 1)
    if not payload or len(payload) > maximum:
        raise HTTPException(413, f"Choose a nonempty file under {maximum // 1024**2} MiB.")
    return payload


@app.post("/api/image")
def image(
    settings: Annotated[str, Form()],
    upload: Annotated[UploadFile | None, File()] = None,
):
    options = parse_options(settings)
    source = limited_read(upload, 32 * 1024**2)
    try:
        original = read_image(source, options)
        # Only the experiment treats the supplied image as ground truth and creates a degraded input.
        reference, image = sampling_pair(original, options.scale) if options.experiment else (None, original)
        if options.input_quality is not None and reference is not None:
            # Simulate lossy compression on the LR image before feeding it to both methods.
            _, image = jpeg_roundtrip(image, options.input_quality)
        with processing:
            processor = make_processor(options)
            result = processor.process(image, intermediates=True)
            # Reuse the output when the selected run already is plain Bicubic.
            baseline = result.image
            if options.model != "bicubic" or processor.restoration.steps or processor.refiner:
                baseline = make_processor(options, baseline=True).process(image).image
        # PSNR and SSIM need aligned reference pixels, which ordinary uploads do not provide.
        metrics = None if reference is None else {
            "processed": comparison_metrics(reference, result.image),
            "bicubic": comparison_metrics(reference, baseline),
        }
        return {
            "input": encode(image, preview=True), "input_size": size(image),
            "baseline": encode(baseline, preview=True),
            "reference": encode(reference, preview=True) if reference is not None else None,
            "output": encode(result.image), "output_size": size(result.image),
            "seconds": round(result.seconds, 3), "metrics": metrics,
            "stages": {name: encode(stage, preview=True) for name, stage in result.intermediates.items()},
            "diagnostics": diagnostics(baseline, result.image),
        }
    except (cv2.error, OSError, RuntimeError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, f"Could not process the image. {exc}") from exc


@app.post("/api/jpeg")
def jpeg(upload: Annotated[UploadFile, File()], quality: Annotated[int, Form(ge=5, le=95)]):
    try:
        payload = limited_read(upload, 64 * 1024**2)
        with Image.open(BytesIO(payload)) as bitmap:
            if bitmap.width * bitmap.height > MAX_OUTPUT_PIXELS:
                raise ValueError("Image exceeds 16 megapixels.")
            image = np.asarray(bitmap.convert("RGB"))
        encoded, decoded = jpeg_roundtrip(image, quality)
        # Here the reference is the pre-export output. This score measures JPEG loss only.
        psnr = MetricCalculator.calculate_psnr(decoded, image)
        filename = "enhanced-image.jpg"
        return Response(encoded, media_type="image/jpeg", headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-PSNR": f"{psnr:.2f}",
        })
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, f"Could not export JPEG. {exc}") from exc


@app.post("/api/video")
def video(
    settings: Annotated[str, Form()],
    upload: Annotated[UploadFile | None, File()] = None,
):
    options = parse_options(settings)
    extension = Path(upload.filename or "").suffix.lower() if upload else ".mp4"
    if extension not in VIDEO_FORMATS:
        raise HTTPException(422, "Choose an MP4, MOV, AVI, MKV, or WebM clip.")
    source = limited_read(upload, 50 * 1024**2)
    OUTPUT.mkdir(exist_ok=True)
    # Keep the source and generated MP4 alive until the response has been sent.
    temporary = TemporaryDirectory(dir=OUTPUT, prefix="video-")
    try:
        root = Path(temporary.name)
        if source is not None:
            path = root / f"source{extension}"
            path.write_bytes(source)
        else:
            path = ASSETS / "demo-clip.mp4"

        output = root / "enhanced.mp4"
        with processing:
            result = process_video(path, output, make_processor(options), crf=options.crf)
        return FileResponse(
            output, media_type="video/mp4", filename="enhanced-video.mp4",
            headers={
                "X-Frame-Count": str(result.frame_count), "X-FPS": f"{result.fps:g}",
                "X-Input-Size": "×".join(map(str, result.source_size)),
                "X-Output-Size": "×".join(map(str, result.output_size)),
            },
            # Remove the temporary media after the download response completes.
            background=BackgroundTask(temporary.cleanup),
        )
    except (cv2.error, OSError, RuntimeError, ValueError, StopIteration) as exc:
        temporary.cleanup()
        raise HTTPException(422, f"Could not process the clip. {exc}") from exc
