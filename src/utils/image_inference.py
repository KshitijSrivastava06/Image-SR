"""Shared RGB restoration and super-resolution for stills, video, and benchmarks."""

from collections.abc import Mapping
from dataclasses import dataclass
import time

import numpy as np
import torch

from models.pretrained_weights import PRETRAINED_REGISTRY
from src.restoration.pipeline import RestorationPipeline
from src.restoration.post_processor import PostProcessor
from src.utils.common import numpy_to_tensor, tensor_to_numpy
from src.utils.inference_utils import pad_to_mod, postprocess_output, prepare_input, unpad
from src.utils.tiled_inference import input_modulus, supports_tiling, tiled_forward

MAX_OUTPUT_PIXELS = 16_000_000


@dataclass
class ImageResult:
    image: np.ndarray
    intermediates: dict[str, np.ndarray]
    seconds: float


class ImageProcessor:
    """Reuse the loaded model and configured filters across frames."""

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        scale: int,
        *,
        restoration: Mapping | None = None,
        refinement: float = 0,
    ):
        self.model = model.eval()
        self.name = model_name
        self.scale = scale
        self.device = next(model.parameters(), torch.empty(0)).device
        self.restoration = RestorationPipeline(**(restoration or {}))
        self.refiner = PostProcessor(strength=refinement) if refinement else None

    # Inference mode disables gradient tracking while producing images.
    @torch.inference_mode()
    def process(self, image: np.ndarray, *, intermediates: bool = False) -> ImageResult:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError("Provide a nonempty RGB uint8 image.")
        height, width = image.shape[:2]
        # Scale applies to each axis, so a 4x image has sixteen times as many pixels.
        if height * width * self.scale**2 > MAX_OUTPUT_PIXELS:
            raise ValueError("Output would exceed 16 megapixels. Use a smaller image or scale.")

        start = time.perf_counter()
        steps = {}
        if self.restoration.steps:
            image, steps = self.restoration.process(image, return_intermediates=intermediates)
        # Convert HWC bytes to a normalized NCHW tensor, then apply model-specific formatting.
        tensor = prepare_input(
            numpy_to_tensor(image, device=self.device), self.name, self.scale, PRETRAINED_REGISTRY,
        )
        # Large neural inputs use tiles with extra context to reduce peak accelerator memory.
        if self.name != "bicubic" and max(height, width) > 192 and supports_tiling(self.model):
            output = tiled_forward(
                self.model, tensor, tile_size=192, overlap=24,
                scale_factor=self.scale, output_device="cpu",
            )
        else:
            # Pad only if the model needs aligned input dimensions, then remove the scaled padding.
            padded, pad_h, pad_w = pad_to_mod(tensor, input_modulus(self.model))
            output = unpad(self.model(padded), pad_h, pad_w, self.scale)

        # Check both the expected shape and finite values before turning model output into pixels.
        expected = (1, 3, height * self.scale, width * self.scale)
        if output.shape != expected or not torch.isfinite(output).all():
            raise RuntimeError("The model returned invalid image data.")
        rgb = postprocess_output(output, self.name, self.scale, PRETRAINED_REGISTRY)
        # Return rounded 8-bit RGB pixels. Optional refinement runs after enlargement.
        result = np.rint(tensor_to_numpy(rgb) * 255).astype(np.uint8)
        if self.refiner:
            # Capture the same upscaled image on either side of post-processing.
            if intermediates:
                steps["before_refinement"] = result
            result = self.refiner.process(result)
            if intermediates:
                steps["refinement"] = result
        return ImageResult(result, steps, time.perf_counter() - start)
