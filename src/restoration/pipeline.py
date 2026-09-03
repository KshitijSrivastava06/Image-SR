"""
Restoration Pipeline — Chain restoration steps with super-resolution.

Provides a configurable pipeline:
    Input → [Denoise] → [Deblur] → [Enhance] → [Super-Resolve] → Output

Each step is optional and can be independently configured.
"""

from typing import Optional

import numpy as np
import torch

from src.restoration.denoiser import Denoiser
from src.restoration.deblurrer import Deblurrer
from src.restoration.enhancer import ContrastEnhancer
from src.restoration.jpeg_deblock import JPEGDeBlocker
from src.utils.common import numpy_to_tensor, tensor_to_numpy
from src.utils.logger import get_logger

logger = get_logger("restoration_pipeline")


class RestorationPipeline:
    """
    Configurable image restoration pipeline.

    Usage::

        pipeline = RestorationPipeline(
            denoise=True, denoise_method='nlm', denoise_strength=10,
            deblur=True, deblur_method='unsharp', deblur_strength=1.5,
            enhance=True, enhance_method='clahe', enhance_strength=2.0,
            sr_model=my_sr_model,
        )
        result, intermediates = pipeline.process(image)
    """

    def __init__(
        self,
        jpeg_deblock: bool = False,
        jpeg_quality: int = 50,
        denoise: bool = False,
        denoise_method: str = "nlm",
        denoise_strength: int = 10,
        deblur: bool = False,
        deblur_method: str = "unsharp",
        deblur_strength: float = 1.5,
        enhance: bool = False,
        enhance_method: str = "clahe",
        enhance_strength: float = 2.0,
        sr_model: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.steps = []

        if jpeg_deblock:
            self.jpeg_deblocker = JPEGDeBlocker(quality=jpeg_quality)
            self.steps.append(("jpeg_deblock", self._jpeg_deblock))
        else:
            self.jpeg_deblocker = None

        if denoise:
            self.denoiser = Denoiser(method=denoise_method, strength=denoise_strength)
            self.steps.append(("denoise", self._denoise))
        else:
            self.denoiser = None

        if deblur:
            self.deblurrer = Deblurrer(method=deblur_method, strength=deblur_strength)
            self.steps.append(("deblur", self._deblur))
        else:
            self.deblurrer = None

        if enhance:
            self.enhancer = ContrastEnhancer(method=enhance_method, strength=enhance_strength)
            self.steps.append(("enhance", self._enhance))
        else:
            self.enhancer = None

        self.sr_model = sr_model
        self.device = device or torch.device("cpu")

        if sr_model is not None:
            self.steps.append(("super_resolve", self._super_resolve))

    def process(
        self,
        image: np.ndarray,
        return_intermediates: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Run the full restoration pipeline on an image.

        Args:
            image: Input image (H, W, C) uint8 RGB.
            return_intermediates: Whether to save intermediate results.

        Returns:
            (output_image, intermediates_dict)
        """
        intermediates = {"input": image.copy()} if return_intermediates else {}
        current = image.copy()

        logger.info(f"Running pipeline: {[name for name, _ in self.steps]}")

        for step_name, step_fn in self.steps:
            logger.info(f"  -> {step_name}")
            current = step_fn(current)
            if return_intermediates:
                intermediates[step_name] = current.copy()

        return current, intermediates

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _jpeg_deblock(self, img: np.ndarray) -> np.ndarray:
        return self.jpeg_deblocker.deblock(img)

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        return self.denoiser.denoise(img)

    def _deblur(self, img: np.ndarray) -> np.ndarray:
        return self.deblurrer.deblur(img)

    def _enhance(self, img: np.ndarray) -> np.ndarray:
        return self.enhancer.enhance(img)

    @torch.no_grad()
    def _super_resolve(self, img: np.ndarray) -> np.ndarray:
        """Run SR model on the image."""
        self.sr_model.eval()

        # Convert to tensor
        tensor = numpy_to_tensor(img, device=self.device)

        # Inference
        sr_tensor = self.sr_model(tensor)

        # Convert back
        sr_np = tensor_to_numpy(sr_tensor)

        # Scale to uint8
        sr_np = (sr_np * 255.0).clip(0, 255).astype(np.uint8)
        return sr_np

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_pipeline_description(self) -> str:
        """Return a human-readable description of the active pipeline."""
        if not self.steps:
            return "Empty pipeline (no steps configured)"
        return " -> ".join(name for name, _ in self.steps)
