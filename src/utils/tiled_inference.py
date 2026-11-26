"""Run inference on spatial tiles with surrounding context."""

import torch
import torch.nn as nn

from src.utils.inference_utils import pad_to_mod, unpad


def supports_tiling(model: nn.Module) -> bool:
    """Only use tiling with models whose normalization uses fixed statistics."""
    return not any(
        isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm))
        and (layer.training or not layer.track_running_stats)
        for layer in model.modules()
    )


def input_modulus(model: nn.Module) -> int:
    """Return the input alignment required by the supported SR architectures."""
    # PixelUnshuffle(2) requires even coordinates and dimensions for consistent pixel grouping.
    unshuffle = getattr(model, "pixel_unshuffle", None)
    return unshuffle.downscale_factor if isinstance(unshuffle, nn.PixelUnshuffle) else 1


@torch.no_grad()
def tiled_forward(
    model: nn.Module,
    lr_tensor: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 16,
    scale_factor: int = 4,
    *,
    output_device: str | torch.device | None = None,
) -> torch.Tensor:
    """Give each tile a context halo and copy its center into the result."""
    alignment = input_modulus(model)
    if tile_size <= 0 or overlap < 0 or tile_size % alignment or overlap % alignment:
        raise ValueError("Tile size and overlap must be nonnegative and aligned, with a positive tile size.")
    lr_tensor, pad_h, pad_w = pad_to_mod(lr_tensor, alignment)
    batch, channels, height, width = lr_tensor.shape
    scale = scale_factor
    # Keep the assembled canvas on CPU when requested to free accelerator memory.
    destination = output_device or lr_tensor.device
    result = lr_tensor.new_empty(
        (batch, channels, height * scale, width * scale), device=destination,
    )

    for top in range(0, height, tile_size):
        bottom = min(top + tile_size, height)
        start_y, end_y = max(top - overlap, 0), min(bottom + overlap, height)
        for left in range(0, width, tile_size):
            right = min(left + tile_size, width)
            start_x, end_x = max(left - overlap, 0), min(right + overlap, width)
            # Use neighboring pixels for context, then keep only the tile core in the output canvas.
            patch = model(lr_tensor[..., start_y:end_y, start_x:end_x])
            center = patch[
                ...,
                (top - start_y) * scale:(bottom - start_y) * scale,
                (left - start_x) * scale:(right - start_x) * scale,
            ]
            result[..., top * scale:bottom * scale, left * scale:right * scale].copy_(center)

    return unpad(result, pad_h, pad_w, scale)
