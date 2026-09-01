"""
Tiled Inference — Process large images in patches to prevent CUDA Out-Of-Memory errors.
"""

import torch
import torch.nn as nn
from src.utils.logger import get_logger
from src.utils.inference_utils import pad_to_mod, unpad

logger = get_logger("tiled_inference")


@torch.no_grad()
def tiled_forward(
    model: nn.Module, 
    lr_tensor: torch.Tensor, 
    tile_size: int = 256, 
    overlap: int = 16, 
    scale_factor: int = 4
) -> torch.Tensor:
    """
    Run inference by breaking the image into overlapping tiles, upscaling them individually,
    and blending them back together.
    
    Args:
        model:        The loaded SR model.
        lr_tensor:    (1, C, H, W) Low-resolution input tensor.
        tile_size:    Size of each square tile in LR space.
        overlap:      Number of pixels to overlap between tiles in LR space.
        scale_factor: Upscaling factor of the model.
        
    Returns:
        sr_tensor:    (1, C, H*s, W*s) Super-resolved output tensor.
    """
    # 1. Mod-padding to prevent checkerboard artifacts on the edges of the whole image
    lr_tensor, pad_h, pad_w = pad_to_mod(lr_tensor, scale_factor)
    
    batch_size, channels, h, w = lr_tensor.size()
    out_h, out_w = h * scale_factor, w * scale_factor
    
    device = lr_tensor.device
    
    # 2. Output buffers
    # We use CPU tensors for the output buffer to save GPU memory if the image is huge
    output = torch.zeros((batch_size, channels, out_h, out_w), device="cpu")
    weight = torch.zeros((batch_size, channels, out_h, out_w), device="cpu")
    
    # Generate a linear blend ramp for overlapping regions
    # In HR space, tile and overlap are scaled
    hr_tile = tile_size * scale_factor
    hr_overlap = overlap * scale_factor
    
    tile_weight = _generate_blend_mask(hr_tile, hr_overlap, channels).to("cpu")
    
    stride = tile_size - overlap
    
    # Calculate grid size
    h_idx_list = list(range(0, h - tile_size, stride)) + [max(0, h - tile_size)]
    w_idx_list = list(range(0, w - tile_size, stride)) + [max(0, w - tile_size)]
    
    # Remove duplicates if image is smaller than tile_size
    h_idx_list = sorted(list(set(h_idx_list)))
    w_idx_list = sorted(list(set(w_idx_list)))
    
    total_tiles = len(h_idx_list) * len(w_idx_list)
    logger.info(f"Tiled inference: processing {total_tiles} tiles ({len(h_idx_list)}x{len(w_idx_list)})")
    
    # 3. Process each tile
    for i, h_idx in enumerate(h_idx_list):
        for j, w_idx in enumerate(w_idx_list):
            
            # Extract LR tile
            in_patch = lr_tensor[..., h_idx:h_idx + tile_size, w_idx:w_idx + tile_size]
            
            # Run inference on tile
            out_patch = model(in_patch)
            
            # Calculate HR indices
            out_h_idx = h_idx * scale_factor
            out_w_idx = w_idx * scale_factor
            patch_h, patch_w = out_patch.shape[2:]
            
            # Accumulate output and weight (move patch to CPU for accumulation)
            out_patch_cpu = out_patch.cpu()
            
            output[
                ..., 
                out_h_idx:out_h_idx + patch_h, 
                out_w_idx:out_w_idx + patch_w
            ] += out_patch_cpu * tile_weight[..., :patch_h, :patch_w]
            
            weight[
                ..., 
                out_h_idx:out_h_idx + patch_h, 
                out_w_idx:out_w_idx + patch_w
            ] += tile_weight[..., :patch_h, :patch_w]
            
    # 4. Normalize overlapping regions
    # Add a small epsilon to prevent division by zero in padding areas
    output = output / (weight + 1e-6)
    
    # Move back to original device
    output = output.to(device)
    
    # 5. Remove mod-padding
    output = unpad(output, pad_h, pad_w, scale_factor)
    
    return output


def _generate_blend_mask(tile_size: int, overlap: int, channels: int) -> torch.Tensor:
    """Generate a 2D Hann window-like mask for blending overlapping tiles."""
    weight = torch.ones((1, channels, tile_size, tile_size))
    
    if overlap > 0:
        # Create 1D ramps
        ramp = torch.linspace(0, 1, overlap)
        
        # Top and bottom blending
        weight[..., :overlap, :] *= ramp.view(1, 1, -1, 1)
        weight[..., -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        
        # Left and right blending
        weight[..., :, :overlap] *= ramp.view(1, 1, 1, -1)
        weight[..., :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)
        
    return weight
