"""
Inference Utilities — Handles color space, normalization ranges, and padding.

Different pre-trained weights expect different input distributions (e.g. BGR vs RGB, 
[-1, 1] vs [0, 1]). This module normalizes all inputs to match the checkpoint's expectations.
"""

import torch
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger("inference_utils")


def prepare_input(tensor: torch.Tensor, model_name: str, scale_factor: int, registry: dict) -> torch.Tensor:
    """
    Format a [0, 1] RGB tensor to match the specific model's expectations.
    
    Args:
        tensor: (B, 3, H, W) RGB tensor in [0, 1].
        model_name: Base model name (e.g., 'esrgan_generator').
        scale_factor: Upscaling factor.
        registry: The PRETRAINED_REGISTRY containing model metadata.
    """
    key = f"{model_name}_x{scale_factor}".lower()
    info = registry.get(key, {})
    
    # 1. Color Space (RGB -> BGR if needed)
    color_space = info.get("color_space", "rgb").lower()
    if color_space == "bgr":
        tensor = tensor[:, [2, 1, 0], :, :]
        
    # 2. Input Range ([0, 1] -> [-1, 1] or [0, 255])
    input_range = info.get("input_range", "01").lower()
    if input_range == "neg11":
        tensor = tensor * 2.0 - 1.0
    elif input_range == "0255":
        tensor = tensor * 255.0
        
    # Validation check
    min_val, max_val = tensor.min().item(), tensor.max().item()
    if input_range == "01" and (min_val < -0.01 or max_val > 1.01):
        logger.warning(f"Tensor range [{min_val:.2f}, {max_val:.2f}] outside expected [0, 1]!")
    elif input_range == "neg11" and (min_val < -1.01 or max_val > 1.01):
        logger.warning(f"Tensor range [{min_val:.2f}, {max_val:.2f}] outside expected [-1, 1]!")
        
    return tensor


def postprocess_output(tensor: torch.Tensor, model_name: str, scale_factor: int, registry: dict) -> torch.Tensor:
    """
    Reverse the formatting applied by prepare_input to yield a [0, 1] RGB tensor.
    """
    key = f"{model_name}_x{scale_factor}".lower()
    info = registry.get(key, {})
    
    # 1. Output Range ([-1, 1] or [0, 255] -> [0, 1])
    output_range = info.get("output_range", "01").lower()
    if output_range == "neg11":
        tensor = (tensor + 1.0) / 2.0
    elif output_range == "0255":
        tensor = tensor / 255.0
        
    # 2. Color Space (BGR -> RGB if needed)
    color_space = info.get("color_space", "rgb").lower()
    if color_space == "bgr":
        tensor = tensor[:, [2, 1, 0], :, :]
        
    return tensor.clamp(0.0, 1.0)


def pad_to_mod(tensor: torch.Tensor, scale_factor: int) -> tuple[torch.Tensor, int, int]:
    """
    Pad tensor so its H and W are divisible by scale_factor to prevent 
    PixelShuffle checkerboard artifacts.
    
    Returns:
        padded_tensor: The reflection-padded tensor.
        pad_h: Amount of padding added to height.
        pad_w: Amount of padding added to width.
    """
    _, _, h, w = tensor.size()
    pad_h = (scale_factor - h % scale_factor) % scale_factor
    pad_w = (scale_factor - w % scale_factor) % scale_factor
    
    if pad_h == 0 and pad_w == 0:
        return tensor, 0, 0
        
    # Pad format: (left, right, top, bottom)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return padded, pad_h, pad_w


def unpad(tensor: torch.Tensor, pad_h: int, pad_w: int, scale_factor: int) -> torch.Tensor:
    """
    Remove the padding added by pad_to_mod from the upscaled output.
    Note that pad_h and pad_w are in LR space, so they must be multiplied by scale_factor.
    """
    if pad_h == 0 and pad_w == 0:
        return tensor
        
    _, _, h, w = tensor.size()
    out_h = h - (pad_h * scale_factor)
    out_w = w - (pad_w * scale_factor)
    
    return tensor[:, :, :out_h, :out_w]
