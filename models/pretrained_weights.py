"""
Pre-Trained Weights Manager — Centralized system for downloading and loading pre-trained checkpoints.
"""

import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from models import get_model
from src.utils.logger import get_logger

logger = get_logger("pretrained_weights")


PRETRAINED_REGISTRY = {
    "srcnn_x4": {
        "url": None,  # No compatible public checkpoint — uses random weights
        "sha256": None,
        "filename": "srcnn_x4.pth",
        "source": "none (lightweight model, fast to retrain)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "srresnet_x4": {
        "url": None,  # Original S3 URL is dead (403). No compatible HF mirror available.
        "sha256": None,
        "filename": "srresnet_x4.pth",
        "source": "none (identical arch to SRGAN generator — use SRGAN instead)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "srgan_generator_x4": {
        "url": None,  # Same architecture as SRResNet. No public checkpoint with matching BN layers.
        "sha256": None,
        "filename": "srgan_generator_x4.pth",
        "source": "none (architecture includes BN — incompatible with BasicSR MSRResNet weights)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "esrgan_generator_x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": None,
        "filename": "RealESRGAN_x4plus.pth",
        "source": "xinntao/Real-ESRGAN (Official GitHub Release v0.1.0)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
    "esrgan_generator_x2": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "sha256": None,
        "filename": "RealESRGAN_x2plus.pth",
        "source": "xinntao/Real-ESRGAN (Official GitHub Release v0.2.1)",
        "color_space": "rgb",
        "input_range": "01",
        "output_range": "01",
    },
}


def download_file(url: str, dest_path: Path):
    """Download a file with a progress bar."""
    logger.info(f"Downloading {url} to {dest_path}")
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest_path.name) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)


def download_pretrained(model_name: str, scale_factor: int, cache_dir: str = "models/pretrained") -> Optional[Path]:
    """Download pre-trained weights if not cached."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    key = f"{model_name}_x{scale_factor}"
    if key not in PRETRAINED_REGISTRY:
        logger.warning(f"No pre-trained weights registered for {key}")
        return None
        
    info = PRETRAINED_REGISTRY[key]
    if "url" not in info or not info["url"]:
        logger.warning(f"No URL provided for {key}. Relying on local weights fallback.")
        return None
        
    dest_path = cache_path / info["filename"]
    if not dest_path.exists():
        download_file(info["url"], dest_path)
        
    return dest_path


def remap_esrgan_keys(state_dict):
    """Remap xinntao ESRGAN keys to our ESRGANGenerator keys."""
    new_state_dict = {}
    for k, v in state_dict.items():
        # Standard ESRGAN to our architecture
        if k.startswith("RRDB_trunk."):
            new_k = k.replace("RRDB_trunk.", "trunk.")
            new_k = new_k.replace("RDB", "db")
            
        # Real-ESRGAN to our architecture
        elif k.startswith("body."):
            new_k = k.replace("body.", "trunk.")
            new_k = new_k.replace(".rdb", ".db")
        elif k.startswith("trunk_conv."):
            new_k = k.replace("trunk_conv.", "conv_trunk.")
        elif k.startswith("conv_body."):
            new_k = k.replace("conv_body.", "conv_trunk.")
        elif k.startswith("upconv1."):
            new_k = k.replace("upconv1.", "upsample.1.")
        elif k.startswith("conv_up1."):
            new_k = k.replace("conv_up1.", "upsample.1.")
        elif k.startswith("upconv2."):
            new_k = k.replace("upconv2.", "upsample.4.")
        elif k.startswith("conv_up2."):
            new_k = k.replace("conv_up2.", "upsample.4.")
        elif k.startswith("HRconv."):
            new_k = k.replace("HRconv.", "conv_last.0.")
        elif k.startswith("conv_hr."):
            new_k = k.replace("conv_hr.", "conv_last.0.")
        elif k.startswith("conv_last."):
            new_k = k.replace("conv_last.", "conv_last.2.")
        else:
            new_k = k
            
        new_state_dict[new_k] = v
        
    return new_state_dict


def load_pretrained(model_name: str, scale_factor: int, device: Optional[torch.device] = None, cache_dir: str = "models/pretrained") -> nn.Module:
    """Instantiate model and load pre-trained weights."""
    model = get_model(model_name, scale_factor=scale_factor)
    
    weight_path = download_pretrained(model_name, scale_factor, cache_dir)
    
    state_dict = None
    if weight_path and weight_path.exists():
        logger.info(f"Loading pre-trained weights from {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    else:
        logger.warning(f"Could not download weights for {model_name}_x{scale_factor}. Trying local fallback.")
        local_name = model_name.replace("_generator", "")
        local_paths = [
            Path(f"models/{local_name}/final.pth"),
            Path(f"outputs/checkpoints/{local_name}/best_model.pth"),
            Path(f"outputs/checkpoints/{local_name}/final.pth"),
        ]
        for lp in local_paths:
            if lp.exists():
                logger.info(f"Loading local weights from {lp}")
                state_dict = torch.load(lp, map_location="cpu", weights_only=False)
                break
            
    if state_dict is not None:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            
        if "esrgan" in model_name.lower() and weight_path and ("RRDB" in weight_path.name or "RealESRGAN" in weight_path.name):
            state_dict = remap_esrgan_keys(state_dict)

        if ("srresnet" in model_name.lower() or "srgan" in model_name.lower()) and getattr(model, "use_batchnorm", True) is False:
            from models.srresnet import fold_srresnet_bn
            state_dict = fold_srresnet_bn(state_dict)
            
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Successfully loaded pre-trained weights (strict=True)")
        except RuntimeError as e:
            logger.warning(f"Strict loading failed. Falling back to strict=False. Reason: {e}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            # Diagnostic logging
            total_keys = len(model.state_dict())
            loaded_keys = total_keys - len(missing_keys)
            logger.warning(f"Loaded {loaded_keys}/{total_keys} keys.")
            
            if missing_keys:
                logger.warning(f"Missing {len(missing_keys)} keys (e.g. {missing_keys[:3]})")
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys (e.g. {unexpected_keys[:3]})")
                
            # Hard failure if too much is missing
            if len(missing_keys) > total_keys * 0.5:
                raise RuntimeError(
                    f"Catastrophic state dict mismatch: >50% keys missing ({len(missing_keys)} missing). "
                    "This checkpoint architecture is incompatible with the model."
                )
    else:
        logger.warning(f"No weights found for {model_name}. Model is randomly initialized.")
        
    # Set to eval and freeze parameters for inference
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    if device:
        model = model.to(device)
        
    return model
