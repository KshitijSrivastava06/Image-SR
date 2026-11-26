"""
Training entry point for Image Super-Resolution.
Supports SRCNN, SRResNet, and SRGAN.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import random
import yaml

import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_manager import DatasetDownloader
from data import DIV2KTrainDataset, SRTestDataset
from models import get_model
from models.srgan import Discriminator
from src.training import Trainer, GANTrainer
from src.utils.logger import get_logger
from src.utils.common import get_device, save_checkpoint

logger = get_logger("train_script")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(opt_config: str, params: iter, lr: float, betas: tuple = (0.9, 0.999)):
    if opt_config.lower() == "adam":
        return optim.Adam(params, lr=lr, betas=betas)
    raise ValueError(f"Unsupported optimizer: {opt_config}")


def build_scheduler(sched_config: str, optimizer: optim.Optimizer, config: dict):
    if sched_config.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config["training"].get("lr_step_size", 50),
            gamma=config["training"].get("lr_gamma", 0.5)
        )
    elif sched_config.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
            eta_min=1e-7
        )
    return None


def initialize_reconstruction(layer: torch.nn.Conv2d, *, color_preserving: bool = False):
    torch.nn.init.normal_(layer.weight, std=0.002)
    torch.nn.init.constant_(layer.bias, 0.0 if color_preserving else 0.5)


def run_training(config_path: str, resume_path: str = None):
    config = load_config(config_path)
    device = get_device()
    if "seed" in config:
        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    
    model_name = config["model"]["name"]
    scale_factor = config["model"]["scale_factor"]
    
    logger.info(f"--- Starting training for {model_name.upper()} ---")
    
    # 1. Datasets
    logger.info("Setting up datasets...")
    dl = DatasetDownloader()
    # Ensure datasets exist
    if not Path(config["data"]["train_hr_dir"]).exists():
        dl.download("div2k", subset="train_hr")
    if not Path(config["data"]["val_hr_dir"]).exists():
        dl.download("div2k", subset="valid_hr")
        
    train_dataset = DIV2KTrainDataset(
        hr_dir=config["data"]["train_hr_dir"],
        patch_size=config["data"]["patch_size"],
        scale_factor=scale_factor,
        cache_in_ram=config["data"].get("cache_in_ram", True),
        antialias=config["data"].get("antialias", False),
    )
    val_dataset = SRTestDataset(
        hr_dir=config["data"]["val_hr_dir"],
        scale_factor=scale_factor,
        antialias=config["data"].get("antialias", False),
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=device.type == "cuda"
    )
    
    # 2. Model
    # Use the YAML architecture settings to build the model and later reconstruct it for inference.
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("name", "generator_weights")}
    if model_name == "srgan":
        generator = get_model("srgan_generator", **model_kwargs).to(device)
        if "generator_weights" in config["model"]:
            weights_target = Path(config["model"]["generator_weights"])
            if not weights_target.exists():
                fallback_target = Path("outputs/checkpoints/srresnet/best_model.pth")
                if fallback_target.exists():
                    logger.warning(f"Specified generator_weights '{weights_target}' not found. Falling back to '{fallback_target}'")
                    weights_target = fallback_target
                else:
                    logger.warning(f"Generator warm-start weights '{weights_target}' not found. Training from scratch.")
                    weights_target = None
            if weights_target is not None:
                ckpt = torch.load(weights_target, map_location=device, weights_only=False)
                # Support loading from best_model.pth or direct state_dict
                state_dict = ckpt.get("model_state_dict", ckpt.get("generator_state_dict", ckpt))
                if getattr(generator, "use_batchnorm", True) is False:
                    from models.srresnet import fold_srresnet_bn
                    state_dict = fold_srresnet_bn(state_dict, num_blocks=generator.num_residual_blocks)
                # Warm-start SRGAN from SRResNet so adversarial training begins with a working reconstructor.
                generator.load_state_dict(state_dict)
                logger.info(f"Loaded generator warm-start weights from {weights_target}")
            
        if config.get("initialize_output") and (not config["model"].get("generator_weights") or weights_target is None):
            initialize_reconstruction(generator.final, color_preserving=generator.color_preserving)
        discriminator = Discriminator(
            pool_size=config["training"].get("discriminator_pool_size", 6),
            num_channels=config["training"].get("discriminator_channels", 64),
        ).to(device)
        
        opt_g = build_optimizer(
            config["training"]["generator"]["optimizer"],
            generator.parameters(),
            config["training"]["generator"]["lr"],
            tuple(config["training"]["generator"].get("betas", [0.9, 0.999]))
        )
        opt_d = build_optimizer(
            config["training"]["discriminator"]["optimizer"],
            discriminator.parameters(),
            config["training"]["discriminator"]["lr"],
            tuple(config["training"]["discriminator"].get("betas", [0.9, 0.999]))
        )
        
        sched_g = build_scheduler(config["training"].get("lr_scheduler", ""), opt_g, config)
        sched_d = build_scheduler(config["training"].get("lr_scheduler", ""), opt_d, config)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            opt_g=opt_g,
            opt_d=opt_d,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=config["training"]["epochs"],
            scheduler_g=sched_g,
            scheduler_d=sched_d,
            loss_weights=config["training"].get("loss_weights"),
            label_smoothing=config["training"].get("label_smoothing", True),
            val_every=config["training"].get("val_every", 1),
            amp=config["training"].get("amp", False),
            grad_clip=config["training"].get("grad_clip", 1.0),
            checkpoint_dir=config["output"]["checkpoint_dir"],
            early_stopping_patience=config["training"].get("early_stopping_patience", 50),
            perceptual_feature_layer=config["training"].get("perceptual_feature_layer", 35),
        )
        
    else:
        model = get_model(model_name, **model_kwargs).to(device)
        if config.get("initialize_output"):
            initialize_reconstruction(
                model.reconstruction if model_name == "srcnn" else model.final,
                color_preserving=model.color_preserving,
            )
        
        opt = build_optimizer(
            config["training"]["optimizer"],
            model.parameters(),
            config["training"]["lr"]
        )
        sched = build_scheduler(config["training"].get("lr_scheduler", ""), opt, config)
        
        from src.training.losses import PixelLoss
        criterion = PixelLoss(criterion=config["training"].get("loss", "l1")).to(device)
        
        trainer = Trainer(
            model=model,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epochs=config["training"]["epochs"],
            scheduler=sched,
            val_every=config["training"].get("val_every", 1),
            amp=config["training"].get("amp", False),
            grad_clip=config["training"].get("grad_clip", 0.0),
            checkpoint_dir=config["output"]["checkpoint_dir"],
            early_stopping_patience=config["training"].get("early_stopping_patience", 20)
        )
        
    if resume_path:
        trainer.load_checkpoint(resume_path)
        
    best_path = Path(config["output"]["checkpoint_dir"]) / "best_model.pth"
    previous_best = best_path.stat().st_mtime_ns if best_path.is_file() else None
    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Training interrupted. Exporting weights from completed epochs.")

    if trainer.history["train_loss"]:
        state_dict = trainer.model.state_dict()
        best = None
        if best_path.is_file() and best_path.stat().st_mtime_ns != previous_best:
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            # Export the generator for SRGAN. Its discriminator is needed only during training.
            state_dict = best.get("model_state_dict", best.get("generator_state_dict"))

        final_dir = Path(config["output"].get("model_dir", "models")) / model_name
        if scale_factor != 4:
            final_dir /= f"x{scale_factor}"
        final_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(config.get("provenance", {}))
        metadata.update(
            epochs_completed=len(trainer.history["train_loss"]),
            selected_epoch=best["epoch"] if best is not None
            else trainer.start_epoch + len(trainer.history["train_loss"]) - 1,
            validation_psnr=trainer.history["val_psnr"],
            validation_ssim=trainer.history["val_ssim"],
        )
        save_checkpoint({
            "model_state_dict": state_dict, "model_kwargs": model_kwargs, "provenance": metadata,
        }, final_dir / "final.pth")
        logger.info(f"Saved trained {model_name} weights to {final_dir / 'final.pth'}")

    if interrupted:
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description="Train Super-Resolution Models")
    parser.add_argument("--config", nargs="+", required=True, help="Path(s) to YAML config files")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    for cfg in args.config:
        run_training(cfg, args.resume)


if __name__ == "__main__":
    main()
