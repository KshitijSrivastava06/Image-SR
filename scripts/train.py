"""
Training entry point for Image Super-Resolution.
Supports SRCNN, SRResNet, and SRGAN.
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.dataset_manager import DatasetDownloader
from data import DIV2KTrainDataset, SRTestDataset
from models import get_model, get_pretrained
from models.srgan import Discriminator
from src.training import Trainer, GANTrainer
from src.utils.logger import get_logger
from src.utils.common import get_device

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


def run_training(config_path: str, resume_path: str = None):
    config = load_config(config_path)
    device = get_device()
    
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
        cache_in_ram=config["data"].get("cache_in_ram", True)
    )
    val_dataset = SRTestDataset(
        hr_dir=config["data"]["val_hr_dir"],
        scale_factor=scale_factor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=True
    )
    
    # 2. Model
    model_kwargs = {k: v for k, v in config["model"].items() if k not in ("name", "generator_weights")}
    if model_name == "srgan":
        generator = get_model("srgan_generator", **model_kwargs).to(device)
        if "generator_weights" in config["model"]:
            ckpt = torch.load(config["model"]["generator_weights"], map_location=device)
            # Support loading from best_model.pth or direct state_dict
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            if getattr(generator, "use_batchnorm", True) is False:
                from models.srresnet import fold_srresnet_bn
                state_dict = fold_srresnet_bn(state_dict)
            generator.load_state_dict(state_dict)
            logger.info(f"Loaded generator warm-start weights from {config['model']['generator_weights']}")
            
        discriminator = Discriminator().to(device)
        
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
            early_stopping_patience=config["training"].get("early_stopping_patience", 50)
        )
        
    else:
        model = get_model(model_name, **model_kwargs).to(device)
        
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
        
    # 3. Train
    trainer.train()
    
    # 4. Save Final Weights (Preferring Best Checkpoint over Last Epoch)
    final_dir = Path(f"models/{model_name}")
    final_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    best_ckpt_path = ckpt_dir / "best_model.pth"
    
    final_sd = None
    if best_ckpt_path.exists():
        try:
            ckpt_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt_data:
                final_sd = ckpt_data["model_state_dict"]
            elif "generator_state_dict" in ckpt_data:
                final_sd = ckpt_data["generator_state_dict"]
            logger.info(f"Loaded best checkpoint weights from {best_ckpt_path} for final deployment")
        except Exception as e:
            logger.warning(f"Could not load best_model.pth, falling back to final training state: {e}")

    if final_sd is None:
        if model_name == "srgan":
            final_sd = generator.state_dict()
        else:
            final_sd = model.state_dict()

    target_m = generator if model_name == "srgan" else model
    if ("srresnet" in model_name.lower() or "srgan" in model_name.lower()) and getattr(target_m, "use_batchnorm", True) is False:
        from models.srresnet import fold_srresnet_bn
        final_sd = fold_srresnet_bn(final_sd)
        
    torch.save(final_sd, final_dir / "final.pth")
    logger.info(f"Saved optimized final {model_name} weights to {final_dir / 'final.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train Super-Resolution Models")
    parser.add_argument("--config", nargs="+", required=True, help="Path(s) to YAML config files")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    for cfg in args.config:
        run_training(cfg, args.resume)


if __name__ == "__main__":
    main()
