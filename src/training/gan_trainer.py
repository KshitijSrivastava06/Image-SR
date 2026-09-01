"""
GAN Trainer — Specialized trainer for Generative Adversarial Networks (SRGAN).
Extends the standard Trainer with alternating G/D updates and multiple losses.
"""

import time
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.trainer import Trainer
from src.training.losses import PixelLoss, VGGPerceptualLoss, AdversarialLoss
from src.utils.logger import get_logger

logger = get_logger("gan_trainer")


class GANTrainer(Trainer):
    """
    Specialized trainer for SRGAN models.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        opt_g: torch.optim.Optimizer,
        opt_d: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        epochs: int = 100,
        scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        label_smoothing: bool = True,
        val_every: int = 1,
        amp: bool = False,
        grad_clip: float = 1.0,
        checkpoint_dir: str = "outputs/checkpoints",
        early_stopping_patience: int = 50,
    ):
        # We pass generator to base class to reuse validation and state dict saving logic
        super().__init__(
            model=generator,
            optimizer=opt_g,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=None, # handled custom
            device=device,
            epochs=epochs,
            scheduler=scheduler_g,
            val_every=val_every,
            amp=amp,
            grad_clip=grad_clip,
            checkpoint_dir=checkpoint_dir,
            early_stopping_patience=early_stopping_patience,
        )
        self.discriminator = discriminator
        self.opt_d = opt_d
        self.scheduler_d = scheduler_d
        self.label_smoothing = label_smoothing
        
        # Default SRGAN paper weights
        self.loss_weights = loss_weights or {
            "pixel": 1.0,
            "perceptual": 0.006,
            "adversarial": 0.001
        }
        
        # Initialize Losses
        self.pixel_loss = PixelLoss(criterion='l1').to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)
        self.adversarial_loss = AdversarialLoss().to(device)
        
        if self.amp:
            self.scaler_d = torch.amp.GradScaler(device='cuda')
        
        self.history.update({
            "train_loss_g": [],
            "train_loss_d": [],
        })

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.discriminator.train()
        
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for lr, hr in pbar:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            # ---------------------
            # 1. Train Discriminator
            # ---------------------
            self.opt_d.zero_grad(set_to_none=True)
            
            if self.amp:
                with torch.amp.autocast(device_type='cuda'):
                    # Generate fake HR
                    fake_hr = self.model(lr).detach()
                    
                    # Real and Fake passes
                    real_logits = self.discriminator(hr)
                    fake_logits = self.discriminator(fake_hr)
                    
                    # Discriminator losses
                    loss_d_real = self.adversarial_loss(real_logits, is_real=True, label_smoothing=self.label_smoothing)
                    loss_d_fake = self.adversarial_loss(fake_logits, is_real=False)
                    loss_d = loss_d_real + loss_d_fake
                    
                self.scaler_d.scale(loss_d).backward()
                if self.grad_clip > 0:
                    self.scaler_d.unscale_(self.opt_d)
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                self.scaler_d.step(self.opt_d)
                self.scaler_d.update()
            else:
                fake_hr = self.model(lr).detach()
                real_logits = self.discriminator(hr)
                fake_logits = self.discriminator(fake_hr)
                loss_d_real = self.adversarial_loss(real_logits, is_real=True, label_smoothing=self.label_smoothing)
                loss_d_fake = self.adversarial_loss(fake_logits, is_real=False)
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip)
                self.opt_d.step()
                
            epoch_loss_d += loss_d.item()
            
            # ---------------------
            # 2. Train Generator
            # ---------------------
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.amp:
                with torch.amp.autocast(device_type='cuda'):
                    # Generate fake HR (require gradients this time)
                    fake_hr = self.model(lr)
                    
                    # G losses
                    p_loss = self.pixel_loss(fake_hr, hr)
                    vgg_loss = self.perceptual_loss(fake_hr, hr)
                    
                    # Generator wants discriminator to think it's real
                    fake_logits_for_g = self.discriminator(fake_hr)
                    adv_loss = self.adversarial_loss(fake_logits_for_g, is_real=True)
                    
                    # Total G loss
                    loss_g = (
                        self.loss_weights["pixel"] * p_loss +
                        self.loss_weights["perceptual"] * vgg_loss +
                        self.loss_weights["adversarial"] * adv_loss
                    )
                    
                self.scaler.scale(loss_g).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                fake_hr = self.model(lr)
                p_loss = self.pixel_loss(fake_hr, hr)
                vgg_loss = self.perceptual_loss(fake_hr, hr)
                fake_logits_for_g = self.discriminator(fake_hr)
                adv_loss = self.adversarial_loss(fake_logits_for_g, is_real=True)
                loss_g = (
                    self.loss_weights["pixel"] * p_loss +
                    self.loss_weights["perceptual"] * vgg_loss +
                    self.loss_weights["adversarial"] * adv_loss
                )
                loss_g.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                
            epoch_loss_g += loss_g.item()
            pbar.set_postfix({"G_loss": f"{loss_g.item():.4f}", "D_loss": f"{loss_d.item():.4f}"})
            
        # Step D scheduler as well
        if self.scheduler_d is not None:
            self.scheduler_d.step()
            
        avg_loss_g = epoch_loss_g / len(self.train_loader)
        avg_loss_d = epoch_loss_d / len(self.train_loader)
        self.history["train_loss_g"].append(avg_loss_g)
        self.history["train_loss_d"].append(avg_loss_d)
        
        return avg_loss_g

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.start_epoch + len(self.history["train_loss_g"]) - 1,
            "model_state_dict": self.model.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer.state_dict(),
            "optimizer_d_state_dict": self.opt_d.state_dict(),
            "best_psnr": self.best_psnr,
        }, path)

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading GAN checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        if "optimizer_g_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_g_state_dict"])
        if "optimizer_d_state_dict" in checkpoint:
            self.opt_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        if "best_psnr" in checkpoint:
            self.best_psnr = checkpoint["best_psnr"]
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming GAN training from epoch {self.start_epoch} with best PSNR {self.best_psnr:.2f}")
