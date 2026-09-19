"""
Unified Trainer — Handles single-model SR training (SRCNN, SRResNet).
Includes mixed-precision, checkpointing, validation, and progress tracking.
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import MetricCalculator
from src.utils.logger import get_logger

logger = get_logger("trainer")


class Trainer:
    """
    Standard trainer for pixel-loss based models (SRCNN, SRResNet).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epochs: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        val_every: int = 1,
        amp: bool = False,
        grad_clip: float = 0.0,
        checkpoint_dir: str = "outputs/checkpoints",
        early_stopping_patience: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.val_every = val_every
        self.amp = amp
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_patience = early_stopping_patience
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize AMP scaler
        self.scaler = torch.amp.GradScaler(device='cuda') if amp else None
        
        # Metrics
        self.metric_calc = MetricCalculator(device=device)
        
        # State tracking
        self.best_psnr = 0.0
        self.patience_counter = 0
        self.start_epoch = 1
        self.history = {
            "train_loss": [],
            "val_psnr": [],
            "val_ssim": [],
        }

    def train(self):
        """Run the full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs on {self.device}")
        logger.info(f"AMP enabled: {self.amp}")
        logger.info(f"Batches per epoch: {len(self.train_loader)}")

        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            train_loss = self._train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            epoch_time = time.time() - start_time
            logger.info(f"Epoch [{epoch}/{self.epochs}] - Time: {epoch_time:.1f}s - Loss: {train_loss:.4f}")
            
            # Validation
            if epoch % self.val_every == 0:
                val_psnr, val_ssim = self._validate()
                self.history["val_psnr"].append(val_psnr)
                self.history["val_ssim"].append(val_ssim)
                
                logger.info(f"Validation - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}")
                
                # Checkpointing and Early Stopping
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.patience_counter = 0
                    self._save_checkpoint("best_model.pth")
                    logger.info(f"New best model saved! (PSNR: {val_psnr:.2f})")
                else:
                    self.patience_counter += 1
                    
                self._save_checkpoint("latest.pth")
                
                if self.early_stopping_patience > 0 and self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        logger.info(f"Training completed. Best PSNR: {self.best_psnr:.2f}")
        return self.history

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
        for lr, hr in pbar:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.amp:
                with torch.amp.autocast(device_type='cuda'):
                    sr = self.model(lr)
                    loss = self.criterion(sr, hr)
                    
                self.scaler.scale(loss).backward()
                
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                sr = self.model(lr)
                loss = self.criterion(sr, hr)
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                self.optimizer.step()
                
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return epoch_loss / len(self.train_loader)

    @torch.no_grad()
    def _validate(self) -> tuple[float, float]:
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        
        for lr, hr, _ in tqdm(self.val_loader, desc="Validating", leave=False):
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            sr = self.model(lr)
            sr = sr.clamp(0, 1)
            
            # Convert to numpy (H, W, C) for metric calculator
            sr_np = sr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hr_np = hr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            metrics = self.metric_calc.calculate_all(sr_np, hr_np, include_lpips=False)
            total_psnr += metrics["psnr"]
            total_ssim += metrics["ssim"]
            
        n = len(self.val_loader)
        return total_psnr / n, total_ssim / n

    def _save_checkpoint(self, filename: str):
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.start_epoch + len(self.history["train_loss"]) - 1, # The epoch that just finished
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_psnr": self.best_psnr,
        }, path)

    def load_checkpoint(self, checkpoint_path: str):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "best_psnr" in checkpoint:
            self.best_psnr = checkpoint["best_psnr"]
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming from epoch {self.start_epoch} with best PSNR {self.best_psnr:.2f}")

    def save_best_weights(self, export_path: str | Path) -> bool:
        """
        Export the best model state dict directly to a standalone weight file for inference.
        If best_model.pth exists, loads its weights. Otherwise uses the current model state.
        Folds BatchNorm if model is BN-free.
        """
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state_dict", self.model.state_dict())
        else:
            sd = self.model.state_dict()
            
        if getattr(self.model, "use_batchnorm", True) is False:
            from models.srresnet import fold_srresnet_bn
            sd = fold_srresnet_bn(sd)
            
        torch.save(sd, export_path)
        logger.info(f"Exported best weights to {export_path}")
        return True

