"""
Post-Processor — Refines the image after Super-Resolution.

SR models (especially those trained with L1/MSE) can sometimes produce slightly 
soft outputs or slightly shifted color vibrance. This module applies very mild
sharpening and color correction to give the final output a "commercial" polish.
"""

import cv2
import numpy as np


class PostProcessor:
    """
    Post-SR refinement module.
    
    Args:
        strength: Overall strength of the post-processing (0.0 to 1.0).
                  Default is 0.5 (mild).
    """
    def __init__(self, strength: float = 0.5):
        self.strength = np.clip(strength, 0.0, 1.0)
        
    def process(self, img: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the SR output image.
        
        Args:
            img: (H, W, C) uint8 RGB image.
            
        Returns:
            Refined (H, W, C) uint8 RGB image.
        """
        if self.strength == 0:
            return img
            
        # 1. Mild Edge-Aware Sharpening (Unsharp Mask)
        # We use a very light touch here because SR models already attempt to sharpen.
        # We just want to recover micro-contrast.
        sharpened = self._mild_unsharp_mask(img)
        
        # 2. Subtle Color Vibrance Boost
        # SR models sometimes slightly desaturate high-frequency details.
        refined = self._boost_vibrance(sharpened)
        
        return refined

    def _mild_unsharp_mask(self, img: np.ndarray) -> np.ndarray:
        # Sigma controls the radius of the blur. 1.0 is a tight radius.
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        
        # Base amount of sharpening (max 1.5 at full strength)
        amount = 1.0 + (0.5 * self.strength)
        
        # Add weighted
        sharpened = cv2.addWeighted(
            img, amount, 
            blurred, -(amount - 1.0), 
            0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _boost_vibrance(self, img: np.ndarray) -> np.ndarray:
        # Convert to HSV to easily boost saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Boost saturation by a small percentage (max 10% boost at full strength)
        sat_boost = 1.0 + (0.10 * self.strength)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_boost, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
