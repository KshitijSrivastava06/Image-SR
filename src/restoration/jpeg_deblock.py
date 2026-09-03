"""
JPEG Deblocker — Removes JPEG compression artifacts before SR.

JPEG compression creates 8x8 blocking artifacts and ringing around edges.
SR models often amplify these artifacts, creating grid-like patterns.
This module uses a tailored bilateral filter to smooth blocks while
preserving actual edges.
"""

import cv2
import numpy as np


class JPEGDeBlocker:
    """
    Removes JPEG compression artifacts.
    
    Args:
        quality: Estimated JPEG quality (0-100). Lower quality means
                 stronger deblocking is needed. Default is 50.
    """
    def __init__(self, quality: int = 50):
        self.quality = np.clip(quality, 10, 100)
        
    def deblock(self, img: np.ndarray) -> np.ndarray:
        """
        Apply deblocking to an image.
        
        Args:
            img: (H, W, C) uint8 RGB image.
            
        Returns:
            Deblocked (H, W, C) uint8 RGB image.
        """
        # If quality is high, minimal or no deblocking is needed
        if self.quality >= 95:
            return img
            
        # Convert to YCrCb because JPEG artifacts are often worse in chroma channels
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        
        y, cr, cb = cv2.split(ycrcb)
        
        # Determine filter strengths based on estimated quality
        # Lower quality = stronger filtering
        strength = max(3.0, 15.0 - (self.quality / 10.0))
        
        # Apply bilateral filter to Luma (Y) channel
        # d=7 is a good window to cover an 8x8 JPEG block boundary
        y_filtered = cv2.bilateralFilter(
            y, d=7, 
            sigmaColor=strength * 2, 
            sigmaSpace=strength * 2
        )
        
        # Apply stronger filter to Chroma channels (Cr, Cb) since they are usually subsampled
        cr_filtered = cv2.bilateralFilter(
            cr, d=7, 
            sigmaColor=strength * 3, 
            sigmaSpace=strength * 3
        )
        cb_filtered = cv2.bilateralFilter(
            cb, d=7, 
            sigmaColor=strength * 3, 
            sigmaSpace=strength * 3
        )
        
        ycrcb_filtered = cv2.merge([y_filtered, cr_filtered, cb_filtered])
        bgr_filtered = cv2.cvtColor(ycrcb_filtered, cv2.COLOR_YCrCb2BGR)
        
        return cv2.cvtColor(bgr_filtered, cv2.COLOR_BGR2RGB)
