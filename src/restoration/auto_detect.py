"""
Auto-Detect — Analyzes images to suggest restoration presets.
"""

import cv2
import numpy as np


class ImageQualityAnalyzer:
    """Analyze input image and recommend restoration settings."""
    
    def analyze(self, img: np.ndarray) -> dict:
        """
        Analyze the image for noise, blur, and JPEG artifacts.
        
        Args:
            img: (H, W, C) uint8 RGB image.
            
        Returns:
            Dict containing scores and a recommended preset.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 1. Blur estimation (Variance of Laplacian)
        # Lower means more blurry. Threshold usually around 100.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 150
        
        # 2. Noise estimation
        # We use a simple high-pass filter. 
        # High noise = high variance in high-frequency domains when blur is low.
        # This is a basic heuristic; true blind noise estimation is complex.
        noise_score = np.mean(cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0)))
        is_noisy = noise_score > 5.0
        
        # 3. JPEG Artifact estimation (Blockiness)
        # Check differences across 8x8 block boundaries vs internal pixel differences.
        blockiness = self._estimate_blockiness(gray)
        is_jpeg_heavy = blockiness > 1.2
        
        # 4. Determine Recommended Preset
        preset = "None"
        reason = "Image quality appears good."
        
        if is_jpeg_heavy:
            preset = "JPEG Fix"
            reason = "Detected strong 8x8 blocking artifacts."
        elif is_noisy and is_blurry:
            preset = "Old Photo Restore"
            reason = "Detected significant noise and blur."
        elif is_noisy or is_blurry:
            preset = "Photo Cleanup"
            reason = "Detected minor noise or softness."
            
        return {
            "laplacian_var": float(laplacian_var),
            "noise_score": float(noise_score),
            "blockiness": float(blockiness),
            "is_blurry": is_blurry,
            "is_noisy": is_noisy,
            "is_jpeg_heavy": is_jpeg_heavy,
            "recommended_preset": preset,
            "reason": reason
        }
        
    def _estimate_blockiness(self, gray: np.ndarray) -> float:
        """Estimate JPEG blocking artifacts by comparing boundary vs internal differences."""
        h, w = gray.shape
        # Need at least a few blocks
        if h < 16 or w < 16:
            return 1.0
            
        # Ensure divisible by 8
        h_crop = (h // 8) * 8
        w_crop = (w // 8) * 8
        img = gray[:h_crop, :w_crop].astype(np.float32)
        
        # Differences across 8x8 horizontal boundaries (exclude the very last row edge)
        diff_h_boundaries = np.abs(img[7:-1:8, :] - img[8::8, :]).mean()
        # Differences across internal rows (non-boundaries)
        diff_h_internal = np.abs(img[1::8, :] - img[2::8, :]).mean()
        
        # Differences across 8x8 vertical boundaries (exclude the very last col edge)
        diff_v_boundaries = np.abs(img[:, 7:-1:8] - img[:, 8::8]).mean()
        # Differences across internal columns
        diff_v_internal = np.abs(img[:, 1::8] - img[:, 2::8]).mean()
        
        # Ratio of boundary difference to internal difference
        # If it's heavily JPEG compressed, boundaries will have higher artificial differences
        ratio_h = diff_h_boundaries / (diff_h_internal + 1e-5)
        ratio_v = diff_v_boundaries / (diff_v_internal + 1e-5)
        
        return (ratio_h + ratio_v) / 2.0
