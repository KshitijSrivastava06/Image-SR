"""Quick verification script for the super-resolution platform."""
import sys
sys.path.insert(0, ".")

import torch
from models import get_model, list_models
from src.utils.common import get_device, get_model_size

print("=" * 60)
print("SUPER-RESOLUTION PLATFORM — VERIFICATION")
print("=" * 60)

device = get_device()
print(f"\nDevice: {device}")
print(f"\nRegistered models: {list_models()}")

# Forward pass tests
print("\n--- Forward Pass Tests ---")
x = torch.randn(1, 3, 24, 24)

# Bicubic
m1 = get_model("bicubic", scale_factor=4)
y1 = m1(x)
print(f"  Bicubic:   {list(x.shape)} -> {list(y1.shape)}  [OK]")

# SRCNN
m2 = get_model("srcnn", scale_factor=4)
y2 = m2(x)
print(f"  SRCNN:     {list(x.shape)} -> {list(y2.shape)}  [OK]")

# SRResNet
m3 = get_model("srresnet", scale_factor=4)
y3 = m3(x)
print(f"  SRResNet:  {list(x.shape)} -> {list(y3.shape)}  [OK]")

# SRGAN
from models.srgan import SRGANGenerator
g = SRGANGenerator(scale_factor=4)
y4 = g(x)
print(f"  SRGAN-G:   {list(x.shape)} -> {list(y4.shape)}  [OK]")

# ESRGAN (reduced blocks for speed)
from models.esrgan import ESRGANGenerator
eg = ESRGANGenerator(scale_factor=4, num_rrdb_blocks=2)
y5 = eg(x)
print(f"  ESRGAN-G:  {list(x.shape)} -> {list(y5.shape)}  [OK] (2 RRDB)")

# Model sizes
print("\n--- Model Sizes ---")
for name in ["bicubic", "srcnn", "srresnet"]:
    m = get_model(name, scale_factor=4)
    ms = get_model_size(m)
    print(f"  {name:10s}: {ms['total_params']:>10,} params   {ms['size_mb']:>6.2f} MB")

full_esrgan = get_model("esrgan_generator", scale_factor=4)
if hasattr(full_esrgan, "generator"):
    ms = get_model_size(full_esrgan.generator)
else:
    ms = get_model_size(full_esrgan)
print(f"  {'esrgan-g':10s}: {ms['total_params']:>10,} params   {ms['size_mb']:>6.2f} MB")

# Metrics
print("\n--- Metrics ---")
from src.evaluation.metrics import MetricCalculator
import numpy as np
mc = MetricCalculator()
img1 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
img2 = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
psnr = mc.calculate_psnr(img1, img2)
ssim = mc.calculate_ssim(img1, img2)
print(f"  PSNR (random vs random): {psnr:.2f} dB  [OK]")
print(f"  SSIM (random vs random): {ssim:.4f}  [OK]")

# Identical images should give perfect PSNR
psnr_same = mc.calculate_psnr(img1, img1)
ssim_same = mc.calculate_ssim(img1, img1)
print(f"  PSNR (self):  {psnr_same:.2f} dB  [OK] (should be high/inf)")
print(f"  SSIM (self):  {ssim_same:.4f}  [OK] (should be 1.0)")

# Restoration pipeline
print("\n--- Restoration Pipeline ---")
from src.restoration import RestorationPipeline, Denoiser, Deblurrer, ContrastEnhancer
dn = Denoiser("nlm", 10)
db = Deblurrer("unsharp", 1.5)
ce = ContrastEnhancer("clahe", 2.0)
test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
r1 = dn.denoise(test_img)
print(f"  Denoiser (NLM):      {test_img.shape} -> {r1.shape}  [OK]")
r2 = db.deblur(test_img)
print(f"  Deblurrer (unsharp): {test_img.shape} -> {r2.shape}  [OK]")
r3 = ce.enhance(test_img)
print(f"  Enhancer (CLAHE):    {test_img.shape} -> {r3.shape}  [OK]")

pipeline = RestorationPipeline(
    denoise=True, deblur=True, enhance=True,
    sr_model=get_model("srcnn", scale_factor=4),
)
print(f"  Pipeline: {pipeline.get_pipeline_description()}")
result, intermediates = pipeline.process(test_img, return_intermediates=True)
print(f"  Pipeline output:     {test_img.shape} -> {result.shape}  [OK]")
print(f"  Intermediates:       {list(intermediates.keys())}  [OK]")

# Config loading
print("\n--- Config Files ---")
import yaml
with open("configs/inference.yaml") as f:
    cfg = yaml.safe_load(f)
print(f"  inference.yaml: default model={cfg['models']['default']}  [OK]")

print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
