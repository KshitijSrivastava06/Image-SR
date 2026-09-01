# 🔬 Technical Deep Dive: How the Super-Resolution Pipeline Works

A complete, step-by-step walkthrough of every mathematical and engineering decision in this project — from uploading an image in the browser to receiving a high-resolution output.

---

## Table of Contents

1. [Inference Pipeline (Upload → Output)](#1-inference-pipeline-upload--output)
2. [Dataset Management](#2-dataset-management)
3. [Training Pipeline](#3-training-pipeline)
4. [Model Architectures](#4-model-architectures)
5. [Loss Functions](#5-loss-functions)
6. [Evaluation Metrics](#6-evaluation-metrics)

---

## 1. Inference Pipeline (Upload → Output)

When a user uploads an image in the Streamlit web app and clicks "🚀 Enhance Image", the following sequence of operations executes:

### Step 1: Image Upload & Decoding

```
User uploads PNG/JPG/BMP/WebP → PIL.Image.open() → .convert("RGB")
```

**What happens:** The raw binary file is decoded into a PIL Image object and forced into **RGB** color mode (3 channels: Red, Green, Blue). This guarantees a consistent format regardless of whether the user uploads a grayscale, RGBA, or palette-based image.

**Why RGB?** All our neural networks were trained on 3-channel RGB inputs. If we accepted a 4-channel RGBA image or a single-channel grayscale, the tensor dimensions would not match the model's first convolutional layer (`nn.Conv2d(3, 64, ...)`), and PyTorch would throw a shape mismatch error.

---

### Step 2: PIL → NumPy Conversion

```python
img_np = np.array(pil_image.convert("RGB"))  # Shape: (H, W, 3), dtype: uint8, range: [0, 255]
```

**What happens:** The PIL Image is converted to a NumPy array with shape `(Height, Width, 3)` and data type `uint8` (unsigned 8-bit integer, values 0–255).

**Why NumPy?** OpenCV (which powers our restoration pipeline) and scikit-image (which powers our metrics) both operate on NumPy arrays. PIL is great for I/O but not for mathematical image processing.

---

### Step 3: Mod-Crop

```python
def mod_crop(img, scale_factor):
    h, w = img.shape[:2]
    h = h - (h % scale_factor)
    w = w - (w % scale_factor)
    return img[:h, :w]
```

**What happens:** The image dimensions are trimmed so that both height and width are exactly divisible by the scale factor (e.g., 4). For a 1021×763 image at 4× scale, it becomes 1020×760.

**Why?** Our models use **PixelShuffle** (also called sub-pixel convolution) for upsampling. PixelShuffle rearranges a tensor of shape `(B, C×r², H, W)` into `(B, C, H×r, W×r)`, where `r` is the scale factor. If H or W is not divisible by `r`, the rearrangement produces fractional dimensions, causing a runtime crash. Mod-cropping prevents this.

**Mathematical detail:**
```
PixelShuffle: (1, 256, H, W) → (1, 64, H×2, W×2)    (for r=2)
Requires: 256 = 64 × 2² ✓
But H and W must be integers after multiplication.
```

---

### Step 4: Restoration Pipeline (Optional)

If the user enables denoising, deblurring, or contrast enhancement, the image passes through a chained restoration pipeline **before** super-resolution.

```
Input → [Denoise] → [Deblur] → [Enhance] → Output
```

**Why before SR?** Noise and blur are low-frequency artifacts. If we super-resolve a noisy image, the model amplifies the noise by 4× along with the image content. By cleaning the image first, we give the SR model a cleaner input to work with, producing sharper outputs.

#### 4a. Denoising (Non-Local Means)

```python
cv2.fastNlMeansDenoisingColored(img, h=strength, hForColorComponents=strength, templateWindowSize=7, searchWindowSize=21)
```

**The math:** Non-Local Means (NLM) computes a weighted average of all pixels in the image, where the weight depends on the **similarity of local patches** around each pixel:

$$\hat{I}(x) = \frac{1}{C(x)} \sum_{y \in \Omega} w(x, y) \cdot I(y)$$

where $w(x, y) = \exp\left(-\frac{\|P(x) - P(y)\|^2}{h^2}\right)$

- `P(x)` is the 7×7 patch centered at pixel `x`
- `h` is the filtering strength (user-controlled slider)
- The search window is 21×21 pixels

**Why NLM?** Unlike Gaussian blur (which blindly averages neighbors), NLM preserves edges because it only averages pixels that have **similar local neighborhoods**. A pixel on an edge will only be averaged with other edge pixels, not smooth background pixels.

#### 4b. Deblurring (Unsharp Masking)

```python
blurred = cv2.GaussianBlur(img, (0, 0), sigma)
sharpened = cv2.addWeighted(img, 1 + strength, blurred, -strength, 0)
```

**The math:** Unsharp masking enhances edges by subtracting a blurred version of the image:

$$I_{sharp} = I_{original} + \alpha \cdot (I_{original} - I_{blurred})$$

- `α` is the sharpening strength
- The "unsharp mask" is `I_original - I_blurred`, which isolates high-frequency details (edges, textures)

**Why unsharp masking?** It is computationally cheap (just a Gaussian blur + weighted addition) and reliably enhances edge contrast without introducing ringing artifacts that more aggressive methods (like Wiener deconvolution) can produce.

#### 4c. Contrast Enhancement (CLAHE)

```python
clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(8, 8))
```

**The math:** CLAHE (Contrast Limited Adaptive Histogram Equalization) divides the image into 8×8 tiles and equalizes the histogram of each tile independently. The "clip limit" prevents over-amplification of noise by capping the histogram bins:

1. Compute histogram for each tile
2. Clip histogram bins that exceed the clip limit
3. Redistribute the clipped pixels uniformly across all bins
4. Compute the CDF (cumulative distribution function) as the mapping function
5. Apply bilinear interpolation at tile boundaries to eliminate seams

**Why CLAHE over global histogram equalization?** Global equalization applies a single mapping to the entire image, which can wash out locally-optimal contrast. CLAHE adapts to local regions, preserving details in both shadows and highlights simultaneously.

---

### Step 5: NumPy → Tensor Conversion

```python
img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0   # [0, 255] → [0, 1]
tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)          # (H,W,3) → (1,3,H,W)
```

**What happens:** Two critical transformations occur:

1. **Normalization to [0, 1]:** Pixel values are divided by 255 to map from integer range `[0, 255]` to float range `[0.0, 1.0]`.
2. **Axis reordering:** NumPy/PIL uses `(H, W, C)` format (channels last). PyTorch uses `(B, C, H, W)` format (batch first, channels second). The `.permute(2, 0, 1)` operation rearranges the axes, and `.unsqueeze(0)` adds a batch dimension.

**Why [0, 1] normalization?** Neural networks learn most efficiently when inputs are in a small, consistent range. The weights are initialized with small values (via Kaiming initialization), and large input values (0–255) would cause massive activations, gradient explosions, and unstable training. Normalizing to [0, 1] keeps everything numerically stable.

**Why channels-first?** PyTorch's `nn.Conv2d` expects `(B, C, H, W)` because the GPU can process all spatial positions of a single channel in parallel using SIMD instructions. This memory layout is optimized for cuDNN's convolution algorithms.

---

### Step 6: Model-Specific Input Formatting

```python
# For models trained on BGR (e.g., some OpenCV-based checkpoints):
tensor = tensor[:, [2, 1, 0], :, :]   # RGB → BGR (swap channels 0 and 2)

# For models expecting [-1, 1] range:
tensor = tensor * 2.0 - 1.0            # [0, 1] → [-1, 1]
```

**What happens:** Different pre-trained checkpoints were trained with different input conventions. Our `prepare_input()` function reads the model's metadata from the `PRETRAINED_REGISTRY` and automatically applies the correct transformations.

**Why does this vary?** Historical reasons. OpenCV loads images in BGR order (a legacy from early camera hardware), so models trained using OpenCV pipelines expect BGR. Some GAN models use `tanh` as their final activation (which outputs in [-1, 1]), so they are trained with [-1, 1] inputs for symmetry.

| Model           | Color Space | Input Range |
|-----------------|-------------|-------------|
| SRCNN           | RGB         | [0, 1]      |
| SRResNet        | RGB         | [0, 1]      |
| SRGAN           | RGB         | [0, 1]      |
| ESRGAN (Real)   | RGB         | [0, 1]      |

---

### Step 7: Forward Pass (Super-Resolution)

The input tensor is passed through the neural network. The exact computation depends on the selected model (see [Model Architectures](#4-model-architectures)).

For small images (< 512×512 pixels), the entire image is processed in a single forward pass:

```python
sr_tensor = model(lr_tensor)   # (1, 3, H, W) → (1, 3, H×4, W×4)
```

For large images (≥ 512×512), **tiled inference** is used.

#### Tiled Inference

**The problem:** A 1920×1080 image at 4× upscaling requires the model to output a 7680×4320 tensor. For a 16.7M parameter model like ESRGAN, the intermediate activations alone can consume 6+ GB of VRAM, exceeding the 8 GB budget of our RTX 4060.

**The solution:** Split the image into overlapping 256×256 tiles, process each tile independently, and blend them back together using a linear ramp mask:

```
1. Pad image to mod(scale_factor)
2. Generate tile grid with `stride = tile_size - overlap`
3. For each tile:
   a. Extract 256×256 LR patch
   b. Run model → 1024×1024 HR patch (at 4×)
   c. Multiply by blend mask (linear ramp at borders)
   d. Accumulate into output buffer
4. Divide by weight buffer to normalize overlapping regions
5. Remove padding
```

**Why overlapping tiles?** Without overlap, each tile is processed independently, and the model has no context about pixels outside the tile boundary. This creates visible **seam artifacts** (hard edges where tiles meet). Overlapping by 16 pixels and blending with a linear ramp creates a smooth transition between tiles.

**The blend mask math:**
```python
# 1D linear ramp from 0→1 over the overlap region
ramp = torch.linspace(0, 1, overlap)   # e.g., [0.0, 0.067, 0.133, ..., 1.0]

# Applied to all 4 borders of each tile
weight[top_border]   *= ramp            # Fade in from top
weight[bottom_border] *= reversed(ramp)  # Fade out at bottom
weight[left_border]  *= ramp            # Fade in from left
weight[right_border] *= reversed(ramp)  # Fade out at right
```

---

### Step 8: Post-Processing

```python
sr_tensor = postprocess_output(sr_tensor, ...)   # Reverse color/range transforms
sr_tensor = sr_tensor.clamp(0.0, 1.0)            # Clip to valid range
```

**What happens:** The inverse of Step 6 is applied (BGR→RGB, [-1,1]→[0,1] if needed), and the tensor is clamped to `[0, 1]` to prevent invalid pixel values.

**Why clamp?** Neural networks can output values slightly outside [0, 1] (e.g., -0.02 or 1.03). Without clamping, these would become corrupt pixels (wrap-around to 253 or 255 in uint8), causing visible bright/dark speckles in the output image.

---

### Step 9: Tensor → PIL Conversion & Display

```python
img = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()   # (3,H,W) → (H,W,3)
pil_image = Image.fromarray((img * 255).astype(np.uint8))           # [0,1] → [0,255]
```

**What happens:** The reverse of Step 5. The tensor is moved from GPU→CPU, reordered from `(C,H,W)` → `(H,W,C)`, scaled back to [0, 255], and converted to a PIL Image for display in the Streamlit UI.

---

## 2. Dataset Management

### DIV2K Dataset

The primary training dataset is **DIV2K** (DIVerse 2K resolution), containing 800 high-resolution training images and 100 validation images.

**Why DIV2K?** It is the standard academic benchmark for super-resolution research. The images are 2K resolution (~2040×1356), providing enough diversity and resolution for patch-based training. Every major SR paper (SRResNet, SRGAN, ESRGAN) uses DIV2K for training.

### On-the-Fly LR/HR Pair Generation

We do **not** pre-generate low-resolution images. Instead, each training iteration:

1. **Load HR image** from disk (OpenCV, BGR→RGB conversion)
2. **Random crop** a 96×96 HR patch from the full image
3. **Data augmentation**: Random horizontal flip, vertical flip, 90° rotation
4. **Generate LR pair**: Bicubic downsample the 96×96 HR patch → 24×24 LR patch
5. **Normalize**: uint8 [0, 255] → float32 [0, 1]
6. **To tensor**: (H, W, C) → (C, H, W)

```python
# HR: 96×96 patch (ground truth)
# LR: 24×24 patch (model input, 4× smaller)
lr_patch = cv2.resize(hr_patch, (24, 24), interpolation=cv2.INTER_CUBIC)
```

**Why on-the-fly generation?**
- **Memory efficiency:** Storing all possible 96×96 crops from 800 images would require terabytes.
- **Infinite variety:** Every epoch sees different random crops and augmentations, effectively giving the model an infinite dataset and reducing overfitting.

**Why bicubic downsampling?**
- Bicubic interpolation uses a 4×4 neighborhood of pixels and a cubic polynomial kernel, producing smooth, natural-looking degradation that mimics real-world image downscaling (e.g., resizing a photo in Photoshop).

**Why 96×96 patch size?**
- At 4× scale, the LR input is 24×24, which is large enough for the model to learn meaningful features but small enough to fit many patches in a batch.
- The receptive field of SRResNet (with 16 residual blocks of 3×3 convolutions) is ~70 pixels, so a 24×24 input patch covers the model's full receptive field.

### Data Augmentation

```python
# Random horizontal flip (50% chance)
if random.random() > 0.5:
    img = cv2.flip(img, 1)

# Random vertical flip (50% chance)
if random.random() > 0.5:
    img = cv2.flip(img, 0)

# Random 90° rotation (0°, 90°, 180°, or 270°)
k = random.randint(0, 3)
img = np.rot90(img, k)
```

**Why?** Super-resolution should work regardless of image orientation. Without augmentation, the model might learn orientation-specific patterns (e.g., "horizontal edges are more common than vertical edges") and fail to generalize. Flips and rotations give us 8× data diversity for free (2 flips × 4 rotations).

---

## 3. Training Pipeline

### Training Loop

The training loop follows a standard supervised learning pattern:

```
For each epoch:
    For each batch (LR, HR):
        1. Move LR, HR tensors to GPU
        2. Zero gradients
        3. Forward pass: SR = model(LR)
        4. Compute loss: L = criterion(SR, HR)
        5. Backward pass: compute gradients ∂L/∂θ
        6. Gradient clipping
        7. Optimizer step: θ ← θ - α·∂L/∂θ
    Validate every N epochs
    Save checkpoint if PSNR improved
    Check early stopping
```

### Automatic Mixed Precision (AMP)

```python
with torch.amp.autocast(device_type='cuda'):
    sr = model(lr)
    loss = criterion(sr, hr)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**What happens:** AMP automatically converts eligible operations (convolutions, matrix multiplications) from float32 to **float16**, halving their memory usage and doubling throughput on NVIDIA Tensor Cores.

**Why?** The RTX 4060's Tensor Cores can perform float16 matrix multiplications at 2× the speed of float32. For training, most operations don't need 32-bit precision. The `GradScaler` prevents underflow by dynamically scaling the loss before backward pass and unscaling gradients before the optimizer step.

**The math of gradient scaling:**
```
Without AMP:  loss.backward()  →  gradients may underflow to 0 in float16
With AMP:     (loss × scale).backward()  →  gradients are large enough to survive float16
              gradients /= scale  →  restore original magnitude before optimizer step
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**What happens:** If the L2 norm of the gradient vector exceeds `max_norm`, all gradients are scaled down proportionally:

$$\hat{g} = \begin{cases} g & \text{if } \|g\| \leq \text{max\_norm} \\ \frac{g \cdot \text{max\_norm}}{\|g\|} & \text{otherwise} \end{cases}$$

**Why?** Deep networks (especially GANs) can produce **exploding gradients** where a single bad batch causes the gradient norm to spike to thousands. Without clipping, the optimizer would take an enormous step, potentially destroying all learned weights in a single iteration.

### Checkpointing

Two checkpoint files are maintained:
- **`best_model.pth`**: Saved only when validation PSNR improves (performance checkpoint)
- **`latest.pth`**: Saved after every validation epoch (crash recovery)

Each checkpoint contains:
```python
{
    "epoch": current_epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_psnr": best_psnr_so_far,
}
```

**Why save optimizer state?** The Adam optimizer maintains per-parameter momentum estimates (`m` and `v`). Without saving these, resuming training would reset momentum to zero, causing a temporary spike in loss and erratic learning behavior.

### Early Stopping

```python
if val_psnr > best_psnr:
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= patience_limit:
    break   # Stop training
```

**Why?** After a certain point, continued training only overfits to the training data without improving generalization. Early stopping saves hours of wasted computation.

---

## 4. Model Architectures

### SRCNN (2014) — The Baseline

```
Input (LR) → Bicubic Upsample → Conv(9×9, 64, ReLU) → Conv(1×1, 32, ReLU) → Conv(5×5, 3) → Output
```

**Step 1: Bicubic Pre-Upsampling.** The LR image is first upscaled to the target resolution using bicubic interpolation (`F.interpolate`). The CNN then acts as a **refinement** network, sharpening the blurry bicubic output.

**Why bicubic first?** The original 2014 paper was designed this way because the authors viewed SR as "post-processing on top of interpolation." This approach is simple but wasteful — all convolutions operate at the full HR resolution, increasing computation by 16× (for 4× upscaling).

**Step 2: Feature Extraction (9×9 conv, 64 filters).** A large 9×9 kernel extracts overlapping local patches, each represented as a 64-dimensional feature vector. This is analogous to the "patch extraction" step in sparse-coding-based SR methods.

**Step 3: Non-linear Mapping (1×1 conv, 32 filters).** A 1×1 convolution maps each 64-dimensional feature to a 32-dimensional representation. This is a dimensionality-reducing bottleneck that forces the network to learn a compact representation.

**Step 4: Reconstruction (5×5 conv, 3 filters).** The 32-dimensional features are mapped back to RGB pixel values using a 5×5 convolution. No activation function is used here — the output should be in [0, 1], and ReLU would clip negative values.

**Weight Initialization: Kaiming/He Normal**
```python
nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
```

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

**Why Kaiming?** Standard random initialization (e.g., uniform or Xavier) causes the variance of activations to shrink or explode as they propagate through layers. Kaiming initialization accounts for the ReLU activation (which zeroes out ~50% of values), ensuring that the variance of activations remains constant across layers.

---

### SRResNet (2017) — Deep Residual Learning

```
Input → Conv(9×9, 64, PReLU) →
  [16 × ResBlock(Conv(3×3,64,BN,PReLU) → Conv(3×3,64,BN) + skip)] →
  Conv(3×3, 64, BN) + global_skip →
  [2 × UpsampleBlock(Conv → PixelShuffle × 2 → PReLU)] →
  Conv(9×9, 3) → Output
```

**Key innovation: Residual Learning.** Each residual block computes:

$$y = x + F(x)$$

where $F(x) = \text{Conv}(\text{PReLU}(\text{BN}(\text{Conv}(x))))$

**Why residual connections?** Without them, gradients must flow through every layer during backpropagation, getting multiplied by weight matrices at each step. After 16 blocks (~32 layers), gradients vanish to near-zero, making early layers untrainable. The skip connection provides a "gradient highway" that bypasses the block entirely:

$$\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}$$

Even if $\frac{\partial F}{\partial x} \approx 0$, the gradient is at least 1, preventing vanishing.

**PixelShuffle Upsampling:**
```python
nn.Conv2d(64, 64 * 4, 3, 1, 1)    # 64 → 256 channels
nn.PixelShuffle(upscale_factor=2)  # 256 channels → 64 channels, 2× spatial
```

**The math:** PixelShuffle rearranges elements from the channel dimension into spatial dimensions:

$$\text{PixelShuffle}: (B, C \cdot r^2, H, W) \rightarrow (B, C, H \cdot r, W \cdot r)$$

**Why PixelShuffle over transposed convolution?** Transposed convolution (`nn.ConvTranspose2d`) is known to produce **checkerboard artifacts** due to uneven overlap patterns. PixelShuffle avoids this entirely because it simply reshuffles existing values without any learned upsampling kernel.

**Batch Normalization:**
$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

**Why BN?** Normalizes activations within each mini-batch, stabilizing training by reducing internal covariate shift. However, note that ESRGAN later removed BN because it can introduce artifacts at inference time when batch statistics differ from training statistics.

---

### SRGAN (2017) — Adversarial Training

SRGAN uses the exact same generator as SRResNet but trains it with an additional **discriminator** and **perceptual loss**.

**Discriminator Architecture:**
```
Input (96×96 HR/SR) →
  8 × ConvBlock(3×3, stride 1 or 2, BN, LeakyReLU(0.2)) →
  AdaptiveAvgPool(6×6) → Flatten → Linear(18432, 1024) → LeakyReLU → Linear(1024, 1) → Logit
```

**The training game:**
- **Generator goal:** Produce images that the discriminator classifies as "real"
- **Discriminator goal:** Correctly distinguish between real HR images and generated SR images

$$\min_G \max_D \mathbb{E}[\log D(I^{HR})] + \mathbb{E}[\log(1 - D(G(I^{LR})))]$$

**Why adversarial training?** Models trained with only pixel loss (MSE/L1) produce blurry outputs because they learn the **average** of all possible HR images. The adversarial loss pushes the generator to produce outputs that lie on the **natural image manifold** — images that look realistic to a discriminator trained on real photos.

---

### ESRGAN (2018) — The Best

```
Input → Conv(3×3, 64) →
  [23 × RRDB(3 × DenseBlock(5 convs each) + β·skip)] →
  Conv(3×3, 64) + global_skip →
  [2 × Nearest↑ → Conv(3×3, 64, LReLU)] →
  Conv(3×3, 64, LReLU) → Conv(3×3, 3) → Output
```

**Key improvements over SRGAN:**

1. **RRDB (Residual-in-Residual Dense Block):** Replaces simple ResBlocks with three cascaded DenseBlocks, each containing 5 convolutions with dense connections (every conv receives the concatenated output of all previous convs in the block).

2. **No Batch Normalization:** BN is removed entirely because its batch statistics during inference can differ from training, causing unwanted artifacts.

3. **Residual Scaling (β = 0.2):** Each residual connection is scaled by 0.2 before adding:
   $$y = 0.2 \cdot F(x) + x$$
   This prevents the network from "overshooting" during the early, unstable phases of training.

4. **Nearest-Neighbor Upsampling:** Instead of PixelShuffle, ESRGAN uses nearest-neighbor upsampling followed by a convolution. This is simpler and avoids the periodic patterns that PixelShuffle can introduce.

---

## 5. Loss Functions

### Pixel Loss (L1 / MSE)

$$L_{L1} = \frac{1}{N} \sum_{i=1}^{N} |I^{SR}_i - I^{HR}_i|$$

$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (I^{SR}_i - I^{HR}_i)^2$$

**Why L1 over MSE for SRResNet?** MSE squares the error, disproportionately penalizing outliers (large pixel errors). This causes the model to "play it safe" and predict the blurry average. L1 loss treats all errors linearly, producing sharper results.

**Why MSE for SRCNN?** The original 2014 paper used MSE, and we follow it for historical accuracy.

### VGG Perceptual Loss

```python
sr_features = vgg19_features(normalize(sr_image))   # Extract conv5_4 features
hr_features = vgg19_features(normalize(hr_image))   # Extract conv5_4 features
perceptual_loss = MSE(sr_features, hr_features)
```

$$L_{perceptual} = \frac{1}{C \cdot H \cdot W} \|\phi_{5,4}(I^{SR}) - \phi_{5,4}(I^{HR})\|^2$$

where $\phi_{5,4}$ extracts features from VGG19's 35th layer (conv5_4, before ReLU activation).

**Why?** Pixel loss measures low-level similarity (exact color values), but humans perceive image quality based on high-level features (edges, textures, structures). VGG perceptual loss measures similarity in **feature space** — two images that look perceptually similar to a human will also have similar VGG features, even if their exact pixel values differ.

**Why before activation (pre-ReLU)?** ESRGAN showed that extracting features **before** the ReLU activation preserves more information about the feature magnitudes, leading to brighter, more vivid outputs.

### Adversarial Loss (BCE)

$$L_{adv} = -\log(\sigma(D(G(I^{LR}))))$$

where $\sigma$ is the sigmoid function and $D$ is the discriminator.

**Why Binary Cross-Entropy?** This is the standard loss for binary classification (real vs. fake). When the generator fools the discriminator (output close to 1.0), the loss is low. When the discriminator correctly identifies a fake, the loss is high, pushing the generator to improve.

### Label Smoothing

```python
# Instead of target = 1.0 for real images:
target = uniform(0.8, 1.0)   # Smoothed label
```

**Why?** A discriminator that is "too confident" (always outputting exactly 0 or 1) provides useless gradients to the generator. Label smoothing forces the discriminator to remain uncertain, providing more informative gradient signals.

---

## 6. Evaluation Metrics

### PSNR (Peak Signal-to-Noise Ratio)

$$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{MAX^2}{MSE}\right) \text{ dB}$$

where $MAX = 255$ for uint8 images.

**Interpretation:** Higher PSNR = less pixel-level distortion. A PSNR of 30 dB means the average pixel error is ~8 intensity levels (out of 255). Typical ranges: Bicubic ~26 dB, SRCNN ~28 dB, SRResNet ~30 dB.

**Limitation:** PSNR can be misleading. A perfectly blurry image (averaged with a large Gaussian kernel) can have higher PSNR than a sharp, detailed image because the blur reduces pixel-level MSE without producing sharp edges.

### SSIM (Structural Similarity Index)

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

where:
- $\mu_x, \mu_y$ = local means (luminance)
- $\sigma_x, \sigma_y$ = local standard deviations (contrast)
- $\sigma_{xy}$ = local cross-correlation (structure)
- $C_1, C_2$ = stabilization constants

**Why SSIM?** Unlike PSNR (which treats each pixel independently), SSIM measures **structural** similarity by comparing local patches. It is closer to human visual perception because our eyes are more sensitive to structural distortions (blurring, blocking) than to uniform intensity shifts.

### LPIPS (Learned Perceptual Image Patch Similarity)

```python
lpips_model = lpips.LPIPS(net="alex")   # Pre-trained AlexNet-based metric
distance = lpips_model(sr_tensor, hr_tensor)
```

**What it does:** Extracts deep features from a pre-trained AlexNet and computes a learned weighted distance between the feature representations of two images.

**Why LPIPS?** It was specifically trained to match human perceptual judgments. In blind tests, LPIPS scores correlate more strongly with human quality ratings than PSNR or SSIM. Lower LPIPS = more perceptually similar.

---

## Summary: The Complete Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                               │
│                                                                         │
│  Upload (PNG/JPG)                                                       │
│     ↓                                                                   │
│  PIL.Image.open() → .convert("RGB")                                     │
│     ↓                                                                   │
│  np.array() → (H, W, 3) uint8 [0, 255]                                │
│     ↓                                                                   │
│  mod_crop(scale_factor=4) → dimensions divisible by 4                   │
│     ↓                                                                   │
│  [Optional] Restoration: Denoise → Deblur → Enhance                    │
│     ↓                                                                   │
│  float32 / 255.0 → [0, 1]                                              │
│     ↓                                                                   │
│  .permute(2,0,1).unsqueeze(0) → (1, 3, H, W) tensor                   │
│     ↓                                                                   │
│  prepare_input() → match model's color space & range                    │
│     ↓                                                                   │
│  model(tensor) or tiled_forward() → (1, 3, H×4, W×4)                  │
│     ↓                                                                   │
│  postprocess_output() → back to RGB [0, 1]                             │
│     ↓                                                                   │
│  .clamp(0, 1) → remove out-of-range values                             │
│     ↓                                                                   │
│  .permute(1,2,0) × 255 → uint8 → PIL Image → Display                  │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```
