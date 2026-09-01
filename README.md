# 🔬 Deep Learning Image Super-Resolution Platform

An end-to-end AI-powered image enhancement platform that increases image resolution and restores visual details using deep learning. Implements SRCNN, SRResNet, SRGAN, and ESRGAN architectures with a complete evaluation and deployment pipeline using pre-trained weights.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)

---

## ✨ Key Features (Full-Stack ML Pipeline)

- **Multi-Model Support**: Compare classical interpolation (Bicubic) against standard CNN baselines (SRCNN, SRResNet) and Adversarial Networks (SRGAN, ESRGAN).
- **Production-Ready Training**: Custom PyTorch training loops featuring Automatic Mixed Precision (AMP) for GPU acceleration, gradient clipping, on-the-fly dataset generation, and automated checkpointing.
- **Tiled Inference**: Automatically handles extremely large images (e.g., 4K/8K) by splitting them into overlapping tiles, completely eliminating GPU Out-Of-Memory (OOM) errors.
- **Dynamic Pre-trained Weights**: The system automatically identifies color space expectations (RGB vs BGR) and normalizations [-1, 1] vs [0, 1] across different model checkpoints.
- **Interactive UI**: A sleek, fully deployed Streamlit web app with side-by-side magnification sliders to visually compare the results.

---

## 🏗️ Architecture Overview

```
Input (LR Image) → [Restoration Pipeline] → [Super-Resolution Model] → Output (HR Image)
```

### Implemented Models

| Model     | Architecture                    | Parameters | Loss Function        | Strength             |
|-----------|---------------------------------|------------|----------------------|----------------------|
| Bicubic   | Classical Interpolation         | 0          | N/A                  | Fast baseline        |
| SRCNN     | 3-layer CNN                     | ~57K       | MSE                  | First DL approach    |
| SRResNet  | 16 Residual Blocks              | ~1.5M      | L1                   | High PSNR            |
| SRGAN     | ResNet Generator + Discriminator| ~5.7M      | L1 + VGG + Adversarial| Perceptually sharp  |
| ESRGAN    | 23 RRDB + Relativistic GAN      | ~16.7M     | L1 + VGG + RaGAN    | Best visual quality  |

---

## 📁 Project Structure

```
image-super-resolution/
├── configs/                    # YAML configuration files
│   └── inference.yaml          # Unified inference config
├── data/                       # Dataset pipeline
│   ├── dataset_manager.py      # Download & manage datasets
│   ├── dataset_validator.py    # Validate image integrity
│   ├── preprocessing.py        # Image cropping utilities
│   ├── dataset_splitter.py     # Train/val/test splitting
│   └── sr_dataset.py           # PyTorch Dataset classes
├── models/                     # Model architectures
│   ├── bicubic.py              # Bicubic interpolation baseline
│   ├── srcnn.py                # SRCNN (Dong et al., 2014)
│   ├── srresnet.py             # SRResNet (Ledig et al., 2017)
│   ├── srgan.py                # SRGAN (Ledig et al., 2017)
│   ├── esrgan.py               # ESRGAN (Wang et al., 2018)
│   └── pretrained_weights.py   # Pre-trained weights manager
├── evaluation/                 # Metrics & benchmarking
│   ├── metrics.py              # PSNR, SSIM, LPIPS
│   ├── evaluator.py            # Model evaluator
│   ├── benchmark.py            # Multi-model benchmark runner
│   └── visualizer.py           # Comparison plots
├── restoration/                # Image restoration pipeline
│   ├── denoiser.py             # Gaussian & salt-pepper denoising
│   ├── deblurrer.py            # Wiener, unsharp, Laplacian
│   ├── enhancer.py             # CLAHE, gamma, white balance
│   └── pipeline.py             # Chained restoration pipeline
├── app/                        # Streamlit web application
│   ├── streamlit_app.py        # Main Streamlit app
│   ├── model_loader.py         # Model loading with caching
│   └── utils.py                # Image conversion helpers
├── utils/                      # Shared utilities
│   ├── common.py               # Device, I/O, tensor conversion
│   └── logger.py               # Configurable logging
├── outputs/                    # Generated outputs
│   ├── images/                 # SR result images
│   └── metrics/                # Benchmark CSVs
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Create Virtual Environment & Install Dependencies

> **Note:** PyTorch CUDA requires **Python 3.12** or below. Python 3.13+ is not supported.

```bash
# Create a virtual environment with Python 3.12
py -3.12 -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate       # Windows (PowerShell / CMD)
# source .venv/bin/activate  # Linux / macOS

# Install PyTorch with CUDA support (for NVIDIA GPUs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Launch Web App (No dataset required!)

You can instantly start enhancing your own images by running the included launch script. The app automatically downloads necessary pre-trained weights on the fly.

```bash
python run_app.py
```

---

## 🏋️ Training Custom Models

The repository includes a highly-optimized MLOps training pipeline built specifically to run on consumer hardware (e.g., RTX 4060 8GB). It uses Automatic Mixed Precision (AMP), gradient clipping, and dynamic dataset loading to maximize GPU utilization.

1. **Configure Hyperparameters**: Modify the YAML files in `configs/` to set your desired batch sizes, learning rates, and losses.
2. **Launch Master Training Pipeline**:
```bash
# Automatically trains SRCNN -> SRResNet -> SRGAN in sequence
python scripts/train_all.py
```

Alternatively, you can launch individual models:
```bash
python scripts/train.py --config configs/train_srresnet.yaml
```

The script will automatically download the academic DIV2K high-resolution dataset, extract random patches on the fly, execute the training loop, and save the best `final.pth` checkpoint to `models/<model_name>/` for the Streamlit app to use.

---

## 📈 Optional: Benchmarking

If you want to formally evaluate model performance (e.g., PSNR/SSIM scores against ground-truth High-Resolution images), you can download a standard dataset and run the evaluator.

### 1. Download a Test Dataset

```python
from data import DatasetDownloader

dl = DatasetDownloader('data/datasets')
dl.download('set5')              # Download a small test set like Set5 or Set14
dl.list_available()              # See all available datasets
```

### 2. Evaluate Models

```python
from evaluation import ModelEvaluator
from models import get_pretrained
from data import SRTestDataset
from torch.utils.data import DataLoader

# Load pre-trained model
model = get_pretrained('srcnn', scale_factor=4)

# Load test dataset
test_ds = SRTestDataset('data/datasets/set5', scale_factor=4)
test_loader = DataLoader(test_ds, batch_size=1)

# Evaluate and save results
evaluator = ModelEvaluator(model, test_loader, model_name='SRCNN')
results = evaluator.evaluate()
evaluator.save_to_db('set5')
```

---

## 📊 Evaluation Metrics

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | 0 – ∞ dB | Higher ↑ |
| **SSIM** | Structural Similarity Index | 0 – 1 | Higher ↑ |
| **LPIPS** | Learned Perceptual Similarity | 0 – 1 | Lower ↓ |

---

## 🔧 Image Restoration Pipeline

```
Input → [Denoise] → [Deblur] → [Enhance] → [Super-Resolve] → Output
```

Each step is independently configurable:

```python
from restoration import RestorationPipeline

pipeline = RestorationPipeline(
    denoise=True, denoise_method='nlm', denoise_strength=10,
    deblur=True, deblur_method='unsharp', deblur_strength=1.5,
    enhance=True, enhance_method='clahe', enhance_strength=2.0,
    sr_model=my_model,
)

result, intermediates = pipeline.process(image, return_intermediates=True)
```

---

## 📚 References

1. **SRCNN**: Dong et al., "Image Super-Resolution Using Deep Convolutional Networks", TPAMI 2016
2. **SRResNet/SRGAN**: Ledig et al., "Photo-Realistic Single Image Super-Resolution Using a GAN", CVPR 2017
3. **ESRGAN**: Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks", ECCVW 2018

---

## 📝 License

This project is for educational and research purposes.
