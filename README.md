# Image and Video Super-Resolution

CSET225 Image and Video Processing semester project. The standalone web app restores and upscales images and short clips at 2× or 4×. FastAPI serves a buildless HTML, CSS, and JavaScript frontend and runs the shared PyTorch/OpenCV pipeline locally. The reconstruction experiment creates a low-resolution sample from a known original so you can measure the result against ground truth.

## Run locally

On Windows, double-click `start.bat` after installing the dependencies. It opens ImageSR Studio in your browser. Keep the launcher window open while using the app.

Use Python 3.12 or 3.14. The app runs on CPU and selects CUDA when the installed PyTorch build and hardware support it.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_web.py
```

Open **http://127.0.0.1:8000**. Use `python run_web.py --port 8001` if port 8000 is in use. The older Streamlit interface is available through `python run_app.py`. On macOS or Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python`. The repository includes Real-ESRGAN checkpoints for both scales. Bicubic interpolation is available immediately. The app lists SRCNN, SRResNet, and SRGAN when a trained checkpoint for the selected scale is present. The supplied training configurations use 4×.

## Demo walkthrough

1. The **Studio** opens with an actual 4× Real-ESRGAN output from DIV2K image 0807. Drag the comparison divider and use **Zoom 2×** to inspect the trees and shoreline. Open **Experiment**, use the sample photo, select **4×**, and run with **Bicubic**. This establishes the baseline using an antialiased bicubic downsample of the original.
2. Select **Real-ESRGAN**, run again, and compare PSNR, SSIM and visual detail. Treat a sharper-looking result and a higher reference score as separate observations.
3. Enable **Simulate JPEG compression** at quality 30. Run with the same model before and after selecting **JPEG Fix**. This isolates the effect of deblocking while holding the reference and degradation fixed.
4. Select **Custom** and try Gaussian or median spatial smoothing, histogram equalization, and Gaussian high-boost filtering. Inspect the intermediate stages. Open **Compare edges, color and frequency** to compare matched center regions.
5. In **Studio**, select **Video**, select **Bicubic** for a quick CPU demo, and use the sample clip or upload your own. Adjust CRF and compare the export size and appearance. Download the H.264 MP4.
6. Open **Models** for checkpoint provenance, the held-out DIV2K benchmark, and syllabus coverage.

## Syllabus coverage

| Module | Topic from the syllabus | Where it appears |
| --- | --- | --- |
| 1 | Image acquisition and basics | The reconstruction experiment reduces spatial sampling using antialiased bicubic interpolation, then reconstructs on a 2× or 4× grid. |
| 1 | Intensity transformations | Global histogram equalization, CLAHE and gamma correction with a before-and-after luma histogram. |
| 1 | Spatial filtering | Gaussian convolution and median filtering before super-resolution. |
| 1 | Edge and corner detection | Canny and Sobel maps for the same region of the bicubic and processed images. |
| 2 | Color image processing | RGB-to-YCbCr conversion, individual Y/Cb/Cr planes, and luma-only histogram equalization and frequency filtering. |
| 3 | Frequency-domain enhancement | A two-dimensional DFT and inverse DFT implement Gaussian low-pass and high-boost filters. Centered log spectra show frequency content. |
| 3 | Compression and fidelity | JPEG quality controls quantization during simulated degradation and export. JPEG deblocking runs before upscaling, and PSNR and SSIM measure reconstruction against ground truth. |
| 4 | Video processing and compression | Decode a video into sampled frames, run the image pipeline on each frame, and encode H.264 at a constant output frame rate derived from the source. CRF controls the encoding quality and file size. |

## Processing and evaluation

The image pipeline applies selected restoration operations before inference and can apply refinement afterward. Bicubic provides a comparison at the same output size. The 2× Real-ESRGAN model pads inputs when pixel unshuffle needs even dimensions, then removes the corresponding output padding. Large neural inputs use tiles with surrounding context.

The experiment aligns the reference by trimming its bottom and right edges to a full sampling grid. PSNR and SSIM use the full aligned RGB reference. The single-image workflow shows the bicubic comparison and diagnostic maps when only the low-resolution original is available. In JPEG export, the displayed PSNR compares the decoded JPEG against the output before export.

Video processing uses FFmpeg and keeps at most the current frame and the first-frame preview in memory. Clips are limited to 15 seconds, 300 frames, and 2.1 megapixels per output frame. The first input audio track is encoded as AAC when available. The frame rate is constant in the export. Frame-by-frame processing may cause temporal flicker with neural upscaling.

## Training and dataset benchmarks

The included presentation checkpoints use a short CPU training run on all 800 DIV2K training originals from `archive`, taking a deterministic 160 × 160 crop per image and sampling 64 × 64 patches for SRCNN and SRResNet and 48 × 48 patches for SRGAN. It selects checkpoints using 13 validation crops. The 13 benchmark crops use different DIV2K validation originals.

```powershell
python scripts/train_presentation.py --epochs 5 8 1
python scripts/verify_models.py
python scripts/benchmark.py --image_dir outputs/presentation/benchmark --scale 4 --num_images 0 --output_dir outputs/presentation/metrics
```

The trained weights are under `models/srcnn`, `models/srresnet`, and `models/srgan`. The benchmark snapshot in `web/assets/div2k-benchmark.json` makes the measured results available on another machine. The bundled Real-ESRGAN weights are under `models/pretrained`. Once Python dependencies are installed, the presentation demos run offline. Checkpoint availability and measured results populate the Models page.

Training is optional. The supplied configurations use DIV2K and train the local 4× checkpoints. SRGAN downloads about 550 MiB of pretrained VGG19 weights on first use. Set `TORCH_HOME` to place the PyTorch cache on a drive with enough space. Set the paths in the YAML files to use an existing copy of the dataset.

```powershell
python scripts/train.py --config configs/train_srcnn.yaml
python scripts/train_all.py
```

Point the benchmark at a folder of high-resolution originals. It generates the low-resolution inputs, applies available models, writes Markdown and JSON reports to `outputs/metrics`, and records results for the legacy dashboard. Use `--output_dir outputs/presentation/metrics` to also show the report on the standalone Models page.

```powershell
python scripts/benchmark.py --image_dir "path/to/reference-images" --models bicubic esrgan --scale 2 --num_images 5
python -m unittest discover -s tests
```

## References

- Dong et al., *Image Super-Resolution Using Deep Convolutional Networks*
- Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*
- Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution With Pure Synthetic Data*
- [FFmpeg documentation](https://ffmpeg.org/ffmpeg.html)
