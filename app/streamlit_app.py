"""
Super-Resolution Platform — Streamlit Web Application.

Features:
    - Image upload with drag-and-drop
    - Model selection (Bicubic, SRCNN, SRResNet, SRGAN, ESRGAN)
    - Scale factor selection (2x, 4x, 8x)
    - Optional restoration pipeline (denoise, deblur, enhance)
    - Before/After comparison view
    - PSNR, SSIM, LPIPS metric display
    - Download enhanced image
    - Model comparison dashboard
"""

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


from app.model_loader import load_sr_model, list_available_models, get_weight_status
from app.utils import (
    format_metric,
    get_image_info,
    numpy_to_pil,
    pil_to_bytes,
    pil_to_numpy,
    pil_to_tensor,
    tensor_to_pil,
)
from src.utils.preprocessing import mod_crop
from models.pretrained_weights import PRETRAINED_REGISTRY
from src.restoration import RestorationPipeline
from src.utils.common import tensor_to_numpy
from src.utils.inference_utils import postprocess_output, prepare_input
from src.utils.tiled_inference import tiled_forward, supports_tiling
from src.evaluation.metrics import MetricCalculator
from src.restoration.post_processor import PostProcessor


# =====================================================================
# Page Configuration
# =====================================================================

st.set_page_config(
    page_title="AI Image Super-Resolution",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Dark-themed premium styling */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
        font-family: 'Inter', sans-serif;
    }

    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label {
        color: #a0aec0;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-ready {
        background: rgba(72, 187, 120, 0.15);
        color: #48bb78;
        border: 1px solid rgba(72, 187, 120, 0.3);
    }

    .status-no-weights {
        background: rgba(236, 201, 75, 0.15);
        color: #ecc94b;
        border: 1px solid rgba(236, 201, 75, 0.3);
    }

    /* Process button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Model info section */
    .model-info {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        margin: 1.5rem 0;
    }

    /* Hide default Streamlit footer */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Cached Model Loading
# =====================================================================

@st.cache_resource
def cached_load_model_v2(model_name: str, scale_factor: int):
    """Load and cache a model (persists across reruns)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_sr_model(model_name, scale_factor, device=device)


# =====================================================================
# Sidebar
# =====================================================================

def render_sidebar():
    """Render the sidebar with configuration controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        # Model Selection
        st.markdown("### 🤖 Model")
        model_name = st.selectbox(
            "Select Model",
            options=["Bicubic", "SRCNN", "SRResNet", "SRGAN", "ESRGAN"],
            index=4,
            help="Choose the super-resolution model",
        )

        # Model status
        weight_status = get_weight_status()
        model_key = model_name.lower()
        if weight_status.get(model_key, False):
            st.markdown('<span class="status-badge status-ready">✓ Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-no-weights">⚠ No Weights</span>', unsafe_allow_html=True)
            st.caption("Model will use random weights. Train first or provide weights.")

        # Model descriptions
        descriptions = {
            "Bicubic": "Classical interpolation baseline. Fast but lacks detail.",
            "SRCNN": "3-layer CNN (~57K params). First DL baseline.",
            "SRResNet": "16 residual blocks (~1.5M params). High PSNR.",
            "SRGAN": "GAN-trained SRResNet. Perceptually sharp.",
            "ESRGAN": "RRDB + RaGAN. Powered by Real-ESRGAN weights for best real-world results.",
        }
        st.caption(descriptions.get(model_name, ""))

        st.markdown("---")

        # Scale Factor
        st.markdown("### 🔍 Scale Factor")
        scale_factor = st.radio(
            "Upscaling",
            options=[2, 4],
            index=1,
            horizontal=True,
            format_func=lambda x: f"{x}×",
        )

        # ESRGAN supports 2× and 4× with pretrained weights.
        # Other models only have pretrained weights for 4×.
        esrgan_supported = model_name.lower() == "esrgan" and scale_factor in (2, 4)
        if scale_factor != 4 and model_name.lower() != "bicubic" and not esrgan_supported:
            st.warning(
                f"⚠️ Pretrained weights are trained for 4× only. "
                f"At {scale_factor}×, random weights will be used and output quality will be poor."
            )
        elif model_name.lower() == "esrgan" and scale_factor == 2:
            st.info("2× mode uses Real-ESRGAN x2plus pretrained weights.")

        st.markdown("---")

        # Restoration Pipeline
        st.markdown("### 🔧 Restoration Pipeline")
        st.caption("Pre-processing before super-resolution")

        preset = st.selectbox(
            "Preset",
            ["None", "Photo Cleanup", "Old Photo Restore", "JPEG Fix", "Custom"],
            help="Auto-configure the restoration pipeline"
        )

        use_jpeg = False
        jpeg_quality = 50
        use_denoise = False
        denoise_method = "nlm"
        denoise_strength = 10
        use_deblur = False
        deblur_method = "unsharp"
        deblur_strength = 1.5
        use_enhance = False
        enhance_method = "clahe"
        enhance_strength = 2.0
        use_post_process = True
        post_process_strength = 0.5

        if preset == "Photo Cleanup":
            use_denoise, denoise_method, denoise_strength = True, "nlm", 5
            use_post_process, post_process_strength = True, 0.4
        elif preset == "Old Photo Restore":
            use_denoise, denoise_method, denoise_strength = True, "nlm", 15
            use_enhance, enhance_method, enhance_strength = True, "clahe", 3.0
            use_post_process, post_process_strength = True, 0.7
        elif preset == "JPEG Fix":
            use_jpeg, jpeg_quality = True, 40
            use_post_process, post_process_strength = True, 0.5
        elif preset == "Custom":
            with st.expander("Advanced Settings", expanded=True):
                use_jpeg = st.checkbox("🧩 JPEG Deblock", value=False)
                if use_jpeg:
                    jpeg_quality = st.slider("JPEG Quality", 10, 100, 50, key="jpeg_q")
                
                use_denoise = st.checkbox("🔇 Denoise", value=False)
                if use_denoise:
                    denoise_method = st.selectbox(
                        "Denoise Method", ["nlm", "median", "bilateral"],
                        key="denoise_method",
                    )
                    denoise_strength = st.slider("Denoise Strength", 1, 30, 10, key="denoise_str")

                use_deblur = st.checkbox("🌀 Deblur", value=False)
                if use_deblur:
                    deblur_method = st.selectbox(
                        "Deblur Method", ["unsharp", "laplacian", "wiener"],
                        key="deblur_method",
                    )
                    deblur_strength = st.slider("Deblur Strength", 0.5, 5.0, 1.5, key="deblur_str")

                use_enhance = st.checkbox("✨ Contrast Enhance", value=False)
                if use_enhance:
                    enhance_method = st.selectbox(
                        "Enhance Method", ["clahe", "gamma", "auto_wb", "histogram"],
                        key="enhance_method",
                    )
                    enhance_strength = st.slider("Enhance Strength", 0.5, 5.0, 2.0, key="enhance_str")
                
                use_post_process = st.checkbox("🪄 Post-SR Refine", value=True)
                if use_post_process:
                    post_process_strength = st.slider("Refinement Strength", 0.1, 1.0, 0.5, key="pp_str")

        st.markdown("---")

        # About
        st.markdown("### ℹ️ About")
        st.caption(
            "Deep Learning Image Super-Resolution Platform. "
            "Built with PyTorch and Streamlit."
        )

    return {
        "model_name": model_name.lower(),
        "scale_factor": scale_factor,
        "preset": preset,
        "jpeg_deblock": use_jpeg,
        "jpeg_quality": jpeg_quality,
        "denoise": use_denoise,
        "denoise_method": denoise_method,
        "denoise_strength": denoise_strength,
        "deblur": use_deblur,
        "deblur_method": deblur_method,
        "deblur_strength": deblur_strength,
        "enhance": use_enhance,
        "enhance_method": enhance_method,
        "enhance_strength": enhance_strength,
        "post_process": use_post_process,
        "post_process_strength": post_process_strength,
    }


# =====================================================================
# Main Content
# =====================================================================

def render_main(config: dict):
    """Render the main content area."""
    # Header
    st.markdown('<h1 class="main-header">🔬 AI Image Super-Resolution</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enhance image resolution using deep learning — '
        'SRCNN • SRResNet • SRGAN • ESRGAN</p>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab_enhance, tab_dashboard = st.tabs(["🖼️ Enhance Image", "📊 Model Dashboard"])

    with tab_enhance:
        render_enhance_tab(config)

    with tab_dashboard:
        render_dashboard_tab()


def render_enhance_tab(config: dict):
    """Render the image enhancement tab."""
    # File Upload
    uploaded_file = st.file_uploader(
        "Upload an image to enhance",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Drag and drop or click to upload. Supports PNG, JPG, BMP, WebP.",
    )

    if uploaded_file is None:
        # Show placeholder
        st.markdown(
            """
            <div style="
                border: 2px dashed rgba(102, 126, 234, 0.3);
                border-radius: 16px;
                padding: 4rem 2rem;
                text-align: center;
                margin: 2rem 0;
                background: rgba(255, 255, 255, 0.02);
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
                <div style="color: #a0aec0; font-size: 1.1rem;">
                    Upload an image to get started
                </div>
                <div style="color: #718096; font-size: 0.9rem; margin-top: 0.5rem;">
                    Supports PNG, JPG, BMP, WebP
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Load and display original image
    original_image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(original_image)
    
    # Display image metadata
    st.info(f"Image loaded: {original_image.width}×{original_image.height} pixels.")
    
    # Auto-Detect Image Quality
    with st.expander("📊 Auto-Detect Image Quality", expanded=False):
        from src.restoration.auto_detect import ImageQualityAnalyzer
        analyzer = ImageQualityAnalyzer()
        analysis = analyzer.analyze(img_np)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Blur:** {'Detected' if analysis['is_blurry'] else 'None'}")
            st.markdown(f"**Noise:** {'Detected' if analysis['is_noisy'] else 'Low'}")
        with col2:
            st.markdown(f"**JPEG Compression:** {'High' if analysis['is_jpeg_heavy'] else 'Normal'}")
            st.markdown(f"**Suggested Preset:** `{analysis['recommended_preset']}`")
        
        if analysis['recommended_preset'] != "None":
            st.warning(analysis['reason'] + f" Consider selecting the **{analysis['recommended_preset']}** preset in the sidebar.")


    # Invalidate cached results when image changes
    img_hash = hash(uploaded_file.getvalue())
    if st.session_state.get("_last_img_hash") != img_hash:
        for k in ("sr_result", "metrics", "elapsed"):
            st.session_state.pop(k, None)
        st.session_state["_last_img_hash"] = img_hash

    img_info = get_image_info(original_image)

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Original Size", f"{img_info['width']}×{img_info['height']}")
    with col_info2:
        expected_w = img_info["width"] * config["scale_factor"]
        expected_h = img_info["height"] * config["scale_factor"]
        st.metric("Output Size", f"{expected_w}×{expected_h}")
    with col_info3:
        st.metric("Model", config["model_name"].upper())

    # Invalidate cached results when config changes
    config_key = (
        f"{config['model_name']}_{config['scale_factor']}_"
        f"{config['denoise']}_{config['denoise_method']}_{config['denoise_strength']}_"
        f"{config['deblur']}_{config['deblur_method']}_{config['deblur_strength']}_"
        f"{config['enhance']}_{config['enhance_method']}_{config['enhance_strength']}"
    )
    if st.session_state.get("_last_config_key") != config_key:
        for k in ("sr_result", "metrics", "elapsed"):
            st.session_state.pop(k, None)
        st.session_state["_last_config_key"] = config_key

    # Process button
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        process = st.button("🚀 Enhance Image", use_container_width=True, type="primary")

    if process:
        _run_enhancement(original_image, config, uploaded_file.name if hasattr(uploaded_file, "name") else "unknown")
    elif "sr_result" in st.session_state:
        # Show cached results (only when config and image match)
        _display_results(
            original_image,
            st.session_state["sr_result"],
            st.session_state.get("metrics", {}),
            st.session_state.get("elapsed", 0),
        )


@torch.no_grad()
def _run_enhancement(original_image: Image.Image, config: dict, input_filename: str = "unknown"):
    """Run the full enhancement pipeline."""
    progress = st.progress(0, text="Loading model...")

    # Step 1: Load model
    model = load_sr_model(config["model_name"], config["scale_factor"])
    # Determine device from system availability (simpler and correct for all model types)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress.progress(20, text="Model loaded. Preparing image...")

    img_np = pil_to_numpy(original_image)
    use_restoration = config["jpeg_deblock"] or config["denoise"] or config["deblur"] or config["enhance"]

    # Step 2: Mod-crop & Restoration pipeline
    # Mod-crop prevents checkerboard artifacts from PixelShuffle
    img_np = mod_crop(img_np, config["scale_factor"])
    
    intermediates = {}
    if use_restoration:
        progress.progress(30, text="Running restoration pipeline...")
        from src.restoration.pipeline import RestorationPipeline
        pipeline = RestorationPipeline(
            jpeg_deblock=config["jpeg_deblock"],
            jpeg_quality=config["jpeg_quality"],
            denoise=config["denoise"],
            denoise_method=config["denoise_method"],
            denoise_strength=config["denoise_strength"],
            deblur=config["deblur"],
            deblur_method=config["deblur_method"],
            deblur_strength=config["deblur_strength"],
            enhance=config["enhance"],
            enhance_method=config["enhance_method"],
            enhance_strength=config["enhance_strength"],
        )
        img_np, intermediates = pipeline.process(img_np, return_intermediates=True)

    progress.progress(50, text="Running super-resolution...")

    # Step 3: Super-resolution
    lr_tensor = pil_to_tensor(numpy_to_pil(img_np), device=device)
    
    # Apply color space / normalization formatting based on model
    lr_tensor = prepare_input(lr_tensor, config["model_name"], config["scale_factor"], PRETRAINED_REGISTRY)

    start_time = time.perf_counter()
    
    # Determine inference mode: tiled vs direct
    tile_size = config.get("tile_size", 256)
    tile_overlap = config.get("tile_overlap", 16)
    
    # Only use tiled inference if model supports it (no BatchNorm) and either:
    # 1) Image is very large (> 1024x1024), or
    # 2) Model is heavy (ESRGAN) where tile processing prevents GPU OOM
    is_large_image = (original_image.width * original_image.height > 1024 * 1024)
    model_is_esrgan = config["model_name"].lower() in ("esrgan", "esrgan_generator")
    use_tiling = (
        tile_size > 0 
        and supports_tiling(model)
        and (is_large_image or (model_is_esrgan and original_image.width * original_image.height > 512 * 512))
    )

    if use_tiling:
        sr_tensor = tiled_forward(model, lr_tensor, tile_size=tile_size, overlap=tile_overlap, scale_factor=config["scale_factor"])
    else:
        sr_tensor = model(lr_tensor)
        
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # Reverse color space / normalization formatting
    sr_tensor = postprocess_output(sr_tensor, config["model_name"], config["scale_factor"], PRETRAINED_REGISTRY)

    # Convert to numpy for post-processing/metrics
    sr_np = tensor_to_numpy(sr_tensor)
    sr_np = (sr_np * 255.0).clip(0, 255).astype(np.uint8)

    # Step 3.5: Post-SR Refinement
    if config["post_process"]:
        progress.progress(75, text="Refining output...")
        pp = PostProcessor(strength=config["post_process_strength"])
        sr_np = pp.process(sr_np)

    progress.progress(80, text="Computing metrics...")

    # Convert result to PIL for display
    sr_image = numpy_to_pil(sr_np)

    # Step 4: Compute similarity metrics vs bicubic baseline
    # NOTE: These compare SR output against bicubic upscale, NOT a ground-truth HR image.
    # True PSNR/SSIM require a ground-truth HR reference.
    metrics = {}
    try:
        mc = MetricCalculator()
        from PIL import Image as PILImage
        ref = original_image.resize(
            (sr_image.width, sr_image.height),
            PILImage.BICUBIC,
        )
        ref_np = np.array(ref)

        if sr_np.shape == ref_np.shape:
            metrics["psnr"] = mc.calculate_psnr(sr_np, ref_np)
            metrics["ssim"] = mc.calculate_ssim(sr_np, ref_np)
        else:
            st.warning(
                f"Metric shape mismatch: SR={sr_np.shape} vs ref={ref_np.shape}. "
                f"Metrics skipped."
            )
    except Exception as exc:
        st.warning(f"Metric computation failed: {exc}")

    metrics["inference_time"] = elapsed

    progress.progress(100, text="Done!")

    # Log to SQLite DB
    try:
        from src.utils.inference_db import InferenceTracker
        tracker = InferenceTracker()
        
        tracker.log_inference(
            model_arch=config["model_name"],
            scale_factor=config["scale_factor"],
            device=device.type,
            inference_time_ms=elapsed * 1000,
            input_filename=input_filename,
            input_width=original_image.width,
            input_height=original_image.height,
            psnr=metrics.get("psnr"),
            ssim=metrics.get("ssim"),
        )
    except Exception as exc:
        st.warning(f"Database logging failed: {exc}")

    # Cache results
    st.session_state["sr_result"] = sr_image
    st.session_state["metrics"] = metrics
    st.session_state["elapsed"] = elapsed
    st.session_state["intermediates"] = intermediates if 'intermediates' in locals() else {}

    _display_results(original_image, sr_image, metrics, elapsed, st.session_state["intermediates"])


def _display_results(
    original: Image.Image,
    sr_image: Image.Image,
    metrics: dict,
    elapsed: float,
    intermediates: dict = None,
):
    """Display enhancement results with before/after comparison and optional pipeline steps."""
    st.markdown("---")
    st.markdown("### 📊 Results")

    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{elapsed*1000:.0f}ms</div>
                <div class="metric-label">Inference Time</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m2:
        psnr_val = metrics.get("psnr", 0)
        psnr_display = "\u221e" if psnr_val == float("inf") else f"{psnr_val:.2f}"
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{psnr_display}dB</div>
                <div class="metric-label">PSNR vs Bicubic</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m3:
        ssim_val = metrics.get("ssim", 0)
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{ssim_val:.4f}</div>
                <div class="metric-label">SSIM vs Bicubic</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_m4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{sr_image.width}×{sr_image.height}</div>
                <div class="metric-label">Output Size</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.caption("ℹ️ *PSNR and SSIM are calculated against a Bicubic upscale of the input image (no ground-truth HR image is available). Perceptually sharper models like SRGAN/ESRGAN add realistic high-frequency textures that naturally diverge from smooth bicubic baseline.*")

    st.markdown("")

    # Before / After comparison
    st.markdown("### 🔄 Before & After")
    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Original**")
        st.image(original, use_container_width=True)
        st.caption(f"{original.width}×{original.height}")

    with col_after:
        st.markdown("**Enhanced**")
        st.image(sr_image, use_container_width=True)
        st.caption(f"{sr_image.width}×{sr_image.height}")

    # Download button
    st.markdown("---")
    img_bytes = pil_to_bytes(sr_image, format="PNG")
    st.download_button(
        label="⬇️ Download Enhanced Image",
        data=img_bytes,
        file_name="enhanced_image.png",
        mime="image/png",
        use_container_width=True,
    )

    # Optional Pipeline Visualization
    if intermediates and len(intermediates) > 1:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Pipeline Steps Visualization", expanded=False):
            st.markdown("See the intermediate output after each active restoration step.")
            steps = list(intermediates.keys())
            
            # Show steps in columns
            cols = st.columns(len(steps))
            for i, (step_name, img_arr) in enumerate(intermediates.items()):
                with cols[i]:
                    st.image(img_arr, caption=step_name.replace("_", " ").title(), use_container_width=True)
                    if i < len(steps) - 1:
                        st.markdown("<div style='text-align:center;'>⬇️</div>", unsafe_allow_html=True)


# =====================================================================
# Dashboard Tab
# =====================================================================

def render_dashboard_tab():
    """Render the model comparison dashboard."""
    st.markdown("### 📊 Model Comparison Dashboard")
    st.markdown("Compare super-resolution models across key metrics.")

    # Model comparison table
    import pandas as pd

    data = {
        "Model": ["Bicubic", "SRCNN", "SRResNet", "SRGAN", "ESRGAN"],
        "Architecture": [
            "Classical Interpolation",
            "3-layer CNN",
            "16 Residual Blocks",
            "ResNet + Discriminator",
            "23 RRDB + RaGAN",
        ],
        "Parameters": ["0", "~57K", "~1.5M", "~1.5M + 4.2M", "~16.7M + 4.2M"],
        "Loss Function": [
            "N/A",
            "MSE",
            "L1",
            "L1 + VGG + BCE",
            "L1 + VGG + RaGAN",
        ],
        "Typical PSNR (×4)": [
            "~28.4 dB",
            "~30.5 dB",
            "~32.0 dB",
            "~29.4 dB",
            "~28.8 dB",
        ],
        "Typical SSIM (×4)": [
            "~0.810",
            "~0.862",
            "~0.903",
            "~0.847",
            "~0.838",
        ],
        "Perceptual Quality": [
            "⭐",
            "⭐⭐",
            "⭐⭐⭐",
            "⭐⭐⭐⭐",
            "⭐⭐⭐⭐⭐",
        ],
    }

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Architecture comparison
    st.markdown("---")
    st.markdown("### 🏗️ Architecture Progression")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **PSNR-Oriented Models** (Higher PSNR, smoother output):
        - **Bicubic** → Classical, no learning
        - **SRCNN** → First deep learning approach
        - **SRResNet** → Deep residual learning, skip connections

        These models optimize pixel-level accuracy (L1/MSE loss),
        producing high PSNR but sometimes over-smooth textures.
        """)

    with col2:
        st.markdown("""
        **Perceptual-Oriented Models** (Lower PSNR, sharper textures):
        - **SRGAN** → Adversarial training for realism
        - **ESRGAN** → RRDB + relativistic discriminator

        These models add perceptual and adversarial losses,
        generating sharper textures at the cost of slightly lower PSNR.
        ESRGAN produces the most visually convincing results.
        """)

    # Check benchmark results from database
    st.markdown("---")
    st.markdown("### 📈 Benchmark Results")
    
    try:
        from src.utils.inference_db import InferenceTracker
        tracker = InferenceTracker()
        benchmark_df = tracker.get_benchmark_stats()
        
        if not benchmark_df.empty:
            st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

            # Bar charts
            if "avg_psnr" in benchmark_df.columns:
                col_c1, col_c2 = st.columns(2)
                
                # We can group by model_arch for simple bar charts
                summary_df = benchmark_df.groupby("model_arch")[["avg_psnr", "avg_ssim"]].mean()
                
                with col_c1:
                    st.bar_chart(summary_df["avg_psnr"])
                    st.caption("PSNR Comparison (Higher is better)")
                with col_c2:
                    st.bar_chart(summary_df["avg_ssim"])
                    st.caption("SSIM Comparison (Higher is better)")
        else:
            st.info(
                "No benchmark results found in the database. Run the benchmark to see actual comparison data:\n\n"
                "```python\nfrom src.evaluation.benchmark import BenchmarkRunner\n"
                "runner = BenchmarkRunner(models, test_loader)\n"
                "runner.save_to_db('DatasetName')\n```"
            )
    except Exception as exc:
        st.error(f"Failed to load benchmark results from database: {exc}")


# =====================================================================
# Entry Point
# =====================================================================

def main():
    config = render_sidebar()
    render_main(config)


if __name__ == "__main__":
    main()
else:
    # When run via `streamlit run`, __name__ is the module path
    main()
