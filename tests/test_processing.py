"""Regression tests for the image and video inference paths."""

from contextlib import closing
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

from app.model_loader import load_sr_model, model_identifier
from models import get_model
from src.evaluation.image_analysis import color_channels, edge_map, fidelity, jpeg_roundtrip, sampling_pair
from src.restoration.frequency import gaussian_frequency_filter
from src.utils.common import save_checkpoint
from src.utils.image_inference import ImageProcessor
from src.utils.tiled_inference import tiled_forward
from src.utils.video import process_video


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(min(cls.threads, 4))
        cls.rng = np.random.default_rng(30)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_bicubic_preserves_odd_and_tiny_dimensions(self):
        processor = ImageProcessor(get_model("bicubic", scale_factor=4), "bicubic", 4)
        for height, width in [(1, 3), (17, 19), (31, 23)]:
            with self.subTest(height=height, width=width):
                image = self.rng.integers(256, size=(height, width, 3), dtype=np.uint8)
                result = processor.process(image)
                self.assertEqual(result.image.shape, (height * 4, width * 4, 3))

    def test_bicubic_tiles_agree_at_borders(self):
        model = get_model("bicubic", scale_factor=2)
        image = torch.full((1, 3, 17, 33), 0.8)
        expected = model(image)
        actual = tiled_forward(model, image, tile_size=10, overlap=4, scale_factor=2)
        torch.testing.assert_close(actual, expected)

    def test_pixel_unshuffle_tiles_preserve_global_alignment(self):
        class UnshuffleUpsampler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pixel_unshuffle = torch.nn.PixelUnshuffle(2)

            def forward(self, image):
                # Retain one pixel phase per channel so misalignment is observable.
                phase_zero = self.pixel_unshuffle(image)[:, ::4]
                return phase_zero.repeat_interleave(4, dim=-1).repeat_interleave(4, dim=-2)

        model = UnshuffleUpsampler()
        image = torch.rand(1, 3, 17, 21)
        actual = tiled_forward(model, image, tile_size=8, overlap=2, scale_factor=2)
        padded = torch.nn.functional.pad(image, (0, 1, 0, 1), mode="reflect")
        torch.testing.assert_close(actual, model(padded)[..., :34, :42])

    def test_explicit_missing_weights_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            with TemporaryDirectory() as directory:
                load_sr_model("srcnn", 4, str(Path(directory) / "missing.pth"))

    def test_checkpoint_restores_architecture_options_and_checks_scale(self):
        options = {"scale_factor": 4, "num_residual_blocks": 2, "residual_clamp": 8.0}
        model = get_model("srresnet", **options)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trained.pth"
            torch.save({"model_state_dict": model.state_dict(), "model_kwargs": options}, path)
            loaded = load_sr_model("srresnet", 4, str(path), device=torch.device("cpu"))
            self.assertEqual(loaded.residual_clamp, 8.0)
            self.assertEqual(len(loaded.residual_blocks), 2)
            with self.assertRaisesRegex(ValueError, "different scale"):
                load_sr_model("srresnet", 2, str(path))

    def test_failed_checkpoint_write_preserves_existing_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            path.write_bytes(b"existing checkpoint")
            with patch("src.utils.common.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "Check disk space"):
                    save_checkpoint({}, path)
            self.assertEqual(path.read_bytes(), b"existing checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_bundled_checkpoints_with_odd_dimensions(self):
        for scale in (2, 4):
            with self.subTest(scale=scale):
                model = load_sr_model("esrgan", scale, device=torch.device("cpu"))
                processor = ImageProcessor(model, "esrgan_generator", scale)
                image = self.rng.integers(256, size=(9, 13, 3), dtype=np.uint8)
                self.assertEqual(processor.process(image).image.shape, (9 * scale, 13 * scale, 3))

    def test_local_checkpoints_enhance_detail_without_shifting_chroma(self):
        device = torch.device("cpu")
        flat_color = np.full((7, 9, 3), [40, 156, 230], dtype=np.uint8)
        texture = self.rng.integers(16, 160, size=(195, 19, 1), dtype=np.uint8)
        # The long axis triggers tiled inference on an image with a fixed RGB channel offset.
        image = np.concatenate((texture, texture + 16, texture + 48), axis=2)
        baseline = ImageProcessor(load_sr_model("bicubic", 4, device=device), "bicubic", 4)
        expected = baseline.process(image).image.astype(np.int16)

        for name in ("srcnn", "srresnet", "srgan"):
            with self.subTest(model=name):
                model = load_sr_model(name, 4, device=device)
                processor = ImageProcessor(model, model_identifier(name), 4, refinement=0.4)
                flat = processor.process(flat_color).image
                np.testing.assert_array_equal(flat, np.broadcast_to(flat_color[0, 0], (28, 36, 3)))

                result = processor.process(image, intermediates=True)
                details = result.intermediates["before_refinement"].astype(np.int16) - expected
                self.assertTrue(np.any(details))
                for output in (result.image.astype(np.int16) - expected, details):
                    # Common brightness offsets may differ by one after 8-bit rounding.
                    self.assertLessEqual(np.abs(output[..., :1] - output[..., 1:]).max(), 1)

    def test_frequency_lowpass_reduces_checkerboard_contrast(self):
        checkerboard = (48 + 160 * (np.indices((31, 33)).sum(axis=0) % 2)).astype(np.uint8)
        rgb = np.repeat(checkerboard[..., None], 3, axis=2)
        lowpass = gaussian_frequency_filter(rgb, "lowpass", cutoff=0.10)
        highboost = gaussian_frequency_filter(rgb, "highboost", cutoff=0.10)
        self.assertLess(lowpass.std(), rgb.std() / 10)
        self.assertGreater(highboost.std(), rgb.std())

    def test_restoration_previews_for_smoothing_and_refinement(self):
        image = self.rng.integers(256, size=(13, 17, 3), dtype=np.uint8)
        model = get_model("bicubic", scale_factor=2)
        plain = ImageProcessor(model, "bicubic", 2).process(image, intermediates=True)
        self.assertEqual(plain.intermediates, {})

        denoiser = ImageProcessor(
            model, "bicubic", 2, restoration={"denoise": True, "denoise_method": "median"},
        )
        denoised = denoiser.process(image, intermediates=True)
        self.assertEqual(list(denoised.intermediates), ["input", "denoise"])
        np.testing.assert_array_equal(denoised.intermediates["input"], image)

        refiner = ImageProcessor(model, "bicubic", 2, refinement=0.4)
        refined = refiner.process(image, intermediates=True)
        self.assertEqual(list(refined.intermediates), ["before_refinement", "refinement"])
        np.testing.assert_array_equal(refined.intermediates["before_refinement"], plain.image)
        np.testing.assert_array_equal(refined.intermediates["refinement"], refined.image)
        self.assertEqual(refiner.process(image).intermediates, {})

    def test_sampling_jpeg_color_edges_and_reference_scores(self):
        image = self.rng.integers(256, size=(67, 71, 3), dtype=np.uint8)
        reference, low = sampling_pair(image, 4)
        self.assertEqual(reference.shape, (64, 68, 3))
        self.assertEqual(low.shape, (16, 17, 3))
        payload, decoded = jpeg_roundtrip(low, 20)
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        self.assertEqual(decoded.shape, low.shape)
        identical = fidelity(reference, reference)
        self.assertEqual(identical["PSNR (dB)"], float("inf"))
        self.assertEqual(identical["SSIM"], 1.0)
        self.assertLess(fidelity(low, decoded)["SSIM"], 1.0)
        y, cb, cr = color_channels(low)
        self.assertEqual(y.shape, cb.shape)
        self.assertEqual(cb.shape, cr.shape)
        self.assertEqual(edge_map(y, "Canny").shape, y.shape)
        self.assertEqual(edge_map(y, "Sobel").shape, y.shape)

    def test_video_dimensions_frame_count_and_audio(self):
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        processor = ImageProcessor(get_model("bicubic", scale_factor=2), "bicubic", 2)
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            target = Path(directory) / "output.mp4"
            subprocess.run([
                ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc=size=33x25:rate=12:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.25",
                "-c:v", "libx264", "-pix_fmt", "yuv444p", "-c:a", "aac", str(source),
            ], check=True, timeout=20)
            result = process_video(source, target, processor)
            self.assertEqual(result.frame_count, 12)
            self.assertEqual(result.output_size, (66, 50))
            with closing(imageio_ffmpeg.read_frames(str(target))) as frames:
                meta = next(frames)
                self.assertEqual(meta["size"], (66, 50))
                self.assertEqual(meta["fps"], 12)
                self.assertEqual(sum(1 for _ in frames), 12)
            subprocess.run(
                [ffmpeg, "-v", "error", "-i", str(target), "-map", "0:a:0", "-f", "null", "-"],
                check=True, timeout=20,
            )
