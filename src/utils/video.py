"""Decode frames, reuse the still-image pipeline, and export H.264 MP4."""

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np

from src.utils.image_inference import ImageProcessor

MAX_VIDEO_FRAMES = 300
MAX_VIDEO_PIXELS = 2_100_000
MAX_VIDEO_DURATION = 15
VIDEO_FORMATS = {
    ".mp4": "mov", ".mov": "mov", ".avi": "avi", ".mkv": "matroska", ".webm": "matroska",
}


@dataclass
class VideoResult:
    frame_count: int
    fps: float
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    input_frame: np.ndarray
    output_frame: np.ndarray


def process_video(
    source: Path,
    destination: Path,
    processor: ImageProcessor,
    *,
    crf: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> VideoResult:
    """Process a short clip with bounded memory and publish only a complete output."""
    if not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be between 0 and 51.")
    if source.resolve() == destination.resolve():
        raise ValueError("Choose a different output path from the input video.")
    container = VIDEO_FORMATS.get(source.suffix.lower())
    if container is None:
        raise ValueError("Use an MP4, MOV, AVI, MKV, or WebM video.")
    source_options = ["-protocol_whitelist", "file,pipe", "-f", container]
    # Read the first video stream as RGB frames without holding the whole clip in memory.
    frames = imageio_ffmpeg.read_frames(
        str(source), input_params=source_options,
        output_params=["-map", "0:V:0", "-fps_mode", "passthrough"],
    )
    with closing(frames):
        metadata = next(frames)
        width, height = metadata["size"]
        fps = metadata["fps"]
        if not math.isfinite(fps) or fps <= 0 or min(width, height) <= 0:
            raise ValueError("The video dimensions or frame rate could not be read.")
        out_width, out_height = width * processor.scale, height * processor.scale
        encoded_width, encoded_height = out_width + out_width % 2, out_height + out_height % 2
        if encoded_width * encoded_height > MAX_VIDEO_PIXELS:
            raise ValueError("Video output exceeds 2.1 megapixels per frame. Use a smaller clip or scale.")
        # The smaller of the frame cap and the duration-derived cap controls processing.
        limit = min(MAX_VIDEO_FRAMES, int(MAX_VIDEO_DURATION * fps + 0.5))
        estimated = max(1, min(limit, round(metadata["duration"] * fps)))

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
            pending = Path(temp_dir) / "encoded.mp4"
            # FFmpeg encodes piped RGB frames as H.264 and optionally maps the first source audio track to AAC.
            command = [
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_width}x{out_height}",
                "-r", str(fps), "-i", "pipe:0", *source_options, "-i", str(source),
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast",
                "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:a", "aac", "-af", "apad", "-shortest", "-movflags", "+faststart", str(pending),
            ]
            count = 0
            first_input = first_output = None
            with tempfile.TemporaryFile() as log:
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                try:
                    for count, raw in enumerate(frames, 1):
                        if count > limit:
                            raise ValueError("Use a clip up to 15 seconds and 300 frames.")
                        # One decoded RGB frame uses the same image processor as the still-image route.
                        image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                        result = processor.process(image).image
                        if count == 1:
                            first_input, first_output = image, result
                        encoder.stdin.write(result.tobytes())
                        if progress:
                            progress(count, estimated)
                    # Close the input stream so the encoder can flush and finalize the MP4.
                    encoder.stdin.close()
                    if encoder.wait(timeout=60) != 0:
                        raise BrokenPipeError("FFmpeg exited with an error.")
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("FFmpeg timed out while finalizing the video.") from exc
                except BrokenPipeError as exc:
                    log.seek(0)
                    detail = log.read().decode(errors="replace").strip()
                    raise RuntimeError(f"Video encoding failed. {detail}") from exc
                finally:
                    if encoder.poll() is None:
                        encoder.kill()
                        encoder.wait()
                    encoder.stdin.close()

            if first_input is None or not pending.is_file():
                raise ValueError("No video frames could be decoded.")
            # Publish the video only after encoding succeeds and a frame was processed.
            pending.replace(destination)

    return VideoResult(
        count, fps, (width, height), (encoded_width, encoded_height), first_input, first_output,
    )
