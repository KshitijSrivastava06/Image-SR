"""Load real checkpoints and verify inference on a DIV2K sample at every ready scale."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from app.model_loader import MODEL_LABELS, get_weight_status, load_sr_model, model_identifier
from src.utils.common import get_device
from src.utils.image_inference import ImageProcessor


def main() -> int:
    device = get_device()
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    with Image.open(ROOT / "web" / "assets" / "demo-input.png") as example:
        image = np.asarray(example.crop((87, 28, 136, 63)).convert("RGB"))
    ready = {scale: get_weight_status(scale) for scale in (2, 4)}
    failed = False
    print(f"Checkpoint verification on {device}")
    for name, label in MODEL_LABELS.items():
        if not ready[4][name]:
            print(f"{label} 4x: missing checkpoint")
            failed = True
        for scale, availability in ready.items():
            if not availability[name]:
                continue
            try:
                model = load_sr_model(name, scale, device=device)
                # Exercise real saved checkpoints on an actual sample. This is a functionality check.
                result = ImageProcessor(model, model_identifier(name), scale).process(image)
                if float(result.image.std()) < 1.0:
                    raise RuntimeError("The output has almost no variation on the sample.")
                height, width = result.image.shape[:2]
                print(f"{label} {scale}x: passed, {width} x {height}, {result.seconds:.3f} s")
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"{label} {scale}x: failed, {exc}")
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
