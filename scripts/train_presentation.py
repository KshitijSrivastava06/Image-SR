"""Train the three local 4× models on deterministic DIV2K crops."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import sys

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "presentation"
ARCHIVE = ROOT / "archive"
MODELS = ("srcnn", "srresnet", "srgan")


def source_images(split):
    directory = f"DIV2K_{split}_HR"
    folder = ARCHIVE / directory
    candidates = sorted((folder / directory).glob("*.png")) or sorted(folder.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No DIV2K images found in {folder}")
    return candidates


def prepare_crops(sources, group, size):
    folder = OUTPUT / group
    folder.mkdir(parents=True, exist_ok=True)

    def copy_crop(source):
        target = folder / source.name
        if target.is_file():
            return
        with Image.open(source) as image:
            rng = random.Random(source.stem)
            x = rng.randint(0, image.width - size)
            y = rng.randint(0, image.height - size)
            pending = target.with_suffix(".tmp")
            image.crop((x, y, x + size, y + size)).convert("RGB").save(
                pending, format="PNG", compress_level=1,
            )
            pending.replace(target)

    with ThreadPoolExecutor(max_workers=5) as pool:
        for _ in pool.map(copy_crop, sources):
            pass
    print(f"Prepared {len(sources)} DIV2K crops for {group}.", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--epochs", nargs=3, type=int, default=[5, 8, 1], metavar=("SRCNN", "SRRESNET", "SRGAN"))
    args = parser.parse_args()

    train = source_images("train")
    val = source_images("valid")
    # Prepare one deterministic crop per training original. The dataset samples smaller patches per epoch.
    train_folder = prepare_crops(train, "train", 160)
    # Use disjoint original images for checkpoint selection and the held-out benchmark.
    validation = prepare_crops(val[::8], "validation", 160)
    prepare_crops([image for i, image in enumerate(val) if i % 8 == 3], "benchmark", 256)
    if args.prepare_only:
        return

    os.environ["TORCH_HOME"] = str(ROOT / "outputs" / "torch-cache")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import torch
    from scripts.train import run_training

    gpu = torch.cuda.is_available()
    print(f"Training on {torch.cuda.get_device_name() if gpu else 'CPU'}", flush=True)
    for name in args.models:
        config = yaml.safe_load((ROOT / "configs" / f"train_{name}.yaml").read_text(encoding="utf-8"))
        epochs = args.epochs[MODELS.index(name)]
        batch = {"srcnn": 24, "srresnet": 16, "srgan": 16}[name]
        patch = 48 if name == "srgan" else 64
        config["seed"] = 2026
        config["initialize_output"] = True
        config["data"].update(
            train_hr_dir=str(train_folder), val_hr_dir=str(validation), patch_size=patch,
            batch_size=batch, num_workers=0, cache_in_ram=True,
        )
        config["training"].update(epochs=epochs, amp=gpu, val_every=1)
        if name == "srgan":
            # Reuse SRResNet architecture options so the SRGAN warm start matches its generator.
            backbone = yaml.safe_load((ROOT / "configs" / "train_srresnet.yaml").read_text(encoding="utf-8"))
            config["model"].update({
                key: value for key, value in backbone["model"].items() if key not in ("name", "scale_factor")
            })
            config["training"].update(
                discriminator_pool_size=1, perceptual_feature_layer=9,
                loss_weights={"pixel": 1.0, "perceptual": 0.001, "adversarial": 0.0005},
            )
            config["training"]["generator"]["lr"] = 0.00001
            config["training"]["discriminator"]["lr"] = 0.00005
        config["output"]["checkpoint_dir"] = str(OUTPUT / "checkpoints" / (name + "-presentation" if name == "srgan" else name))
        config["provenance"] = {
            "source": "DIV2K archive",
            "training_images": len(train),
            "validation_images": len(list(validation.glob("*.png"))),
            "patch_size": patch,
        }
        path = OUTPUT / f"train_{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        os.chdir(ROOT)
        run_training(str(path))


if __name__ == "__main__":
    main()
