"""
Synthetic dataset generator for training-pipeline smoke tests.

Creates a minimal folder layout compatible with ImageFolder:

    <output>/
        train/class_a/*.png
        train/class_b/*.png
        valid/class_a/*.png
        valid/class_b/*.png

Images are solid RGB fills with class-dependent hues.
That is enough to exercise data loading, forward/backward passes,
checkpoint saving, and device selection (CPU/MPS).

Used by:
- configs/mac_test_resnet18_cpu.yaml
- configs/mac_test_resnet18_mps.yaml

Example:
    python tools/create_minimal_dataset.py --output data/minimal_test
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def create_class_images(class_dir: Path, class_idx: int, count: int, size: int = 224) -> None:
    """
    Create a series of PNG images for a single class.

    Colors depend on the class index and sample number so classes
    remain linearly separable even without augmentations.

    Args:
        class_dir: Class folder (created if missing).
        class_idx: Numeric class index (0, 1, ...).
        count: Number of images to create.
        size: Side length of the square image in pixels.
    """
    class_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        base = (class_idx * 40) % 255
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 0] = base
        image[:, :, 1] = (base + i * 10) % 255
        image[:, :, 2] = (base + class_idx * 25) % 255
        Image.fromarray(image).save(class_dir / f"sample_{i:03d}.png")


def create_minimal_dataset(output_dir: str, train_per_class: int = 8, val_per_class: int = 4) -> None:
    """
    Create a complete minimal dataset with two classes.

    Args:
        output_dir: Dataset root directory.
        train_per_class: Number of train images per class.
        val_per_class: Number of valid images per class.
    """
    root = Path(output_dir)
    classes = ["class_a", "class_b"]

    for split, count in (("train", train_per_class), ("valid", val_per_class)):
        for class_idx, class_name in enumerate(classes):
            create_class_images(root / split / class_name, class_idx, count)

    print(f"Dataset created at: {root.resolve()}")
    print(f"Classes: {', '.join(classes)}")
    print(f"Train images per class: {train_per_class}")
    print(f"Valid images per class: {val_per_class}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a minimal RFUAV test dataset")
    parser.add_argument(
        "--output",
        default="data/minimal_test",
        help="Output dataset directory",
    )
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=4)
    args = parser.parse_args()

    create_minimal_dataset(
        output_dir=args.output,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
    )


if __name__ == "__main__":
    main()
