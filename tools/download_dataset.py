"""
CLI for selective RFUAV dataset download from Hugging Face.

Examples:

    # List all available classes (37)
    python tools/download_dataset.py --list-classes

    # Download 2 classes, 50 images per split
    python tools/download_dataset.py \\
        --classes "DJI AVATA2" "DJI MINI4 PRO" \\
        --max-per-class 50 \\
        --output data/rfuav_subset

    # Train split only
    python tools/download_dataset.py \\
        --classes "DJI AVATA2" \\
        --split train \\
        --max-per-class 100

After download, the layout is ready for `CustomTrainer` and `ImageFolder`.
See also `configs/mac_test_resnet18_hf.yaml`.
"""

import argparse
import logging
import sys
from pathlib import Path

# Put the project root on sys.path so this script can be run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.hf_dataset import (  # noqa: E402
    DEFAULT_IMAGE_SET,
    download_subset,
    list_available_classes,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a subset of kitofrank/RFUAV from Hugging Face",
    )
    parser.add_argument(
        "--output",
        default="data/rfuav_subset",
        help="Output directory with train/ and valid/ folders",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="Drone class names to download (default: all available classes)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        help="Maximum number of images per class in each split",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "both"],
        default="both",
        help="Which dataset split(s) to download",
    )
    parser.add_argument(
        "--image-set",
        default=DEFAULT_IMAGE_SET,
        help="Image set prefix inside the Hugging Face repository",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Print available class names and exit",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: list classes or start a download."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.list_classes:
        for class_name in list_available_classes(image_set=args.image_set):
            print(class_name)
        return

    splits = ("train", "valid") if args.split == "both" else (args.split,)
    summary = download_subset(
        output_dir=args.output,
        classes=args.classes,
        max_per_class=args.max_per_class,
        splits=splits,
        image_set=args.image_set,
    )

    print(f"Dataset saved to: {summary['output_dir']}")
    print(f"Classes: {len(summary['classes'])}")
    for class_name, split_counts in summary["files"].items():
        counts = ", ".join(f"{split}={count}" for split, count in split_counts.items())
        print(f"  - {class_name}: {counts}")


if __name__ == "__main__":
    main()
