"""
Entry point for RFUAV model training.

Supports two modes:
- classify — spectrogram classification (ResNet, ViT, etc.) via a YAML config
- detect   — detection with YOLOv5

Examples:

    # CPU smoke test (synthetic dataset)
    python tools/create_minimal_dataset.py
    mkdir -p runs/minimal_test_cpu
    python train.py --cfg configs/mac_test_resnet18_cpu.yaml

    # Training on a Hugging Face subset
    python tools/download_dataset.py --classes "DJI AVATA2" --max-per-class 50
    mkdir -p runs/rfuav_subset_mps
    python train.py --cfg configs/mac_test_resnet18_hf.yaml

    # Detection (YOLO)
    python train.py --mode detect --save-dir runs/yolo --dataset-dir path/to/data
"""
import argparse

from utils.trainer import CustomTrainer
from utils.trainer import DetTrainer


def main():
    parser = argparse.ArgumentParser(description="Train RFUAV classification or detection model")
    parser.add_argument(
        "--cfg",
        default="",
        help="Path to classification YAML config (e.g. configs/mac_test_resnet18_cpu.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "detect"],
        default="classify",
        help="Training mode",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Output directory for detection training",
    )
    parser.add_argument(
        "--dataset-dir",
        default="",
        help="Dataset directory for detection training",
    )
    args = parser.parse_args()

    if args.mode == "classify":
        if not args.cfg:
            raise ValueError("Classification training requires --cfg")
        trainer = CustomTrainer(cfg=args.cfg)
        trainer.train()
        return

    save_dir = args.save_dir
    if not save_dir:
        raise ValueError("Detection training requires --save-dir")

    model = DetTrainer(model_name="yolo", dataset_dir=args.dataset_dir)
    model.train(save_dir=save_dir)


if __name__ == "__main__":
    main()
