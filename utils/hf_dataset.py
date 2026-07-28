"""
Selective RFUAV dataset download from Hugging Face.

Repository: https://huggingface.co/datasets/kitofrank/RFUAV

Classification spectrograms live under
`ImageSet-AllDrones-MatlabPipeline` with the layout:

    ImageSet-AllDrones-MatlabPipeline/
        train/<class_name>/*.jpg
        valid/<class_name>/*.jpg

After download, files are arranged in an ImageFolder-compatible layout
expected by the project trainer:

    <output_dir>/
        train/<class_name>/*.jpg
        valid/<class_name>/*.jpg
        download_manifest.json

The manifest records which classes and how many files were downloaded.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download, list_repo_files

# Hugging Face Hub dataset identifier.
REPO_ID = "kitofrank/RFUAV"

# Repository subset with spectrograms (Matlab pipeline).
DEFAULT_IMAGE_SET = "ImageSet-AllDrones-MatlabPipeline"

# Allowed image extensions when filtering repository files.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Splits supported by this image set.
SPLITS = ("train", "valid")


def _available_classes_from_repo_files(
    repo_files: Sequence[str],
    image_set: str = DEFAULT_IMAGE_SET,
) -> list[str]:
    """
    Extract class names from an already-fetched repository file list.

    Parses paths of the form `<image_set>/<split>/<class_name>/<file>.jpg`.
    Keeping this separate avoids re-listing files inside `download_subset()`.

    Args:
        repo_files: Full list of file paths from `list_repo_files()`.
        image_set: Image-set prefix inside the repository.

    Returns:
        Sorted list of unique class names.
    """
    classes: set[str] = set()
    prefix = f"{image_set}/"
    for path in repo_files:
        if not path.startswith(prefix):
            continue
        parts = Path(path).parts
        if len(parts) < 4:
            continue
        split, class_name = parts[1], parts[2]
        if split in SPLITS and Path(path).suffix.lower() in IMAGE_EXTENSIONS:
            classes.add(class_name)
    return sorted(classes)


def list_available_classes(
    image_set: str = DEFAULT_IMAGE_SET,
    repo_id: str = REPO_ID,
) -> list[str]:
    """
    Return all classes available in the given Hugging Face image set.

    Performs a network request to the Hub API (~7 s for the full RFUAV repo).

    Args:
        image_set: Image-set prefix (defaults to the Matlab pipeline).
        repo_id: Hugging Face dataset ID.

    Returns:
        Sorted list of class names (37 for DEFAULT_IMAGE_SET).
    """
    return _available_classes_from_repo_files(
        list_repo_files(repo_id, repo_type="dataset"),
        image_set=image_set,
    )


def _class_files(
    class_name: str,
    split: str,
    image_set: str,
    repo_files: Sequence[str],
) -> list[str]:
    """
    Filter repository files for a specific class and split.

    Args:
        class_name: Class name (must match the Hub folder name).
        split: `train` or `valid`.
        image_set: Image-set prefix.
        repo_files: Cached list of all repository files.

    Returns:
        Sorted relative paths of image files.
    """
    prefix = f"{image_set}/{split}/{class_name}/"
    return sorted(
        path
        for path in repo_files
        if path.startswith(prefix) and Path(path).suffix.lower() in IMAGE_EXTENSIONS
    )


def download_subset(
    output_dir: str | Path,
    classes: Sequence[str] | None = None,
    max_per_class: int | None = None,
    splits: Sequence[str] = SPLITS,
    image_set: str = DEFAULT_IMAGE_SET,
    repo_id: str = REPO_ID,
) -> dict:
    """
    Download selected spectrograms and arrange them for training.

    Files are fetched via `hf_hub_download` into the local Hub cache,
    then copied into `output_dir` with a flat train/valid layout.
    Re-running updates/overwrites files with the same names.

    Args:
        output_dir: Root folder for the resulting dataset.
        classes: Classes to download. None means all available classes.
        max_per_class: Max images per class in each split.
            None downloads every file for the class.
        splits: Which splits to download (`train`, `valid`, or both).
        image_set: Subdirectory inside the HF repository.
        repo_id: Hugging Face dataset ID.

    Returns:
        Manifest dict with `classes`, `files`, `output_dir`, and related fields.
        Also written to `output_dir/download_manifest.json`.

    Raises:
        ValueError: If an unknown class or unsupported split is requested.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # One repository file-list request for the entire download.
    repo_files = list_repo_files(repo_id, repo_type="dataset")
    available_classes = _available_classes_from_repo_files(repo_files, image_set=image_set)

    if classes:
        requested = list(classes)
        missing = sorted(set(requested) - set(available_classes))
        if missing:
            raise ValueError(
                "Unknown classes: "
                + ", ".join(missing)
                + ". Use --list-classes to see available class names."
            )
        selected_classes = requested
    else:
        selected_classes = available_classes

    summary = {
        "repo_id": repo_id,
        "image_set": image_set,
        "output_dir": str(output_path.resolve()),
        "classes": selected_classes,
        "splits": list(splits),
        "max_per_class": max_per_class,
        "files": {},
    }

    for class_name in selected_classes:
        summary["files"][class_name] = {}
        for split in splits:
            if split not in SPLITS:
                raise ValueError(f"Unsupported split '{split}'. Use: {', '.join(SPLITS)}")

            files = _class_files(class_name, split, image_set, repo_files)
            if max_per_class is not None:
                # Take the first N files in lexicographic order.
                files = files[:max_per_class]

            split_dir = output_path / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0

            for remote_path in files:
                # hf_hub_download stores the file under ~/.cache/huggingface/hub/.
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                )
                destination = split_dir / Path(remote_path).name
                shutil.copy2(local_file, destination)
                downloaded += 1

            summary["files"][class_name][split] = downloaded
            logging.info("Downloaded %s/%s: %d files", split, class_name, downloaded)

    manifest_path = output_path / "download_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
