"""Shared fixtures for the RFUAV functional smoke suite.

Everything here is self-contained: synthetic IQ (interleaved float32), a
synthetic ImageFolder, and a *tiny* trained-from-scratch classifier `.pth`
so that inference / benchmark / metrics can run without any Hugging Face or
network dependency. Tests that genuinely need weights or the network are
marked (`net`) and skip cleanly with a printed reason.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force a non-interactive matplotlib backend before anything imports pyplot
# (PySide6 is installed, so matplotlib could otherwise pick a Qt backend and
# try to open a display).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

# --- make the repo root importable regardless of pytest import mode ----------
# tests/ has no __init__.py, so with the default "prepend" mode pytest would put
# tests/ (not the repo root) on sys.path. Flat modules like `graphic`, `utils`,
# `ChannelSimulation` live at the repo root, so insert it explicitly.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# device / GPU
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def require_gpu():
    """Skip a test unless a CUDA GPU is present; also expose its capability."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU available")
    return torch.cuda.get_device_capability()


# ---------------------------------------------------------------------------
# synthetic IQ  (interleaved float32 I,Q,I,Q ... == complex64 on disk)
# ---------------------------------------------------------------------------
def _make_complex_signal(n: int = 65536, fs: float = 100e6, tone_hz: float = 5e6) -> np.ndarray:
    """A narrowband complex tone + a little noise, unit-ish power, complex64."""
    t = np.arange(n, dtype=np.float64) / fs
    sig = np.exp(2j * np.pi * tone_hz * t)
    rng = np.random.default_rng(0)
    sig = sig + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig.astype(np.complex64)


@pytest.fixture(scope="session")
def iq_array() -> np.ndarray:
    return _make_complex_signal()


@pytest.fixture(scope="session")
def iq_file(tmp_path_factory) -> Path:
    """Write a synthetic complex signal as interleaved float32 to <tmp>/synth.iq."""
    sig = _make_complex_signal()
    out = tmp_path_factory.mktemp("iq") / "synth.iq"
    interleaved = np.empty(sig.size * 2, dtype=np.float32)
    interleaved[0::2] = sig.real
    interleaved[1::2] = sig.imag
    interleaved.tofile(out)
    return out


# ---------------------------------------------------------------------------
# synthetic ImageFolder  (5 classes so benchmark's topk=(1,2,3) is valid)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["DroneA", "DroneB", "DroneC", "DroneD", "DroneE"]


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory) -> Path:
    """<root>/<class>/*.png  — a few 64x64 RGB PNGs per class."""
    from PIL import Image

    root = tmp_path_factory.mktemp("imgfolder")
    rng = np.random.default_rng(1)
    for ci, name in enumerate(CLASS_NAMES):
        d = root / name
        d.mkdir()
        for k in range(6):
            # give each class a distinct colour bias so a tiny net can fit it
            base = np.full((64, 64, 3), 30 * ci, dtype=np.uint8)
            noise = rng.integers(0, 60, size=(64, 64, 3), dtype=np.uint8)
            arr = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(d / f"{name}_{k}.png")
    return root


# ---------------------------------------------------------------------------
# tiny classifier: build resnet18 (5 classes, 64px), save a plain state_dict
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_classifier(tmp_path_factory, image_folder):
    """Return a dict describing a ready-to-load classifier:

    ``{"weights": <.pth>, "cfg": <.yaml>, "num_classes": 5,
       "class_names": {..}, "image_size": 64, "data_root": <ImageFolder>}``

    The .pth is a plain ``state_dict`` (what ``ClassifyModel`` expects). We do
    NOT train — random init is fine for a forward/inference smoke test and keeps
    the fixture fast and CPU-friendly.
    """
    import torch
    import yaml

    from utils.trainer import model_init_

    num_classes = len(CLASS_NAMES)
    class_names = {name: i for i, name in enumerate(CLASS_NAMES)}
    image_size = 64

    model = model_init_("resnet18", num_class=num_classes, pretrained=False,
                        image_size=image_size)
    workdir = tmp_path_factory.mktemp("clf")
    weights = workdir / "tiny_resnet18.pth"
    torch.save(model.state_dict(), weights)

    save_path = workdir / "runs"
    save_path.mkdir()

    cfg = {
        "batch_size": 4,
        "class_names": class_names,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_size": image_size,
        "lr": 1e-4,
        "model": "resnet18",
        "num_classes": num_classes,
        "num_epochs": 1,
        "optimizer": "Adam",
        # keep training offline: model_init_ pretrained weights are an ImageNet
        # download; ClassifyModel ignores this key (derives it from weight file).
        "pretrained": False,
        "save_path": str(save_path) + os.sep,
        "shuffle": False,
        "train": str(image_folder),
        "val": str(image_folder),
        "weights": "",
        "num_workers": 0,
    }
    cfg_path = workdir / "tiny_resnet18.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    return {
        "weights": weights,
        "cfg": cfg_path,
        "cfg_dict": cfg,
        "num_classes": num_classes,
        "class_names": class_names,
        "image_size": image_size,
        "data_root": image_folder,
    }


@pytest.fixture(scope="session")
def benchmark_tree(tmp_path_factory, image_folder):
    """A benchmark()-shaped tree: <root>/<snr>/<CM>/<class>/*.png

    benchmark() walks data_path/<snr>/<CM>/ and treats the next level as an
    ImageFolder, so we nest the synthetic ImageFolder two levels down. Class
    names match the classifier's training classes.
    """
    import shutil

    root = tmp_path_factory.mktemp("bench")
    leaf = root / "snr_10dB" / "CM_1"
    leaf.mkdir(parents=True)
    for cls_dir in sorted(Path(image_folder).iterdir()):
        if cls_dir.is_dir():
            shutil.copytree(cls_dir, leaf / cls_dir.name)
    return root


# ---------------------------------------------------------------------------
# ffmpeg availability (for video-producing paths)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    import shutil

    return shutil.which("ffmpeg") is not None
