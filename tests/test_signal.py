"""Signal / channel-simulation capability smoke tests.

Covers: STFT + generate_images (buffer mode), io_utils IQ readers, all 8
channel `apply_*` functions (complex64 invariant), the in-memory channel
pipeline, SNR estimation, and the preprocessor split/augment helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from graphic.RawDataProcessor import STFT, generate_images
from graphic.io_utils import _complex_from_file, _count_complex_samples

# Small framing so the 65536-sample synthetic file yields several frames.
FS = 1e6
DUR = 0.02          # -> slice_point = 20000 samples/frame
STFT_POINT = 256


# ---------------------------------------------------------------------------
# graphic: STFT + spectrogram generation + IQ readers
# ---------------------------------------------------------------------------
def test_stft_shapes(iq_array):
    data = iq_array[: int(FS * DUR)]
    f, t, Zxx = STFT(data, onside=False, stft_point=STFT_POINT, fs=FS, duration_time=DUR)
    assert Zxx.ndim == 2
    assert Zxx.shape[0] == STFT_POINT
    assert np.isfinite(np.abs(Zxx)).all()


def test_io_utils_roundtrip(iq_file, iq_array):
    n = _count_complex_samples(str(iq_file))
    assert n == iq_array.size
    chunk = _complex_from_file(str(iq_file), 0, 1024)
    assert chunk.dtype == np.complex64
    assert chunk.size == 1024
    np.testing.assert_allclose(chunk, iq_array[:1024], rtol=1e-4, atol=1e-4)


def test_generate_images_buffer(iq_file):
    from PIL import Image

    images = generate_images(datapack=str(iq_file), fs=FS, stft_point=STFT_POINT,
                             duration_time=DUR, location="buffer")
    assert images, "no spectrogram frames produced"
    assert all(isinstance(im, Image.Image) for im in images)


# ---------------------------------------------------------------------------
# ChannelSimulation: each apply_* preserves the complex64 invariant
# ---------------------------------------------------------------------------
def _x(iq_array):
    return iq_array[:8192].copy()


def test_apply_awgn(iq_array):
    from ChannelSimulation.channels.awgn import apply_awgn

    y = apply_awgn(_x(iq_array), snr_db=10.0, rng=np.random.default_rng(0))
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_cfo(iq_array):
    from ChannelSimulation.channels.cfo import apply_cfo

    y = apply_cfo(_x(iq_array), fs=FS, cfo_hz=1e3)
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_doppler(iq_array):
    from ChannelSimulation.channels.doppler import apply_doppler

    y = apply_doppler(_x(iq_array), fs=FS, doppler_shift_hz=0.0,
                      doppler_spread_hz=50.0, rng=np.random.default_rng(0))
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_fading(iq_array):
    from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading

    for kind in ("rayleigh", "rician"):
        y = apply_rayleigh_or_rician_fading(_x(iq_array), fading_type=kind,
                                            rng=np.random.default_rng(0))
        assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_iq_imbalance(iq_array):
    from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance

    y = apply_iq_imbalance(_x(iq_array), gain_imbalance_db=1.0, phase_imbalance_deg=3.0)
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_multipath(iq_array):
    from ChannelSimulation.channels.multipath import apply_multipath

    y = apply_multipath(_x(iq_array), rng=np.random.default_rng(0))
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_phase_noise(iq_array):
    from ChannelSimulation.channels.phase_noise import apply_phase_noise

    y = apply_phase_noise(_x(iq_array), rng=np.random.default_rng(0))
    assert y.dtype == np.complex64 and y.shape == (8192,)


def test_apply_crop_shift(iq_array):
    from ChannelSimulation.channels.crop_shift import apply_random_crop_and_time_shift

    y = apply_random_crop_and_time_shift(_x(iq_array), rng=np.random.default_rng(0))
    # crop changes length; only the dtype invariant must hold.
    assert y.dtype == np.complex64
    assert y.size <= 8192


# ---------------------------------------------------------------------------
# in-memory channel pipeline (subset of stages, fast)
# ---------------------------------------------------------------------------
def test_run_channel_pipeline(iq_array, tmp_path):
    from ChannelSimulation.channel_pipeline import (
        default_stage_configs,
        run_channel_pipeline,
    )

    out = run_channel_pipeline(
        iq_array[:16384].copy(), fs=FS, out_dir=tmp_path,
        stage_configs=default_stage_configs(),
        enabled_stages=["original", "awgn"], seed=42,
    )
    assert out.dtype == np.complex64
    # the "original" stage always renders a time/frequency plot
    assert list(Path(tmp_path).glob("*.png")), "pipeline wrote no plots"


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------
def test_snr_estimator(iq_array):
    from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator

    res = SNREstimator(fs=100e6, bw=20e6).estimate(iq_array)
    assert isinstance(res.snr_db, float)
    assert np.isfinite(res.snr_db)
    assert res.f1 <= res.f2


# ---------------------------------------------------------------------------
# preprocessor: split_images (+ best-effort data_augmentation)
# ---------------------------------------------------------------------------
def test_split_images(image_folder, tmp_path):
    from utils.preprocessor import split_images

    out = tmp_path / "split"
    split_images(str(image_folder), str(out), train_ratio=0.7)
    assert (out / "train").is_dir() and (out / "valid").is_dir()
    # each class present under train
    train_classes = {p.name for p in (out / "train").iterdir() if p.is_dir()}
    assert train_classes, "no class dirs under train/"


def test_data_augmentation(tmp_path):
    """dataset/{train,valid}/<class>/*.png layout -> augmented copy.

    ``data_augmentation`` treats the entries directly under ``train/`` and
    ``valid/`` as class directories, so ``dataset_path`` is the dataset root.
    """
    from PIL import Image
    import albumentations as A

    from utils.preprocessor import data_augmentation

    ds = tmp_path / "dataset"
    classes = ("DroneA", "DroneB")
    for split in ("train", "valid"):
        for cls in classes:
            imgs = ds / split / cls
            imgs.mkdir(parents=True)
            for k in range(3):
                arr = np.random.default_rng(k).integers(0, 255, (32, 32, 3)).astype(np.uint8)
                Image.fromarray(arr).save(imgs / f"img_{k}.png")

    out = tmp_path / "dataset_aug"
    data_augmentation(dataset_path=str(ds),
                      output_path=str(out),
                      methods=[A.HorizontalFlip(p=1.0)])

    # Every class dir is reproduced under train/ and valid/ with output images.
    for split in ("train", "valid"):
        for cls in classes:
            produced = list((out / split / cls).glob("*.png"))
            assert produced, f"no augmented images written for {split}/{cls}"


def test_data_augmentation_default_methods(tmp_path):
    """The documented ``methods=None`` default pipeline runs under the pinned
    Albumentations version (AdvancedBlur/CLAHE/ColorJitter/GaussNoise/ISONoise/
    Sharpen) without raising."""
    from PIL import Image

    from utils.preprocessor import data_augmentation

    ds = tmp_path / "dataset"
    for split in ("train", "valid"):
        imgs = ds / split / "DroneA"
        imgs.mkdir(parents=True)
        for k in range(2):
            arr = np.random.default_rng(k).integers(0, 255, (48, 48, 3)).astype(np.uint8)
            Image.fromarray(arr).save(imgs / f"img_{k}.png")

    out = tmp_path / "dataset_aug"
    data_augmentation(dataset_path=str(ds), output_path=str(out), methods=None)
    assert list((out / "train" / "DroneA").glob("*.png"))

