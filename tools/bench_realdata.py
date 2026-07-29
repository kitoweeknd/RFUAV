#!/usr/bin/env python
"""Benchmark every RFUAV functionality point on REAL raw IQ recordings.

This is a *standalone* script, deliberately not a pytest module: the real data
lives outside the repo (mounted read-only, e.g. ``-v D:/ML/tmp:/data:ro``) and
``testpaths=["tests"]`` only collects the 55 hermetic synthetic tests. Running
real-data benchmarks here keeps that suite fast and reproducible.

Each functionality point is timed with ``time.perf_counter`` (per-phase, the
same pattern as ``ChannelSimulation/viewer/signal_proc/_bench.py``), the current
RSS is sampled with ``psutil`` and the process-wide peak with
``resource.getrusage(RUSAGE_SELF).ru_maxrss``. Every point runs inside
try/except so a missing weight file, missing ffmpeg, or an unreachable model
download is recorded as ``skipped``/``error`` instead of aborting the run.

Bounds: real files here are ~800 MB float32 (100 M complex samples ≈ 1.05 s @
100 MHz). Every whole-file path is capped — either by ``--max-samples``, by a
bounded window, or by ``step`` / ``target_frame_gap`` — so nothing OOMs a
shared box.

Usage (from repo root; in the GPU container add ``-v D:/ML/tmp:/data:ro``)::

    python tools/bench_realdata.py --data-dir /data --files 1
    python tools/bench_realdata.py --data-dir /data --snr-full --do-detect --do-two-stage

Heavy / networked points are opt-in: ``--snr-full`` (whole-file complex128 SNR),
``--do-detect`` (COCO weights download), ``--do-two-stage`` (S1+S2 end-to-end).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
import traceback
from typing import Callable

import numpy as np

# Allow running directly as `python tools/bench_realdata.py` (add the repo root
# to the path so the sibling packages `graphic` / `utils` / `ChannelSimulation`
# are importable, not just via `python -m`). Mirrors tools/DroneMixing.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psutil
    _PROC = psutil.Process()
except Exception:  # psutil is a declared dep, but never let its absence abort
    psutil = None
    _PROC = None


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------
def _rss_mb() -> float:
    """Current resident set size of this process, in MB (0.0 if unavailable)."""
    if _PROC is None:
        return 0.0
    return _PROC.memory_info().rss / (1024 * 1024)


def _peak_rss_mb() -> float:
    """Process-wide peak RSS (high-water mark) in MB.

    ``ru_maxrss`` is KiB on Linux (the container) and bytes on macOS.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return peak * scale / (1024 * 1024)


class Bench:
    """Runs and records timed functionality points."""

    def __init__(self):
        self.results: list[dict] = []

    def run(self, name: str, fn: Callable[[], dict], note: str = ""):
        """Run ``fn`` under timing/memory instrumentation, catching failures.

        ``fn`` returns a dict of point-specific metrics (throughput, frame count,
        output sizes, ...). Missing prerequisites should raise ``_Skip``.
        """
        gc.collect()
        rss_before = _rss_mb()
        print(f"\n=== {name} ===", flush=True)
        if note:
            print(f"    ({note})", flush=True)
        t0 = time.perf_counter()
        status, metrics, err = "ok", {}, None
        try:
            metrics = fn() or {}
        except _Skip as e:
            status, err = "skipped", str(e)
            print(f"    SKIPPED: {err}", flush=True)
        except Exception as e:  # noqa: BLE001 — a bench point must not abort the run
            status, err = "error", f"{type(e).__name__}: {e}"
            print(f"    ERROR: {err}", flush=True)
            traceback.print_exc()
        dt = time.perf_counter() - t0
        rss_after = _rss_mb()
        record = {
            "point": name,
            "status": status,
            "seconds": round(dt, 4),
            "rss_before_mb": round(rss_before, 1),
            "rss_after_mb": round(rss_after, 1),
            "rss_delta_mb": round(rss_after - rss_before, 1),
            "peak_rss_mb": round(_peak_rss_mb(), 1),
        }
        if err:
            record["detail"] = err
        record.update({k: _jsonable(v) for k, v in metrics.items()})
        self.results.append(record)
        if status == "ok":
            print(f"    ok  {dt:.3f}s  rssΔ={record['rss_delta_mb']:+.1f}MB "
                  f"peak={record['peak_rss_mb']:.0f}MB "
                  + "  ".join(f"{k}={record[k]}" for k in metrics), flush=True)
        return record


class _Skip(Exception):
    """Raised inside a bench point to record it as 'skipped' (not an error)."""


def _jsonable(v):
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, float):
        return round(v, 4)
    return v


# ---------------------------------------------------------------------------
# functionality points
# ---------------------------------------------------------------------------
def point_io_read(datapack: str, fs: float, max_samples: int) -> dict:
    """1. IQ read throughput: bounded complex64 slice vs whole-file complex128.

    Exposes the upcast cost of ``read_interleaved_iq`` (interleaved float32 ->
    complex128, ~2x the bytes) against the streaming complex64 reader used by
    the graphic pipeline.
    """
    from graphic.io_utils import _complex_from_file, _count_complex_samples

    total = _count_complex_samples(datapack, np.float32)
    n = min(total, max_samples)

    t0 = time.perf_counter()
    win = _complex_from_file(datapack, 0, n, np.float32)
    t_c64 = time.perf_counter() - t0
    assert win.dtype == np.complex64, win.dtype

    metrics = {
        "total_complex": total,
        "read_samples": int(n),
        "complex64_dtype": str(win.dtype),
        "complex64_MSamples_per_s": (n / t_c64) / 1e6 if t_c64 else None,
        "complex64_seconds": round(t_c64, 4),
    }
    del win
    gc.collect()
    return metrics


def point_io_read_full_upcast(datapack: str, max_samples: int) -> dict:
    """1b. Whole-file complex128 read (the shared ChannelSimulation reader).

    Bounded by ``--max-samples`` via a manual bounded read that mirrors
    ``read_interleaved_iq``'s complex128 output, so we measure the upcast cost
    without pulling a full 1.6 GB array on an 800 MB file.
    """
    from graphic.io_utils import _count_complex_samples

    total = _count_complex_samples(datapack, np.float32)
    n = min(total, max_samples)
    t0 = time.perf_counter()
    raw = np.fromfile(datapack, dtype=np.float32, count=2 * n)
    m = raw.size // 2
    iq = raw[0:2 * m:2] + 1j * raw[1:2 * m:2]  # complex128, same as read_interleaved_iq
    t = time.perf_counter() - t0
    metrics = {
        "read_samples": int(m),
        "complex128_dtype": str(iq.dtype),
        "complex128_MSamples_per_s": (m / t) / 1e6 if t else None,
        "complex128_seconds": round(t, 4),
        "bytes_in_ram_MB": round(iq.nbytes / (1024 * 1024), 1),
    }
    del raw, iq
    gc.collect()
    return metrics


def point_stft(datapack: str, fs: float, stft_point: int, duration_time: float) -> dict:
    """2. Single-slice STFT (graphic/RawDataProcessor.STFT)."""
    from graphic.RawDataProcessor import STFT
    from graphic.io_utils import _complex_from_file

    slice_point = int(fs * duration_time)
    data = _complex_from_file(datapack, 0, slice_point, np.float32)
    t0 = time.perf_counter()
    f, t, Zxx = STFT(data, stft_point=stft_point, fs=fs,
                     duration_time=duration_time, onside=False)
    dt = time.perf_counter() - t0
    return {
        "slice_samples": slice_point,
        "Zxx_shape": list(Zxx.shape),
        "seconds": round(dt, 4),
        "finite": bool(np.isfinite(Zxx).all()),
    }


def point_generate_images(datapack: str, fs: float, stft_point: int,
                          duration_time: float, dpi: int, target_frames: int) -> dict:
    """3. generate_images(location='buffer') for a bounded number of frames.

    Uses ``step`` to cap the frame count while keeping each frame's work
    identical to production (same slice length / STFT / dpi).
    """
    from graphic.RawDataProcessor import generate_images
    from graphic.io_utils import _count_complex_samples

    slice_point = int(fs * duration_time)
    total = _count_complex_samples(datapack, np.float32)
    n_frames = max(1, total // slice_point)
    step = max(1, n_frames // target_frames)

    t0 = time.perf_counter()
    images = generate_images(datapack=datapack, fs=fs, stft_point=stft_point,
                             duration_time=duration_time, step=step,
                             location='buffer', dpi=dpi)
    dt = time.perf_counter() - t0

    n = len(images) if images else 0
    size = images[0].size if n else None  # (W, H) from PIL
    for im in images or []:
        im.close()
    return {
        "n_frames_total": int(n_frames),
        "step": int(step),
        "frames_rendered": n,
        "image_size_wh": list(size) if size else None,
        "dpi": dpi if dpi is not None else 300,
        "seconds": round(dt, 4),
        "frames_per_s": (n / dt) if dt and n else None,
    }


def point_save_as_video(datapack: str, fs: float, stft_point: int,
                        duration_time: float, dpi: int, fps: int) -> dict:
    """4. save_as_video -> mp4 (ffmpeg stdin pipe). Skips if ffmpeg is absent."""
    from graphic.RawDataProcessor import save_as_video
    from graphic.io_utils import _count_complex_samples

    if shutil.which("ffmpeg") is None:
        raise _Skip("ffmpeg not on PATH")

    slice_point = int(fs * duration_time)
    total = _count_complex_samples(datapack, np.float32)
    n_frames = max(1, total // slice_point)

    out_dir = tempfile.mkdtemp(prefix="bench_video_")
    try:
        t0 = time.perf_counter()
        save_as_video(datapack=datapack, save_path=out_dir, fs=fs,
                      stft_point=stft_point, duration_time=duration_time,
                      fps=fps, dpi=dpi)
        dt = time.perf_counter() - t0
        video = os.path.join(out_dir, "video.mp4")
        size_kb = os.path.getsize(video) / 1024 if os.path.exists(video) else 0
        return {
            "frames": int(n_frames),
            "seconds": round(dt, 4),
            "frames_per_s": (n_frames / dt) if dt else None,
            "video_kb": round(size_kb, 1),
            "dpi": dpi if dpi is not None else 100,
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def point_waterfall(datapack: str, fs: float) -> dict:
    """5. waterfall_spectrogram_optimized (disk mode, large frame gap)."""
    from graphic.RawDataProcessor import waterfall_spectrogram_optimized

    out_dir = tempfile.mkdtemp(prefix="bench_waterfall_")
    try:
        t0 = time.perf_counter()
        waterfall_spectrogram_optimized(
            datapack=datapack, fft_size=256, fs=int(fs), location=out_dir,
            time_scale=39062, target_frame_gap=100_000, chunk_size=50_000_000)
        dt = time.perf_counter() - t0
        # The disk branch saves JPEGs (RGBA PNG buffer -> convert('RGB') -> JPEG);
        # count that extension, not .png, so the frame count is truthful.
        n_img = len(glob.glob(os.path.join(out_dir, "*_waterfall_spectrogram.jpg")))
        return {
            "seconds": round(dt, 4),
            "frames_written": n_img,
            "frames_per_s": (n_img / dt) if dt and n_img else None,
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def point_snr_window(datapack: str, fs: float, bw: float, max_samples: int) -> dict:
    """6. SNR on a bounded complex64 window (default; cheap, no upcast)."""
    from graphic.io_utils import _complex_from_file, _count_complex_samples
    from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator

    total = _count_complex_samples(datapack, np.float32)
    n = min(total, max_samples)
    win = _complex_from_file(datapack, 0, n, np.float32)
    t0 = time.perf_counter()
    res = SNREstimator(fs=fs, bw=bw).estimate(win)
    dt = time.perf_counter() - t0
    return {
        "window_samples": int(n),
        "seconds": round(dt, 4),
        "snr_db": round(float(res.snr_db), 3),
        "MSamples_per_s": (n / dt) / 1e6 if dt else None,
    }


def point_snr_full(datapack: str, fs: float, bw: float, max_samples: int) -> dict:
    """6b. Whole-file SNR (--snr-full) via the optimized ``estimate_file_snr``.

    Times two modes back to back so the optimization is self-evident in the JSON:

      * whole-file (``max_samples=None``): the in-place ``complex64`` read — same
        whole-file semantics as before, but without the ~4 GB complex128 read
        transient the old ``a + 1j*b`` path incurred.
      * bounded (``max_samples=N``): a leading-N-sample prefix — the sub-second,
        sub-GB fast path. ``snr_db`` should track the whole-file estimate for a
        stationary recording, so ``snr_db_delta`` being small is the evidence the
        bound speeds things up *without* distorting the result.
    """
    from SNREstimation.SNREstimation import estimate_file_snr

    t0 = time.perf_counter()
    res_full = estimate_file_snr(datapack, fs=fs, bw=bw)
    dt_full = time.perf_counter() - t0

    t1 = time.perf_counter()
    res_cap = estimate_file_snr(datapack, fs=fs, bw=bw, max_samples=max_samples)
    dt_cap = time.perf_counter() - t1

    return {
        "whole_file_seconds": round(dt_full, 4),
        "whole_file_snr_db": round(float(res_full.snr_db), 3),
        "bounded_samples": int(max_samples),
        "bounded_seconds": round(dt_cap, 4),
        "bounded_snr_db": round(float(res_cap.snr_db), 3),
        "snr_db_delta": round(abs(float(res_full.snr_db) - float(res_cap.snr_db)), 3),
        "speedup": round(dt_full / dt_cap, 1) if dt_cap else None,
    }


def point_channelsim(datapack: str, fs: float, chunk_samples: int) -> dict:
    """7. ChannelSimulation on one bounded chunk (original + awgn + fading)."""
    from ChannelSimulation import (default_stage_configs, iter_iq_chunks,
                                    run_channel_pipeline)

    chunk = next(iter_iq_chunks(datapack, chunk_size=chunk_samples,
                                dtype=np.float32))
    cfg = default_stage_configs()
    out_dir = tempfile.mkdtemp(prefix="bench_chsim_")
    try:
        t0 = time.perf_counter()
        out = run_channel_pipeline(
            chunk, fs=fs, out_dir=out_dir, stage_configs=cfg,
            enabled_stages=["original", "awgn", "fading"], seed=42)
        dt = time.perf_counter() - t0
        return {
            "chunk_samples": int(len(chunk)),
            "chunk_dtype": str(chunk.dtype),
            "out_samples": int(len(out)),
            "seconds": round(dt, 4),
            "MSamples_per_s": (len(chunk) / dt) / 1e6 if dt else None,
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def _make_tiny_classifier(workdir: str):
    """Build a random-init resnet18 (.pth) + minimal cfg + tiny ImageFolder.

    Mirrors tests/conftest.py::tiny_classifier without pytest, so classification
    can be timed end-to-end with no Hugging Face / network dependency.
    Returns (cfg_path, weight_path).
    """
    import torch
    import yaml
    from PIL import Image
    from utils.trainer import model_init_

    class_names = ["DroneA", "DroneB", "DroneC", "DroneD", "DroneE"]
    num_classes = len(class_names)
    image_size = 64

    # tiny ImageFolder (5 classes x 1 png) so cfg's train/val paths resolve
    data_root = os.path.join(workdir, "imgfolder")
    rng = np.random.default_rng(1)
    for ci, name in enumerate(class_names):
        d = os.path.join(data_root, name)
        os.makedirs(d, exist_ok=True)
        arr = np.clip(np.full((64, 64, 3), 30 * ci, dtype=int)
                      + rng.integers(0, 60, size=(64, 64, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(d, f"{name}.png"))

    model = model_init_("resnet18", num_class=num_classes, pretrained=False,
                        image_size=image_size)
    weight_path = os.path.join(workdir, "tiny_resnet18.pth")
    torch.save(model.state_dict(), weight_path)

    save_path = os.path.join(workdir, "runs")
    os.makedirs(save_path, exist_ok=True)
    cfg = {
        "batch_size": 8,
        "class_names": {n: i for i, n in enumerate(class_names)},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "image_size": image_size,
        "lr": 1e-4,
        "model": "resnet18",
        "num_classes": num_classes,
        "num_epochs": 1,
        "optimizer": "Adam",
        "pretrained": False,
        "save_path": save_path + os.sep,
        "shuffle": False,
        "train": data_root,
        "val": data_root,
        "weights": "",
        "num_workers": 0,
    }
    cfg_path = os.path.join(workdir, "tiny_resnet18.yaml")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return cfg_path, weight_path


def point_classify(datapack: str) -> dict:
    """8. Classification end-to-end: IQ -> generate_images -> batched inference.

    ClassifyModel.inference on a raw .iq file routes through RawdataProcess ->
    generate_images(source) at the DEFAULT duration_time (0.1 s), i.e. ~10 frames
    for an 800 MB file — bounded. Uses a random-init resnet18 (no network).
    """
    from utils.benchmark import ClassifyModel

    workdir = tempfile.mkdtemp(prefix="bench_clf_")
    save_path = os.path.join(workdir, "res")
    try:
        cfg_path, weight_path = _make_tiny_classifier(workdir)
        model = ClassifyModel(cfg=cfg_path, weight_path=weight_path, save=True)
        t0 = time.perf_counter()
        model.inference(source=datapack, save_path=save_path)
        dt = time.perf_counter() - t0
        outputs = glob.glob(os.path.join(save_path, "*"))
        return {
            "seconds": round(dt, 4),
            "device": model.cfg.get("device", "?") if hasattr(model, "cfg") else "?",
            "outputs": len(outputs),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _bgr_frames_from_iq(datapack: str, fs: float, stft_point: int,
                        duration_time: float, target_frames: int):
    """Derive a bounded list of BGR frames (np.uint8 HxWx3) from real IQ."""
    import cv2
    from graphic.RawDataProcessor import generate_images
    from graphic.io_utils import _count_complex_samples

    slice_point = int(fs * duration_time)
    total = _count_complex_samples(datapack, np.float32)
    n_frames = max(1, total // slice_point)
    step = max(1, n_frames // target_frames)
    images = generate_images(datapack=datapack, fs=fs, stft_point=stft_point,
                             duration_time=duration_time, step=step,
                             location='buffer')
    frames = [cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
              for im in (images or [])]
    for im in images or []:
        im.close()
    return frames


def point_detect(datapack: str, fs: float, stft_point: int,
                 duration_time: float, detector: str = "rtdetr") -> dict:
    """9. Detection end-to-end (--do-detect): IQ -> frames -> detect_batch.

    Backend selected by ``--detector`` (default ``rtdetr``). On this box the
    fasterrcnn COCO weights fail their hash check via the proxy, so rtdetr is
    the default; whichever backend can't fetch its weights offline is skipped.
    """
    try:
        from utils.benchmark import DetectionModel
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"DetectionModel unavailable: {e}")

    frames = _bgr_frames_from_iq(datapack, fs, stft_point, duration_time,
                                 target_frames=4)
    if not frames:
        raise _Skip("no frames derived from IQ")

    try:
        det = DetectionModel(model_name=detector)
    except Exception as e:  # noqa: BLE001 — offline / weights download failed
        raise _Skip(f"detector build failed (offline?): {e}")

    t0 = time.perf_counter()
    annotated = det.detect_batch(frames)
    dt = time.perf_counter() - t0
    return {
        "frames": len(frames),
        "seconds": round(dt, 4),
        "frames_per_s": (len(frames) / dt) if dt else None,
        "annotated": len(annotated) if annotated is not None else 0,
    }


def point_two_stage(datapack: str, fs: float, stft_point: int,
                    duration_time: float, detector: str = "rtdetr") -> dict:
    """10. Two-stage end-to-end (--do-two-stage), BOUNDED.

    Never runs the default whole-file TwoStagesDetector (which decodes every
    frame into a list -> tens of GB). Instead derives a few BGR frames, runs S1
    (``--detector`` backend detect_batch) then S2 (the random-init resnet18
    classifier), measuring the combined per-frame cost without the OOM path.
    """
    try:
        from utils.benchmark import ClassifyModel, DetectionModel
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"models unavailable: {e}")

    frames = _bgr_frames_from_iq(datapack, fs, stft_point, duration_time,
                                 target_frames=3)
    if not frames:
        raise _Skip("no frames derived from IQ")

    try:
        det = DetectionModel(model_name=detector)
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"S1 detector build failed (offline?): {e}")

    import cv2
    import torch

    workdir = tempfile.mkdtemp(prefix="bench_two_")
    try:
        cfg_path, weight_path = _make_tiny_classifier(workdir)
        clf = ClassifyModel(cfg=cfg_path, weight_path=weight_path, save=False)

        t0 = time.perf_counter()
        # Stage 1: detect on all frames.
        _ = det.detect_batch(frames)
        # Stage 2: classify each frame (mirrors the per-crop classify step).
        with torch.no_grad():
            for fr in frames:
                rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                from PIL import Image as _Img
                tensor = clf._transform(_Img.fromarray(rgb)).unsqueeze(0).to(clf.device)
                with clf._autocast():
                    clf.model(tensor)
        dt = time.perf_counter() - t0
        return {
            "frames": len(frames),
            "seconds": round(dt, 4),
            "frames_per_s": (len(frames) / dt) if dt else None,
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Benchmark RFUAV functionality points on real raw IQ data.")
    ap.add_argument("--data-dir", required=True,
                    help="Directory of real .iq recordings (mount read-only).")
    ap.add_argument("--glob", default="*.iq", help="Glob for IQ files (default *.iq).")
    ap.add_argument("--files", type=int, default=1,
                    help="How many files to benchmark (default 1).")
    ap.add_argument("--fs", type=float, default=100e6, help="Sample rate Hz.")
    ap.add_argument("--bw", type=float, default=20e6, help="Signal bandwidth Hz (SNR).")
    ap.add_argument("--stft-point", type=int, default=1024, help="STFT window size.")
    ap.add_argument("--duration-time", type=float, default=0.1,
                    help="Seconds per spectrogram frame (default 0.1).")
    ap.add_argument("--max-samples", type=int, default=5_000_000,
                    help="Upper bound on complex samples for whole-file paths.")
    ap.add_argument("--chunk-samples", type=int, default=10_000_000,
                    help="Complex samples per ChannelSimulation chunk.")
    ap.add_argument("--dpi", type=int, default=None,
                    help="dpi for generate_images/save_as_video (None keeps defaults).")
    ap.add_argument("--fps", type=int, default=5, help="Video frame rate.")
    ap.add_argument("--target-frames", type=int, default=3,
                    help="Approx frame count for bounded image/detect points.")
    ap.add_argument("--snr-full", action="store_true",
                    help="Also run whole-file SNR (complex128 upcast path).")
    ap.add_argument("--do-detect", action="store_true",
                    help="Run detection end-to-end (may download COCO weights).")
    ap.add_argument("--detector", default="rtdetr",
                    choices=["rtdetr", "fasterrcnn", "yolov5"],
                    help="Backend for detect / two-stage points (default rtdetr; "
                         "fasterrcnn COCO weights fail the hash check via proxy).")
    ap.add_argument("--do-two-stage", action="store_true",
                    help="Run bounded two-stage end-to-end.")
    ap.add_argument("--out", default="bench_realdata_results.json",
                    help="Where to write the JSON report.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, args.glob)))
    if not files:
        print(f"No files matching {args.glob!r} in {args.data_dir}", file=sys.stderr)
        sys.exit(2)
    files = files[:args.files]

    print(f"Platform: {platform.platform()}")
    print(f"Files ({len(files)}): " + ", ".join(os.path.basename(f) for f in files))
    print(f"fs={args.fs:g} bw={args.bw:g} stft_point={args.stft_point} "
          f"duration_time={args.duration_time} max_samples={args.max_samples:,} "
          f"dpi={args.dpi}")

    bench = Bench()
    report = {
        "platform": platform.platform(),
        "args": vars(args),
        "files": [os.path.basename(f) for f in files],
        "points": bench.results,
    }

    for path in files:
        base = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\n########## {base}  ({size_mb:.0f} MB) ##########", flush=True)

        def tag(name):  # prefix each point with the file it ran on
            return f"{base} :: {name}"

        bench.run(tag("1. IQ read (complex64 window)"),
                  lambda: point_io_read(path, args.fs, args.max_samples))
        bench.run(tag("1b. IQ read (complex128 whole-file upcast)"),
                  lambda: point_io_read_full_upcast(path, args.max_samples))
        bench.run(tag("2. STFT (single slice)"),
                  lambda: point_stft(path, args.fs, args.stft_point, args.duration_time))
        bench.run(tag("3. generate_images (buffer)"),
                  lambda: point_generate_images(path, args.fs, args.stft_point,
                                                args.duration_time, args.dpi,
                                                args.target_frames))
        bench.run(tag("4. save_as_video (mp4)"),
                  lambda: point_save_as_video(path, args.fs, args.stft_point,
                                              args.duration_time, args.dpi, args.fps))
        bench.run(tag("5. waterfall_spectrogram_optimized"),
                  lambda: point_waterfall(path, args.fs))
        bench.run(tag("6. SNR (bounded complex64 window)"),
                  lambda: point_snr_window(path, args.fs, args.bw, args.max_samples))
        if args.snr_full:
            bench.run(tag("6b. SNR (whole-file complex64 vs bounded)"),
                      lambda: point_snr_full(path, args.fs, args.bw,
                                             args.max_samples))
        bench.run(tag("7. ChannelSimulation (bounded chunk)"),
                  lambda: point_channelsim(path, args.fs, args.chunk_samples))
        bench.run(tag("8. Classify end-to-end (IQ->frames->infer)"),
                  lambda: point_classify(path))
        if args.do_detect:
            bench.run(tag(f"9. Detect end-to-end ({args.detector})"),
                      lambda: point_detect(path, args.fs, args.stft_point,
                                           args.duration_time, args.detector))
        if args.do_two_stage:
            bench.run(tag(f"10. Two-stage end-to-end (bounded, {args.detector})"),
                      lambda: point_two_stage(path, args.fs, args.stft_point,
                                              args.duration_time, args.detector))

    report["peak_rss_mb_final"] = round(_peak_rss_mb(), 1)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # summary table
    print("\n\n==================== SUMMARY ====================")
    ok = sum(1 for r in bench.results if r["status"] == "ok")
    sk = sum(1 for r in bench.results if r["status"] == "skipped")
    er = sum(1 for r in bench.results if r["status"] == "error")
    for r in bench.results:
        mark = {"ok": "  ", "skipped": "SK", "error": "!!"}[r["status"]]
        print(f"[{mark}] {r['seconds']:8.3f}s  {r['point']}")
    print(f"\n{ok} ok, {sk} skipped, {er} error   peak RSS "
          f"{report['peak_rss_mb_final']:.0f} MB")
    print(f"Report written to {args.out}")
    sys.exit(1 if er else 0)


if __name__ == "__main__":
    main()
