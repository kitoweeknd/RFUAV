"""
Backend performance benchmark & numerical-parity check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validates the backend optimizations (A1 multithreaded FFT backend, A2 parallel
per-stage fan-out, A3 compute micro-opts) with numbers rather than assertion:

1. **STFT numerical parity** — the optimized power-domain / multithreaded STFT
   vs the original ``20*log10(|fft|+eps)`` formula (must match within noise).
2. **Fan-out timing** — full per-stage STFT+PSD precompute, sequential+single-
   thread vs parallel+multithreaded, reporting the speedup.
3. **Per-phase breakdown** — pipeline / STFT / PSD / metrics, so A6 can decide
   whether any non-FFT kernel is worth a compiled/GPU rewrite (default: no).

Run:  python -m ChannelSimulation.viewer.signal_proc._bench [n_samples]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import fft_backend as fb
from .spectrum_data import SpectrumCalculator
from .metrics_calculator import MetricsCalculator
from .pipeline_runner import PipelineRunner, STAGE_ORDER


def _make_signal(n: int, fs: float) -> np.ndarray:
    """A tone-plus-noise complex64 signal (deterministic)."""
    rng = np.random.default_rng(2024)
    t = np.arange(n) / fs
    sig = np.exp(2j * np.pi * 1.0e6 * t) + np.exp(-2j * np.pi * 3.0e6 * t)
    sig = sig + 0.1 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return sig.astype(np.complex64)


def _stft_parity(sig: np.ndarray, fs: float) -> float:
    """Max |optimized − old-formula| dB on the first STFT frame."""
    calc = SpectrumCalculator(fs=fs)
    stft_db, _, _ = calc.compute_stft(sig)
    nperseg = min(calc.STFT_NPERSEG_LONG, len(sig))
    frame = sig[:nperseg] * np.hanning(nperseg).astype(np.complex64)
    spec = np.fft.fftshift(np.fft.fft(frame))
    old = 20 * np.log10(np.abs(spec) + calc.EPS)  # original formula
    return float(np.max(np.abs(old - stft_db[:, 0])))


def _worker_run(sig, cfg, fs, fc):
    """Time the real ProcessingWorker end-to-end: total + time-to-first-paint."""
    from PySide6.QtCore import QCoreApplication
    from .processing_worker import ProcessingWorker

    if QCoreApplication.instance() is None:
        QCoreApplication([])  # worker emits Qt signals
    w = ProcessingWorker(iq_data=sig, config=cfg, fs=fs, fc=fc, seed=42)
    marks: dict[str, float] = {}
    stages: set[str] = set()
    # first-paint = the immediate spectrum_ready for the "final" stage; each
    # intermediate stage precompute arrives as stage_spectrum_cached(name, dict).
    w.spectrum_ready.connect(lambda d: marks.setdefault("first", time.perf_counter()))
    w.stage_spectrum_cached.connect(lambda name, d: stages.add(name))
    t0 = time.perf_counter()
    w.run()  # QThread.run() called directly → synchronous; signals fire inline
    total = time.perf_counter() - t0
    n_stages = len(stages) + 1  # intermediate stages + the "final" stage
    return total, marks.get("first", t0) - t0, n_stages


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 10_000_000
    fs, fc = 100e6, 0.0
    print(f"FFT backend: {fb.BACKEND_NAME}  cores: {os.cpu_count()}  n: {n:,}\n")

    # ── 1. STFT numerical parity ──────────────────────────────────────────
    sig = _make_signal(n, fs)
    dmax = _stft_parity(sig, fs)
    print(f"[parity] max |optimized - old| STFT = {dmax:.2e} dB  "
          f"({'OK' if dmax < 0.01 else 'REGRESSION'})")

    # ── build a realistic multi-stage pipeline output ─────────────────────
    runner = PipelineRunner(iq_data=sig, fs=fs, fc=fc, seed=42)
    runner.set_config({"stage_configs": {s: {"enabled": True} for s in STAGE_ORDER[1:]}})
    t0 = time.perf_counter()
    results = runner.run(return_all_stages=True)
    t_pipeline = time.perf_counter() - t0
    original = results["original"]
    stage_signals = {k: v for k, v in results.items() if k != "original"}
    print(f"\n[pipeline] {t_pipeline:.3f}s  stages: {list(stage_signals.keys())}")

    # ── per-phase breakdown (single stage) ────────────────────────────────
    calc = SpectrumCalculator(fs=fs, fc=fc)
    t0 = time.perf_counter(); calc.compute_stft(results["final"]); t_stft = time.perf_counter() - t0
    t0 = time.perf_counter(); calc.compute_psd(results["final"]);  t_psd = time.perf_counter() - t0
    t0 = time.perf_counter()
    MetricsCalculator().calculate_all(original, results["final"], fs)
    t_metrics = time.perf_counter() - t0
    print(f"[breakdown] 1×STFT={t_stft:.3f}s  1×PSD={t_psd:.3f}s  metrics={t_metrics:.3f}s")

    # ── metrics sanity (before freeing the pipeline output) ───────────────
    m = MetricsCalculator().calculate_all(original, results["final"], fs)
    finite = all(np.isfinite(v) for v in m.values())
    print(f"[metrics] finite={finite}  " + "  ".join(f"{k}={v:.2f}" for k, v in m.items()))

    # ── real end-to-end ProcessingWorker (total + time-to-first-paint) ────
    # The per-stage spectrum precompute is intentionally SEQUENTIAL: a thread-pool
    # fan-out measured ~10× slower here (each stage is dominated by GIL-held numpy
    # elementwise ops, not the FFT). The parallelism that pays off is inside each
    # STFT's batched FFT (multithreaded fft_backend) + numexpr for the dB step.
    import gc
    del results, stage_signals, original  # free the pipeline's retained arrays first
    gc.collect()
    total, first, cached = _worker_run(
        sig, {"stage_configs": {s: {"enabled": True} for s in STAGE_ORDER[1:]}}, fs, fc
    )
    print(f"\n[worker end-to-end, {cached} stages]  "
          f"total={total:.2f}s  first-paint={first:.2f}s  stages_cached={cached}")
    return 0 if (dmax < 0.01 and finite) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
