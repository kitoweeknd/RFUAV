"""
Background Processing Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements background signal processing using QThread to avoid blocking the main UI thread.
Supports progress reporting, error handling, metrics calculation, and pre-computed STFT/PSD cache.

Memory optimisation notes
~~~~~~~~~~~~~~~~~~~~~~~~~~
- A spectral cache stores computed STFT/PSD results, keyed by signal id + view params.
- The cache is bounded (max entries) and uses LRU eviction when full.
- All Qt signals pass float32 arrays (half the memory of float64).
- Arrays are registered with MemoryManager for explicit tracking.
"""

from __future__ import annotations

import logging
import time
import threading
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Spectral Cache (LRU bounded cache for STFT/PSD results, keyed by stage)
# ----------------------------------------------------------------------

# view_type values (consistent with SpectrumViewer)
_VIEW_TIME = 0
_VIEW_PSD = 1
_VIEW_STFT = 2
_VIEW_RESIDUAL = 3
_VIEW_STFT_ZOOM = 4  # STFT zoomed to a specific time/freq window


# Monotonic signal-identity counter. Using id(array) as the cache identity is
# unsafe: once an array is garbage-collected, CPython may reuse its id() for a
# new array, causing the cache to return a previous signal's spectra. A
# never-reused counter avoids that aliasing entirely.
_signal_id_lock = threading.Lock()
_signal_id_seq = 0


def _next_signal_id() -> int:
    global _signal_id_seq
    with _signal_id_lock:
        _signal_id_seq += 1
        return _signal_id_seq


class _SpectralCache:
    """Thread-safe LRU cache for spectrum computation results.

    Key design decisions
    ~~~~~~~~~~~~~~~~~~~~
    - Bounded size: at most ``_max_entries`` entries; evict LRU when full.
    - Primary key: (stage_name, view_type, signal_id) → cached result.
      For zoom entries the key extends to
      (stage_name, _VIEW_STFT_ZOOM, signal_id, zoom_key).
    - Stale entries are auto-detected by comparing ``signal_id`` (id of the
      raw original array).  When the signal changes, all entries are
      treated as invalid regardless of key match.
    - Stores pre-computed STFT / PSD / residual for a given stage + view
      combination; zoom recompute is handled separately by StftWorker.
    """

    _lock = threading.RLock()
    _max_entries = 64  # full-spectrum + zoom entries; LRU eviction when full

    def __init__(self):
        self._cache: dict[tuple, tuple[float, int, dict]] = {}
        self._order: list[tuple] = []  # most-recently-used at end

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_key(
        stage_name: str,
        view_type: int,
        signal_id: int,
        zoom_key: str | None = None,
    ) -> tuple:
        """Build a cache key, optionally including zoom_key for zoom entries."""
        if view_type == _VIEW_STFT_ZOOM:
            assert zoom_key is not None, "zoom_key required for STFT_ZOOM view_type"
            return (stage_name, view_type, signal_id, zoom_key)
        return (stage_name, view_type, signal_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(
        self,
        stage_name: str,
        view_type: int,
        signal_id: int,
        zoom_key: str | None = None,
    ) -> dict | None:
        """Return cached spectrum for the given key if still valid."""
        key = self._make_key(stage_name, view_type, signal_id, zoom_key)
        with self._lock:
            if key not in self._cache:
                return None
            ts, cached_sig_id, result = self._cache[key]
            if cached_sig_id != signal_id:
                self._cache.pop(key, None)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                return None
            try:
                self._order.remove(key)
            except ValueError:
                pass
            self._order.append(key)
            self._cache[key] = (time.monotonic(), cached_sig_id, result)
            return result

    def put(
        self,
        stage_name: str,
        view_type: int,
        signal_id: int,
        result: dict,
        zoom_key: str | None = None,
    ) -> None:
        """Store a computed spectrum result."""
        key = self._make_key(stage_name, view_type, signal_id, zoom_key)
        with self._lock:
            if key in self._cache:
                self._cache[key] = (time.monotonic(), signal_id, result)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass
                self._order.append(key)
                return

            if len(self._cache) >= self._max_entries:
                evict_key = self._order.pop(0)
                self._cache.pop(evict_key, None)

            self._cache[key] = (time.monotonic(), signal_id, result)
            self._order.append(key)

    def clear(self) -> None:
        """Clear all entries (call when loading a new IQ file)."""
        with self._lock:
            self._cache.clear()
            self._order.clear()

    def clear_for_signal(self, signal_id: int) -> None:
        """Remove only entries belonging to a specific signal (signal replacement)."""
        with self._lock:
            stale: list[tuple] = []
            for key, (_, cached_sig_id, _) in self._cache.items():
                if cached_sig_id == signal_id:
                    stale.append(key)
            for key in stale:
                self._cache.pop(key, None)
                try:
                    self._order.remove(key)
                except ValueError:
                    pass


# Module-level cache singleton
_SPECTRAL_CACHE = _SpectralCache()


try:
    from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner
    from ChannelSimulation.viewer.signal_proc.metrics_calculator import MetricsCalculator
    from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator, SpectrumData
except ImportError:
    from pipeline_runner import PipelineRunner
    from metrics_calculator import MetricsCalculator
    from spectrum_data import SpectrumCalculator, SpectrumData


class ProcessingWorker(QThread):
    """Background processing thread

    Signals:
        finished: Processing completed with result data
        progress: Progress update (percentage, current stage name)
        error: Error message
        metrics_ready: Metrics calculation completed
        spectrum_ready: Pre-computed STFT/PSD for final stage (immediate display)
        stage_spectrum_cached: Intermediate stage spectrum cached (stage_name, spectrum_dict)
    """

    finished = Signal(dict)
    progress = Signal(float, str)
    error = Signal(str)
    metrics_ready = Signal(dict)
    spectrum_ready = Signal(dict)
    stage_spectrum_cached = Signal(str, dict)

    def __init__(
        self,
        iq_data: np.ndarray | None,
        config: dict[str, Any],
        fs: float = 100_000_000,
        fc: float = 0.0,
        seed: int = 42,
        parent=None,
    ):
        super().__init__(parent)
        self._iq_data = iq_data
        self._config = config
        self._fs = fs
        self._fc = fc
        self._seed = seed
        self._input_file: str | None = None
        self._should_stop = False

    def set_input_file(self, file_path: str) -> None:
        """Set input file path (alternative to passing iq_data directly)"""
        self._input_file = file_path

    def stop(self) -> None:
        """Request to stop processing"""
        self._should_stop = True

    def run(self) -> None:
        """Main loop for the processing thread"""
        start_time = time.perf_counter()

        try:
            runner = PipelineRunner(
                iq_data=self._iq_data,
                fs=self._fs,
                fc=self._fc,
                seed=self._seed,
            )
            runner.set_config(self._config)

            if self._input_file is not None:
                self.progress.emit(0.05, "Loading file...")
                runner.load_iq_file(self._input_file)

            if self._should_stop:
                return

            self.progress.emit(0.15, "Running Pipeline...")
            results = runner.run(return_all_stages=True)

            if self._should_stop:
                return

            original = results.get("original")
            processed = results.get("final")

            if original is None or processed is None:
                return

            # ── Build a SINGLE SpectrumCalculator and reuse for all stages ─────
            # This is critical: creating a new SpectrumCalculator() inside each
            # _compute_all_spectra_for_stage call re-creates scipy signal objects.
            sig_id = _next_signal_id()
            calc = SpectrumCalculator(fs=self._fs, fc=self._fc)

            # ── Compute original spectra ONCE (reuse for final + every stage) ───
            self.progress.emit(0.35, "Computing original spectrum...")
            stft_o, freq_stft, time_stft = calc.compute_stft(original)
            psd_o, freq_psd = calc.compute_psd(original)
            orig_data = {
                "stft_original": stft_o,
                "freq_axis_stft": freq_stft,
                "time_axis_stft": time_stft,
                "psd_original": psd_o,
                "freq_axis_psd": freq_psd,
            }

            # Cache original spectra so stage switches are instant
            _SPECTRAL_CACHE.put("original", _VIEW_STFT, sig_id, {**orig_data})
            _SPECTRAL_CACHE.put("original", _VIEW_PSD, sig_id, {**orig_data})

            # ── Compute processed (final) stage spectra ──────────────────────────
            self.progress.emit(0.60, "Computing processed spectrum...")
            stft_p, _, _ = calc.compute_stft(processed)
            psd_p, _ = calc.compute_psd(processed)

            result_final = self._build_spectrum_dict(
                original, processed,
                "final", sig_id,
                orig_data, stft_p, psd_p, calc,
            )

            # Emit immediately so UI updates right away
            self.spectrum_ready.emit(result_final)

            if self._should_stop:
                return

            # ── Pre-compute intermediate stage spectra (populate L2 cache) ─────
            stage_items = [
                (k, v)
                for k, v in results.items()
                if k not in ("original", "final") and isinstance(v, np.ndarray)
            ]
            n_stages = len(stage_items)
            if n_stages > 0:
                for i, (sname, ssignal) in enumerate(stage_items):
                    frac = 0.60 + 0.25 * ((i + 1) / max(n_stages, 1))
                    self.progress.emit(frac, f"Computing {sname}...")
                    if self._should_stop:
                        return
                    stft_s, _, _ = calc.compute_stft(ssignal)
                    psd_s, _ = calc.compute_psd(ssignal)
                    stage_result = self._build_spectrum_dict(
                        original, ssignal,
                        sname, sig_id,
                        orig_data, stft_s, psd_s, calc,
                    )
                    _SPECTRAL_CACHE.put(sname, _VIEW_STFT, sig_id, stage_result)
                    _SPECTRAL_CACHE.put(sname, _VIEW_PSD, sig_id, stage_result)
                    self.stage_spectrum_cached.emit(sname, stage_result)

            if self._should_stop:
                return

            # Emit the finished result (spectra + signals) FIRST so the UI becomes
            # interactive immediately; metrics flow separately via metrics_ready and
            # fill in a beat later, instead of blocking completion at "90%".
            elapsed = time.perf_counter() - start_time
            result = {
                "original": original,
                "processed": processed,
                "stage_signals": {
                    k: v for k, v in results.items() if k not in ("original", "final")
                },
                "fs": self._fs,
                "fc": self._fc,
                "seed": self._seed,
                "elapsed": elapsed,
                "info": {
                    "n_samples": len(original),
                    "duration": len(original) / self._fs if self._fs > 0 else 0,
                    "power_original": self._calc_power(original),
                    "power_processed": self._calc_power(processed),
                },
            }
            self.finished.emit(result)

            # Metrics last (bounded-window, cheap now, but still after first paint).
            self.progress.emit(0.90, "Calculating metrics...")
            metrics = self._calculate_metrics(original, processed)
            self.metrics_ready.emit(metrics)
            self.progress.emit(1.0, "Complete")

        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_spectrum_dict(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        stage_name: str,
        sig_id: int,
        orig_data: dict,
        stft_p: np.ndarray,
        psd_p: np.ndarray,
        calc: SpectrumCalculator,
    ) -> dict[str, Any]:
        """Build the spectrum result dict for a stage, computing residual lazily."""
        result: dict[str, Any] = {
            "stage_name": stage_name,
            "signal_id": sig_id,
            "time_axis": self._decimate_time_axis(len(original)),
            "time_domain_original": self._decimate_1d(np.abs(original), 2_000_000),
            "time_domain_processed": self._decimate_1d(np.abs(processed), 2_000_000),
            # Reuse original spectra from precompute (no recompute)
            "stft_original": orig_data["stft_original"],
            "stft_processed": stft_p,
            "freq_axis_stft": orig_data["freq_axis_stft"],
            "time_axis_stft": orig_data["time_axis_stft"],
            "psd_original": orig_data["psd_original"],
            "psd_processed": psd_p,
            "freq_axis_psd": orig_data["freq_axis_psd"],
            # Residual deferred — computed on demand by SpectrumViewer when needed
            "residual_stft": None,
            "residual_psd": None,
            "zoom_key": None,
        }
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def clear_spectral_cache() -> None:
        """Clear the module-level spectrum cache; call on new file load."""
        _SPECTRAL_CACHE.clear()

    def _decimate_time_axis(self, n_samples: int) -> np.ndarray:
        """Build decimated time axis (float32)."""
        max_pts = 2_000_000
        step = max(1, n_samples // max_pts)
        n_dec = n_samples // step
        return np.arange(n_dec, dtype=np.float32) * (step / self._fs)

    @staticmethod
    def _decimate_1d(arr: np.ndarray, max_pts: int) -> np.ndarray:
        """Downsample 1-D array to at most max_pts samples."""
        if len(arr) <= max_pts:
            return arr
        step = max(1, len(arr) // max_pts)
        return arr[::step]

    def _calculate_metrics(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> dict[str, float]:
        """Calculate signal quality metrics"""
        try:
            if original is None or processed is None:
                return {}
            calculator = MetricsCalculator()
            return calculator.calculate_all(original, processed, self._fs)
        except Exception:
            return {}

    def _calc_power(self, signal: np.ndarray | None) -> float | None:
        """Calculate signal power"""
        if signal is None or len(signal) == 0:
            return None
        power = float(np.mean(np.abs(signal) ** 2))
        if power <= 0:
            return None
        return 10 * np.log10(power)


# ─────────────────────────────────────────────────────────────────────────────
#  Zoom-recompute worker — dedicated thread so STFT never blocks the UI
# ─────────────────────────────────────────────────────────────────────────────
class StftWorker(QThread):
    """Background thread for zoom-recompute STFT.

    SpectrumViewer calls ``start_stft(sig_key, sig_orig, sig_proc, f_window,
    nperseg, fs)``.  When done emits ``stft_done(sig_key, so, sp, ff, tt)``
    on the main thread.
    """

    stft_done = Signal(object, object, object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._task: tuple | None = None
        self._should_stop = False

    def start_stft(
        self,
        sig_key: object,
        sig_orig: np.ndarray,
        sig_proc: np.ndarray,
        f_window: tuple,
        nperseg: int,
        fs: float,
    ) -> None:
        """Discard any pending task, submit a new one."""
        self._should_stop = True
        self.wait()
        self._should_stop = False
        self._task = (sig_key, sig_orig, sig_proc, f_window, nperseg, fs)
        self.start()

    def stop(self) -> None:
        self._should_stop = True
        self.wait()

    def run(self) -> None:
        if self._task is None:
            return
        sig_key, sig_orig, sig_proc, f_window, nperseg, fs = self._task

        try:
            calc = SpectrumCalculator(fs=fs)
            so, ff, tt = calc.compute_stft_windowed(sig_orig, f_window, nperseg, fs)
            sp, _, _ = calc.compute_stft_windowed(sig_proc, f_window, nperseg, fs)
            if self._should_stop:
                return
            self.stft_done.emit(sig_key, so, sp, ff, tt)
        except Exception:
            logger.exception("Zoom STFT computation failed")


class StreamingWorker(QThread):
    """Streaming processing thread (for large files)"""

    finished = Signal(dict)
    progress = Signal(float, str)
    error = Signal(str)

    def __init__(
        self,
        input_file: str,
        output_file: str,
        config: dict[str, Any],
        fs: float = 100_000_000,
        fc: float = 0.0,
        seed: int = 42,
        chunk_size: int = 10_000_000,
        parent=None,
    ):
        super().__init__(parent)
        self._input_file = input_file
        self._output_file = output_file
        self._config = config
        self._fs = fs
        self._fc = fc
        self._seed = seed
        self._chunk_size = chunk_size
        self._should_stop = False

    def stop(self) -> None:
        """Request to stop processing"""
        self._should_stop = True

    def run(self) -> None:
        """Main loop for streaming processing"""
        start_time = time.perf_counter()

        try:
            self.progress.emit(0.1, "Preparing streaming processing...")

            from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner

            runner = PipelineRunner(fs=self._fs, fc=self._fc, seed=self._seed)
            runner.load_iq_file(self._input_file)
            runner.set_config(self._config)

            self.progress.emit(0.3, "Processing...")

            results = runner.run()

            self.progress.emit(0.8, "Saving results...")

            from ChannelSimulation.core.io_utils import save_interleaved_iq

            save_interleaved_iq(self._output_file, results["final"])

            elapsed = time.perf_counter() - start_time

            self.progress.emit(1.0, "Complete")

            self.finished.emit(
                {
                    "output_file": self._output_file,
                    "elapsed": elapsed,
                    "n_samples": len(results["final"]),
                }
            )

        except Exception as e:
            import traceback

            self.error.emit(f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")
