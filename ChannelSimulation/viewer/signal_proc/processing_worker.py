"""
Background Processing Thread
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implements background signal processing using QThread to avoid blocking the main UI thread.
Supports progress reporting, error handling, metrics calculation, and pre-computed STFT/PSD cache.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal

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
        spectrum_ready: Pre-computed spectrum data (dict)
    """

    finished = Signal(dict)
    progress = Signal(float, str)
    error = Signal(str)
    metrics_ready = Signal(dict)
    spectrum_ready = Signal(dict)  # pre-computed STFT/PSD in background

    def __init__(
        self,
        iq_data: np.ndarray | None,
        config: Dict[str, Any],
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

            # ── Pre-compute STFT / PSD in this background thread ─────────────
            self.progress.emit(0.85, "Computing spectrum...")
            spectrum_data = self._compute_spectrum_background(original, processed)
            self.spectrum_ready.emit(spectrum_data)

            if self._should_stop:
                return

            metrics = self._calculate_metrics(original, processed)

            elapsed = time.perf_counter() - start_time
            self.progress.emit(1.0, "Complete")

            result = {
                "original": original,
                "processed": processed,
                "stage_signals": {k: v for k, v in results.items() if k not in ("original", "final")},
                "fs": self._fs,
                "fc": self._fc,
                "seed": self._seed,
                "elapsed": elapsed,
                "info": {
                    "n_samples": len(original) if original is not None else 0,
                    "duration": (len(original) / self._fs) if (original is not None and self._fs > 0) else 0,
                    "power_original": self._calc_power(original),
                    "power_processed": self._calc_power(processed),
                },
            }

            self.finished.emit(result)
            self.metrics_ready.emit(metrics)

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    # ------------------------------------------------------------------ #
    #  Spectrum pre-computation (runs in background thread)              #
    # ------------------------------------------------------------------ #
    def _compute_spectrum_background(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> Dict[str, Any]:
        """STFT / PSD / residual computed entirely in the background thread."""
        if original is None or processed is None:
            return {}

        calc = SpectrumCalculator(fs=self._fs, fc=self._fc)
        data = calc.calculate_all(original, processed)

        return {
            "time_axis": data.time_axis,
            "stft_original": data.stft_original,
            "stft_processed": data.stft_processed,
            "freq_axis_stft": data.freq_axis_stft,
            "time_axis_stft": data.time_axis_stft,
            "psd_original": data.psd_original,
            "psd_processed": data.psd_processed,
            "freq_axis_psd": data.freq_axis_psd,
            "residual_psd": data.residual_psd,
            "residual_stft": data.residual_stft,
        }

    def _calculate_metrics(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> Dict[str, float]:
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
    nperseg, fs)``.  When done emits ``stft_done(sig_key, stft_orig, freq_axis,
    time_axis)`` on the main thread.
    """

    stft_done = Signal(object, object, object, object, object)  # sig_key, so, sp, ff, tt

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
            pass


class StreamingWorker(QThread):
    """Streaming processing thread (for large files)"""

    finished = Signal(dict)
    progress = Signal(float, str)
    error = Signal(str)

    def __init__(
        self,
        input_file: str,
        output_file: str,
        config: Dict[str, Any],
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
            import tempfile
            from pathlib import Path

            temp_dir = Path(tempfile.mkdtemp())

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

            self.finished.emit({
                "output_file": self._output_file,
                "elapsed": elapsed,
                "n_samples": len(results["final"]),
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")
