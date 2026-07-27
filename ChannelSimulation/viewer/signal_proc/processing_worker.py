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
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PySide6.QtCore import QThread, Signal


try:
    from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner
    from ChannelSimulation.viewer.signal_proc.metrics_calculator import MetricsCalculator
    from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator
except ImportError:
    from pipeline_runner import PipelineRunner
    from metrics_calculator import MetricsCalculator
    from spectrum_data import SpectrumCalculator


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
        max_samples: int | None = None,
    ):
        super().__init__(parent)
        self._iq_data = iq_data
        self._config = config
        self._fs = fs
        self._fc = fc
        self._seed = seed
        self._input_file: str | None = None
        self._should_stop = False
        # Preview truncation: process only the first N samples for an on-screen
        # spectrogram. None = process the whole signal (used by export/save).
        self._max_samples = max_samples

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
                # Preview passes max_samples so only the leading slice is
                # materialised from disk (cheap); export passes None (full).
                runner.load_iq_file(self._input_file, max_samples=self._max_samples)

            # Safety net for the set_iq_data / small-file paths.
            runner.limit_samples(self._max_samples)

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
            # Creating a new SpectrumCalculator() per stage would re-create scipy
            # signal objects each time.
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

            # ── Compute processed (final) stage spectra ──────────────────────────
            self.progress.emit(0.60, "Computing processed spectrum...")
            stft_p, _, _ = calc.compute_stft(processed)
            psd_p, _ = calc.compute_psd(processed)

            result_final = self._build_spectrum_dict(
                original, processed, "final",
                orig_data, stft_p, psd_p, calc,
            )

            # Emit immediately so UI updates right away
            self.spectrum_ready.emit(result_final)

            if self._should_stop:
                return

            # ── Pre-compute intermediate stage spectra, emitted to the viewer ──
            # NOTE: this runs SEQUENTIALLY on purpose. A thread-pool fan-out was
            # measured to be ~10× SLOWER here: the per-stage cost is dominated by
            # GIL-held numpy elementwise ops (log10/|·|²/fftshift), not the FFT, so
            # concurrent threads just contend on the GIL and thrash memory bandwidth.
            # The parallelism that *does* pay off is inside each STFT's batched FFT
            # (the multithreaded fft_backend), which each stage already uses.
            stage_items = [
                (k, v)
                for k, v in results.items()
                if k not in ("original", "final") and isinstance(v, np.ndarray)
            ]
            n_stages = len(stage_items)
            for i, (sname, ssignal) in enumerate(stage_items):
                if self._should_stop:
                    return
                frac = 0.60 + 0.25 * ((i + 1) / max(n_stages, 1))
                self.progress.emit(frac, f"Computing {sname}...")
                stft_s, _, _ = calc.compute_stft(ssignal)
                psd_s, _ = calc.compute_psd(ssignal)
                stage_result = self._build_spectrum_dict(
                    original, ssignal, sname, orig_data, stft_s, psd_s, calc,
                )
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
        orig_data: dict,
        stft_p: np.ndarray,
        psd_p: np.ndarray,
        calc: SpectrumCalculator,
    ) -> dict[str, Any]:
        """Build the spectrum result dict for a stage, computing residual lazily."""
        result: dict[str, Any] = {
            "stage_name": stage_name,
            "time_axis": self._decimate_time_axis(len(original)),
            # Decimate first, then take |.| — avoids an O(N) abs over the full signal.
            "time_domain_original": np.abs(self._decimate_1d(original, 2_000_000)),
            "time_domain_processed": np.abs(self._decimate_1d(processed, 2_000_000)),
            # Reuse original spectra from precompute (no recompute)
            "stft_original": orig_data["stft_original"],
            "stft_processed": stft_p,
            "freq_axis_stft": orig_data["freq_axis_stft"],
            "time_axis_stft": orig_data["time_axis_stft"],
            "psd_original": orig_data["psd_original"],
            "psd_processed": psd_p,
            "freq_axis_psd": orig_data["freq_axis_psd"],
            # Residual (this stage vs the original) — nearly free: both STFT/PSD dB
            # arrays already exist, so filling it here makes the Residual view a pure
            # cache hit instead of a synchronous GUI-thread recompute on view switch.
            "residual_stft": calc.compute_residual_stft(orig_data["stft_original"], stft_p),
            "residual_psd": calc.compute_residual_psd(orig_data["psd_original"], psd_p),
        }
        return result

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
        stage_name: str = "final",
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
        # Which pipeline stage to save ("final" = fully processed). The full
        # signal is always processed — no preview truncation on export.
        self._stage_name = stage_name

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
            if self._should_stop:
                return

            self.progress.emit(0.3, "Processing...")

            # Full signal, no truncation. Fetch intermediate stages only when a
            # non-final stage was requested (keeps memory low for plain exports).
            want_stages = self._stage_name not in ("final", "", None)
            results = runner.run(return_all_stages=want_stages)
            signal = results.get(self._stage_name, results["final"])
            if self._should_stop:
                return  # cancelled before writing — don't emit a file

            self.progress.emit(0.8, "Saving results...")

            from ChannelSimulation.core.io_utils import save_interleaved_iq

            save_interleaved_iq(self._output_file, signal)

            elapsed = time.perf_counter() - start_time

            self.progress.emit(1.0, "Complete")

            self.finished.emit(
                {
                    "output_file": self._output_file,
                    "elapsed": elapsed,
                    "n_samples": len(signal),
                }
            )

        except Exception as e:
            import traceback

            self.error.emit(f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}")
