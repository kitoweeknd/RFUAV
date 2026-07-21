"""
Spectrum Data Computation Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
|
| Spectrum data computation module decoupled from plotting logic.
| Provides FFT, PSD, STFT, and residual spectrum computation.
|
| Memory optimisation notes
| ~~~~~~~~~~~~~~~~~~~~~~~~~~
| - All display arrays (PSD dB, STFT dB) use float32, halving memory vs float64.
| - Sliding-window blocks are allocated from the ArrayPool.
| - Window arrays are registered as fixed pool slots so they are reused across calls.
| - For PSD/STFT the output is always float32 regardless of input dtype.
| - NumPy array views are used wherever possible to avoid data copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .array_pool import get_pool
from . import fft_backend as fb

try:
    import numexpr as _ne  # multithreaded, GIL-free, fused elementwise (log10/|·|²)
except ImportError:
    _ne = None


def _complex_to_db(spec: np.ndarray, factor: float, eps: float, dtype) -> np.ndarray:
    """``factor * log10(|spec|² + eps)`` — the dominant STFT elementwise step.

    numexpr fuses the power + log10 into one multithreaded pass (no temporaries,
    releases the GIL); this is the biggest single cost of an STFT and runs once per
    stage. Falls back to numpy when numexpr is absent.
    """
    if _ne is not None:
        re, im = spec.real, spec.imag
        return _ne.evaluate("factor * log10(re * re + im * im + eps)").astype(dtype, copy=False)
    power = spec.real ** 2 + spec.imag ** 2
    return (factor * np.log10(power + eps)).astype(dtype, copy=False)


def _diff_to_db(a: np.ndarray, b: np.ndarray, factor: float, eps: float, dtype) -> np.ndarray:
    """``factor * log10(|a - b| + eps)`` for residuals, multithreaded via numexpr."""
    if _ne is not None:
        return _ne.evaluate("factor * log10(abs(a - b) + eps)").astype(dtype, copy=False)
    return (factor * np.log10(np.abs(a - b) + eps)).astype(dtype, copy=False)


@dataclass
class SpectrumData:
    """Spectrum data structure"""
    # Time domain data
    time_domain_original: np.ndarray | None = None
    time_domain_processed: np.ndarray | None = None
    time_axis: np.ndarray | None = None

    # PSD data
    psd_original: np.ndarray | None = None
    psd_processed: np.ndarray | None = None
    freq_axis_psd: np.ndarray | None = None

    # STFT data
    stft_original: np.ndarray | None = None
    stft_processed: np.ndarray | None = None
    freq_axis_stft: np.ndarray | None = None
    time_axis_stft: np.ndarray | None = None

    # Residual data
    residual_psd: np.ndarray | None = None
    residual_stft: np.ndarray | None = None

    # Metadata
    fs: float = 100_000_000
    fc: float = 0.0


class SpectrumCalculator:
    """Spectrum data calculator

    Provides spectrum data computation decoupled from plotting logic.
    """

    # Rendering limits (prevent Matplotlib OOM)
    MAX_PLOT_POINTS = 2_000_000
    MAX_STFT_COLS = 3_000

    # Display dtype for all spectrum / time-domain arrays.
    # float32 halves memory vs float64 with negligible precision loss for UI.
    DISPLAY_DTYPE = np.float32

    # Beyond this length, PSD uses Welch instead of a full-length FFT (avoids 1e8-point FFT).
    MAX_PSD_DIRECT_FFT = 524_288
    PSD_WELCH_NPERSEG = 8192
    # Welch averages every segment (O(N)). For display we only need a smooth estimate, so we
    # average a fixed number of evenly-spaced segments → O(PSD_MAX_SEGMENTS·nperseg), size-independent.
    PSD_MAX_SEGMENTS = 256

    # Long captures: never decimate IQ before STFT (that shrinks Nyquist and breaks MHz axis).
    # Instead use sparse window placement (hop may exceed nperseg) to cap frame count.
    # Threshold kept low: a full overlapped scipy.stft over millions of samples computes
    # ~N/hop frames and then throws 99% away when decimating to MAX_STFT_COLS. Above this
    # size we compute only the ~MAX_STFT_COLS frames we actually display (O(display), not O(N)).
    # Below it, the dense overlapped path is cheap and gives finer columns for short captures.
    STFT_LONG_THRESHOLD = 500_000  # ~5 ms @ 100 Msps
    STFT_NPERSEG_LONG = 4096  # FFT size per column; full [-fs/2, fs/2] at original fs

    # Zoom-recompute STFT: must never let nfft grow with 1/bandwidth (was allocating 100+ GiB).
    MAX_ZOOM_STFT_SAMPLES = 4_000_000
    MAX_ZOOM_NFFT = 8192

    # STFT parameters — optimised for 100 Msps UAV OFDM signals
    # nperseg=2048 → freq resolution ≈ 100MHz/2048 ≈ 48.8 kHz
    # enough to resolve typical OFDM subcarrier spacing (≥ 10 kHz)
    DEFAULT_NPERSEG = 2048
    DEFAULT_NOVERLAP = 1536   # 75% overlap for smooth spectrogram
    DEFAULT_NFFT = 2048

    # Small value for numerical stability
    EPS = 1e-12

    def __init__(
        self,
        fs: float = 100_000_000,
        fc: float = 0.0,
        nperseg: int = DEFAULT_NPERSEG,
        noverlap: int = DEFAULT_NOVERLAP,
    ):
        self._fs = fs
        self._fc = fc
        self._nperseg = nperseg
        self._noverlap = noverlap

    @property
    def fs(self) -> float:
        """Sampling frequency"""
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        self._fs = float(value)

    @property
    def fc(self) -> float:
        """Center frequency"""
        return self._fc

    @fc.setter
    def fc(self, value: float) -> None:
        self._fc = float(value)

    def calculate_all(
        self,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> SpectrumData:
        """Calculate all spectrum data (full cache; used by batch export)."""
        data = SpectrumData(fs=self._fs, fc=self._fc)
        self.fill_time_domain(data, original, processed)
        self.fill_psd(data, original, processed)
        self.fill_stft(data, original, processed)
        self.fill_residual(data)
        return data

    def ensure_for_view(
        self,
        data: SpectrumData,
        original: np.ndarray,
        processed: np.ndarray,
        view_type: int,
    ) -> None:
        """Compute only what the given view mode needs (0–3 = Time, PSD, STFT, Residual)."""
        if view_type == 0:
            self.fill_time_domain(data, original, processed)
        elif view_type == 1:
            self.fill_psd(data, original, processed)
        elif view_type == 2:
            self.fill_stft(data, original, processed)
        elif view_type == 3:
            self.fill_psd(data, original, processed)
            self.fill_stft(data, original, processed)
            self.fill_residual(data)

    def fill_time_domain(
        self,
        data: SpectrumData,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> None:
        if data.time_axis is not None:
            return
        data.time_domain_original = self.compute_time_domain(original)
        data.time_domain_processed = self.compute_time_domain(processed)
        data.time_axis = self._compute_time_axis(len(original))

    def fill_psd(
        self,
        data: SpectrumData,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> None:
        if data.psd_original is not None:
            return
        data.psd_original, data.freq_axis_psd = self.compute_psd(original)
        data.psd_processed, _ = self.compute_psd(processed)

    def fill_stft(
        self,
        data: SpectrumData,
        original: np.ndarray,
        processed: np.ndarray,
    ) -> None:
        if data.stft_original is not None:
            return
        data.stft_original, data.freq_axis_stft, data.time_axis_stft = self.compute_stft(original)
        data.stft_processed, _, _ = self.compute_stft(processed)

    def fill_residual(self, data: SpectrumData) -> None:
        if data.residual_psd is not None and data.residual_stft is not None:
            return
        if data.psd_original is None or data.psd_processed is None:
            raise RuntimeError("fill_residual requires PSD data")
        if data.stft_original is None or data.stft_processed is None:
            raise RuntimeError("fill_residual requires STFT data")
        data.residual_psd = self.compute_residual_psd(data.psd_original, data.psd_processed)
        data.residual_stft = self.compute_residual_stft(
            data.stft_original, data.stft_processed
        )

    def compute_time_domain(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """Compute time domain envelope

        Args:
            signal: IQ signal

        Returns:
            Amplitude envelope |x(t)|
        """
        # Decimate *before* taking the magnitude: |x| over the full array then
        # discarding 99% of it wastes an O(N) abs on tens of millions of samples.
        # step matches _compute_time_axis so the axes stay length-aligned.
        n = len(signal)
        step = max(1, n // self.MAX_PLOT_POINTS)
        return np.abs(signal[::step])

    def _compute_time_axis(
        self,
        n_samples: int,
    ) -> np.ndarray:
        """Compute time axis in float32 to save memory."""
        step = max(1, n_samples // self.MAX_PLOT_POINTS)
        n_decimated = n_samples // step
        # float32 is sufficient for time axis display (sub-ms precision at 100 MHz is ~1e-8).
        t = np.arange(n_decimated, dtype=np.float32) * (step / self._fs)
        return t

    def compute_psd(
        self,
        signal: np.ndarray,
        window: str = "hanning",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density (Welch for long captures; direct FFT when short)."""
        n = len(signal)
        if n <= self.MAX_PSD_DIRECT_FFT:
            return self._compute_psd_direct_fft(signal, window)

        try:
            from scipy.signal import welch
        except ImportError:
            return self._compute_psd_decimated_fft(signal, window)

        nperseg = min(self.PSD_WELCH_NPERSEG, n)
        noverlap = min(nperseg // 2, nperseg - 1)
        win = "hann" if window == "hanning" else window

        # Cap the number of averaged segments: Welch over the full signal touches every
        # sample (O(N)). A fixed count of evenly-spaced segments gives a statistically
        # equivalent display PSD at O(PSD_MAX_SEGMENTS·nperseg), independent of N.
        n_blocks = n // nperseg
        if n_blocks > self.PSD_MAX_SEGMENTS:
            starts = np.linspace(0, n - nperseg, self.PSD_MAX_SEGMENTS, dtype=np.int64)
            subset = np.concatenate([signal[s:s + nperseg] for s in starts])
            welch_input = subset
            welch_noverlap = 0  # each nperseg block is one clean, independent segment
        else:
            welch_input = signal
            welch_noverlap = noverlap

        freqs, psd = welch(
            welch_input,
            fs=self._fs,
            nperseg=nperseg,
            noverlap=welch_noverlap,
            window=win,
            return_onesided=False,
            scaling="density",
            average="mean",
        )
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        psd_db = (10 * np.log10(psd + self.EPS)).astype(self.DISPLAY_DTYPE)
        if self._fc != 0:
            freqs = freqs + self._fc
        return psd_db, freqs

    def _compute_psd_direct_fft(
        self,
        signal: np.ndarray,
        window: str = "hanning",
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(signal)
        if window == "hanning":
            window_vals = np.hanning(n)
        elif window == "hamming":
            window_vals = np.hamming(n)
        elif window == "blackman":
            window_vals = np.blackman(n)
        else:
            window_vals = np.ones(n, dtype=np.float64)

        windowed = signal * window_vals
        spectrum = fb.fft(windowed)
        spectrum_single = spectrum[: n // 2]
        # Power directly from real/imag avoids the sqrt in abs()**2.
        psd = spectrum_single.real ** 2 + spectrum_single.imag ** 2
        window_power = np.sum(window_vals**2)
        psd = psd / (window_power * self._fs)
        # float32 output for display (saves ~50 % memory vs float64).
        psd_db = (10 * np.log10(psd + self.EPS)).astype(self.DISPLAY_DTYPE)
        freqs = np.fft.fftfreq(n, 1 / self._fs)[: n // 2]
        if self._fc != 0:
            freqs = freqs + self._fc
        return psd_db, freqs

    def _compute_psd_decimated_fft(
        self,
        signal: np.ndarray,
        window: str = "hanning",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback when SciPy is missing: decimate then direct FFT."""
        n = len(signal)
        target = self.MAX_PSD_DIRECT_FFT
        step = max(1, int(np.ceil(n / target)))
        sig = signal[::step]
        fs_eff = self._fs / step
        calc = SpectrumCalculator(fs=fs_eff, fc=self._fc, nperseg=self._nperseg, noverlap=self._noverlap)
        return calc._compute_psd_direct_fft(sig, window)

    def compute_stft(
        self,
        signal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute short-time Fourier transform

        Args:
            signal: IQ signal

        Returns:
            (STFT magnitude in dB, frequency axis, time axis)
        """
        n = len(signal)
        fs = self._fs

        # Long IQ: sparse-in-time STFT at **original** fs so y-axis stays ±fs/2 (e.g. ±50 MHz @ 100 Msps).
        if n > self.STFT_LONG_THRESHOLD:
            return self._compute_stft_long_sparse(signal, fs)

        try:
            from scipy.signal import stft
        except ImportError:
            return self._compute_stft_fallback_short(signal, fs)

        nperseg = min(self._nperseg, n)
        noverlap = min(self._noverlap, nperseg - 1)

        freqs, times, Zxx = stft(
            signal,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            boundary=None,
            padded=False,
        )

        freqs_shifted = np.fft.fftshift(freqs)
        Zxx_shifted = np.fft.fftshift(Zxx, axes=0)
        stft_db = 20 * np.log10(np.abs(Zxx_shifted) + self.EPS)

        # scipy stft: shape (nfreq, ntime) → matches imshow convention directly.
        stft_db = self._decimate_2d(stft_db, self.MAX_STFT_COLS, axis=1)
        times = self._decimate_1d(times, self.MAX_STFT_COLS)

        if self._fc != 0:
            freqs_shifted = freqs_shifted + self._fc

        return stft_db, freqs_shifted, times

    def _compute_stft_long_sparse(
        self,
        signal: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized STFT for long IQ: no Python for-loop over frames.

        Uses non-overlapping blocks (hop = n // max_cols) so the time axis
        covers the FULL recording from 0 to n/fs — no sparse gaps, no noise
        bands on idle portions.
        """
        n = len(signal)
        nperseg = min(self.STFT_NPERSEG_LONG, n)
        if nperseg < 32:
            return self._compute_stft_fallback_short(signal, fs)

        max_cols = self.MAX_STFT_COLS
        hop = max(1, n // max_cols)          # non-overlapping stride
        n_frames = max(1, n // hop)          # how many frames fit in the recording

        # Build the block: (n_frames × nperseg)  — no linspace, straight slicing
        starts = np.arange(0, n_frames * hop, hop, dtype=np.intp)
        starts = starts[starts + nperseg <= n]  # drop any window that would overflow
        if len(starts) == 0:
            return self._compute_stft_fallback_short(signal, fs)

        # Gather the frames as a (n_frames, nperseg) block. sliding_window_view is
        # a zero-copy strided view of every length-nperseg window; fancy-indexing
        # the chosen starts materialises only the frames we keep — no Python
        # list-comprehension over thousands of slices, no np.stack.
        windows = np.lib.stride_tricks.sliding_window_view(signal, nperseg)
        block = windows[starts]                              # (n_frames, nperseg)

        # Apply Hann window row-wise (cast to the signal's real dtype so a
        # complex64 input stays complex64 — half the memory/bandwidth of c128).
        hann = np.hanning(nperseg).astype(block.real.dtype, copy=False)
        block = block * hann

        # All FFTs at once via the multithreaded backend; output cast to float32.
        spec = fb.fft(block, axis=1)                         # (n_frames, nperseg) = (ntime, nfreq)
        spec = fb.fftshift(spec, axes=1)
        # Power-domain dB (10*log10(|z|^2), no sqrt; EPS**2 keeps the same silent
        # floor as 20*log10(|z|+EPS)), fused + multithreaded via numexpr.
        stft_db = _complex_to_db(spec, 10.0, self.EPS ** 2, self.DISPLAY_DTYPE)
        stft_db = stft_db.T                                   # → (nfreq, ntime)  for imshow

        # Time axis: mid-point of each non-overlapping block
        times = (starts.astype(np.float64) + nperseg * 0.5) / fs

        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(nperseg, 1.0 / fs))
        if self._fc != 0:
            freqs_shifted = freqs_shifted + self._fc

        return stft_db, freqs_shifted, times

    def _compute_stft_fallback_short(
        self,
        signal: np.ndarray,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SciPy-free STFT for short signals (same path as old _compute_stft_fallback)."""
        n = len(signal)
        nperseg = min(self._nperseg, n)
        step = max(1, nperseg - self._noverlap)
        n_frames = (n - nperseg) // step + 1

        # Use pooled buffer for the STFT matrix (avoids repeated np.zeros() calls).
        stft_matrix = get_pool().acquire(
            (nperseg, n_frames), np.complex128, zero=True
        )
        times = np.zeros(n_frames, dtype=self.DISPLAY_DTYPE)

        # Register Hanning window as a fixed pool slot (reused across calls, no allocation each time).
        hann_key = f"hann_{nperseg}"
        hann_win = get_pool().get_fixed(hann_key)
        if hann_win is None:
            hann_win = get_pool().register_fixed(hann_key, (nperseg,), np.float64)
            hann_win[:] = np.hanning(nperseg)
        window_vals = hann_win  # view into fixed slot

        for i in range(n_frames):
            start = i * step
            end = start + nperseg
            if end > n:
                break

            frame = signal[start:end] * window_vals
            stft_matrix[:, i] = np.fft.fft(frame)
            times[i] = (start + nperseg * 0.5) / fs

        freqs = np.fft.fftfreq(nperseg, 1 / fs)
        freqs_shifted = np.fft.fftshift(freqs)
        stft_shifted = np.fft.fftshift(stft_matrix, axes=0)

        # float32 is sufficient for display; saves ~50 % memory vs float64.
        stft_db = (20 * np.log10(np.abs(stft_shifted) + self.EPS)).astype(self.DISPLAY_DTYPE)
        # stft_db is (nfreq, ntime) — matches imshow convention directly; decimate time axis.
        stft_db = self._decimate_2d(stft_db, self.MAX_STFT_COLS, axis=1)
        times = self._decimate_1d(times, self.MAX_STFT_COLS)

        if self._fc != 0:
            freqs_shifted = freqs_shifted + self._fc

        return stft_db, freqs_shifted, times

    def compute_stft_windowed(
        self,
        signal: np.ndarray,
        f_window: Tuple[float, float],
        nperseg: int,
        fs: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """STFT on a time slice, then **crop** rows to the visible frequency band.

        Uses a fixed upper bound on ``nfft`` (never scales as 1/bandwidth), so zooming
        a narrow frequency strip cannot allocate hundreds of GiB.

        ``f_window`` is in the same Hz units as ``compute_stft``'s frequency axis
        (fftshifted baseband bins, plus ``fc`` when non-zero).

        Args:
            signal:   IQ slice for the visible time range.
            f_window: ``(f_lo_hz, f_hi_hz)`` with ``f_lo_hz <= f_hi_hz``.
            nperseg:  Requested window length (clamped internally).
            fs:       Sample rate of ``signal``.

        Returns:
            (STFT dB, cropped frequency axis Hz, time axis s)
        """
        f_lo, f_hi = (min(f_window), max(f_window))
        sig = signal
        fs_use = float(fs)

        # Long zoom window: do not decimate IQ (same Nyquist issue as overview STFT).
        if len(sig) > self.MAX_ZOOM_STFT_SAMPLES:
            stft_full, freqs_line, times = self._compute_stft_long_sparse(sig, fs_use)
            mask = (freqs_line >= f_lo) & (freqs_line <= f_hi)
            if not np.any(mask):
                mask = np.ones_like(freqs_line, dtype=bool)
            return stft_full[mask, :], freqs_line[mask], times

        n = len(sig)
        if n < 8:
            z = np.zeros((1, 1), dtype=np.float64)
            return z, np.array([-fs_use / 2, fs_use / 2]), np.array([0.0])

        nperseg_w = int(np.clip(nperseg, 32, min(4096, n // 2)))
        noverlap = min(max(1, int(nperseg_w * 0.75)), nperseg_w - 1)
        n_fft = min(
            self.MAX_ZOOM_NFFT,
            max(nperseg_w, 1 << (nperseg_w - 1).bit_length()),
        )

        try:
            from scipy.signal import stft
        except ImportError:
            return self._compute_stft_windowed_fallback(sig, fs_use, f_lo, f_hi, nperseg_w, n_fft)

        freqs, times, Zxx = stft(
            sig,
            fs=fs_use,
            nperseg=nperseg_w,
            noverlap=noverlap,
            nfft=n_fft,
            return_onesided=False,
            boundary=None,
            padded=False,
        )
        freqs_shifted = np.fft.fftshift(freqs)
        Zxx_s = np.fft.fftshift(Zxx, axes=0)
        if self._fc != 0:
            freqs_shifted = freqs_shifted + self._fc

        mask = (freqs_shifted >= f_lo) & (freqs_shifted <= f_hi)
        if not np.any(mask):
            mask = np.ones_like(freqs_shifted, dtype=bool)

        stft_db = 20 * np.log10(np.abs(Zxx_s[mask, :]) + self.EPS)
        freqs_crop = freqs_shifted[mask]

        stft_db = self._decimate_2d(stft_db, self.MAX_STFT_COLS, axis=1)
        times = self._decimate_1d(times, self.MAX_STFT_COLS)

        return stft_db, freqs_crop, times

    def _compute_stft_windowed_fallback(
        self,
        signal: np.ndarray,
        fs: float,
        f_lo: float,
        f_hi: float,
        nperseg: int,
        n_fft: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """SciPy-free path: capped n_fft, crop frequencies."""
        n_fft = min(int(n_fft), self.MAX_ZOOM_NFFT)
        n = len(signal)
        if n < 8:
            z = np.zeros((1, 1), dtype=self.DISPLAY_DTYPE)
            return z, np.array([-fs / 2, fs / 2], dtype=self.DISPLAY_DTYPE), np.array([0.0], dtype=self.DISPLAY_DTYPE)
        nperseg_w = min(nperseg, n)
        step = max(1, nperseg_w - nperseg_w * 3 // 4)
        max_frames = self.MAX_STFT_COLS + 1
        n_frames = min((n - nperseg_w) // step + 1, max_frames)

        # Pool the large STFT matrix instead of np.zeros().
        stft_matrix = get_pool().acquire((n_fft, n_frames), np.complex128, zero=True)
        times = np.zeros(n_frames, dtype=self.DISPLAY_DTYPE)

        # Hanning window: registered fixed slot avoids re-computing each time.
        hann_key = f"hann_{nperseg_w}"
        hann_win = get_pool().get_fixed(hann_key)
        if hann_win is None:
            hann_win = get_pool().register_fixed(hann_key, (nperseg_w,), np.float64)
            hann_win[:] = np.hanning(nperseg_w)
        window_vals = hann_win

        # Pool the padding buffer.
        padded = get_pool().acquire((n_fft,), np.complex128, zero=True)
        fi = 0
        for i in range(n_frames):
            start = i * step
            end = start + nperseg_w
            if end > n:
                break
            frame = signal[start:end] * window_vals
            padded.fill(0)  # clear pool buffer
            padded[:nperseg_w] = frame
            stft_matrix[:, fi] = np.fft.fft(padded)
            times[fi] = (start + nperseg_w * 0.5) / fs
            fi += 1
        stft_matrix = stft_matrix[:, :fi]
        times = times[:fi]

        freqs = np.fft.fftfreq(n_fft, 1 / fs)
        freqs_shifted = np.fft.fftshift(freqs)
        stft_shifted = np.fft.fftshift(stft_matrix, axes=0)
        if self._fc != 0:
            freqs_shifted = freqs_shifted + self._fc
        mask = (freqs_shifted >= f_lo) & (freqs_shifted <= f_hi)
        if not np.any(mask):
            mask = np.ones_like(freqs_shifted, dtype=bool)
        stft_db = 20 * np.log10(np.abs(stft_shifted[mask, :]) + self.EPS)
        freqs_crop = freqs_shifted[mask]
        stft_db = self._decimate_2d(stft_db, self.MAX_STFT_COLS, axis=1)
        times = self._decimate_1d(times, self.MAX_STFT_COLS)
        return stft_db, freqs_crop, times

    def compute_residual_psd(
        self,
        psd_original: np.ndarray,
        psd_processed: np.ndarray,
    ) -> np.ndarray:
        """Compute PSD residual

        Args:
            psd_original: Original PSD
            psd_processed: Processed PSD

        Returns:
            Residual PSD (dB, float32)
        """
        # float32 is sufficient for display residual; numexpr-fused when available.
        return _diff_to_db(psd_original, psd_processed, 10.0, self.EPS, self.DISPLAY_DTYPE)

    def compute_residual_stft(
        self,
        stft_original: np.ndarray,
        stft_processed: np.ndarray,
    ) -> np.ndarray:
        """Compute STFT residual

        Args:
            stft_original: Original STFT
            stft_processed: Processed STFT

        Returns:
            Residual STFT (dB, float32)
        """
        # float32 is sufficient for display residual; numexpr-fused when available.
        return _diff_to_db(stft_original, stft_processed, 20.0, self.EPS, self.DISPLAY_DTYPE)

    def _decimate_1d(
        self,
        arr: np.ndarray,
        max_points: int,
        axis: int = 0,
    ) -> np.ndarray:
        """1D array downsampling

        Args:
            arr: Input array
            max_points: Maximum number of points
            axis: Downsampling axis

        Returns:
            Downsampled array
        """
        n = arr.shape[axis]
        if n <= max_points:
            return arr

        step = max(1, n // max_points)
        return arr[::step]

    def _decimate_2d(
        self,
        arr: np.ndarray,
        max_cols: int,
        axis: int = 1,
    ) -> np.ndarray:
        """2D array downsampling (along time axis)"""
        n_cols = arr.shape[axis]
        if n_cols <= max_cols:
            return arr

        step = max(1, n_cols // max_cols)

        if axis == 1:
            return arr[:, ::step]
        else:
            return arr[::step, :]

    def find_peaks(
        self,
        psd: np.ndarray,
        freqs: np.ndarray,
        n_peaks: int = 5,
        prominence: float | None = None,
    ) -> list[dict]:
        """Find spectral peaks

        Args:
            psd: PSD data
            freqs: Frequency axis
            n_peaks: Maximum number of peaks
            prominence: Peak prominence threshold

        Returns:
            List of peaks [{freq, power, ...}, ...]
        """
        try:
            from scipy.signal import find_peaks
        except ImportError:
            return self._find_peaks_fallback(psd, freqs, n_peaks)

        # Find peaks
        peaks, properties = find_peaks(
            psd,
            height=np.nanmean(psd),
            prominence=prominence if prominence else 10,
        )

        if len(peaks) == 0:
            return []

        # Sort by height
        heights = psd[peaks]
        sorted_idx = np.argsort(heights)[::-1]
        peaks = peaks[sorted_idx[:n_peaks]]

        return [
            {
                "freq": float(freqs[p]),
                "power": float(psd[p]),
                "index": int(p),
            }
            for p in peaks
        ]

    def _find_peaks_fallback(
        self,
        psd: np.ndarray,
        freqs: np.ndarray,
        n_peaks: int = 5,
    ) -> list[dict]:
        """Peak finding fallback without scipy"""
        # Simple peak detection
        peaks = []
        for i in range(1, len(psd) - 1):
            if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
                peaks.append(i)

        if not peaks:
            return []

        # Sort by height
        peaks.sort(key=lambda i: psd[i], reverse=True)
        peaks = peaks[:n_peaks]

        return [
            {
                "freq": float(freqs[p]),
                "power": float(psd[p]),
                "index": int(p),
            }
            for p in peaks
        ]
