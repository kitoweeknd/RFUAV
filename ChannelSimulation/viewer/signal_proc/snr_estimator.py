"""
SNR Estimator
~~~~~~~~~~~~~

Port of the MATLAB OFDM UAV signal SNR estimation algorithm with
performance improvements (cached FFT bins, vectorized operations).

Algorithm flow
~~~~~~~~~~~~~
1. drones_ofdm_freq_shift_esti
   → PSD via Welch, sliding energy window to find OFDM signal band edges [f1, f2]

2. drones_ofdm_freq_compensation
   → Down-convert to baseband: x * exp(-j*2π*f_center*t)

3. position_find (locate_signal)
   → Downsample 2x, find FFT bin indices for signal band [idx1, idx2]
     and noise band [idx3, idx4] (offset by bw_noise below signal band)

4. snr_esti
   → FFT on downsampled signal, compute:
       SNR = 10*log10((P_signal - P_noise) / P_noise)
   where  P_signal = mean(|FFT[idx1:idx2]|^2)
          P_noise  = mean(|FFT[idx3:idx4]|^2)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window, resample_poly, welch


_EPS = 1e-20
# Default parameters — optimised for 100 Msps UAV OFDM signals
_DEFAULT_BW = 20e6          # OFDM signal bandwidth (Hz)
_DEFAULT_NFFT = 16384       # FFT size; at 100 Msps → ~6.1 kHz resolution
_NOISE_BW = 2e6             # Noise bandwidth below signal (Hz)
_NOISE_OFFSET = 1e6         # Offset from signal band to noise band (Hz)


def _db10(x: np.ndarray | float) -> np.ndarray | float:
    """Power → dB."""
    scalar_in = np.isscalar(x)
    x_arr = np.asarray(x, dtype=np.float64)
    x_arr = np.where(x_arr < _EPS, _EPS, x_arr)
    result = 10.0 * np.log10(x_arr)
    if scalar_in:
        return float(result)
    return result


def _pwelch_centered(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD with frequency centred on zero."""
    # Clamp nfft to signal length; ensure nperseg <= nfft
    nfft = min(nfft, len(x))
    nperseg = max(8, min(nfft, int(round(len(x) / 10))))
    window = get_window("hamming", nperseg, fftbins=True)
    f, pxx = welch(
        x, fs=fs, window=window, nperseg=nperseg,
        noverlap=None, nfft=nfft, detrend=False,
        return_onesided=False, scaling="density",
    )
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    return pxx, f


class SNRResult:
    """Container for SNR estimation results."""

    __slots__ = (
        "snr_db", "f1", "f2", "f_center",
        "idx1", "idx2", "idx3", "idx4",
        "p_signal", "p_noise",
        "psd_full", "freq_full",   # raw PSD before compensation
        "psd_comp", "freq_comp",   # compensated PSD
        "psd_ds", "freq_ds",      # downsampled PSD
    )

    def __init__(
        self,
        snr_db: float,
        f1: float, f2: float,
        idx1: int, idx2: int, idx3: int, idx4: int,
        p_signal: float, p_noise: float,
        psd_full: np.ndarray, freq_full: np.ndarray,
        psd_comp: np.ndarray, freq_comp: np.ndarray,
        psd_ds: np.ndarray, freq_ds: np.ndarray,
    ):
        self.snr_db = snr_db
        self.f1 = f1
        self.f2 = f2
        self.f_center = (f1 + f2) / 2.0
        self.idx1 = idx1
        self.idx2 = idx2
        self.idx3 = idx3
        self.idx4 = idx4
        self.p_signal = p_signal
        self.p_noise = p_noise
        self.psd_full = psd_full
        self.freq_full = freq_full
        self.psd_comp = psd_comp
        self.freq_comp = freq_comp
        self.psd_ds = psd_ds
        self.freq_ds = freq_ds


class SNREstimator:
    """OFDM UAV signal SNR estimator.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    bw : float, optional
        Estimated OFDM signal bandwidth in Hz (default 20 MHz).
    nfft : int, optional
        FFT size for Welch PSD. If None, auto-sized from signal length.
    noise_bw : float, optional
        Noise bandwidth in Hz below signal band (default 1 MHz).
    noise_offset : float, optional
        Offset from signal lower edge to noise band centre (default 0.75 MHz).
    """

    def __init__(
        self,
        fs: float,
        bw: float = _DEFAULT_BW,
        nfft: int | None = None,
        noise_bw: float = _NOISE_BW,
        noise_offset: float = _NOISE_OFFSET,
    ):
        self.fs = fs
        self.bw = bw
        self.nfft = nfft or _DEFAULT_NFFT
        self.noise_bw = noise_bw
        self.noise_offset = noise_offset

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def estimate(self, signal: np.ndarray) -> SNRResult:
        """Run full SNR estimation pipeline.

        Args:
            signal: Complex IQ baseband signal.

        Returns:
            SNRResult with all intermediate data for visualisation.
        """
        fs, bw = self.fs, self.bw
        nfft = min(self.nfft, len(signal))

        # ── Step 1: find signal band edges ───────────────────────────────
        f1, f2, psd_full, freq_full = self._find_band_edges(signal, fs, nfft, bw)

        # ── Step 2: frequency compensation ───────────────────────────────
        f_center = (f1 + f2) / 2.0
        sig_comp, psd_comp, freq_comp = self._compensate(signal, fs, f_center)

        # ── Step 3: downsample + find FFT bin indices ───────────────────
        refs = fs / 2.0
        re_nfft = nfft // 2
        sig_ds, psd_ds, freq_ds = self._downsample(sig_comp, refs, re_nfft)

        idx1, idx2, idx3, idx4 = self._find_indices(
            f1, f2, f_center, freq_ds, re_nfft, refs
        )

        # ── Step 4: SNR calculation ─────────────────────────────────────
        snr_db, p_sig, p_nos = self._calc_snr(sig_ds, re_nfft, idx1, idx2, idx3, idx4)

        return SNRResult(
            snr_db=snr_db,
            f1=f1, f2=f2,
            idx1=idx1, idx2=idx2, idx3=idx3, idx4=idx4,
            p_signal=p_sig, p_noise=p_nos,
            psd_full=psd_full, freq_full=freq_full,
            psd_comp=psd_comp, freq_comp=freq_comp,
            psd_ds=psd_ds, freq_ds=freq_ds,
        )

    # ------------------------------------------------------------------ #
    #  Internal steps                                                     #
    # ------------------------------------------------------------------ #

    def _find_band_edges(
        self, x: np.ndarray, fs: float, nfft: int, bw: float
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        """Step 1: find signal band edges via PSD threshold crossing."""
        nfft = min(nfft, len(x))
        psd, fvec = _pwelch_centered(x, fs, nfft)
        psd_db_raw = _db10(psd)

        # Noise floor estimate (avoid edges where spectral leakage occurs)
        margin = max(10, nfft // 20)
        core = psd_db_raw[margin:-margin] if len(psd_db_raw) > 2 * margin else psd_db_raw
        noise_floor = float(np.percentile(core, 15))

        # Threshold = noise floor + 6 dB
        threshold = noise_floor + 6.0
        above = psd_db_raw > threshold

        # Find first and last index above threshold
        if not np.any(above):
            # Fallback: use argmax
            pk = int(np.argmax(psd_db_raw))
            bw_nfft = max(1, int(round(nfft * (bw / fs) * 0.9)))
            f1 = float(fvec[max(0, pk - bw_nfft // 2)])
            f2 = float(fvec[min(pk + bw_nfft // 2, nfft - 1)])
            return f1, f2, psd_db_raw, fvec

        idx_above = np.where(above)[0]
        f1 = float(fvec[idx_above[0]])
        f2 = float(fvec[idx_above[-1]])
        return f1, f2, psd_db_raw, fvec

    def _compensate(
        self, x: np.ndarray, fs: float, f_center: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 2: down-convert to baseband and compute its PSD.

        ``psd_comp`` is plotted against a frequency axis by the SNR view, so it
        must be a real (frequency-domain) PSD — not the time-domain power
        ``|sig|**2``. Use the same Welch helper and linear-power convention as
        ``_downsample`` / ``psd_ds`` (also plotted directly), which yields a
        zero-centred baseband axis spanning [-fs/2, fs/2).
        """
        n = len(x)
        t = np.arange(n, dtype=np.float64) / fs
        sig = x * np.exp(-1j * 2.0 * np.pi * f_center * t)
        psd_comp, freq_comp = _pwelch_centered(sig, fs, min(self.nfft, n))
        return sig, psd_comp, freq_comp

    def _downsample(
        self, sig: np.ndarray, refs: float, re_nfft: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Step 3: 2× downsample and compute PSD."""
        sig_ds = resample_poly(sig, up=1, down=2)
        psd_ds, fvec_ds = _pwelch_centered(sig_ds, refs, re_nfft)
        return sig_ds, psd_ds, fvec_ds

    def _find_indices(
        self,
        f1: float, f2: float, f_center: float,
        freq_ds: np.ndarray, re_nfft: int, refs: float,
    ) -> tuple[int, int, int, int]:
        """Step 3b: map Hz edges to FFT bin indices."""
        f11 = f1 - f_center
        f22 = f2 - f_center

        idx1 = int(np.clip(np.argmin(np.abs(freq_ds - f11)), 0, re_nfft - 1))
        idx2 = int(np.clip(np.argmin(np.abs(freq_ds - f22)), 0, re_nfft - 1))

        bw_noise_nfft = int(round(re_nfft * (self.noise_bw / refs)))
        idx4 = idx1 - int(round(re_nfft * (self.noise_offset / refs)))
        idx3 = idx4 - bw_noise_nfft

        idx1, idx2 = min(idx1, idx2), max(idx1, idx2)
        idx3, idx4 = min(idx3, idx4), max(idx3, idx4)

        idx1 = int(np.clip(idx1, 0, re_nfft - 1))
        idx2 = int(np.clip(idx2, 0, re_nfft - 1))
        idx3 = int(np.clip(idx3, 0, re_nfft - 1))
        idx4 = int(np.clip(idx4, 0, re_nfft - 1))

        return idx1, idx2, idx3, idx4

    def _calc_snr(
        self, sig_ds: np.ndarray, re_nfft: int,
        idx1: int, idx2: int, idx3: int, idx4: int,
    ) -> tuple[float, float, float]:
        """Step 4: compute SNR from FFT bins."""
        fft_arr = np.fft.fftshift(np.fft.fft(sig_ds, re_nfft) / re_nfft)

        # Indices are already clamped to [0, re_nfft-1] by _find_indices, which is
        # exactly the length of fft_arr — no further clamping (the old min(idx,
        # len(sig_ds)-1) used the wrong bound, since fft_arr has re_nfft bins).
        p_sig = float(np.mean(np.abs(fft_arr[idx1:idx2 + 1]) ** 2))
        p_nos = float(np.mean(np.abs(fft_arr[idx3:idx4 + 1]) ** 2))

        lin = (p_sig - p_nos) / max(p_nos, _EPS)
        snr_db = float(_db10(max(lin, _EPS)))
        return snr_db, p_sig, p_nos
