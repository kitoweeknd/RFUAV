"""
Dual-path SNR estimation for OFDM UAV signals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FCME (Forward Consecutive Mean Excision) + CP-based (cyclic prefix
autocorrelation) cross-validated estimator.

Path 1 — FCME (iterative threshold, band-aware, primary)
  → Welch PSD → iterative noise-bin convergence → SNR from linear power

Path 2 — CP correlation (OFDM-specific, blind, no band detection)
  → autocorrelation at lag N_FFT → P_signal = corr, P_noise = total − corr
  → OFDM CP creates a correlation peak that AWGN cannot produce

Cross-validation
  If |FCME − CP| > 10 dB → BAND LOCK WARNING
  (CP correlation is diluted by the CP ratio, so the absolute scale differs
  from FCME; the cross-validation threshold is wider than the 3 dB used for
  M2M4.  What matters is that the two estimators move together — if FCME
  says 30 dB and CP says −30 dB, the band detection failed.)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from scipy.signal import get_window, welch

_EPS = 1e-20


@dataclass
class DualSNRResult:
    snr_fcme_db: float
    snr_cp_db: float
    snr_db: float
    reliable: bool
    band_warning: bool
    method: str
    fcme_iterations: int = 0
    fcme_noise_bins: int = 0
    fcme_signal_bins: int = 0
    psd_linear: Optional[np.ndarray] = None
    freq_axis: Optional[np.ndarray] = None
    signal_mask: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# FCME
# ---------------------------------------------------------------------------

class FCMEEstimator:
    """Iterative FCME noise-floor and SNR estimator with adaptive threshold.

    T_factor=None (default) enables adaptive mode: tries T ∈ [2.5, 2.0, 1.5, 1.0]
    until signal bins > 10 are found.  Pass a float to fix the threshold.

    Parameters
    ----------
    fs : float
    nfft : int, optional
    T_factor : float or None
    max_iter : int
    nperseg : int, optional
    """

    def __init__(self, fs: float, nfft: Optional[int] = None,
                 T_factor: Optional[float] = None, max_iter: int = 20,
                 nperseg: Optional[int] = None):
        self.fs = fs
        self.nfft = nfft
        self.T_factor = T_factor
        self.max_iter = max_iter
        self.nperseg = nperseg

    def _run_fcme(self, signal: np.ndarray, T_factor: float):
        """Run one FCME pass with a fixed T_factor.  Returns (snr_db, diag)."""
        n = len(signal)
        nfft = self.nfft or min(16384, n)
        nperseg = self.nperseg or max(8, min(nfft, int(round(n / 10))))

        window = get_window("hamming", nperseg, fftbins=True)
        f, psd_linear = welch(
            signal, fs=self.fs, window=window,
            nperseg=nperseg, nfft=nfft,
            return_onesided=False, scaling="density", detrend=False,
        )
        f = np.fft.fftshift(f)
        psd_linear = np.fft.fftshift(psd_linear)
        psd_linear = np.maximum(psd_linear, _EPS)
        psd_db = 10.0 * np.log10(psd_linear)

        noise_mask = np.ones(len(psd_db), dtype=bool)
        n_bins = len(psd_db)
        converged = False
        threshold_db = 0.0
        noise_floor_db = 0.0

        for iteration in range(self.max_iter):
            noise_vals = psd_db[noise_mask]
            if len(noise_vals) < 2:
                break
            noise_mean = float(np.mean(noise_vals))
            noise_std = float(np.std(noise_vals))
            threshold_db = noise_mean + T_factor * noise_std
            noise_floor_db = noise_mean
            new_mask = psd_db < threshold_db
            if np.array_equal(new_mask, noise_mask):
                converged = True
                break
            if new_mask.sum() < max(5, n_bins * 0.05):
                threshold_db = noise_mean + T_factor * 0.5 * noise_std
                new_mask = psd_db < threshold_db
                if np.array_equal(new_mask, noise_mask):
                    converged = True
                    break
            if new_mask.sum() < 2:
                # Never let the noise set empty out: with zero noise bins the
                # SNR is undefined (mean of empty slice -> NaN).  Keep the
                # previous mask instead — happens on full-band signals where
                # every bin sits above the shrinking threshold.
                break
            noise_mask = new_mask

        signal_mask = ~noise_mask
        noise_bins = int(noise_mask.sum())
        signal_bins = int(signal_mask.sum())

        if signal_bins == 0:
            return -30.0, {
                "iterations": iteration + 1, "noise_bins": noise_bins,
                "signal_bins": signal_bins, "threshold_db": threshold_db,
                "noise_floor_db": noise_floor_db, "psd_linear": psd_linear,
                "freq_axis": f, "signal_mask": signal_mask,
                "converged": converged, "T_factor": T_factor,
            }

        p_sig = float(np.mean(psd_linear[signal_mask]))
        p_noise = float(np.mean(psd_linear[noise_mask]))
        p_sig_clean = max(p_sig - p_noise, _EPS)
        snr_db = float(10.0 * np.log10(p_sig_clean / max(p_noise, _EPS)))

        return snr_db, {
            "iterations": iteration + 1, "noise_bins": noise_bins,
            "signal_bins": signal_bins, "threshold_db": threshold_db,
            "noise_floor_db": noise_floor_db, "psd_linear": psd_linear,
            "freq_axis": f, "signal_mask": signal_mask,
            "converged": converged, "T_factor": T_factor,
        }

    def estimate(self, signal: np.ndarray):
        if self.T_factor is not None:
            return self._run_fcme(signal, self.T_factor)

        # Adaptive: try progressively lower thresholds until we find signal.
        # A result only counts if the noise set survived too — full-band
        # signals can classify every bin as signal, which leaves the SNR
        # undefined (this produced NaN for DroneRFa RF0 recordings).
        for T in [2.5, 2.0, 1.5, 1.0]:
            snr, diag = self._run_fcme(signal, T)
            n_bins = diag["signal_bins"] + diag["noise_bins"]
            if (diag["signal_bins"] > 10
                    and diag["noise_bins"] >= max(16, int(n_bins * 0.01))
                    and np.isfinite(snr)):
                return snr, diag

        # All thresholds failed (signal occupies nearly the whole band).
        # Percentile-floor fallback: bottom-decile bins define the noise
        # floor; bins >= 6 dB above it are signal.  Biased when the band is
        # truly full, but finite and monotone — flagged in the diagnostics.
        psd = diag.get("psd_linear")
        if psd is not None and len(psd):
            floor = float(np.percentile(psd, 10))
            noise_mask = psd <= floor * (10 ** (3 / 10.0))
            signal_mask = psd > floor * (10 ** (6 / 10.0))
            if noise_mask.sum() >= 2 and signal_mask.sum() > 0:
                p_noise = float(np.mean(psd[noise_mask]))
                p_sig = float(np.mean(psd[signal_mask]))
                snr = float(10.0 * np.log10(
                    max(p_sig - p_noise, _EPS) / max(p_noise, _EPS)))
                diag = dict(diag)
                diag["signal_bins"] = int(signal_mask.sum())
                diag["noise_bins"] = int(noise_mask.sum())
                diag["signal_mask"] = signal_mask
                diag["method_fallback"] = "percentile_floor"
        return snr, diag


# ---------------------------------------------------------------------------
# CP-based SNR estimator
# ---------------------------------------------------------------------------

def estimate_cp_snr(signal: np.ndarray, nfft: int = 2048) -> float:
    x = np.asarray(signal, dtype=np.complex128).ravel()
    n = len(x)
    if n < 2 * nfft:
        return -30.0

    chunk_size = min(200_000, n - nfft)
    corr_sum = 0.0
    corr_count = 0
    pwr_sum = 0.0

    for start in range(0, n - nfft, chunk_size):
        end = min(start + chunk_size, n - nfft)
        a = x[start:end]
        b = x[start + nfft:end + nfft]
        if len(a) < 2:
            break
        corr_sum += float(np.abs(np.sum(a * np.conj(b))))
        corr_count += len(a)
        pwr_sum += float(np.sum(np.abs(a) ** 2))

    if corr_count == 0:
        return -30.0

    avg_corr = corr_sum / corr_count
    avg_pwr = pwr_sum / corr_count

    P_signal = avg_corr
    P_noise = max(avg_pwr - avg_corr, _EPS)
    P_signal = max(P_signal, _EPS)

    snr_db = float(10.0 * np.log10(P_signal / P_noise))
    return max(snr_db, -30.0)


# ---------------------------------------------------------------------------
# Dual-path
# ---------------------------------------------------------------------------

def estimate_snr_dual(
    signal: np.ndarray,
    fs: float = 100e6,
    nfft: Optional[int] = None,
    max_samples: Optional[int] = None,
    warn_threshold_db: float = 10.0,
) -> DualSNRResult:
    snr_cp = estimate_cp_snr(signal)

    sig_fcme = signal[:max_samples] if max_samples else signal
    fcme = FCMEEstimator(fs=fs, nfft=nfft)  # adaptive T_factor
    snr_fcme, diag = fcme.estimate(sig_fcme)

    diff = abs(snr_fcme - snr_cp)
    band_warning = diff > warn_threshold_db

    return DualSNRResult(
        snr_fcme_db=snr_fcme,
        snr_cp_db=snr_cp,
        snr_db=snr_fcme,
        reliable=not band_warning,
        band_warning=band_warning,
        method="fcme+cp_warn" if band_warning else "fcme",
        fcme_iterations=diag["iterations"],
        fcme_noise_bins=diag["noise_bins"],
        fcme_signal_bins=diag["signal_bins"],
        psd_linear=diag["psd_linear"],
        freq_axis=diag["freq_axis"],
        signal_mask=diag["signal_mask"],
    )