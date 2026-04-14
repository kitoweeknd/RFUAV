"""
CFO Estimation and Compensation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Provides CFO (Carrier Frequency Offset) estimation and compensation functionality.
Supports two modes: FFT peak method and cross-correlation method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CFOEstimateResult:
    """CFO estimation result"""
    estimated_cfo_hz: float
    true_cfo_hz: float
    error_hz: float
    method: str
    confidence: float  # 0-1, confidence level


class CFOEstimator:
    """CFO Estimator

    Supports two estimation methods:
    - FFT Peak Method: Compute signal spectrum, find peak frequency
    - Cross-correlation Method: Estimate CFO based on phase difference between first and second halves
    """

    def __init__(self, fs: float = 100_000_000):
        self._fs = fs

    @property
    def fs(self) -> float:
        return self._fs

    @fs.setter
    def fs(self, value: float) -> None:
        self._fs = float(value)

    def estimate_fft_peak(
        self,
        signal: np.ndarray,
        search_range: tuple[float, float] | None = None,
    ) -> CFOEstimateResult:
        """Estimate CFO using FFT peak method

        Args:
            signal: Input IQ signal
            search_range: Search range (min, max) in Hz, None means full range

        Returns:
            Estimation result
        """
        n = len(signal)

        # Apply window
        window = np.hanning(n)
        windowed = signal * window

        # FFT
        spectrum = np.fft.fft(windowed)
        freqs = np.fft.fftfreq(n, 1 / self._fs)

        # Take positive frequency part
        half_n = n // 2
        freqs_pos = freqs[:half_n]
        spectrum_pos = spectrum[:half_n]

        # Compute magnitude spectrum
        magnitude = np.abs(spectrum_pos)

        # Determine search range
        if search_range:
            freq_min, freq_max = search_range
            mask = (freqs_pos >= freq_min) & (freqs_pos <= freq_max)
        else:
            mask = np.ones_like(freqs_pos, dtype=bool)

        # Find peak
        masked_magnitude = np.where(mask, magnitude, 0)
        peak_idx = np.argmax(masked_magnitude)
        peak_freq = freqs_pos[peak_idx]

        # Compute confidence (ratio of peak to mean)
        mean_mag = np.mean(magnitude[mask])
        peak_mag = magnitude[peak_idx]
        confidence = min(1.0, peak_mag / (mean_mag * 10))

        return CFOEstimateResult(
            estimated_cfo_hz=float(peak_freq),
            true_cfo_hz=0.0,  # Unknown
            error_hz=0.0,
            method="fft_peak",
            confidence=float(confidence),
        )

    def estimate_correlation(
        self,
        signal: np.ndarray,
        segment_ratio: float = 0.5,
    ) -> CFOEstimateResult:
        """Estimate CFO using cross-correlation method

        Estimates CFO based on phase difference between first and second halves of the signal.
        Suitable for continuous wave (CW) signals.

        Args:
            signal: Input IQ signal
            segment_ratio: Split ratio (proportion of first half to total length)

        Returns:
            Estimation result
        """
        n = len(signal)
        seg_len = int(n * segment_ratio)

        # Split signal
        seg1 = signal[:seg_len]
        seg2 = signal[seg_len:2 * seg_len]

        # FFT
        n_fft = seg_len
        spec1 = np.fft.fft(seg1, n=n_fft)
        spec2 = np.fft.fft(seg2, n=n_fft)

        # Cross-correlation (computed in frequency domain)
        correlation = spec1 * np.conj(spec2)

        # Inverse FFT to get correlation function
        corr = np.fft.ifft(correlation)

        # Find maximum correlation point
        peak_idx = np.argmax(np.abs(corr))

        # Convert to frequency offset
        # Position of correlation peak corresponds to time delay, which corresponds to frequency offset
        delta_t = peak_idx / self._fs
        cfo_hz = peak_idx / (seg_len / self._fs) if seg_len > 0 else 0

        # Alternative calculation: use phase difference
        phase_diff = np.angle(correlation[peak_idx])
        cfo_phase = phase_diff * self._fs / (2 * np.pi * seg_len * (1 - segment_ratio))

        # Average of two methods
        estimated_cfo = (cfo_hz + cfo_phase) / 2

        # Compute confidence
        max_corr = np.abs(corr[peak_idx])
        mean_corr = np.mean(np.abs(corr))
        confidence = min(1.0, max_corr / (mean_corr * 5))

        return CFOEstimateResult(
            estimated_cfo_hz=float(estimated_cfo),
            true_cfo_hz=0.0,
            error_hz=0.0,
            method="correlation",
            confidence=float(confidence),
        )

    def compensate(
        self,
        signal: np.ndarray,
        cfo_hz: float,
    ) -> np.ndarray:
        """Compensate CFO

        Args:
            signal: Input IQ signal
            cfo_hz: Estimated CFO value (Hz)

        Returns:
            Compensated signal
        """
        n = len(signal)
        t = np.arange(n) / self._fs

        # Generate phase rotation factor
        phase = -2 * np.pi * cfo_hz * t
        rotation = np.exp(1j * phase)

        return signal * rotation


class CFOEstimatorDialog:
    """CFO Estimator Dialog

    Used to display CFO estimation and compensation results in UI.
    """

    def __init__(self, parent=None):
        self._parent = parent
        self._estimator = CFOEstimator()
        self._result: Optional[CFOEstimateResult] = None

    def estimate_and_compare(
        self,
        original: np.ndarray,
        degraded: np.ndarray,
        fs: float,
        true_cfo_hz: float = 0.0,
    ) -> dict:
        """Estimate and compare results

        Args:
            original: Original signal
            degraded: Signal with CFO
            fs: Sampling rate
            true_cfo_hz: True CFO value (if known)

        Returns:
            Dictionary containing estimation results
        """
        self._estimator.fs = fs

        # FFT peak method
        result_fft = self._estimator.estimate_fft_peak(degraded)
        if true_cfo_hz != 0.0:
            result_fft.true_cfo_hz = true_cfo_hz
            result_fft.error_hz = result_fft.estimated_cfo_hz - true_cfo_hz

        # Cross-correlation method
        result_corr = self._estimator.estimate_correlation(degraded)
        if true_cfo_hz != 0.0:
            result_corr.true_cfo_hz = true_cfo_hz
            result_corr.error_hz = result_corr.estimated_cfo_hz - true_cfo_hz

        # Compensation
        compensated = self._estimator.compensate(degraded, result_fft.estimated_cfo_hz)

        return {
            "fft_peak": {
                "estimated": result_fft.estimated_cfo_hz,
                "true": result_fft.true_cfo_hz,
                "error": result_fft.error_hz,
                "confidence": result_fft.confidence,
            },
            "correlation": {
                "estimated": result_corr.estimated_cfo_hz,
                "true": result_corr.true_cfo_hz,
                "error": result_corr.error_hz,
                "confidence": result_corr.confidence,
            },
            "compensated_signal": compensated,
        }
