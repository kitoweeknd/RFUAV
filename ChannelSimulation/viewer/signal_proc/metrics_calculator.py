"""
Signal Quality Metrics Calculator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates signal quality metrics such as EVM, ACLR, and crest factor.
"""

from __future__ import annotations

import numpy as np


class MetricsCalculator:
    """Signal quality metrics calculator"""

    def __init__(self):
        self._eps = 1e-12

    def _align_signals(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align ``degraded`` to ``reference`` by their best cross-correlation lag.

        Stages such as ``crop_time_shift`` translate the signal in time, so a
        sample-wise ``degraded - reference`` would measure the shift instead of
        the channel degradation. We estimate the integer lag that maximises the
        cross-correlation, undo it, and return the overlapping region. When the
        signals are already aligned the estimated lag is 0 and this is a no-op.
        """
        n = min(len(reference), len(degraded))
        if n == 0:
            return reference[:0], degraded[:0]
        a = reference[:n]
        b = degraded[:n]

        # FFT-based cross-correlation. Zero-pad to 2n to get linear (not
        # circular) correlation; the argmax gives the lag of b relative to a.
        size = 1 << int(np.ceil(np.log2(2 * n)))
        fa = np.fft.fft(a, size)
        fb = np.fft.fft(b, size)
        xcorr = np.fft.ifft(fb * np.conj(fa))
        mag = np.abs(xcorr)
        # Lags [0, n) are positive delays; the tail maps to negative delays.
        peak = int(np.argmax(mag))
        lag = peak if peak < n else peak - size

        if lag > 0:      # degraded delayed: drop its leading `lag` samples
            a = a[: n - lag]
            b = b[lag:]
        elif lag < 0:    # degraded advanced: drop reference's leading samples
            a = a[-lag:]
            b = b[: n + lag]
        return a, b

    def calculate_all(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
        fs: float,
    ) -> dict[str, float]:
        """Calculate all metrics

        Args:
            reference: Original reference signal
            degraded: Processed signal
            fs: Sampling rate

        Returns:
            Dictionary containing various metrics
        """
        metrics = {}

        # Align once and share between EVM and SNR (both difference the signals);
        # avoids running the FFT cross-correlation twice per call.
        ref_aligned, deg_aligned = self._align_signals(reference, degraded)

        metrics["evm"] = self._evm_from_aligned(ref_aligned, deg_aligned)
        metrics["crest_factor"] = self.calculate_crest_factor(degraded)
        metrics["aclr_main"] = self.calculate_acl_main(degraded, fs)
        metrics["aclr_lower"] = self.calculate_acl_lower(degraded, fs)
        metrics["aclr_upper"] = self.calculate_acl_upper(degraded, fs)
        metrics["power_dbfs"] = self.calculate_power_dbfs(degraded)
        metrics["snr"] = self._snr_from_aligned(ref_aligned, deg_aligned)

        return metrics

    def calculate_evm(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> float:
        """Calculate EVM (Error Vector Magnitude)

        EVM = sqrt(P_error / P_reference) * 100%

        Args:
            reference: Original signal
            degraded: Processed signal

        Returns:
            EVM percentage
        """
        # _align_signals also trims to the common length, so no separate
        # length-matching step is needed here.
        reference, degraded = self._align_signals(reference, degraded)
        return self._evm_from_aligned(reference, degraded)

    def _evm_from_aligned(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """EVM for already time-aligned, equal-length signals."""
        if len(reference) == 0:
            return 100.0

        error = degraded - reference
        p_error = np.mean(np.abs(error) ** 2)
        p_reference = np.mean(np.abs(reference) ** 2)

        if p_reference < self._eps:
            return 100.0

        evm = np.sqrt(p_error / p_reference) * 100
        return float(evm)

    def calculate_crest_factor(self, signal: np.ndarray) -> float:
        """Calculate Crest Factor

        Crest Factor = 10 * log10(P_peak / P_avg)

        Args:
            signal: Input signal

        Returns:
            Crest factor (dB)
        """
        if len(signal) == 0:
            return 0.0

        # Peak power
        peak_amplitude = np.max(np.abs(signal))
        p_peak = peak_amplitude ** 2

        # Average power
        p_avg = np.mean(np.abs(signal) ** 2)

        if p_avg < self._eps:
            return 0.0

        crest_factor = 10 * np.log10(p_peak / p_avg)
        return float(crest_factor)

    def calculate_acl_main(self, signal: np.ndarray, fs: float) -> float:
        """Calculate ACLR main channel power ratio

        Args:
            signal: Input signal
            fs: Sampling rate

        Returns:
            Main channel power (dBc)
        """
        return self._calculate_channel_power(signal, fs, center_ratio=0.0, width_ratio=0.1)

    def calculate_acl_lower(self, signal: np.ndarray, fs: float) -> float:
        """Calculate ACLR lower adjacent channel power

        Args:
            signal: Input signal
            fs: Sampling rate

        Returns:
            Lower adjacent channel power (dBc)
        """
        return self._calculate_channel_power(signal, fs, center_ratio=-0.15, width_ratio=0.05)

    def calculate_acl_upper(self, signal: np.ndarray, fs: float) -> float:
        """Calculate ACLR upper adjacent channel power

        Args:
            signal: Input signal
            fs: Sampling rate

        Returns:
            Upper adjacent channel power (dBc)
        """
        return self._calculate_channel_power(signal, fs, center_ratio=0.15, width_ratio=0.05)

    def _calculate_channel_power(
        self,
        signal: np.ndarray,
        fs: float,
        center_ratio: float,
        width_ratio: float,
    ) -> float:
        """Calculate power in specified frequency band

        Args:
            signal: Input signal
            fs: Sampling rate
            center_ratio: Center frequency ratio relative to fs/2
            width_ratio: Bandwidth ratio relative to fs/2

        Returns:
            Power (dBc)
        """
        if len(signal) == 0:
            return -np.inf

        # FFT — keep the full two-sided spectrum. For complex IQ signals the
        # negative frequencies are physically meaningful (they carry the lower
        # adjacent channel), so truncating to the positive half would make any
        # negative center_ratio band never match a bin.
        n = len(signal)
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1 / fs)

        # Calculate power spectral density over the full spectrum
        psd = np.abs(spectrum) ** 2

        # Determine frequency range
        max_freq = fs / 2
        center = center_ratio * max_freq
        width = width_ratio * max_freq
        freq_min = center - width / 2
        freq_max = center + width / 2

        # Integrate power in the frequency band
        mask = (freqs >= freq_min) & (freqs < freq_max)
        power = np.sum(psd[mask])

        # Normalize
        total_power = np.sum(psd)
        if total_power < self._eps:
            return -np.inf

        power_ratio = power / total_power
        if power_ratio < self._eps:
            return -100.0

        return float(10 * np.log10(power_ratio))

    def calculate_power_dbfs(self, signal: np.ndarray) -> float:
        """Calculate signal power (dBFS, Full Scale)

        Args:
            signal: Input signal

        Returns:
            Power (dBFS)
        """
        if len(signal) == 0:
            return -np.inf

        # Assume full scale is 1.0 (normalized signal)
        p_signal = np.mean(np.abs(signal) ** 2)
        p_fullscale = 1.0

        if p_signal < self._eps:
            return -np.inf

        return float(10 * np.log10(p_signal / p_fullscale))

    def calculate_snr(
        self,
        reference: np.ndarray,
        degraded: np.ndarray,
    ) -> float:
        """Calculate SNR (Signal-to-Noise Ratio)

        Args:
            reference: Original signal
            degraded: Processed signal

        Returns:
            SNR (dB)
        """
        # _align_signals also trims to the common length.
        reference, degraded = self._align_signals(reference, degraded)
        return self._snr_from_aligned(reference, degraded)

    def _snr_from_aligned(self, reference: np.ndarray, degraded: np.ndarray) -> float:
        """SNR for already time-aligned, equal-length signals."""
        if len(reference) == 0:
            return 100.0

        # Signal power
        p_signal = np.mean(np.abs(reference) ** 2)

        # Noise power (difference between before and after processing)
        error = degraded - reference
        p_noise = np.mean(np.abs(error) ** 2)

        if p_noise < self._eps:
            return 100.0

        snr = p_signal / p_noise
        return float(10 * np.log10(snr))

    def calculate_evm_percent(self, evm: float) -> str:
        """Convert EVM value to quality assessment

        Args:
            evm: EVM value (percentage)

        Returns:
            Quality assessment string
        """
        if evm <= 5:
            return "Excellent"
        elif evm <= 10:
            return "Good"
        elif evm <= 20:
            return "Fair"
        else:
            return "Poor"

    def calculate_acl_quality(self, aclr: float) -> str:
        """Convert ACLR value to quality assessment

        Args:
            aclr: ACLR value (dBc)

        Returns:
            Quality assessment string
        """
        if aclr >= 50:
            return "Excellent"
        elif aclr >= 45:
            return "Good"
        elif aclr >= 35:
            return "Fair"
        else:
            return "Poor"
