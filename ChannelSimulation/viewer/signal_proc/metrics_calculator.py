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

        metrics["evm"] = self.calculate_evm(reference, degraded)
        metrics["crest_factor"] = self.calculate_crest_factor(degraded)
        metrics["aclr_main"] = self.calculate_acl_main(degraded, fs)
        metrics["aclr_lower"] = self.calculate_acl_lower(degraded, fs)
        metrics["aclr_upper"] = self.calculate_acl_upper(degraded, fs)
        metrics["power_dbfs"] = self.calculate_power_dbfs(degraded)
        metrics["snr"] = self.calculate_snr(reference, degraded)

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
        if len(reference) != len(degraded):
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

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

        # FFT
        n = len(signal)
        spectrum = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n, 1 / fs)

        # Convert to positive frequencies
        half_n = n // 2
        freqs_pos = freqs[:half_n]
        spectrum_pos = spectrum[:half_n]

        # Calculate power spectral density
        psd = np.abs(spectrum_pos) ** 2

        # Determine frequency range
        max_freq = fs / 2
        center = center_ratio * max_freq
        width = width_ratio * max_freq
        freq_min = center - width / 2
        freq_max = center + width / 2

        # Integrate power in the frequency band
        mask = (freqs_pos >= freq_min) & (freqs_pos < freq_max)
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
        if len(reference) != len(degraded):
            min_len = min(len(reference), len(degraded))
            reference = reference[:min_len]
            degraded = degraded[:min_len]

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
