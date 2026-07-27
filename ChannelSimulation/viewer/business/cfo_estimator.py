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

    Provides a UI for CFO estimation and compensation functionality.
    """

    def __init__(self, parent=None):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QGroupBox, QFormLayout, QTabWidget, QWidget, QTextEdit, QSpinBox
        from PySide6.QtCore import Qt

        self._parent = parent
        self._estimator = CFOEstimator()
        self._original_signal = None
        self._processed_signal = None
        self._fs = 100_000_000

        # Create dialog
        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("CFO Estimator & Compensation")
        self._dialog.setMinimumSize(600, 450)

        layout = QVBoxLayout(self._dialog)

        # Method selection
        method_group = QGroupBox("Estimation Method")
        method_layout = QHBoxLayout(method_group)

        self._method_combo = QComboBox()
        self._method_combo.addItems(["FFT Peak Method", "Cross-Correlation Method"])
        method_layout.addWidget(QLabel("Method:"))
        method_layout.addWidget(self._method_combo)
        method_layout.addStretch()
        layout.addWidget(method_group)

        # Search range settings
        range_group = QGroupBox("Search Range (Optional)")
        range_layout = QFormLayout(range_group)

        self._range_min = QSpinBox()
        self._range_min.setRange(-1000000, 1000000)
        self._range_min.setValue(-500000)
        self._range_min.setSuffix(" Hz")
        self._range_min.setMaximumWidth(150)

        self._range_max = QSpinBox()
        self._range_max.setRange(-1000000, 1000000)
        self._range_max.setValue(500000)
        self._range_max.setSuffix(" Hz")
        self._range_max.setMaximumWidth(150)

        range_layout.addRow("Min Frequency:", self._range_min)
        range_layout.addRow("Max Frequency:", self._range_max)
        layout.addWidget(range_group)

        # Results display
        results_group = QGroupBox("Estimation Results")
        results_layout = QVBoxLayout(results_group)

        # FFT Peak results
        self._fft_result_label = QLabel("FFT Peak: Not estimated")
        self._fft_result_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        results_layout.addWidget(self._fft_result_label)

        # Correlation results
        self._corr_result_label = QLabel("Correlation: Not estimated")
        self._corr_result_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        results_layout.addWidget(self._corr_result_label)

        # True CFO input
        true_layout = QHBoxLayout()
        true_layout.addWidget(QLabel("True CFO (for comparison):"))
        self._true_cfo_input = QSpinBox()
        self._true_cfo_input.setRange(-1000000, 1000000)
        self._true_cfo_input.setValue(15000)
        self._true_cfo_input.setSuffix(" Hz")
        self._true_cfo_input.setMaximumWidth(150)
        true_layout.addWidget(self._true_cfo_input)
        true_layout.addStretch()
        results_layout.addLayout(true_layout)

        layout.addWidget(results_group)

        # Compensation section
        comp_group = QGroupBox("Compensation")
        comp_layout = QVBoxLayout(comp_group)

        self._comp_result_label = QLabel("Compensation status: Not applied")
        self._comp_result_label.setStyleSheet("font-family: monospace; font-size: 12px;")
        comp_layout.addWidget(self._comp_result_label)

        comp_btn_layout = QHBoxLayout()
        self._apply_comp_btn = QPushButton("Apply Compensation")
        self._apply_comp_btn.clicked.connect(self._apply_compensation)
        comp_btn_layout.addWidget(self._apply_comp_btn)
        comp_btn_layout.addStretch()
        comp_layout.addLayout(comp_btn_layout)
        layout.addWidget(comp_group)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        run_btn = QPushButton("Run Estimation")
        run_btn.setObjectName("btnPrimary")
        run_btn.clicked.connect(self._run_estimation)
        btn_layout.addWidget(run_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._dialog.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def set_signals(self, original: "np.ndarray | None", processed: "np.ndarray | None", fs: float) -> None:
        """Set signals for estimation.

        Args:
            original: Original signal without CFO
            processed: Processed signal with CFO
            fs: Sampling rate
        """
        self._original_signal = original
        self._processed_signal = processed
        self._fs = fs
        self._estimator.fs = fs

    def exec(self) -> None:
        """Show dialog."""
        self._dialog.exec()

    def _run_estimation(self) -> None:
        """Run CFO estimation."""
        if self._processed_signal is None or len(self._processed_signal) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self._dialog, "No Signal", "Please load a signal first.")
            return

        signal = self._processed_signal
        true_cfo = float(self._true_cfo_input.value())

        # FFT Peak method
        result_fft = self._estimator.estimate_fft_peak(signal)
        result_fft.true_cfo_hz = true_cfo
        result_fft.error_hz = result_fft.estimated_cfo_hz - true_cfo

        # Correlation method
        result_corr = self._estimator.estimate_correlation(signal)
        result_corr.true_cfo_hz = true_cfo
        result_corr.error_hz = result_corr.estimated_cfo_hz - true_cfo

        # Update labels
        self._fft_result_label.setText(
            f"FFT Peak: Estimated={result_fft.estimated_cfo_hz:.2f} Hz | "
            f"True={result_fft.true_cfo_hz:.2f} Hz | "
            f"Error={result_fft.error_hz:.2f} Hz | "
            f"Confidence={result_fft.confidence:.2%}"
        )

        self._corr_result_label.setText(
            f"Correlation: Estimated={result_corr.estimated_cfo_hz:.2f} Hz | "
            f"True={result_corr.true_cfo_hz:.2f} Hz | "
            f"Error={result_corr.error_hz:.2f} Hz | "
            f"Confidence={result_corr.confidence:.2%}"
        )

    def _apply_compensation(self) -> None:
        """Apply CFO compensation."""
        if self._processed_signal is None or len(self._processed_signal) == 0:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self._dialog, "No Signal", "Please load a signal first.")
            return

        # Use FFT peak estimate
        result = self._estimator.estimate_fft_peak(self._processed_signal)
        compensated = self._estimator.compensate(self._processed_signal, result.estimated_cfo_hz)

        self._comp_result_label.setText(
            f"Compensation applied with estimated CFO: {result.estimated_cfo_hz:.2f} Hz"
        )
