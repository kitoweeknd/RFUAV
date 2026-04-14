"""
Help Panel
~~~~~~~~~
|
|Provides collapsible help panel and parameter descriptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.core.ui_style import get_color


class HelpPanel(QWidget):
    """Help Panel

    Provides parameter descriptions and usage guide.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(16)
        content_layout.setContentsMargins(16, 16, 16, 16)

        # Add each section of help content
        content_layout.addWidget(self._create_quick_start())
        content_layout.addWidget(self._create_channel_help())
        content_layout.addWidget(self._create_parameter_help())
        content_layout.addWidget(self._create_shortcuts_help())

        # Add stretch
        content_layout.addStretch()

        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_quick_start(self) -> QWidget:
        """Quick Start"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        title = QLabel("Quick Start")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {get_color('text_primary')};")
        layout.addWidget(title)

        steps = [
            "1. Click 'Load' or use Ctrl+O to load IQ file",
            "2. Configure channel effects in left panel (enable desired channels and adjust parameters)",
            "3. Click 'Preview' or press F5 to view processing results",
            "4. Use view switch to view Time/PSD/STFT/Residual spectrum",
            "5. Export IQ data or spectrum to save results",
        ]

        for step in steps:
            label = QLabel(step)
            label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 12px;")
            layout.addWidget(label)

        return group

    def _create_channel_help(self) -> QWidget:
        """Channel Description"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        title = QLabel("Channel Effects")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {get_color('text_primary')};")
        layout.addWidget(title)

        channels = [
            ("AWGN", "Additive Gaussian White Noise, simulates thermal noise. Smaller SNR means more noise."),
            ("CFO", "Carrier Frequency Offset, simulates oscillator mismatch or Doppler effect."),
            ("Fading", "Rayleigh/Rician fading, simulates multipath effect. Rayleigh for NLOS, Rician for LOS."),
            ("Multipath", "Multipath fading, simulates signal arriving via multiple paths. Parameters include max delay, number of paths, and attenuation coefficient."),
            ("Phase Noise", "Phase noise, simulates oscillator phase jitter."),
            ("IQ Imbalance", "IQ imbalance, simulates I/Q branch amplitude and phase mismatch."),
            ("Crop+TimeShift", "Random cropping and time shift, simulates data truncation and sync errors."),
            ("Doppler", "Doppler shift and spread, simulates frequency variation due to high-speed movement."),
        ]

        for name, desc in channels:
            header = QLabel(f"<b>{name}</b>")
            header.setStyleSheet(f"color: {get_color('accent')}; font-size: 13px;")
            layout.addWidget(header)

            text = QLabel(desc)
            text.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
            text.setWordWrap(True)
            layout.addWidget(text)

        return group

    def _create_parameter_help(self) -> QWidget:
        """Parameter Description"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        title = QLabel("Key Parameters")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {get_color('text_primary')};")
        layout.addWidget(title)

        params = [
            ("SNR (dB)", "Signal-to-noise ratio, -50~50 dB. 15 dB means signal power is ~31.6x noise power."),
            ("CFO (Hz)", "Carrier frequency offset, recommended not to exceed fs/4."),
            ("K Factor", "K factor for Rician fading, ratio of LOS power to scattered power. K=0 equivalent to Rayleigh, K→∞ no fading."),
            ("Block Size", "Coherent block length for fading simulation in samples. Channel parameters stay constant within each block."),
            ("Delay (samples)", "Maximum delay spread of multipath. At fs=100 MHz, 20 samples = 0.2 µs."),
            ("Num Paths", "Number of independent multipath. Each path has different delay and attenuation."),
        ]

        for name, desc in params:
            header = QLabel(f"<b>{name}</b>")
            header.setStyleSheet(f"color: {get_color('accent')}; font-size: 13px;")
            layout.addWidget(header)

            text = QLabel(desc)
            text.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
            text.setWordWrap(True)
            layout.addWidget(text)

        return group

    def _create_shortcuts_help(self) -> QWidget:
        """Keyboard Shortcuts"""
        group = QWidget()
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        title = QLabel("Keyboard Shortcuts")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {get_color('text_primary')};")
        layout.addWidget(title)

        shortcuts = [
            ("Ctrl+O", "Load IQ file"),
            ("Ctrl+S", "Export IQ data"),
            ("Ctrl+E", "Export spectrum"),
            ("Ctrl+T", "Toggle theme"),
            ("F5", "Preview"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Ctrl+R", "Reset parameters"),
            ("Space", "Play/Pause animation"),
            ("R", "Reset spectrum zoom"),
        ]

        for key, desc in shortcuts:
            row = QWidget()
            row_layout = QVBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)

            key_label = QLabel(f"<b>{key}</b>")
            key_label.setStyleSheet(f"color: {get_color('accent')}; font-family: monospace;")

            desc_label = QLabel(desc)
            desc_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")

            row_layout.addWidget(key_label)
            row_layout.addWidget(desc_label)

            layout.addWidget(row)

        return group


class HelpDialog(QDialog):
    """Help Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help")
        self.setMinimumSize(600, 500)
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)

        # Help panel
        help_panel = HelpPanel()
        layout.addWidget(help_panel)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class AboutDialog(QDialog):
    """About Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About")
        self.setFixedSize(400, 300)
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        # Title
        title = QLabel("RFUAV ChannelSim Viewer")
        title.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {get_color('accent')};
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Version
        version = QLabel("Version 1.3.0")
        version.setStyleSheet(f"color: {get_color('text_secondary')};")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        # Date
        date = QLabel("2026-04-09")
        date.setStyleSheet(f"color: {get_color('text_secondary')};")
        date.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(date)

        # Description
        desc = QLabel(
            "Desktop visualization tool for wireless communication signal channel simulation and spectrum analysis."
        )
        desc.setStyleSheet(f"color: {get_color('text_primary')};")
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # License
        license = QLabel("Licensed under MIT open source license")
        license.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        license.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(license)

        # Add stretch
        layout.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class ParameterTooltip:
    """Parameter Tooltip

    Provides tooltip content for each parameter.
    """

    TOOLTIPS = {
        "awgn.snr_db": {
            "name": "SNR (dB)",
            "description": "Signal-to-noise ratio, ratio of signal power to noise power (decibels).",
            "range": "Range: -50 ~ 50 dB",
            "typical": "Typical: 10-30 dB (communication link)",
            "standard": "Reference: 3GPP typically requires ≥ 20 dB",
        },
        "cfo.cfo_hz": {
            "name": "CFO (Hz)",
            "description": "Carrier frequency offset, difference between transmit and receive carrier frequencies.",
            "range": "Range: 0 ~ fs/4 Hz",
            "typical": "Typical: 5-50 kHz (oscillator accuracy)",
            "standard": "Reference: Crystal oscillator accuracy typically ±10-50 ppm",
        },
        "fading.k_factor": {
            "name": "K Factor",
            "description": "K factor for Rician fading, ratio of LOS power to scattered power.",
            "range": "Range: 0 ~ 100",
            "typical": "K=0: Rayleigh (NLOS); K→∞: No fading",
            "standard": "Indoor: K=3-7; Outdoor LOS: K=5-20",
        },
        "fading.block_size": {
            "name": "Block Size",
            "description": "Coherent block size for fading in samples. Channel parameters remain constant within each block.",
            "range": "Range: 64 ~ 2048 samples",
            "typical": "Typical: 128-512 samples",
            "standard": "Block size should be smaller than samples corresponding to coherence time",
        },
        "multipath.max_delay_samp": {
            "name": "Max Delay (samples)",
            "description": "Maximum delay spread of multipath signal (in samples).",
            "range": "Range: 1 ~ 200 samples",
            "typical": "At fs=100 MHz, 20 samples = 0.2 µs = 60 m",
            "standard": "Urban environments can reach hundreds of meters, rural smaller",
        },
        "multipath.n_paths": {
            "name": "Num Paths",
            "description": "Number of independent multipath. Each path has different delay and attenuation.",
            "range": "Range: 1 ~ 20",
            "typical": "Typical: 3-8 significant paths",
            "standard": "TDL (Tapped Delay Line) channel model",
        },
        "phase_noise.phase_std_per_sample": {
            "name": "Phase Std Dev (rad)",
            "description": "Standard deviation of random phase perturbation per sample (Wiener process increment).",
            "range": "Range: 0 ~ 0.1 rad",
            "typical": "Typical: 0.001-0.01 rad",
            "standard": "Refer to oscillator phase noise specifications",
        },
        "iq_imbalance.gain_imbalance_db": {
            "name": "Amplitude Imbalance (dB)",
            "description": "Amplitude difference between I and Q channels.",
            "range": "Range: -10 ~ 10 dB",
            "typical": "Typical: < 1 dB (high quality hardware)",
            "standard": "3GPP: Typically requires < 0.5 dB",
        },
        "iq_imbalance.phase_imbalance_deg": {
            "name": "Phase Imbalance (deg)",
            "description": "90° phase error between I and Q channels.",
            "range": "Range: -30 ~ 30 degrees",
            "typical": "Typical: < 3° (high quality hardware)",
            "standard": "3GPP: Typically requires < 2°",
        },
        "crop_time_shift.min_keep_ratio": {
            "name": "Min Keep Ratio",
            "description": "Ratio of signal length retained after cropping.",
            "range": "Range: 0.1 ~ 1.0",
            "typical": "Typical: 0.7-0.9",
            "standard": "1.0 means no cropping",
        },
        "crop_time_shift.max_shift": {
            "name": "Max Shift (samples)",
            "description": "Maximum range of time shift (in samples).",
            "range": "Range: 0 ~ 5000 samples",
            "typical": "Typical: 100-500 samples",
            "standard": "Should be less than coherence time",
        },
        "doppler.doppler_shift_hz": {
            "name": "Doppler Shift (Hz)",
            "description": "Frequency offset due to relative motion.",
            "range": "Range: -100000 ~ 100000 Hz",
            "typical": "At v=100 km/h, f=2 GHz, f_d ≈ 185 Hz",
            "standard": "Reference: f_d = v * f_c / c",
        },
        "doppler.doppler_spread_hz": {
            "name": "Doppler Spread (Hz)",
            "description": "Frequency broadening due to multipath.",
            "range": "Range: 0 ~ 10000 Hz",
            "typical": "Typical: 10-500 Hz (depending on environment)",
            "standard": "Commonly used in Jakes model",
        },
    }

    @classmethod
    def get_tooltip(cls, param_key: str) -> str:
        """Get tooltip for parameter

        Args:
            param_key: Parameter key, e.g., "awgn.snr_db"

        Returns:
            Tooltip text
        """
        info = cls.TOOLTIPS.get(param_key)
        if not info:
            return ""

        parts = [
            f"<b>{info['name']}</b>",
            "",
            info["description"],
            "",
            f"<font color='gray'>{info['range']}</font>",
            f"<font color='gray'>{info['typical']}</font>",
            f"<font color='gray'>{info['standard']}</font>",
        ]

        return "<br/>".join(parts)
