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
