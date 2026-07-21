"""
Signal Information Panel
~~~~~~~~~~~~~~~~~~~~~~~~
|
Displays IQ file information and signal quality metric cards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.core.ui_style import get_color
from ChannelSimulation.viewer.core.utils import (
    format_duration,
    format_frequency,
    format_power,
    format_samples,
)
from ChannelSimulation.viewer.ui.collapsible_section import CollapsibleSection


class SignalInfoPanel(QWidget):
    """Signal Information Panel

    Displays IQ file basic information and signal quality metrics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._reset_info()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # File info (collapsible)
        overview_inner = QWidget()
        info_layout = QFormLayout(overview_inner)
        info_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        info_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        info_layout.setHorizontalSpacing(16)
        info_layout.setVerticalSpacing(8)
        info_layout.setContentsMargins(4, 0, 4, 4)

        # File info labels
        self._label_file = QLabel("--")
        self._label_file.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("File:", self._label_file)

        self._label_samples = QLabel("--")
        self._label_samples.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("Samples:", self._label_samples)

        self._label_duration = QLabel("--")
        self._label_duration.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("Duration:", self._label_duration)

        self._label_fs = QLabel("--")
        self._label_fs.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("Sample Rate:", self._label_fs)

        self._label_power = QLabel("--")
        self._label_power.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("Power:", self._label_power)

        self._label_fc = QLabel("--")
        self._label_fc.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        info_layout.addRow("Center Frequency:", self._label_fc)

        self._overview_section = CollapsibleSection(
            "Signal Overview", overview_inner, expanded=True
        )
        layout.addWidget(self._overview_section)

        # Metrics cards (collapsible; default collapsed to leave room for preprocessing)
        metrics_inner = QWidget()
        metrics_layout = QGridLayout(metrics_inner)
        metrics_layout.setSpacing(12)
        metrics_layout.setContentsMargins(4, 0, 4, 4)
        metrics_layout.setColumnStretch(0, 1)
        metrics_layout.setColumnStretch(1, 1)

        # Create metric cards
        self._metrics_cards = {}

        # EVM card
        self._metrics_cards["evm"] = self._create_metric_card(
            "EVM (%)",
            "--",
            "Modulation signal quality assessment",
            metrics_layout,
            row=0, col=0,
        )

        # Crest factor card
        self._metrics_cards["crest_factor"] = self._create_metric_card(
            "Peak Factor",
            "--",
            "Signal dynamic range (dB)",
            metrics_layout,
            row=0, col=1,
        )

        # ACLR card
        self._metrics_cards["aclr"] = self._create_metric_card(
            "ACLR (dBc)",
            "--",
            "Adjacent channel leakage ratio",
            metrics_layout,
            row=1, col=0,
        )

        # Power dBFS card
        self._metrics_cards["power_dbfs"] = self._create_metric_card(
            "Power (dBFS)",
            "--",
            "Signal level (full scale)",
            metrics_layout,
            row=1, col=1,
        )

        # SNR card
        self._metrics_cards["snr"] = self._create_metric_card(
            "SNR (dB)",
            "--",
            "Signal-to-noise ratio",
            metrics_layout,
            row=2, col=0,
        )

        self._metrics_section = CollapsibleSection(
            "Signal Quality Metrics", metrics_inner, expanded=True
        )
        layout.addWidget(self._metrics_section)

    def _create_metric_card(
        self,
        title: str,
        value: str,
        description: str,
        parent_layout: QGridLayout,
        row: int,
        col: int,
    ) -> dict[str, QWidget]:
        """Create a metric card"""
        card = QFrame()
        card.setFrameShape(QFrame.Shape.StyledPanel)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {get_color('bg_secondary')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        card.setMinimumWidth(120)
        card.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        title_label.setWordWrap(False)
        title_label.setMinimumHeight(title_label.sizeHint().height())

        # Value
        value_label = QLabel(value)
        value_label.setStyleSheet(f"""
            color: {get_color('accent')};
            font-size: 18px;
            font-weight: bold;
            font-family: "Consolas", "Courier New", monospace;
        """)
        layout.addWidget(value_label)

        # Description
        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 9px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        parent_layout.addWidget(card, row, col)

        return {"card": card, "title": title_label, "value": value_label, "desc": desc_label}

    def _reset_info(self) -> None:
        """Reset info display"""
        self._label_file.setText("--")
        self._label_samples.setText("--")
        self._label_duration.setText("--")
        self._label_fs.setText("--")
        self._label_power.setText("--")
        self._label_fc.setText("--")

        # Raw (unformatted) signal state, kept so get_info() can round-trip.
        self._raw_info: Dict[str, Any] = {}

        # Tracks whether the metrics section was auto-expanded once already.
        self._metrics_expanded_once = False

        # Reset metric cards
        for key in self._metrics_cards:
            self._metrics_cards[key]["value"].setText("--")

    def update_info(self, info: Dict[str, Any]) -> None:
        """Update signal information

        Args:
            info: Dictionary containing signal information
        """
        # File info
        if "file_path" in info:
            self._raw_info["file_path"] = info["file_path"]
            self._label_file.setText(Path(info["file_path"]).name)

        if "n_samples" in info:
            n_samples = info["n_samples"]
            self._raw_info["n_samples"] = n_samples
            self._label_samples.setText(format_samples(n_samples))

        if "duration" in info:
            duration = info["duration"]
            self._raw_info["duration"] = duration
            self._label_duration.setText(format_duration(duration))

        if "fs" in info:
            fs = info["fs"]
            self._raw_info["fs"] = fs
            self._label_fs.setText(format_frequency(fs))

        if "fc" in info:
            fc = info.get("fc", 0.0)
            self._raw_info["fc"] = fc
            if fc > 0:
                self._label_fc.setText(format_frequency(fc))
            else:
                self._label_fc.setText("Baseband")

        if "power" in info:
            power = info["power"]
            self._raw_info["power"] = power
            if power is not None:
                self._label_power.setText(format_power(power, "dB"))
            else:
                self._label_power.setText("--")

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update signal quality metrics

        Args:
            metrics: Dictionary containing metrics
        """
        # Auto-expand the section only the first time real data arrives, so we
        # do not override the user manually collapsing it on later updates.
        if metrics and not self._metrics_expanded_once and hasattr(self, '_metrics_section'):
            if not self._metrics_section.is_expanded():
                self._metrics_section.set_expanded(True)
            self._metrics_expanded_once = True

        # EVM
        if "evm" in metrics:
            evm = metrics["evm"]
            self._metrics_cards["evm"]["value"].setText(f"{evm:.2f}%")
            # Set color based on threshold
            if evm <= 5:
                color = get_color("success")
            elif evm <= 10:
                color = get_color("accent")
            else:
                color = get_color("danger")
            self._metrics_cards["evm"]["value"].setStyleSheet(f"""
                color: {color};
                font-size: 18px;
                font-weight: bold;
                font-family: "Consolas", "Courier New", monospace;
            """)

        # Crest factor
        if "crest_factor" in metrics:
            cf = metrics["crest_factor"]
            self._metrics_cards["crest_factor"]["value"].setText(f"{cf:.2f} dB")

        # ACLR (using main channel)
        if "aclr_main" in metrics:
            aclr = metrics["aclr_main"]
            self._metrics_cards["aclr"]["value"].setText(f"{aclr:.2f} dBc")

        # Power dBFS
        if "power_dbfs" in metrics:
            power = metrics["power_dbfs"]
            if power > -100:
                self._metrics_cards["power_dbfs"]["value"].setText(f"{power:.2f}")
            else:
                self._metrics_cards["power_dbfs"]["value"].setText("--")

        # SNR
        if "snr" in metrics:
            snr = metrics["snr"]
            self._metrics_cards["snr"]["value"].setText(f"{snr:.2f} dB")

    def set_info(self, info: Dict[str, Any]) -> None:
        """Set signal information (one-time update)"""
        self.update_info(info)

    def get_info(self) -> Dict[str, Any]:
        """Get current signal information.

        Returns raw numeric/native values (not formatted display strings) so
        the state can round-trip. Keys not yet set are returned as ``None``.
        """
        return {
            "n_samples": self._raw_info.get("n_samples"),
            "duration": self._raw_info.get("duration"),
            "fs": self._raw_info.get("fs"),
            "fc": self._raw_info.get("fc"),
            "power": self._raw_info.get("power"),
        }
