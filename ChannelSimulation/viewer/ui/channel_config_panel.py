"""
Channel Configuration Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration panel containing 8 channel collapse cards and 4 preprocessing cards.
Each channel/preprocessing has independent toggle switches and parameter controls.
"""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.core.ui_style import get_color
from ChannelSimulation.viewer.ui.collapsible_section import CollapsibleSection


class ChannelConfigPanel(QWidget):
    """Channel Configuration Panel

    Contains:
    - 4 preprocessing cards (DC removal, digital filtering, resampling, gain control)
    - 8 channel cards (AWGN, CFO, Fading, Multipath, Phase Noise, IQ Imbalance, Crop+TimeShift, Doppler)
    """

    config_changed = Signal(dict)  # Configuration change signal

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(0, 0, 8, 0)

        # Preprocessing (collapsible; expanded by default — main user focus)
        preprocess_inner = QWidget()
        preprocess_layout = QVBoxLayout(preprocess_inner)
        preprocess_layout.setSpacing(4)
        preprocess_layout.setContentsMargins(0, 0, 0, 0)

        self._preprocess_dc = PreprocessCard("DC Bias Removal", "Remove IQ DC component")
        self._preprocess_dc.setChecked(True)  # Enabled by default
        preprocess_layout.addWidget(self._preprocess_dc)

        self._preprocess_filter = PreprocessCard("Digital Filtering", "Bandpass/lowpass/highpass FIR/IIR filtering")
        preprocess_layout.addWidget(self._preprocess_filter)

        self._preprocess_resample = PreprocessCard("Resampling", "Change sampling rate")
        preprocess_layout.addWidget(self._preprocess_resample)

        self._preprocess_gain = PreprocessCard("Gain Control", "Manual amplitude scaling")
        preprocess_layout.addWidget(self._preprocess_gain)

        preprocess_section = CollapsibleSection(
            "Signal Preprocessing", preprocess_inner, expanded=True
        )
        scroll_layout.addWidget(preprocess_section)

        # Channel simulation (collapsible)
        channel_inner = QWidget()
        channel_layout = QVBoxLayout(channel_inner)
        channel_layout.setSpacing(4)
        channel_layout.setContentsMargins(0, 0, 0, 0)

        # Create channel cards
        self._channels: Dict[str, ChannelCard] = {}

        self._channels["awgn"] = ChannelCard(
            "AWGN",
            "Additive Gaussian White Noise",
            [
                ParameterDef("snr_db", "SNR", "dB", -50, 50, 15, 0.5),
            ],
        )
        channel_layout.addWidget(self._channels["awgn"])

        self._channels["cfo"] = ChannelCard(
            "CFO",
            "Carrier Frequency Offset",
            [
                ParameterDef("cfo_hz", "CFO", "Hz", 0, 1e6, 15000, 100),
            ],
        )
        channel_layout.addWidget(self._channels["cfo"])

        self._channels["fading"] = ChannelCard(
            "Fading",
            "Rayleigh/Rician Fading",
            [
                ParameterDef("k_factor", "K Factor", "", 0, 100, 5, 0.5),
                ParameterDef("block_size", "Block Size", "samples", 64, 2048, 256, 64),
            ],
            has_type_selector=True,
            type_options=["Rayleigh", "Rician"],
        )
        channel_layout.addWidget(self._channels["fading"])

        self._channels["multipath"] = ChannelCard(
            "Multipath",
            "Multipath Fading",
            [
                ParameterDef("max_delay_samp", "Max Delay", "samples", 1, 200, 20, 1),
                ParameterDef("n_paths", "Num Paths", "", 1, 20, 4, 1),
                ParameterDef("exp_decay", "Decay Factor", "", 0, 1, 0.35, 0.01),
            ],
        )
        channel_layout.addWidget(self._channels["multipath"])

        self._channels["phase_noise"] = ChannelCard(
            "Phase Noise",
            "Phase Noise",
            [
                ParameterDef("phase_std_per_sample", "Phase Std Dev", "rad", 0, 0.1, 0.002, 0.0001),
            ],
        )
        channel_layout.addWidget(self._channels["phase_noise"])

        self._channels["iq_imbalance"] = ChannelCard(
            "IQ Imbalance",
            "IQ Imbalance",
            [
                ParameterDef("gain_imbalance_db", "Gain Imbalance", "dB", -10, 10, 1.0, 0.1),
                ParameterDef("phase_imbalance_deg", "Phase Imbalance", "deg", -30, 30, 3.0, 0.1),
            ],
        )
        channel_layout.addWidget(self._channels["iq_imbalance"])

        self._channels["crop_time_shift"] = ChannelCard(
            "Crop+TimeShift",
            "Random Crop and Time Shift",
            [
                ParameterDef("min_keep_ratio", "Min Keep Ratio", "", 0.1, 1.0, 0.7, 0.01),
                ParameterDef("max_shift", "Max Shift", "samples", 0, 5000, 300, 10),
            ],
        )
        channel_layout.addWidget(self._channels["crop_time_shift"])

        self._channels["doppler"] = ChannelCard(
            "Doppler",
            "Doppler Shift and Spread",
            [
                ParameterDef("doppler_shift_hz", "Doppler Shift", "Hz", -100000, 100000, 0, 100),
                ParameterDef("doppler_spread_hz", "Doppler Spread", "Hz", 0, 10000, 100, 10),
            ],
        )
        channel_layout.addWidget(self._channels["doppler"])

        channel_section = CollapsibleSection(
            "Channel Simulation", channel_inner, expanded=True
        )
        scroll_layout.addWidget(channel_section)
        scroll_layout.addStretch(1)

        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def _connect_signals(self) -> None:
        """Connect signals"""
        for channel in self._channels.values():
            channel.value_changed.connect(self._on_value_changed)

        self._preprocess_dc.value_changed.connect(self._on_value_changed)
        self._preprocess_filter.value_changed.connect(self._on_value_changed)
        self._preprocess_resample.value_changed.connect(self._on_value_changed)
        self._preprocess_gain.value_changed.connect(self._on_value_changed)

    def _on_value_changed(self) -> None:
        """Handle value change"""
        self.config_changed.emit(self.get_config())

    def get_config(self) -> dict[str, Any]:
        """Get current configuration"""
        config = {
            "preprocess": {
                "dc_removal": {"enabled": self._preprocess_dc.isChecked()},
                "digital_filter": self._preprocess_filter.get_filter_config(),
                "resampling": self._preprocess_resample.get_resample_config(),
                "gain_control": self._preprocess_gain.get_gain_config(),
            },
            "stage_configs": {},
        }

        for name, card in self._channels.items():
            cfg = card.get_stage_config()
            if cfg:
                config["stage_configs"][name] = cfg

        return config

    def set_config(self, config: dict[str, Any]) -> None:
        """Set configuration"""
        # Preprocess config
        preprocess = config.get("preprocess", {})
        if "dc_removal" in preprocess:
            self._preprocess_dc.setChecked(preprocess["dc_removal"].get("enabled", True))
        if "digital_filter" in preprocess:
            self._preprocess_filter.set_filter_config(preprocess["digital_filter"])
        if "resampling" in preprocess:
            self._preprocess_resample.set_resample_config(preprocess["resampling"])
        if "gain_control" in preprocess:
            self._preprocess_gain.set_gain_config(preprocess["gain_control"])

        # Stage config
        stage_configs = config.get("stage_configs", {})
        for name, card in self._channels.items():
            if name in stage_configs:
                card.set_stage_config(stage_configs[name])

    def reset(self) -> None:
        """Reset to default values"""
        self._preprocess_dc.setChecked(True)
        self._preprocess_filter.reset()
        self._preprocess_resample.reset()
        self._preprocess_gain.reset()

        for card in self._channels.values():
            card.reset()


class ParameterDef:
    """Parameter definition"""

    def __init__(
        self,
        key: str,
        label: str,
        unit: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float = 1.0,
        decimals: int | None = None,
    ):
        self.key = key
        self.label = label
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.default = default
        self.step = step
        # Auto-determine decimal places
        if decimals is None:
            if step < 1:
                decimals = len(str(step).split(".")[-1])
            else:
                decimals = 0
        self.decimals = decimals


class ChannelCard(QWidget):
    """Channel collapse card"""

    value_changed = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        parameters: list[ParameterDef],
        has_type_selector: bool = False,
        type_options: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._description = description
        self._parameters = parameters
        self._has_type_selector = has_type_selector
        self._type_options = type_options or ["Default"]

        self._enabled = False
        self._param_values: Dict[str, float] = {p.key: p.default for p in parameters}
        self._selected_type = type_options[0] if type_options else "Default"

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        """Initialize UI"""
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Card frame
        self._card_frame = QFrame()
        self._card_frame.setObjectName("card")
        self._card_frame.setStyleSheet(f"""
            QFrame#card {{
                background-color: {get_color('card_bg')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
            }}
        """)

        card_layout = QVBoxLayout(self._card_frame)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Title bar (clickable)
        self._title_bar = QWidget()
        self._title_bar.setCursor(Qt.CursorShape.PointingHandCursor)
        title_layout = QHBoxLayout(self._title_bar)
        title_layout.setContentsMargins(12, 10, 12, 10)
        title_layout.setSpacing(8)

        # Expand/collapse button
        self._expand_btn = QLabel("▼")
        self._expand_btn.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        title_layout.addWidget(self._expand_btn)

        # Switch
        self._switch = QCheckBox()
        self._switch.setObjectName("channelSwitch")
        self._switch.setText("")
        title_layout.addWidget(self._switch)

        # Title
        title_label = QLabel(self._title)
        title_label.setStyleSheet(f"color: {get_color('text_primary')}; font-weight: bold; font-size: 13px;")
        title_layout.addWidget(title_label)

        # Type selector (optional)
        if self._has_type_selector:
            self._type_selector = QComboBox()
            self._type_selector.addItems(self._type_options)
            self._type_selector.setMaximumWidth(100)
            self._type_selector.currentTextChanged.connect(self._on_type_changed)
            title_layout.addWidget(QLabel("Type:"))
            title_layout.addWidget(self._type_selector)

        # Add stretch space
        title_layout.addStretch()

        # Status indicator
        self._status_dot = QLabel("○")
        self._status_dot.setStyleSheet(f"color: {get_color('text_disabled')}; font-size: 12px;")
        title_layout.addWidget(self._status_dot)

        card_layout.addWidget(self._title_bar)

        # Parameters area (collapsible)
        self._params_widget = QWidget()
        self._params_widget.setVisible(False)
        params_layout = QFormLayout(self._params_widget)
        params_layout.setContentsMargins(16, 8, 16, 12)
        params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        params_layout.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        params_layout.setHorizontalSpacing(16)
        params_layout.setVerticalSpacing(8)

        # Create parameter controls
        self._param_widgets: Dict[str, QWidget] = {}
        for param in self._parameters:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(8)

            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimumWidth(120)
            slider.setMaximum(int(param.max_val / param.step))
            slider.setValue(int(param.default / param.step))
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(int((param.max_val - param.min_val) / param.step / 5))
            self._param_widgets[f"{param.key}_slider"] = slider

            # Spinbox input
            if param.decimals > 0:
                spinbox = QDoubleSpinBox()
                spinbox.setDecimals(param.decimals)
            else:
                spinbox = QSpinBox()
            spinbox.setMinimum(param.min_val)
            spinbox.setMaximum(param.max_val)
            spinbox.setValue(param.default)
            spinbox.setSingleStep(param.step)
            spinbox.setSuffix(f" {param.unit}" if param.unit else "")
            spinbox.setMaximumWidth(100)
            self._param_widgets[f"{param.key}_spinbox"] = spinbox

            # Connect slider and spinbox
            slider.valueChanged.connect(lambda v, p=param, s=slider, sb=spinbox: self._on_slider_changed(p, v, sb, s))
            spinbox.valueChanged.connect(lambda v, p=param, sb=spinbox, s=slider: self._on_spinbox_changed(p, v, s, sb))

            row_layout.addWidget(slider)
            row_layout.addWidget(spinbox)

            params_layout.addRow(f"{param.label}:", row_layout)

        card_layout.addWidget(self._params_widget)

        main_layout.addWidget(self._card_frame)

    def _connect_signals(self) -> None:
        """Connect signals"""
        self._title_bar.mousePressEvent = self._on_title_click
        self._switch.toggled.connect(self._on_switch_toggled)

    def _on_title_click(self, event) -> None:
        """Handle title bar click"""
        self._toggle_expand()

    def _on_switch_toggled(self, checked: bool) -> None:
        """Handle switch toggle"""
        self._enabled = checked
        self._update_status()
        self.value_changed.emit()

    def _on_type_changed(self, text: str) -> None:
        """Handle type selection change"""
        self._selected_type = text
        self.value_changed.emit()

    def _on_slider_changed(self, param: ParameterDef, value: int, spinbox, slider) -> None:
        """Handle slider value change"""
        actual_value = value * param.step
        spinbox.blockSignals(True)
        spinbox.setValue(actual_value)
        spinbox.blockSignals(False)
        self._param_values[param.key] = actual_value
        self.value_changed.emit()

    def _on_spinbox_changed(self, param: ParameterDef, value: float, slider, spinbox) -> None:
        """Handle spinbox value change"""
        slider.blockSignals(True)
        slider.setValue(int(value / param.step))
        slider.blockSignals(False)
        self._param_values[param.key] = float(value)
        self.value_changed.emit()

    def _toggle_expand(self) -> None:
        """Toggle expand/collapse"""
        is_visible = self._params_widget.isVisible()
        self._params_widget.setVisible(not is_visible)
        self._expand_btn.setText("▶" if is_visible else "▼")

    def _update_status(self) -> None:
        """Update status indicator"""
        if self._enabled:
            self._status_dot.setText("●")
            self._status_dot.setStyleSheet(f"color: {get_color('accent')}; font-size: 12px;")
            self._card_frame.setStyleSheet(f"""
                QFrame#card {{
                    background-color: {get_color('card_bg')};
                    border: 1px solid {get_color('accent')};
                    border-radius: 8px;
                }}
            """)
        else:
            self._status_dot.setText("○")
            self._status_dot.setStyleSheet(f"color: {get_color('text_disabled')}; font-size: 12px;")
            self._card_frame.setStyleSheet(f"""
                QFrame#card {{
                    background-color: {get_color('card_bg')};
                    border: 1px solid {get_color('border')};
                    border-radius: 8px;
                }}
            """)

    def isChecked(self) -> bool:
        """Check if enabled"""
        return self._enabled

    def setChecked(self, checked: bool) -> None:
        """Set enabled state"""
        self._switch.blockSignals(True)
        self._switch.setChecked(checked)
        self._switch.blockSignals(False)
        self._enabled = checked
        self._update_status()

    def get_stage_config(self) -> dict[str, Any]:
        """Get stage configuration"""
        if not self._enabled:
            return {"enabled": False}

        config = {
            "enabled": True,
            "params": dict(self._param_values),
        }

        if self._has_type_selector:
            config["method"] = self._selected_type.lower()

        return config

    def set_stage_config(self, config: dict[str, Any]) -> None:
        """Set stage configuration"""
        enabled = config.get("enabled", False)
        self.setChecked(enabled)

        if "params" in config:
            for key, value in config["params"].items():
                if key in self._param_values:
                    self._param_values[key] = float(value)
                    spinbox = self._param_widgets.get(f"{key}_spinbox")
                    slider = self._param_widgets.get(f"{key}_slider")
                    if spinbox:
                        spinbox.blockSignals(True)
                        spinbox.setValue(float(value))
                        spinbox.blockSignals(False)
                    if slider:
                        for param in self._parameters:
                            if param.key == key:
                                slider.blockSignals(True)
                                slider.setValue(int(float(value) / param.step))
                                slider.blockSignals(False)
                                break

        if "method" in config:
            method = config["method"].lower()
            if self._has_type_selector and method in [opt.lower() for opt in self._type_options]:
                idx = [opt.lower() for opt in self._type_options].index(method)
                self._type_selector.setCurrentIndex(idx)

    def reset(self) -> None:
        """Reset to default values"""
        self.setChecked(False)
        for param in self._parameters:
            self._param_values[param.key] = param.default
            spinbox = self._param_widgets.get(f"{param.key}_spinbox")
            slider = self._param_widgets.get(f"{param.key}_slider")
            if spinbox:
                spinbox.setValue(param.default)
            if slider:
                slider.setValue(int(param.default / param.step))

        if self._has_type_selector:
            self._type_selector.setCurrentIndex(0)


class PreprocessCard(QWidget):
    """Preprocessing card"""

    value_changed = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._description = description

        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)

        # Switch
        self._switch = QCheckBox()
        layout.addWidget(self._switch)

        # Title
        title_label = QLabel(self._title)
        title_label.setStyleSheet(f"color: {get_color('text_primary')}; font-size: 12px;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(self._description)
        desc_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        layout.addWidget(desc_label)

        # Add stretch space
        layout.addStretch()

        self._switch.toggled.connect(lambda *args: self.value_changed.emit())

    def isChecked(self) -> bool:
        """Check if enabled"""
        return self._switch.isChecked()

    def setChecked(self, checked: bool) -> None:
        """Set enabled state"""
        self._switch.blockSignals(True)
        self._switch.setChecked(checked)
        self._switch.blockSignals(False)

    def get_filter_config(self) -> dict[str, Any]:
        """Get filter configuration"""
        return {
            "enabled": self._switch.isChecked(),
            "type": "lowpass",
            "filter_class": "butterworth",
            "cutoff_hz": 50_000_000,
            "order": 8,
        }

    def set_filter_config(self, config: dict[str, Any]) -> None:
        """Set filter configuration"""
        self._switch.blockSignals(True)
        self._switch.setChecked(config.get("enabled", False))
        self._switch.blockSignals(False)

    def get_resample_config(self) -> dict[str, Any]:
        """Get resampling configuration"""
        return {
            "enabled": self._switch.isChecked(),
            "target_fs": 200_000_000,
            "filter_type": "firls",
        }

    def set_resample_config(self, config: dict[str, Any]) -> None:
        """Set resampling configuration"""
        self._switch.blockSignals(True)
        self._switch.setChecked(config.get("enabled", False))
        self._switch.blockSignals(False)

    def get_gain_config(self) -> dict[str, Any]:
        """Get gain configuration"""
        return {
            "enabled": self._switch.isChecked(),
            "gain_db": 0.0,
        }

    def set_gain_config(self, config: dict[str, Any]) -> None:
        """Set gain configuration"""
        self._switch.blockSignals(True)
        self._switch.setChecked(config.get("enabled", False))
        self._switch.blockSignals(False)

    def reset(self) -> None:
        """Reset"""
        self._switch.blockSignals(True)
        self._switch.setChecked(False)
        self._switch.blockSignals(False)
