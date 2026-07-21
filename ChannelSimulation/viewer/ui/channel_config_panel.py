"""
Channel Configuration Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration panel containing 8 channel collapse cards and 4 preprocessing cards.
Each channel/preprocessing has independent toggle switches and parameter controls.
"""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.core.ui_style import get_color
from ChannelSimulation.viewer.ui.collapsible_section import CollapsibleSection


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


# Data-driven card specifications.
#
# Preprocessing cards map to fixed attribute names on the panel because the
# public API (get_config/set_config/reset) references them directly.
PREPROCESS_CARD_SPECS = [
    {
        "attr": "_preprocess_dc",
        "title": "DC Bias Removal",
        "description": "Remove IQ DC component",
        "default_enabled": True,
    },
    {
        "attr": "_preprocess_filter",
        "title": "Digital Filtering",
        "description": "Bandpass/lowpass/highpass FIR/IIR filtering",
        "default_enabled": False,
    },
    {
        "attr": "_preprocess_resample",
        "title": "Resampling",
        "description": "Change sampling rate",
        "default_enabled": False,
    },
    {
        "attr": "_preprocess_gain",
        "title": "Gain Control",
        "description": "Manual amplitude scaling",
        "default_enabled": False,
    },
]

# Channel cards. ``parameters`` holds ParameterDef constructor argument tuples so
# that a fresh ParameterDef is built per panel instance (avoids sharing mutable
# state such as ``max_val`` across panels via set_param_max).
CHANNEL_CARD_SPECS = [
    {
        "name": "awgn",
        "title": "AWGN",
        "description": "Additive Gaussian White Noise",
        "parameters": [
            ("snr_db", "Output SNR", "dB", -50, 50, 15, 0.5),
        ],
    },
    {
        "name": "cfo",
        "title": "CFO",
        "description": "Carrier Frequency Offset",
        "parameters": [
            ("cfo_hz", "CFO", "Hz", -50e6, 50e6, 0, 1000),
        ],
    },
    {
        "name": "fading",
        "title": "Fading",
        "description": "Rayleigh/Rician Fading",
        "parameters": [
            ("k_factor", "K Factor", "", 0, 100, 5, 0.5),
            ("block_size", "Block Size", "samples", 64, 2048, 256, 64),
        ],
        "has_type_selector": True,
        "type_options": ["Rayleigh", "Rician", "Rice Mild"],
    },
    {
        "name": "multipath",
        "title": "Multipath",
        "description": "Multipath Fading",
        "parameters": [
            ("max_delay_samp", "Max Delay", "samples", 1, 200, 20, 1),
            ("n_paths", "Num Paths", "", 1, 20, 4, 1),
            ("exp_decay", "Decay Factor", "", 0, 1, 0.35, 0.01),
        ],
    },
    {
        "name": "phase_noise",
        "title": "Phase Noise",
        "description": "Phase Noise",
        "parameters": [
            ("phase_std_per_sample", "Phase Std Dev", "rad", 0, 0.1, 0.002, 0.0001),
        ],
    },
    {
        "name": "iq_imbalance",
        "title": "IQ Imbalance",
        "description": "IQ Imbalance",
        "parameters": [
            ("gain_imbalance_db", "Gain Imbalance", "dB", -10, 10, 1.0, 0.1),
            ("phase_imbalance_deg", "Phase Imbalance", "deg", -30, 30, 3.0, 0.1),
        ],
    },
    {
        "name": "crop_time_shift",
        "title": "Crop+TimeShift",
        "description": "Random Crop and Time Shift",
        "parameters": [
            ("min_keep_ratio", "Min Keep Ratio", "", 0.1, 1.0, 0.7, 0.01),
            ("max_shift", "Max Shift", "samples", 0, 5000, 300, 10),
        ],
    },
    {
        "name": "doppler",
        "title": "Doppler",
        "description": "Doppler Shift and Spread",
        "parameters": [
            ("doppler_shift_hz", "Doppler Shift", "Hz", -100000, 100000, 0, 100),
            ("doppler_spread_hz", "Doppler Spread", "Hz", 0, 10000, 100, 10),
        ],
    },
]


class ChannelConfigPanel(QWidget):
    """Channel Configuration Panel

    Contains:
    - 4 preprocessing cards (DC removal, digital filtering, resampling, gain control)
    - 8 channel cards (AWGN, CFO, Fading, Multipath, Phase Noise, IQ Imbalance, Crop+TimeShift, Doppler)
    """

    config_changed = Signal(dict)  # Configuration change signal

    def __init__(self, parent=None):
        super().__init__(parent)
        # Debounce timer: coalesce rapid slider/spinbox changes into a single
        # outward config_changed emit. Live intra-panel UI updates stay immediate.
        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(180)
        self._emit_timer.timeout.connect(self._emit_config_changed)
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

        for spec in PREPROCESS_CARD_SPECS:
            card = PreprocessCard(spec["title"], spec["description"])
            if spec["default_enabled"]:
                card.setChecked(True)  # Enabled by default
            setattr(self, spec["attr"], card)
            preprocess_layout.addWidget(card)

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

        for spec in CHANNEL_CARD_SPECS:
            parameters = [ParameterDef(*args) for args in spec["parameters"]]
            card = ChannelCard(
                spec["title"],
                spec["description"],
                parameters,
                has_type_selector=spec.get("has_type_selector", False),
                type_options=spec.get("type_options"),
            )
            self._channels[spec["name"]] = card
            channel_layout.addWidget(card)

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
        """Handle value change (debounced: coalesce rapid changes into one emit)."""
        self._emit_timer.start()

    def _emit_config_changed(self) -> None:
        """Emit the coalesced configuration change."""
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

    def set_awgn_snr_max(self, max_snr_db: float) -> None:
        """Update AWGN SNR slider max to the current signal SNR (so AWGN cannot improve SNR)."""
        awgn_card = self._channels.get("awgn")
        if awgn_card is not None:
            awgn_card.set_param_max("snr_db", max_snr_db)

    def reset(self) -> None:
        """Reset to default values"""
        self._preprocess_dc.setChecked(True)
        self._preprocess_filter.reset()
        self._preprocess_resample.reset()
        self._preprocess_gain.reset()

        for card in self._channels.values():
            card.reset()

        # Reset AWGN SNR max back to default 50 dB after full panel reset
        self.set_awgn_snr_max(50.0)


class _ClickableWidget(QWidget):
    """A QWidget that emits ``clicked`` on a left mouse press.

    Used for the channel card title bar so clicking toggles expand/collapse
    through a proper Qt event override instead of monkeypatching mousePressEvent.
    """

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


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
        # Two selectors: the base look, and the highlighted ("active") look when
        # the channel is enabled. Toggling only flips a dynamic property so the
        # QSS string is not rebuilt on every switch (see _update_status).
        self._card_frame.setStyleSheet(f"""
            QFrame#card {{
                background-color: {get_color('card_bg')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
            }}
            QFrame#card[active="true"] {{
                background-color: {get_color('card_bg')};
                border: 1px solid {get_color('accent')};
                border-radius: 8px;
            }}
        """)
        self._card_frame.setProperty("active", False)

        card_layout = QVBoxLayout(self._card_frame)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Title bar (clickable)
        self._title_bar = _ClickableWidget()
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

            # Slider — always uses setMinimum + setMaximum so negative ranges work
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimumWidth(120)
            slider.setMinimum(round(param.min_val / param.step))
            slider.setMaximum(round(param.max_val / param.step))
            slider.setValue(round(param.default / param.step))
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
        self._title_bar.clicked.connect(self._toggle_expand)
        self._switch.toggled.connect(self._on_switch_toggled)

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
        self.value_changed.emit()

    def _on_spinbox_changed(self, param: ParameterDef, value: float, slider, spinbox) -> None:
        """Handle spinbox value change"""
        slider.blockSignals(True)
        slider.setValue(round(value / param.step))
        slider.blockSignals(False)
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
        else:
            self._status_dot.setText("○")
            self._status_dot.setStyleSheet(f"color: {get_color('text_disabled')}; font-size: 12px;")

        # Flip the dynamic property and re-polish instead of rebuilding the QSS.
        self._card_frame.setProperty("active", self._enabled)
        self._card_frame.style().unpolish(self._card_frame)
        self._card_frame.style().polish(self._card_frame)

    def set_param_max(self, key: str, max_val: float) -> None:
        """Dynamically update the maximum value of a parameter's slider and spinbox."""
        param: ParameterDef | None = None
        for p in self._parameters:
            if p.key == key:
                param = p
                break
        if param is None:
            return

        slider = self._param_widgets.get(f"{key}_slider")
        spinbox = self._param_widgets.get(f"{key}_spinbox")

        param.max_val = max_val

        # The spinbox is the single source of truth for the current value.
        current_val = spinbox.value() if spinbox is not None else param.default
        clamped_val = min(current_val, max_val)

        if slider is not None:
            slider.blockSignals(True)
            slider.setMinimum(round(param.min_val / param.step))
            slider.setMaximum(round(max_val / param.step))
            slider.setTickInterval(int((max_val - param.min_val) / param.step / 5))
            # Clamp current value to new range
            lo = round(param.min_val / param.step)
            hi = round(max_val / param.step)
            slider.setValue(max(lo, min(round(clamped_val / param.step), hi)))
            slider.blockSignals(False)

        if spinbox is not None:
            spinbox.blockSignals(True)
            spinbox.setMaximum(max_val)
            # Write the clamped value back so a stale over-max value cannot leak
            # into the emitted config.
            spinbox.setValue(clamped_val)
            spinbox.blockSignals(False)

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

        # Read live values straight from the spinboxes (single source of truth).
        params: Dict[str, float] = {}
        for param in self._parameters:
            spinbox = self._param_widgets.get(f"{param.key}_spinbox")
            params[param.key] = spinbox.value() if spinbox is not None else param.default

        config = {
            "enabled": True,
            "params": params,
        }

        if self._has_type_selector:
            config["method"] = self._selected_type.lower().replace(" ", "_")

        return config

    def set_stage_config(self, config: dict[str, Any]) -> None:
        """Set stage configuration"""
        enabled = config.get("enabled", False)
        self.setChecked(enabled)

        if "params" in config:
            valid_keys = {p.key: p for p in self._parameters}
            for key, value in config["params"].items():
                if key in valid_keys:
                    param = valid_keys[key]
                    spinbox = self._param_widgets.get(f"{key}_spinbox")
                    slider = self._param_widgets.get(f"{key}_slider")
                    if spinbox:
                        spinbox.blockSignals(True)
                        spinbox.setValue(float(value))
                        spinbox.blockSignals(False)
                    if slider:
                        slider.blockSignals(True)
                        slider.setValue(round(float(value) / param.step))
                        slider.blockSignals(False)

        if "method" in config:
            method = config["method"].lower().replace("_", " ")
            if self._has_type_selector and method in [opt.lower() for opt in self._type_options]:
                idx = [opt.lower() for opt in self._type_options].index(method)
                self._type_selector.setCurrentIndex(idx)

    def reset(self) -> None:
        """Reset to default values"""
        self.setChecked(False)
        for param in self._parameters:
            spinbox = self._param_widgets.get(f"{param.key}_spinbox")
            slider = self._param_widgets.get(f"{param.key}_slider")
            if spinbox:
                spinbox.setValue(param.default)
            if slider:
                slider.setValue(round(param.default / param.step))

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

        # Switch (clickable area)
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

    def mousePressEvent(self, event) -> None:
        """Handle mouse click on the card to toggle the switch."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._switch.toggle()
        super().mousePressEvent(event)

    def isChecked(self) -> bool:
        """Check if enabled"""
        return self._switch.isChecked()

    def setChecked(self, checked: bool) -> None:
        """Set enabled state"""
        self._switch.blockSignals(True)
        self._switch.setChecked(checked)
        self._switch.blockSignals(False)

    def _set_enabled_config(self, config: dict[str, Any]) -> None:
        """Shared setter: apply only the ``enabled`` flag from a config dict.

        The digital-filter/resampling/gain cards expose no editable controls yet,
        so all three setters behave identically.
        """
        self._switch.blockSignals(True)
        self._switch.setChecked(config.get("enabled", False))
        self._switch.blockSignals(False)

    def get_filter_config(self) -> dict[str, Any]:
        """Get filter configuration.

        The extra keys are constants (no controls expose them yet) but are kept in
        the returned shape because the pipeline runner and preset store read them.
        """
        return {
            "enabled": self._switch.isChecked(),
            "type": "lowpass",
            "filter_class": "butterworth",
            "cutoff_hz": 50_000_000,
            "order": 8,
        }

    def set_filter_config(self, config: dict[str, Any]) -> None:
        """Set filter configuration"""
        self._set_enabled_config(config)

    def get_resample_config(self) -> dict[str, Any]:
        """Get resampling configuration.

        Extra keys are constants kept for pipeline/preset compatibility.
        """
        return {
            "enabled": self._switch.isChecked(),
            "target_fs": 200_000_000,
            "filter_type": "firls",
        }

    def set_resample_config(self, config: dict[str, Any]) -> None:
        """Set resampling configuration"""
        self._set_enabled_config(config)

    def get_gain_config(self) -> dict[str, Any]:
        """Get gain configuration.

        Extra keys are constants kept for pipeline/preset compatibility.
        """
        return {
            "enabled": self._switch.isChecked(),
            "gain_db": 0.0,
        }

    def set_gain_config(self, config: dict[str, Any]) -> None:
        """Set gain configuration"""
        self._set_enabled_config(config)

    def reset(self) -> None:
        """Reset"""
        self._switch.blockSignals(True)
        self._switch.setChecked(False)
        self._switch.blockSignals(False)
