"""
Preset Manager Dialog
~~~~~~~~~~~~~~~~~~~~

Dialog for managing user-defined preset templates:
add, edit, delete, and preview preset configurations.
"""

from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.business.presets_manager import get_presets_manager
from ChannelSimulation.viewer.core.ui_style import get_color


class PresetManagerDialog(QDialog):
    """Preset Manager Dialog"""

    CATEGORY_LABELS = {
        "satellite": "Satellite",
        "uav": "UAV Swarm",
        "cellular": "Cellular",
        "radar": "Radar",
        "custom": "Custom",
        "reference": "Reference",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Presets")
        self.setMinimumSize(700, 480)
        self._presets_mgr = get_presets_manager()
        self._current_preset_name: str | None = None
        self._init_ui()
        self._refresh_list()

    def _status(self, msg: str) -> None:
        """Show status message in window title bar."""
        self.setWindowTitle(f"Manage Presets — {msg}")

    def _init_ui(self) -> None:
        """Initialize UI."""
        layout = QHBoxLayout(self)
        layout.setSpacing(12)

        # ── Left panel: preset list ────────────────────────────────────────
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._preset_list = QListWidget()
        self._preset_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._preset_list.currentItemChanged.connect(self._on_preset_selected)
        left_layout.addWidget(self._preset_list)

        # Buttons below list
        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)

        self._btn_add = QPushButton("Add")
        self._btn_add.clicked.connect(self._on_add_preset)
        btn_row.addWidget(self._btn_add)

        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._on_delete_preset)
        self._btn_delete.setEnabled(False)
        btn_row.addWidget(self._btn_delete)

        self._btn_duplicate = QPushButton("Duplicate")
        self._btn_duplicate.clicked.connect(self._on_duplicate_preset)
        self._btn_duplicate.setEnabled(False)
        btn_row.addWidget(self._btn_duplicate)

        left_layout.addLayout(btn_row)

        # ── Right panel: preset editor ──────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        form = QFormLayout()
        form.setSpacing(8)

        self._input_name = QLineEdit()
        self._input_name.setPlaceholderText("Preset name")
        self._input_name.textChanged.connect(self._on_field_changed)
        form.addRow("Name:", self._input_name)

        self._input_description = QLineEdit()
        self._input_description.setPlaceholderText("Brief description of this preset")
        form.addRow("Description:", self._input_description)

        self._combo_category = self._build_category_combo()
        form.addRow("Category:", self._combo_category)

        right_layout.addLayout(form)

        # Channels summary (read-only for built-in, editable for custom)
        right_layout.addWidget(QLabel("Channel Configuration:"))
        self._channels_text = QTextEdit()
        self._channels_text.setReadOnly(True)
        self._channels_text.setMaximumHeight(120)
        self._channels_text.setPlaceholderText("Channel settings will be shown here when a preset is selected")
        right_layout.addWidget(self._channels_text)

        right_layout.addStretch()

        # Buttons
        btn_box = QDialogButtonBox()
        self._btn_save = btn_box.addButton("Save", QDialogButtonBox.ButtonRole.ActionRole)
        self._btn_save.clicked.connect(self._on_save_preset)
        self._btn_save.setEnabled(False)

        btn_box.addButton(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.accept)
        right_layout.addWidget(btn_box)

        # Set stretch factors
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 2)

    def _build_category_combo(self) -> "QComboBox":
        """Build category combo box."""
        from PySide6.QtWidgets import QComboBox
        combo = QComboBox()
        for key, label in self.CATEGORY_LABELS.items():
            combo.addItem(label, key)
        return combo

    def _refresh_list(self) -> None:
        """Refresh the preset list."""
        self._preset_list.blockSignals(True)
        self._preset_list.clear()

        presets = self._presets_mgr.get_presets()
        for name in presets:
            preset = self._presets_mgr.get_preset(name)
            category = preset.get("category", "custom") if preset else "custom"
            label = self.CATEGORY_LABELS.get(category, category)
            item = QListWidgetItem(f"[{label}] {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._preset_list.addItem(item)

        self._preset_list.blockSignals(False)

    def _on_preset_selected(self, item: QListWidgetItem | None) -> None:
        """Handle preset selection."""
        if item is None:
            self._current_preset_name = None
            self._btn_delete.setEnabled(False)
            self._btn_duplicate.setEnabled(False)
            return

        name = item.data(Qt.ItemDataRole.UserRole)
        self._current_preset_name = name
        preset = self._presets_mgr.get_preset(name)

        self._btn_delete.setEnabled(True)
        self._btn_duplicate.setEnabled(True)

        if preset:
            self._input_name.setText(name)
            self._input_name.setReadOnly(False)
            self._input_description.setText(preset.get("description", ""))
            idx = self._combo_category.findData(preset.get("category", "custom"))
            if idx >= 0:
                self._combo_category.setCurrentIndex(idx)
            self._channels_text.setPlainText(self._format_preset_channels(preset))

    def _format_preset_channels(self, preset: Dict[str, Any]) -> str:
        """Format preset channel settings as readable text."""
        lines = []
        stage_configs = preset.get("stage_configs", {})
        for stage, cfg in stage_configs.items():
            enabled = cfg.get("enabled", False)
            marker = "✓" if enabled else "✗"
            params = cfg.get("params", {})
            param_str = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            lines.append(f"  {marker} {stage}: {param_str}")
        return "\n".join(lines) if lines else "(no channel configs)"

    def _on_field_changed(self, _text: str) -> None:
        """Mark as modified, enable save button."""
        self._btn_save.setEnabled(True)

    def _on_add_preset(self) -> None:
        """Add a new preset from current application config."""
        # Ask user for name
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self, "New Preset", "Enter preset name:",
            QLineEdit.EchoMode.Normal,
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        if name in self._presets_mgr.get_presets():
            QMessageBox.warning(self, "Duplicate Name",
                               f"A preset named '{name}' already exists.")
            return

        # Create a minimal preset skeleton (user fills in channels)
        new_preset = {
            "name": name,
            "description": "",
            "category": "custom",
            "fs": 100_000_000,
            "fc": 0.0,
            "seed": 42,
            "preprocess": {
                "dc_removal": {"enabled": True},
                "digital_filter": {"enabled": False},
                "resampling": {"enabled": False},
                "gain_control": {"enabled": False},
            },
            "stage_configs": {
                "fading": {"enabled": False},
                "doppler": {"enabled": False},
                "multipath": {"enabled": False},
                "cfo": {"enabled": False},
                "phase_noise": {"enabled": False},
                "iq_imbalance": {"enabled": False},
                "awgn": {"enabled": False},
                "crop_time_shift": {"enabled": False},
            },
        }
        self._presets_mgr.add_preset(name, new_preset, description="", category="custom")
        self._refresh_list()
        # Select the new preset
        for i in range(self._preset_list.count()):
            item = self._preset_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == name:
                self._preset_list.setCurrentRow(i)
                break

    def _on_delete_preset(self) -> None:
        """Delete selected preset."""
        if not self._current_preset_name:
            return
        name = self._current_preset_name
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete preset '{name}'?\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._presets_mgr.delete_preset(name)
        self._current_preset_name = None
        self._input_name.clear()
        self._input_description.clear()
        self._channels_text.clear()
        self._btn_delete.setEnabled(False)
        self._btn_duplicate.setEnabled(False)
        self._refresh_list()

    def _on_duplicate_preset(self) -> None:
        """Duplicate selected preset."""
        if not self._current_preset_name:
            return
        original = self._presets_mgr.get_preset(self._current_preset_name)
        if not original:
            return

        from PySide6.QtWidgets import QInputDialog
        new_name, ok = QInputDialog.getText(
            self, "Duplicate Preset",
            f"Enter name for the duplicate of '{self._current_preset_name}':",
            QLineEdit.EchoMode.Normal,
            self._current_preset_name + " (copy)",
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()

        self._presets_mgr.add_preset(
            new_name, original,
            description=original.get("description", ""),
            category=original.get("category", "custom"),
        )
        self._refresh_list()

    def _on_save_preset(self) -> None:
        """Save changes to the current preset."""
        if not self._current_preset_name:
            return

        new_name = self._input_name.text().strip()
        if not new_name:
            QMessageBox.warning(self, "Invalid Name", "Preset name cannot be empty.")
            return

        preset = self._presets_mgr.get_preset(self._current_preset_name)
        if preset is None:
            preset = {}

        preset["name"] = new_name
        preset["description"] = self._input_description.text().strip()
        category_data = self._combo_category.currentData()
        preset["category"] = category_data if category_data else "custom"

        # Rename if name changed
        old_name = self._current_preset_name
        if new_name != old_name:
            # Delete old
            self._presets_mgr.delete_preset(old_name)

        self._presets_mgr.add_preset(
            new_name, preset,
            description=preset.get("description", ""),
            category=preset.get("category", "custom"),
        )
        self._current_preset_name = new_name
        self._btn_save.setEnabled(False)
        self._refresh_list()
        self._status("Preset saved.")
