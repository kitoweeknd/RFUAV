"""
Main Window
~~~~~~~~~~

RFUAV ChannelSim Viewer main window class,
containing menu bar, toolbar, central layout, and signal connections.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    """Main window class."""

    # Signal definitions
    load_iq_triggered = Signal(str)                    # Load IQ file
    export_iq_triggered = Signal(str, int)           # Export IQ (path, stage_id)
    export_fig_triggered = Signal(str, int)          # Export figure (path, view_type)
    stage_config_changed = Signal(dict)                # Stage config changed
    preprocess_config_changed = Signal(dict)            # Preprocess config changed
    theme_changed = Signal(bool)                      # Theme changed

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("RFUAV ChannelSim Viewer v1.3.0")
        self.setMinimumSize(1200, 800)

        # Internal state
        self._current_file: str | None = None
        self._current_config: dict[str, Any] = {}
        self._is_modified: bool = False
        self._is_processing: bool = False
        self._theme_dark: bool = False

        # Initialize UI
        self._init_ui()
        self._init_menu()
        self._init_toolbar()
        self._init_statusbar()
        self._init_central_widget()
        self._connect_signals()

        # Apply stylesheet
        self._apply_stylesheet()

    # =========================================================================
    # UI Initialization
    # =========================================================================

    def _init_ui(self) -> None:
        """Initialize UI components."""
        self.setWindowFlags(Qt.Window)

    def _init_menu(self) -> None:
        """Initialize menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File(&F)")

        self._action_load_iq = QAction("Load IQ File...", self)
        self._action_load_iq.setShortcut(QKeySequence("Ctrl+O"))
        self._action_load_iq.triggered.connect(self._on_load_iq)
        file_menu.addAction(self._action_load_iq)

        file_menu.addSeparator()

        self._action_export_iq = QAction("Export IQ Data...", self)
        self._action_export_iq.setShortcut(QKeySequence("Ctrl+S"))
        self._action_export_iq.setEnabled(False)
        self._action_export_iq.triggered.connect(self._on_export_iq)
        file_menu.addAction(self._action_export_iq)

        self._action_export_fig = QAction("Export Spectrogram...", self)
        self._action_export_fig.setShortcut(QKeySequence("Ctrl+E"))
        self._action_export_fig.setEnabled(False)
        self._action_export_fig.triggered.connect(self._on_export_fig)
        file_menu.addAction(self._action_export_fig)

        file_menu.addSeparator()

        self._action_quit = QAction("Exit", self)
        self._action_quit.setShortcut(QKeySequence("Ctrl+Q"))
        self._action_quit.triggered.connect(self.close)
        file_menu.addAction(self._action_quit)

        # Config menu
        config_menu = menubar.addMenu("Config(&C)")

        self._action_save_config = QAction("Save Config...", self)
        self._action_save_config.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._action_save_config.setEnabled(False)
        self._action_save_config.triggered.connect(self._on_save_config)
        config_menu.addAction(self._action_save_config)

        self._action_load_config = QAction("Load Config...", self)
        self._action_load_config.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self._action_load_config.triggered.connect(self._on_load_config)
        config_menu.addAction(self._action_load_config)

        config_menu.addSeparator()

        self._action_reset = QAction("Reset to Default", self)
        self._action_reset.setShortcut(QKeySequence("Ctrl+R"))
        self._action_reset.triggered.connect(self._on_reset_config)
        config_menu.addAction(self._action_reset)

        # Add modification indicator
        self._action_modified_indicator = QAction("Modified", self)
        self._action_modified_indicator.setEnabled(False)
        self._action_modified_indicator.setVisible(False)
        config_menu.addAction(self._action_modified_indicator)

        # View menu
        view_menu = menubar.addMenu("View(&V)")

        self._action_toggle_theme = QAction("Toggle Theme", self)
        self._action_toggle_theme.setShortcut(QKeySequence("Ctrl+T"))
        self._action_toggle_theme.triggered.connect(self._on_toggle_theme)
        view_menu.addAction(self._action_toggle_theme)

        view_menu.addSeparator()

        self._action_toggle_fullscreen = QAction("Fullscreen", self)
        self._action_toggle_fullscreen.setShortcut(QKeySequence("F11"))
        self._action_toggle_fullscreen.triggered.connect(self._on_toggle_fullscreen)
        view_menu.addAction(self._action_toggle_fullscreen)

        # Tools menu
        tools_menu = menubar.addMenu("Tools(&T)")

        self._action_batch_process = QAction("Batch Processing...", self)
        self._action_batch_process.setShortcut(QKeySequence("Ctrl+B"))
        self._action_batch_process.triggered.connect(self._on_batch_process)
        tools_menu.addAction(self._action_batch_process)

        tools_menu.addSeparator()

        self._action_cfo_estimator = QAction("CFO Estimator...", self)
        self._action_cfo_estimator.setEnabled(False)
        self._action_cfo_estimator.triggered.connect(self._on_cfo_estimator)
        tools_menu.addAction(self._action_cfo_estimator)

        # Help menu
        help_menu = menubar.addMenu("Help(&H)")

        self._action_about = QAction("About...", self)
        self._action_about.triggered.connect(self._on_about)
        help_menu.addAction(self._action_about)

        self._action_shortcuts = QAction("Shortcuts...", self)
        self._action_shortcuts.setShortcut(QKeySequence("Ctrl+/"))
        self._action_shortcuts.triggered.connect(self._on_show_shortcuts)
        help_menu.addAction(self._action_shortcuts)

    def _init_toolbar(self) -> None:
        """Initialize toolbar."""
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        toolbar.setIconSize(self._get_icon_size())
        self.addToolBar(toolbar)

        # Load button
        self._btn_load = QPushButton("Load")
        self._btn_load.setObjectName("btnPrimary")
        self._btn_load.clicked.connect(self._on_load_iq)
        toolbar.addWidget(self._btn_load)

        # Save button
        self._btn_save = QPushButton("Save")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save_config)
        toolbar.addWidget(self._btn_save)

        toolbar.addSeparator()

        # Preview button
        self._btn_preview = QPushButton("Preview")
        self._btn_preview.setObjectName("btnPrimary")
        self._btn_preview.setEnabled(False)
        self._btn_preview.clicked.connect(self._on_preview)
        toolbar.addWidget(self._btn_preview)

        toolbar.addSeparator()

        # Reset button
        self._btn_reset = QPushButton("Reset")
        self._btn_reset.clicked.connect(self._on_reset_config)
        toolbar.addWidget(self._btn_reset)

        # Zoom buttons
        self._btn_zoom_in = QPushButton("Zoom In")
        self._btn_zoom_in.clicked.connect(self._on_zoom_in)
        toolbar.addWidget(self._btn_zoom_in)

        self._btn_zoom_out = QPushButton("Zoom Out")
        self._btn_zoom_out.clicked.connect(self._on_zoom_out)
        toolbar.addWidget(self._btn_zoom_out)

        # Add stretch spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # Sampling rate label
        fs_label = QLabel("Sampling Rate:")
        toolbar.addWidget(fs_label)

        self._input_fs = self._create_line_edit(placeholder="100000000", width=100)
        self._input_fs.setToolTip("Sampling Rate (Hz), e.g., 100000000")
        toolbar.addWidget(self._input_fs)

        fs_unit_label = QLabel("Hz")
        toolbar.addWidget(fs_unit_label)

        spacer = QWidget()
        spacer.setFixedWidth(16)
        toolbar.addWidget(spacer)

        # Center frequency label
        fc_label = QLabel("Center Frequency:")
        toolbar.addWidget(fc_label)

        self._input_fc = self._create_line_edit(placeholder="0", width=100)
        self._input_fc.setToolTip("Center Frequency (Hz), e.g., 5800000000")
        toolbar.addWidget(self._input_fc)

        fc_unit_label = QLabel("Hz")
        toolbar.addWidget(fc_unit_label)

    def _init_statusbar(self) -> None:
        """Initialize status bar."""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # Status indicator
        self._status_indicator = QLabel("Ready")
        statusbar.addWidget(self._status_indicator)

        _sep = QFrame()
        _sep.setFrameShape(QFrame.Shape.VLine)
        _sep.setFrameShadow(QFrame.Shadow.Sunken)
        statusbar.addPermanentWidget(_sep)

        # Processing time
        self._status_time = QLabel("Processing: --")
        statusbar.addPermanentWidget(self._status_time)

        _sep2 = QFrame()
        _sep2.setFrameShape(QFrame.Shape.VLine)
        _sep2.setFrameShadow(QFrame.Shadow.Sunken)
        statusbar.addPermanentWidget(_sep2)

        # Sample count
        self._status_samples = QLabel("Samples: --")
        statusbar.addPermanentWidget(self._status_samples)

        # Add stretch spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        statusbar.addPermanentWidget(spacer)

        # Progress bar (hidden by default)
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        statusbar.addPermanentWidget(self._progress_bar)

    def _init_central_widget(self) -> None:
        """Initialize central widget."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Create horizontal splitter
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self._main_splitter)

        # Left panel (wider so preprocessing / channel cards are usable)
        self._left_panel = self._create_left_panel()
        self._left_panel.setMinimumWidth(300)
        self._left_panel.setMaximumWidth(480)
        self._main_splitter.addWidget(self._left_panel)

        # Right main area
        self._right_panel = self._create_right_panel()
        self._main_splitter.addWidget(self._right_panel)

        # Set stretch factors
        self._main_splitter.setStretchFactor(0, 0)  # Left fixed
        self._main_splitter.setStretchFactor(1, 1)  # Right adaptive

    def _create_left_panel(self) -> QWidget:
        """Create left panel (channel config area)."""
        from ChannelSimulation.viewer.ui.channel_config_panel import ChannelConfigPanel

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Signal overview + metrics (each block collapsible)
        from ChannelSimulation.viewer.ui.signal_info_panel import SignalInfoPanel
        self._signal_info = SignalInfoPanel()
        layout.addWidget(self._signal_info)

        # Random seed (collapsible)
        from ChannelSimulation.viewer.ui.collapsible_section import CollapsibleSection
        seed_core = self._create_seed_widget()
        seed_section = CollapsibleSection("Random Seed", seed_core, expanded=True)
        layout.addWidget(seed_section)

        # Channel / preprocessing — takes all remaining vertical space
        self._channel_config = ChannelConfigPanel()
        self._channel_config.config_changed.connect(self._on_config_changed)
        self._channel_config.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        layout.addWidget(self._channel_config, stretch=1)

        return panel

    def _create_right_panel(self) -> QWidget:
        """Create right panel (spectrogram display area)."""
        from ChannelSimulation.viewer.ui.spectrum_viewer import SpectrumViewer

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Spectrogram viewer
        self._spectrum_viewer = SpectrumViewer()
        layout.addWidget(self._spectrum_viewer)

        return panel

    def _create_seed_widget(self) -> QWidget:
        """Create random seed input widget (content only; title from CollapsibleSection)."""
        from PySide6.QtWidgets import QHBoxLayout, QWidget

        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(8, 4, 8, 8)

        self._input_seed = self._create_line_edit(placeholder="42", width=80)
        self._input_seed.setText("42")
        self._input_seed.setToolTip("Random seed for reproducibility")
        layout.addWidget(self._input_seed)

        return w

    def _create_line_edit(
        self,
        placeholder: str = "",
        width: int = 100,
    ) -> "QLineEdit":
        """Create styled line edit."""
        from PySide6.QtWidgets import QLineEdit
        le = QLineEdit()
        le.setPlaceholderText(placeholder)
        le.setMaximumWidth(width)
        return le

    def _get_icon_size(self) -> "QtCore.QSize":
        """Get toolbar icon size."""
        from PySide6.QtCore import QSize
        return QSize(20, 20)

    # =========================================================================
    # Signal Connections
    # =========================================================================

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.stage_config_changed.connect(self._on_stage_config_changed)
        self.preprocess_config_changed.connect(self._on_preprocess_config_changed)

    # =========================================================================
    # Menu and Toolbar Actions
    # =========================================================================

    def _on_load_iq(self) -> None:
        """Load IQ file."""
        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select IQ File",
            "",
            "IQ Files (*.iq *.cf32 *.cfile *.raw *.bin *.iq16 *.s16);;"
            "NumPy Files (*.npy *.npz);;"
            "MATLAB Files (*.mat);;"
            "All Files (*.*)",
        )

        if file_path:
            self._current_file = file_path
            self._emit_load_iq(file_path)
            self._update_ui_state()

    def _on_export_iq(self) -> None:
        """Export IQ file."""
        stage_id = self._spectrum_viewer.get_current_stage_id() if hasattr(
            self._spectrum_viewer, 'get_current_stage_id'
        ) else 0

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export IQ File",
            "",
            "IQ Files (*.iq);;All Files (*.*)",
        )

        if file_path:
            self.export_iq_triggered.emit(file_path, stage_id)

    def _on_export_fig(self) -> None:
        """Export spectrogram."""
        view_type = self._spectrum_viewer.get_current_view_type() if hasattr(
            self._spectrum_viewer, 'get_current_view_type'
        ) else 0

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Spectrogram",
            "",
            "PNG Images (*.png);;All Files (*.*)",
        )

        if file_path:
            self.export_fig_triggered.emit(file_path, view_type)

    def _on_save_config(self) -> None:
        """Save config."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
        )

        if file_path:
            self._save_config_to_file(file_path)

    def _on_load_config(self) -> None:
        """Load config."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Config",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*.*)",
        )

        if file_path:
            self._load_config_from_file(file_path)

    def _on_reset_config(self) -> None:
        """Reset config."""
        reply = QMessageBox.question(
            self,
            "Confirm Reset",
            "Are you sure you want to reset all parameters to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._reset_config()

    def _on_toggle_theme(self) -> None:
        """Toggle theme."""
        self._theme_dark = not self._theme_dark
        self._apply_theme(self._theme_dark)
        self.theme_changed.emit(self._theme_dark)

    def _on_toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _on_batch_process(self) -> None:
        """Open batch processing dialog."""
        from ChannelSimulation.viewer.advanced.batch_processor import BatchProcessorDialog

        dialog = BatchProcessorDialog(self)
        dialog.exec()

    def _on_cfo_estimator(self) -> None:
        """Open CFO estimator dialog."""
        from ChannelSimulation.viewer.business.cfo_estimator import CFOEstimatorDialog

        dialog = CFOEstimatorDialog(self)
        dialog.exec()

    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About RFUAV ChannelSim Viewer",
            """<h3>RFUAV ChannelSim Viewer</h3>
            <p>Version: 1.3.0</p>
            <p>Date: 2026-04-09</p>
            <p><br/>
            Desktop visualization tool for wireless communication signal
            channel simulation and spectrum analysis.
            </p>
            <p><br/>
            <b>Dependency Versions:</b><br/>
            PySide6 &ge; 6.5<br/>
            NumPy &ge; 1.24<br/>
            SciPy &ge; 1.10<br/>
            Matplotlib &ge; 3.7<br/>
            PyYAML &ge; 6.0
            </p>
            <p><br/>
            Licensed under MIT.
            </p>""",
        )

    def _on_show_shortcuts(self) -> None:
        """Show shortcuts dialog."""
        from ChannelSimulation.viewer.core.shortcuts import get_shortcut_table_html

        QMessageBox.information(
            self,
            "Shortcuts",
            f"<html><body>{get_shortcut_table_html()}</body></html>",
        )

    def _on_preview(self) -> None:
        """Preview button clicked."""
        if self._current_file:
            self._run_pipeline()

    def _on_zoom_in(self) -> None:
        """Zoom in."""
        if hasattr(self._spectrum_viewer, 'zoom_in'):
            self._spectrum_viewer.zoom_in()

    def _on_zoom_out(self) -> None:
        """Zoom out."""
        if hasattr(self._spectrum_viewer, 'zoom_out'):
            self._spectrum_viewer.zoom_out()

    # =========================================================================
    # Config Management
    # =========================================================================

    def _on_config_changed(self, config: dict) -> None:
        """Handle config change."""
        self._current_config = config
        self._is_modified = True
        self._action_modified_indicator.setVisible(True)
        self._update_ui_state()
        self.stage_config_changed.emit(config)

    def _on_stage_config_changed(self, config: dict) -> None:
        """Handle stage config change."""
        self._schedule_pipeline_run()

    def _on_preprocess_config_changed(self, config: dict) -> None:
        """Handle preprocess config change."""
        self._schedule_pipeline_run()

    def _schedule_pipeline_run(self) -> None:
        """Schedule pipeline run with debounce."""
        if hasattr(self, '_debounce_timer'):
            self._debounce_timer.stop()
        else:
            from PySide6.QtCore import QTimer
            self._debounce_timer = QTimer()
            self._debounce_timer.setSingleShot(True)
            self._debounce_timer.timeout.connect(self._run_pipeline)

        self._debounce_timer.start(200)

    def _run_pipeline(self) -> None:
        """Run pipeline."""
        if not self._current_file:
            return

        self._set_processing_state(True)
        self._status_indicator.setText("Processing...")

        # Get config
        config = self._get_current_config()

        # Get sampling rate
        try:
            fs = float(self._input_fs.text()) if self._input_fs.text() else 100_000_000
        except ValueError:
            fs = 100_000_000

        try:
            fc = float(self._input_fc.text()) if self._input_fc.text() else 0.0
        except ValueError:
            fc = 0.0

        # Get random seed
        try:
            seed = int(self._input_seed.text()) if self._input_seed.text() else 42
        except ValueError:
            seed = 42

        # Start background processing
        from ChannelSimulation.viewer.signal_proc.processing_worker import ProcessingWorker

        self._worker = ProcessingWorker(
            iq_data=None,
            config=config,
            fs=fs,
            fc=fc,
            seed=seed,
        )

        # Connect signals
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.progress.connect(self._on_processing_progress)
        self._worker.error.connect(self._on_processing_error)
        self._worker.metrics_ready.connect(self._on_metrics_ready)
        self._worker.spectrum_ready.connect(self._on_spectrum_ready)

        # Set IQ file path
        self._worker.set_input_file(self._current_file)

        self._worker.start()

    def _get_current_config(self) -> dict:
        """Get current config."""
        if hasattr(self._channel_config, 'get_config'):
            return self._channel_config.get_config()
        return {}

    def _set_processing_state(self, processing: bool) -> None:
        """Set processing state."""
        self._is_processing = processing
        self._progress_bar.setVisible(processing)

        if processing:
            self._btn_preview.setEnabled(False)
            self._action_export_iq.setEnabled(False)
        else:
            self._btn_preview.setEnabled(bool(self._current_file))
            self._action_export_iq.setEnabled(bool(self._current_file))

    def _on_processing_finished(self, result: dict) -> None:
        """Processing finished callback."""
        self._set_processing_state(False)
        self._status_indicator.setText("Ready")

        # Store IQ but defer redraw — spectrum arrives via _on_spectrum_ready
        if hasattr(self._spectrum_viewer, 'update_data'):
            self._spectrum_viewer.update_data(
                original=result.get('original'),
                processed=result.get('processed'),
                fs=result.get('fs', 100_000_000),
                fc=result.get('fc', 0.0),
            )

        # Update signal info
        if hasattr(self._signal_info, 'update_info'):
            info = result.get('info', {})
            self._signal_info.update_info(info)

        # Update elapsed time display
        elapsed = result.get('elapsed', 0)
        self._status_time.setText(f"Processing: {elapsed:.2f}s")

    def _on_processing_progress(self, value: float, stage: str) -> None:
        """Processing progress callback."""
        self._progress_bar.setValue(int(value * 100))
        self._status_indicator.setText(f"Processing: {stage}")

    def _on_processing_error(self, error_msg: str) -> None:
        """Processing error callback."""
        self._set_processing_state(False)
        self._status_indicator.setText("Error")

        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred during processing:\n\n{error_msg}",
        )

    def _on_spectrum_ready(self, spectrum_data: dict) -> None:
        """Apply pre-computed spectrum data from the background thread."""
        if hasattr(self._spectrum_viewer, 'apply_spectrum_data'):
            self._spectrum_viewer.apply_spectrum_data(spectrum_data)

    def _on_metrics_ready(self, metrics: dict) -> None:
        """Metrics ready callback."""
        if hasattr(self._signal_info, 'update_metrics'):
            self._signal_info.update_metrics(metrics)

    def _save_config_to_file(self, file_path: str) -> None:
        """Save config to file."""
        try:
            import yaml

            config = self._get_current_config()

            full_config = {
                'version': '1.0',
                'fs': float(self._input_fs.text()) if self._input_fs.text() else 100_000_000,
                'fc': float(self._input_fc.text()) if self._input_fc.text() else 0.0,
                'seed': int(self._input_seed.text()) if self._input_seed.text() else 42,
                **config,
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)

            self._is_modified = False
            self._action_modified_indicator.setVisible(False)
            self._status_indicator.setText(f"Config saved: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Failed to save config:\n\n{str(e)}")

    def _load_config_from_file(self, file_path: str) -> None:
        """Load config from file."""
        try:
            import yaml

            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not isinstance(config, dict):
                raise ValueError("Invalid config file format")

            # Update UI
            if 'fs' in config:
                self._input_fs.setText(str(config['fs']))
            if 'fc' in config:
                self._input_fc.setText(str(config['fc']))
            if 'seed' in config:
                self._input_seed.setText(str(config['seed']))

            # Update channel config
            if hasattr(self._channel_config, 'set_config'):
                self._channel_config.set_config(config)

            self._current_config = config
            self._is_modified = False
            self._action_modified_indicator.setVisible(False)
            self._status_indicator.setText(f"Config loaded: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to load config:\n\n{str(e)}")

    def _reset_config(self) -> None:
        """Reset config to defaults."""
        if hasattr(self._channel_config, 'reset'):
            self._channel_config.reset()

        self._input_fs.setText("100000000")
        self._input_fc.setText("0")
        self._input_seed.setText("42")

        self._current_config = {}
        self._is_modified = False
        self._action_modified_indicator.setVisible(False)
        self._status_indicator.setText("Config reset")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _emit_load_iq(self, file_path: str) -> None:
        """Emit load IQ signal."""
        self.load_iq_triggered.emit(file_path)

        from pathlib import Path
        self._status_samples.setText(f"File: {Path(file_path).name}")

    def _update_ui_state(self) -> None:
        """Update UI state."""
        has_file = self._current_file is not None

        self._btn_save.setEnabled(has_file or self._is_modified)
        self._btn_preview.setEnabled(has_file)
        self._action_export_iq.setEnabled(has_file)
        self._action_export_fig.setEnabled(has_file)
        self._action_save_config.setEnabled(has_file or self._is_modified)

        # Update title bar
        title = "RFUAV ChannelSim Viewer v1.3.0"
        if self._current_file:
            from pathlib import Path
            title += f" - {Path(self._current_file).name}"
        if self._is_modified:
            title += " *"
        self.setWindowTitle(title)

    def _apply_stylesheet(self) -> None:
        """Apply stylesheet."""
        from ChannelSimulation.viewer.core.ui_style import get_all_stylesheet
        self.setStyleSheet(get_all_stylesheet())

    def _apply_theme(self, dark: bool) -> None:
        """Apply theme."""
        from ChannelSimulation.viewer.core.ui_style import apply_stylesheet, set_theme
        set_theme(dark)
        apply_stylesheet(QApplication.instance())

        # Update spectrogram colors
        if hasattr(self._spectrum_viewer, 'set_theme'):
            self._spectrum_viewer.set_theme(dark)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def set_theme(self, dark: bool) -> None:
        """Set theme."""
        self._theme_dark = dark
        self._apply_theme(dark)

    def get_current_file(self) -> str | None:
        """Get current file path."""
        return self._current_file

    def get_current_config(self) -> dict:
        """Get current config."""
        return self._get_current_config()

    def is_modified(self) -> bool:
        """Check if there are unsaved changes."""
        return self._is_modified

    def get_fs(self) -> float:
        """Get sampling rate."""
        try:
            return float(self._input_fs.text()) if self._input_fs.text() else 100_000_000
        except ValueError:
            return 100_000_000

    def get_fc(self) -> float:
        """Get center frequency."""
        try:
            return float(self._input_fc.text()) if self._input_fc.text() else 0.0
        except ValueError:
            return 0.0

    def get_seed(self) -> int:
        """Get random seed."""
        try:
            return int(self._input_seed.text()) if self._input_seed.text() else 42
        except ValueError:
            return 42

    # =========================================================================
    # Window Events
    # =========================================================================

    def closeEvent(self, event) -> None:
        """Handle window close event."""
        from ChannelSimulation.viewer.app import save_window_state

        # Save window state
        save_window_state(self)

        # Check for unsaved changes
        if self._is_modified:
            reply = QMessageBox.question(
                self,
                "Confirm Exit",
                "There are unsaved changes. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return

        # Stop background worker
        if hasattr(self, '_worker') and self._worker is not None:
            self._worker.stop()
            self._worker.wait(5000)

        event.accept()
