"""
Spectrum Viewer
~~~~~~~~~~~~~~~

Matplotlib QtAgg canvas integration with blit-accelerated rendering.
Supports four view modes: time domain, PSD, STFT, and residual spectrum.
Supports left-right comparison of original vs processed signals.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvasQTAgg = object  # type: ignore
    Figure = object  # type: ignore

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QCursor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator, SpectrumData
from ChannelSimulation.viewer.signal_proc.processing_worker import (
    StftWorker, _SPECTRAL_CACHE, ProcessingWorker,
    _VIEW_TIME, _VIEW_PSD, _VIEW_STFT, _VIEW_RESIDUAL, _VIEW_STFT_ZOOM,
)
from ChannelSimulation.viewer.core.ui_style import get_color, get_spectrogram_cmap

# Stage order (matches pipeline_runner STAGE_ORDER)
ALL_STAGE_NAMES = [
    "original",
    "preprocess",
    "fading",
    "doppler",
    "multipath",
    "cfo",
    "phase_noise",
    "iq_imbalance",
    "awgn",
    "crop_time_shift",
]

STAGE_DISPLAY_LABELS = {
    "original": "Original",
    "preprocess": "Preprocess",
    "fading": "Fading",
    "doppler": "Doppler",
    "multipath": "Multipath",
    "cfo": "CFO",
    "phase_noise": "Phase Noise",
    "iq_imbalance": "IQ Imbalance",
    "awgn": "AWGN",
    "crop_time_shift": "Crop+Shift",
    "final": "Processed",
}

PIPELINE_STAGE_ORDER = [
    "original",
    "preprocess",
    "fading",
    "doppler",
    "multipath",
    "cfo",
    "phase_noise",
    "iq_imbalance",
    "awgn",
    "crop_time_shift",
]


class PipelineFlowWidget(QLabel):
    """Mini pipeline flow display showing active stages as colored dots connected by lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active_stages: list[str] = []
        self._enabled_stages: set[str] = set()
        self.setMinimumWidth(160)
        self.setMaximumWidth(160)
        self.setMinimumHeight(20)
        self.setMaximumHeight(20)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("─ ○ ─ ○ ─ ○ ─")
        self._tooltip_stages: list[tuple[str, bool]] = []

    def set_active_stages(self, stages: list[str]) -> None:
        """Set the available active stages list."""
        self._active_stages = stages
        self._update_display()

    def set_enabled_stages(self, enabled: set[str]) -> None:
        """Set which stages are enabled in current config."""
        self._enabled_stages = enabled
        self._update_display()

    def _update_display(self) -> None:
        """Update the pipeline flow display."""
        stages = self._active_stages or PIPELINE_STAGE_ORDER
        # Build mini text: first 4 chars of each stage name with separators
        parts = []
        labels = []
        for s in stages:
            abbrev = {
                "original": "Orig", "preprocess": "Pre", "fading": "Fade",
                "doppler": "Dop", "multipath": "Mul", "cfo": "CFO",
                "phase_noise": "PhN", "iq_imbalance": "IQ", "awgn": "AWGN",
                "crop_time_shift": "Crp",
            }.get(s, s[:3])
            labels.append(abbrev)

        # Show only first 4 and last 2 stages if many
        if len(labels) > 6:
            display = " › ".join(labels[:3]) + " ›…› " + " › ".join(labels[-2:])
        else:
            display = " › ".join(labels)

        self.setText(display)
        self._tooltip_stages = [(s, s in self._enabled_stages) for s in stages]
        self.setToolTip(
            "Pipeline: " + " | ".join(
                f"{s}{'*' if s in self._enabled_stages else ''}" for s in stages
            )
        )

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        self.setStyleSheet(
            f"border: 1px solid {get_color('accent')}; border-radius: 4px; padding: 2px 6px; "
            f"background: {get_color('bg_secondary')}; color: {get_color('accent')};"
        )

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.setStyleSheet(
            f"border: 1px solid {get_color('border')}; border-radius: 4px; padding: 2px 6px; "
            f"background: {get_color('bg_secondary')};"
        )


class SpectrumViewer(QWidget):
    """Spectrum Viewer

    Supports four view modes:
    - Time domain envelope
    - Power Spectral Density (PSD)
    - Short-Time Fourier Transform (STFT)
    - Residual spectrum
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calculator = SpectrumCalculator()
        self._current_view_type = 0  # 0=Time, 1=PSD, 2=STFT, 3=Residual
        self._current_stage_id = 0
        self._current_stage_name = "final"  # tracks current stage name for cache key matching
        self._is_dark_theme = False

        # Data
        self._data: SpectrumData | None = None
        self._raw_original: np.ndarray | None = None
        self._raw_processed: np.ndarray | None = None
        self._full_lim_orig: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._full_lim_proc: tuple[tuple[float, float], tuple[float, float]] | None = None
        # Saved limits are only valid for the view mode that created them (x/y semantics differ).
        self._saved_limits_view_type: int | None = None

        # Locked STFT color scale: survives across stacking processing runs so
        # the scale does not reset each time a new algorithm is added.
        self._stft_vmin_locked: float | None = None
        self._stft_vmax_locked: float | None = None
        self._is_first_processing: bool = True   # locks scale on first run after file load

        # Track the loaded signal id so we only clear the STFT scale lock
        # when the *file* changes, not when the same file is re-processed.
        self._loaded_signal_id: int | None = None

        # Cached imshow images and axes for incremental STFT updates during slider drag
        self._stft_im_orig: "matplotlib.image.AxesImage | None" = None
        self._stft_im_proc: "matplotlib.image.AxesImage | None" = None
        self._stft_ax_orig: "matplotlib.axes.Axes | None" = None
        self._stft_ax_proc: "matplotlib.axes.Axes | None" = None

        # Intermediate stage signals
        self._stage_signals: dict[str, np.ndarray] = {}
        self._available_stages: list[str] = []

        # Box selection state
        self._box_select_mode = False
        self._box_selection_start: tuple[float, float] | None = None
        self._box_selection_rect = None

        # STFT navigation (full recording bounds; not overwritten by zoom-recompute STFT crops)
        self._nav_t0: float | None = None
        self._nav_t1: float | None = None
        self._nav_f_lo_mhz: float | None = None
        self._nav_f_hi_mhz: float | None = None
        self._stft_center_t: float = 0.0
        self._stft_center_f_mhz: float = 0.0
        self._slider_nav_block: bool = False
        self._slider_debounce_timer: "QTimer | None" = None  # Debounce slider drag

        self._init_ui()
        self._setup_figure()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # View switch toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(8)

        # View selector dropdown
        self._view_selector = QComboBox()
        self._view_selector.addItems(["Time Domain", "Power Spectral Density (PSD)", "Spectrogram (STFT)", "Residual Spectrum"])
        self._view_selector.currentIndexChanged.connect(self._on_view_changed)
        toolbar_layout.addWidget(QLabel("View:"))
        toolbar_layout.addWidget(self._view_selector)

        toolbar_layout.addStretch()

        layout.addWidget(toolbar)

        # Spectrogram: time / frequency span sliders (no scroll-wheel zoom)
        self._stft_nav_row = QWidget()
        nav_layout = QHBoxLayout(self._stft_nav_row)
        nav_layout.setContentsMargins(8, 0, 8, 4)
        nav_layout.setSpacing(12)
        nav_layout.addWidget(QLabel("Time Window:"))
        self._slider_time = QSlider(Qt.Orientation.Horizontal)
        self._slider_time.setRange(0, 1000)
        self._slider_time.setValue(1000)
        self._slider_time.setToolTip("Left: zoom in time axis (shorter period); Right: show longer period")
        self._slider_time.valueChanged.connect(self._on_stft_time_slider_changed)
        self._slider_time.sliderMoved.connect(self._on_stft_time_slider_changed)
        nav_layout.addWidget(self._slider_time, stretch=1)
        self._label_time_span = QLabel("—")
        self._label_time_span.setMinimumWidth(120)
        nav_layout.addWidget(self._label_time_span)
        nav_layout.addWidget(QLabel("Freq Window:"))
        self._slider_freq = QSlider(Qt.Orientation.Horizontal)
        self._slider_freq.setRange(0, 1000)
        self._slider_freq.setValue(1000)
        self._slider_freq.setToolTip("Left: zoom in frequency axis (narrower band); Right: wider band")
        self._slider_freq.valueChanged.connect(self._on_stft_freq_slider_changed)
        self._slider_freq.sliderMoved.connect(self._on_stft_freq_slider_changed)
        nav_layout.addWidget(self._slider_freq, stretch=1)
        self._label_freq_span = QLabel("—")
        self._label_freq_span.setMinimumWidth(120)
        nav_layout.addWidget(self._label_freq_span)
        layout.addWidget(self._stft_nav_row)
        self._stft_nav_row.hide()

        # Spectrum area (vertical split: original vs processed)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)

        # Original signal canvas
        self._canvas_original = SpectrumCanvas(parent=self)
        self._splitter.addWidget(self._canvas_original)

        # Processed signal canvas
        self._canvas_processed = SpectrumCanvas(parent=self)
        self._splitter.addWidget(self._canvas_processed)

        # Set initial split ratio
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)

        layout.addWidget(self._splitter)

        # Colorbar
        self._colorbar = ColorBarWidget()
        layout.addWidget(self._colorbar)

        self._canvas_original._spectrum_zoom_reset = lambda: self.reset_zoom()
        self._canvas_processed._spectrum_zoom_reset = lambda: self.reset_zoom()

        # Connect SpectrumCanvas forwarded signals to parent handlers
        for canvas in (self._canvas_original, self._canvas_processed):
            canvas.scroll_zoomed.connect(self._on_canvas_scroll_zoomed)
            canvas.reset_requested.connect(self.reset_zoom)
            canvas.cursor_moved.connect(self._on_canvas_cursor_moved)
            canvas.spectrogram_pan_step.connect(self._on_spectrogram_pan_step)
            canvas.spectrogram_pan_end.connect(self._on_spectrogram_pan_end)

        self._canvas_processed.spectrogram_pan_end.connect(self._on_residual_pan_end)

        # Debounce timer for zoom-recompute (fires after zoom settles)
        self._zoom_recompute_timer = QTimer()
        self._zoom_recompute_timer.setSingleShot(True)
        self._zoom_recompute_timer.timeout.connect(self._do_zoom_recompute)

        # Slider drag debounce: applies view-limits immediately, delays recompute
        self._slider_debounce_timer = QTimer()
        self._slider_debounce_timer.setSingleShot(True)
        self._slider_debounce_timer.timeout.connect(self._do_slider_delayed_recompute)

        # Pan drag debounce: queues draw_idle only after settle (~50 ms)
        self._pan_debounce_timer = QTimer()
        self._pan_debounce_timer.setSingleShot(True)
        self._pan_debounce_timer.timeout.connect(self._do_pan_delayed_draw)

        # Dedicated background thread for zoom-recompute STFT
        self._stft_worker = StftWorker(self)
        self._stft_worker.stft_done.connect(self._on_stft_worker_done)

        self._update_spectrogram_canvas_modes()

    def _setup_figure(self) -> None:
        """Setup Matplotlib canvas"""
        # Use the canvas's own figure (created during SpectrumCanvas init)
        fig_orig = self._canvas_original.figure
        fig_proc = self._canvas_processed.figure

        # Initial draw
        self._canvas_original.initial_draw()
        self._canvas_processed.initial_draw()

    def _on_view_changed(self, index: int) -> None:
        """Handle view mode change"""
        self._current_view_type = index
        self._clear_saved_axis_limits()
        # Data must be populated before drawing, and navigation must reset AFTER drawing
        # (so canvas has an imshow image for _update_stft_extent_from_sliders to find)
        self._ensure_data_for_current_view()
        self._redraw()
        if index in (2, 3) and self._nav_t0 is not None:
            self._reset_stft_navigation()
        self._update_spectrogram_canvas_modes()

    def _ensure_data_for_current_view(self) -> None:
        """Populate _data with spectrum for the current stage + view_type.

        Tries the L2 spectrum cache first (instant, no recompute).
        Falls back to synchronous SpectrumCalculator computation if cache miss.
        """
        if self._raw_original is None or self._raw_processed is None:
            return
        if self._data is None:
            self._data = SpectrumData(fs=self._calculator.fs, fc=self._calculator.fc)

        sig_id = id(self._raw_original)
        # Resolve stage name from selector index
        idx = self._current_stage_id
        stage_name = (
            self._available_stages[idx]
            if self._available_stages and idx < len(self._available_stages)
            else "final"
        )
        view_type = self._current_view_type

        # ── Try L2 cache first ────────────────────────────────────────────
        cached = _SPECTRAL_CACHE.get(stage_name, view_type, sig_id)
        if cached is not None:
            self._apply_cached_spectrum(cached)
            return

        # ── Cache miss: compute synchronously ─────────────────────────────
        self._calculator.ensure_for_view(
            self._data,
            self._raw_original,
            self._raw_processed,
            view_type,
        )

        # Populate STFT navigation bounds whenever STFT data is available,
        # so the sliders work even before the background worker has delivered
        # spectrum_ready (e.g. after switching to STFT view without waiting).
        if view_type in (2, 3) and self._data.stft_original is not None:
            self._sync_stft_nav_bounds()

    def _apply_cached_spectrum(self, cached: dict) -> None:
        """Apply cached spectrum data to _data, then sync navigation bounds.

        For STFT views (type 2/3) also triggers a redraw so the im refs
        (_stft_im_orig / _stft_im_proc) are refreshed — needed so the
        slider drag handler can update extent via those refs.
        """
        if self._data is None:
            return
        self._data.time_axis = cached.get("time_axis", self._data.time_axis)
        self._data.time_domain_original = cached.get("time_domain_original")
        self._data.time_domain_processed = cached.get("time_domain_processed")
        self._data.stft_original = cached.get("stft_original")
        self._data.stft_processed = cached.get("stft_processed")
        self._data.freq_axis_stft = cached.get("freq_axis_stft")
        self._data.time_axis_stft = cached.get("time_axis_stft")
        self._data.psd_original = cached.get("psd_original") if self._current_stage_name == "final" else self._data.psd_original
        self._data.psd_processed = cached.get("psd_processed")
        self._data.freq_axis_psd = cached.get("freq_axis_psd")
        self._data.residual_stft = cached.get("residual_stft")
        self._data.residual_psd = cached.get("residual_psd")

        if self._current_view_type in (2, 3) and self._data.stft_original is not None:
            self._sync_stft_nav_bounds()
            self._redraw()  # so _stft_im_orig/_stft_im_proc refs get set in _draw_stft
        elif self._data.psd_processed is not None:
            self._redraw()

    def _sync_stft_nav_bounds(self) -> None:
        """Update _nav_t0/_nav_t1/_nav_f_* from _data STFT axes."""
        if self._data is None:
            return
        t_ax = self._data.time_axis_stft
        f_ax = self._data.freq_axis_stft
        if t_ax is not None and len(t_ax) and f_ax is not None and len(f_ax):
            self._nav_t0 = float(np.min(t_ax))
            self._nav_t1 = float(np.max(t_ax))
            self._nav_f_lo_mhz = float(np.min(f_ax)) / 1e6
            self._nav_f_hi_mhz = float(np.max(f_ax)) / 1e6

    def _redraw(self) -> None:
        """Redraw spectrum"""
        if self._data is None:
            return

        # Draw based on view type
        if self._current_view_type == 0:
            self._draw_time_domain()
        elif self._current_view_type == 1:
            self._draw_psd()
        elif self._current_view_type == 2:
            self._draw_stft()
        elif self._current_view_type == 3:
            self._draw_residual()

        self._capture_default_limits()

    def _clear_saved_axis_limits(self) -> None:
        self._full_lim_orig = None
        self._full_lim_proc = None
        self._saved_limits_view_type = None

    def _capture_default_limits(self) -> None:
        self._full_lim_orig = None
        self._full_lim_proc = None
        self._saved_limits_view_type = None
        if self._canvas_original.figure.axes:
            ax = self._canvas_original.figure.axes[0]
            self._full_lim_orig = (ax.get_xlim(), ax.get_ylim())
        if self._canvas_processed.figure.axes:
            ax = self._canvas_processed.figure.axes[0]
            self._full_lim_proc = (ax.get_xlim(), ax.get_ylim())
        self._saved_limits_view_type = self._current_view_type

    def _update_spectrogram_canvas_modes(self) -> None:
        """STFT: both canvases hand-pan + sliders. Residual: left PSD wheel-zoom, right STFT hand-pan."""
        is_stft = self._current_view_type == 2
        is_resid = self._current_view_type == 3
        self._stft_nav_row.setVisible(is_stft or is_resid)
        # Original canvas: always default in residual (line PSD), spectrogram otherwise
        self._canvas_original.set_interaction_mode("spectrogram" if is_stft else "default")
        # Processed canvas: spectrogram for both STFT and residual STFT
        self._canvas_processed.set_interaction_mode("spectrogram" if (is_stft or is_resid) else "default")

    def _clear_stft_navigation(self) -> None:
        self._nav_t0 = self._nav_t1 = None
        self._nav_f_lo_mhz = self._nav_f_hi_mhz = None

    def _has_stft_nav(self) -> bool:
        return (
            self._nav_t0 is not None
            and self._nav_t1 is not None
            and self._nav_f_lo_mhz is not None
            and self._nav_f_hi_mhz is not None
        )

    def _reset_stft_navigation(self) -> None:
        """Center view and full span after new overview spectrum."""
        if self._nav_t0 is None or self._nav_t1 is None:
            return
        # Set sliders to 1000 without triggering handler
        self._slider_nav_block = True
        self._slider_time.setValue(1000)
        self._slider_freq.setValue(1000)
        self._slider_nav_block = False
        self._stft_center_t = 0.5 * (self._nav_t0 + self._nav_t1)
        self._stft_center_f_mhz = 0.5 * (self._nav_f_lo_mhz + self._nav_f_hi_mhz)
        self._clamp_stft_centers()
        self._update_stft_nav_labels()
        # Apply the full-range extent to canvases so the image fills the window
        # (slider handler is blocked so we must apply here manually)
        self._update_stft_extent_from_sliders()

    @staticmethod
    def _frac_from_slider(s: int) -> float:
        """Map 0..1000 to visible fraction of full axis.

        slider=0   → 100% visible (full overview)
        slider=500 → 10%  visible (~200 ms / 1 MHz typical)
        slider=1000→ 1%   visible (zoomed in tight)
        """
        s = int(np.clip(s, 0, 1000))
        # Exponential so fine control near the left (zoomed-in) end
        return 1.0 * (0.01 ** (s / 1000.0))  # 100% at s=0 → 1% at s=1000

    def _compute_stft_view_rect(self) -> tuple[float, float, float, float] | None:
        """Return (x_lo, x_hi, y_lo, y_hi) in data units for the current visible window.

        X-axis = TIME (seconds), Y-axis = FREQUENCY (MHz).
        """
        if (
            self._nav_t0 is None
            or self._nav_t1 is None
            or self._nav_f_lo_mhz is None
            or self._nav_f_hi_mhz is None
        ):
            return None
        T = self._nav_t1 - self._nav_t0
        F = self._nav_f_hi_mhz - self._nav_f_lo_mhz
        if T <= 0 or F <= 0:
            return None
        ft = self._frac_from_slider(self._slider_time.value())
        ff = self._frac_from_slider(self._slider_freq.value())
        span_t = min(T, T * ft)
        span_f = min(F, F * ff)
        half_t = span_t * 0.5
        half_f = span_f * 0.5
        ct = float(np.clip(self._stft_center_t, self._nav_t0 + half_t, self._nav_t1 - half_t))
        cf = float(np.clip(self._stft_center_f_mhz, self._nav_f_lo_mhz + half_f, self._nav_f_hi_mhz - half_f))
        x_lo = ct - half_t   # time left
        x_hi = ct + half_t  # time right
        y_lo = cf - half_f  # freq bottom
        y_hi = cf + half_f  # freq top
        return (x_lo, x_hi, y_lo, y_hi)

    def _clamp_stft_centers(self) -> None:
        """Keep pan/zoom center inside allowed range for current slider spans."""
        if (
            self._nav_t0 is None
            or self._nav_t1 is None
            or self._nav_f_lo_mhz is None
            or self._nav_f_hi_mhz is None
        ):
            return
        T = self._nav_t1 - self._nav_t0
        F = self._nav_f_hi_mhz - self._nav_f_lo_mhz
        if T <= 0 or F <= 0:
            return
        ft = self._frac_from_slider(self._slider_time.value())
        ff = self._frac_from_slider(self._slider_freq.value())
        half_t = min(T, T * ft) * 0.5
        half_f = min(F, F * ff) * 0.5
        self._stft_center_t = float(
            np.clip(self._stft_center_t, self._nav_t0 + half_t, self._nav_t1 - half_t)
        )
        self._stft_center_f_mhz = float(
            np.clip(self._stft_center_f_mhz, self._nav_f_lo_mhz + half_f, self._nav_f_hi_mhz - half_f)
        )

    def _update_stft_nav_labels(self) -> None:
        rect = self._compute_stft_view_rect()
        if rect is None:
            self._label_time_span.setText("—")
            self._label_freq_span.setText("—")
            return
        t_lo, t_hi, f_lo, f_hi = rect
        self._label_time_span.setText(f"{(t_hi - t_lo) * 1e3:.2f} ms  ·  [{t_lo:.3f}, {t_hi:.3f}] s")
        self._label_freq_span.setText(f"{(f_hi - f_lo):.2f} MHz  ·  [{f_lo:.1f}, {f_hi:.1f}] MHz")

    def _apply_stft_view_limits_to_figures(self, *, right_only: bool = False) -> None:
        rect = self._compute_stft_view_rect()
        if rect is None:
            return
        x_lo, x_hi, y_lo, y_hi = rect  # x=time(s), y=freq(MHz)
        canvases = [self._canvas_processed] if right_only else [self._canvas_original, self._canvas_processed]
        for c in canvases:
            if not c.figure.axes:
                continue
            ax = c.figure.axes[0]
            ax.set_xlim(x_lo, x_hi)   # time on x-axis
            ax.set_ylim(y_lo, y_hi)   # frequency on y-axis
            # Do NOT call draw_idle() here — debounce timer at the end of
            # _on_stft_slider_moved will fire once after the user settles.

    def _stft_is_full_overview(self) -> bool:
        """True if current STFT matrices still cover (almost) full time & freq."""
        if self._raw_original is None or self._data is None or self._data.time_axis_stft is None:
            return False
        if self._data.freq_axis_stft is None:
            return False
        t_full = len(self._raw_original) / self._calculator.fs
        t_span = float(np.ptp(self._data.time_axis_stft))
        if t_full <= 0 or t_span < 0.5 * t_full:
            return False
        f_hz = float(np.ptp(self._data.freq_axis_stft))
        if f_hz < 0.45 * self._calculator.fs:
            return False
        return True

    def _on_stft_time_slider_changed(self, _value: int) -> None:
        self._on_stft_slider_moved()

    def _on_stft_freq_slider_changed(self, _value: int) -> None:
        self._on_stft_slider_moved()

    def _on_stft_slider_moved(self) -> None:
        if self._slider_nav_block:
            return
        if self._nav_t0 is None:
            return
        self._clamp_stft_centers()
        self._update_stft_nav_labels()
        self._update_stft_extent_from_sliders()
        self._slider_debounce_timer.start(150)

    def _update_stft_extent_from_sliders(self) -> None:
        """Crop STFT data to visible window and redraw — provides true zoom during slider drag."""
        if self._data is None:
            return
        rect = self._compute_stft_view_rect()
        if rect is None:
            return
        x_lo, x_hi, y_lo, y_hi = rect  # x=time(s), y=freq(MHz)

        t_ax = self._data.time_axis_stft
        f_ax = self._data.freq_axis_stft
        stft_orig = self._data.stft_original
        stft_proc = self._data.stft_processed

        if t_ax is None or f_ax is None or stft_orig is None:
            return

        # Crop time axis to visible window
        t_mask = (t_ax >= x_lo) & (t_ax <= x_hi)
        t_idx = np.where(t_mask)[0]
        if len(t_idx) < 2:
            return
        i0, i1 = t_idx[0], t_idx[-1] + 1

        # Crop freq axis to visible window (axis is in MHz)
        f_mhz = f_ax / 1e6
        f_mask = (f_mhz >= y_lo) & (f_mhz <= y_hi)
        f_idx = np.where(f_mask)[0]
        if len(f_idx) < 2:
            return
        j0, j1 = f_idx[0], f_idx[-1] + 1

        # Crop STFT matrices (nfreq, ntime)
        crop_orig = stft_orig[j0:j1, i0:i1]
        crop_t = t_ax[i0:i1]
        crop_f = f_ax[j0:j1]

        stft_proc_crop = stft_proc[j0:j1, i0:i1] if stft_proc is not None else None

        for sc, stft_crop in [
            (self._canvas_original, crop_orig),
            (self._canvas_processed, stft_proc_crop),
        ]:
            if sc is None or not sc.figure.axes or stft_crop is None:
                continue
            mpl_canvas = sc.figure.canvas
            # Prefer cached im ref; fall back to ax.images[-1] only if present.
            # Safe: residual view (type 3) left canvas has no image, so we skip it.
            im = getattr(mpl_canvas, '_cached_im', None)
            if im is None:
                ims = sc.figure.axes[0].images
                im = ims[-1] if ims else None
                if im is not None:
                    mpl_canvas._cached_im = im

            if im is None or im.axes is None:
                continue  # canvas has no image (e.g. PSD plot), skip

            im.set_data(stft_crop)
            im.set_extent([crop_t[0], crop_t[-1], y_lo, y_hi])
            sc.figure.axes[0].set_xlim(crop_t[0], crop_t[-1])
            sc.figure.axes[0].set_ylim(y_lo, y_hi)
            # Coalesce repaints during drag: draw_idle batches to one render per
            # event-loop turn instead of a blocking full draw per motion event.
            sc._background = None
            sc.draw_idle()

    def _do_slider_delayed_recompute(self) -> None:
        """Called ~150 ms after the last slider change — triggers zoom recompute."""
        if self._current_view_type not in (2, 3):
            return
        if self._stft_is_full_overview():
            return
        self._schedule_zoom_recompute()

    def _on_spectrogram_pan_step(self, dx: float, dy: float) -> None:
        if self._current_view_type not in (2, 3):
            return
        if self._nav_t0 is None:
            return
        self._stft_center_t -= dx
        self._stft_center_f_mhz -= dy
        self._clamp_stft_centers()
        self._update_stft_nav_labels()
        # Update imshow extent for instant visual feedback during drag.
        # draw_idle is debounced to avoid a draw per motion event (~60/s).
        self._update_stft_extent_from_sliders()
        self._pan_debounce_timer.start(50)  # batch consecutive pan steps

    def _do_pan_delayed_draw(self) -> None:
        """Called ~50 ms after last pan motion — flushes pending draw."""
        self._canvas_original.figure.canvas.draw()
        self._canvas_processed.figure.canvas.draw()

    def _on_spectrogram_pan_end(self) -> None:
        if self._current_view_type == 2:
            if self._stft_is_full_overview():
                return
            self._redraw()
            self._schedule_zoom_recompute()

    def _on_residual_pan_end(self) -> None:
        if self._current_view_type == 3:
            if self._stft_is_full_overview():
                return
            self._redraw()
            self._schedule_zoom_recompute()

    # =========================================================================
    # Zoom recompute (STFT re-render at higher resolution for visible window)
    # =========================================================================

    def _on_canvas_scroll_zoomed(self, source_canvas: object) -> None:
        """Time/PSD: wheel zoom. Spectrogram: disabled on canvas (use sliders)."""
        if self._current_view_type in (2, 3):
            return
        self._sync_spectrogram_axes_from(source_canvas)
        self._schedule_zoom_recompute()

    def _sync_spectrogram_axes_from(self, source_canvas: object) -> None:
        """Mirror x/y limits from the canvas that was zoomed to the sibling (comparison view)."""
        if not isinstance(source_canvas, SpectrumCanvas):
            return
        if not source_canvas.figure.axes:
            return
        ax_src = source_canvas.figure.axes[0]
        xl, xr = ax_src.get_xlim()
        yb, yt = ax_src.get_ylim()
        for c in (self._canvas_original, self._canvas_processed):
            if c is source_canvas or not c.figure.axes:
                continue
            ax = c.figure.axes[0]
            ax.set_xlim(xl, xr)
            ax.set_ylim(yb, yt)
            c.draw()

    def _on_canvas_cursor_moved(self, x: float, y: float) -> None:
        """Handle cursor position updates from canvas."""
        pass

    def _schedule_zoom_recompute(self) -> None:
        """Debounce: wait for zoom to settle, then recompute STFT for visible window."""
        if self._current_view_type not in (2, 3):
            return
        # Don't restart if slider debounce timer is already pending
        if self._slider_debounce_timer is not None and self._slider_debounce_timer.isActive():
            return
        self._zoom_recompute_timer.start(550)

    def _do_zoom_recompute(self) -> None:
        """Dispatch zoom-recompute to the StftWorker background thread."""
        if self._data is None:
            return
        if self._current_view_type not in (2, 3):
            return
        if self._raw_original is None or self._raw_processed is None:
            return

        rect = self._compute_stft_view_rect()
        if rect is None:
            return
        t_min, t_max, f_min_mhz, f_max_mhz = rect

        fs = self._calculator.fs
        if t_max <= t_min:
            return

        n_total = len(self._raw_original)
        if n_total == 0:
            return
        i_start = int(np.clip(np.floor(t_min * fs), 0, n_total - 1))
        i_end = int(np.clip(np.ceil(t_max * fs), i_start + 1, n_total))
        sig_orig = self._raw_original[i_start:i_end]
        sig_proc = self._raw_processed[i_start:i_end]

        t_total = n_total / fs
        zoom_factor = max(1.0, t_total / max(t_max - t_min, 1e-9))
        nperseg = int(min(8192, max(64, 256 * zoom_factor)))
        nperseg = min(nperseg, max(32, len(sig_orig) // 4))

        f_window = (float(f_min_mhz) * 1e6, float(f_max_mhz) * 1e6)

        # Stamp the zoom params so _on_stft_worker_done can match them
        self._zoom_t_min = t_min
        self._zoom_t_max = t_max

        self._stft_worker.start_stft(
            sig_key=id(self),
            sig_orig=sig_orig,
            sig_proc=sig_proc,
            f_window=f_window,
            nperseg=nperseg,
            fs=fs,
        )

    def _on_stft_worker_done(
        self,
        sig_key: object,
        so: np.ndarray,
        sp: np.ndarray,
        ff: np.ndarray,
        tt: np.ndarray,
    ) -> None:
        """Called on the main thread when the StftWorker finishes.

        Stores the zoomed STFT in L2 cache so repeating the same zoom window
        is instant (cache hit instead of re-compute).
        """
        if sig_key != id(self):
            return  # stale result from a previous load
        if self._data is None:
            return

        times_zoom = np.linspace(self._zoom_t_min, self._zoom_t_max, so.shape[1])
        self._data.stft_original = so
        self._data.stft_processed = sp
        self._data.freq_axis_stft = ff
        self._data.time_axis_stft = times_zoom

        if self._current_view_type == 3:
            self._data.residual_stft = self._calculator.compute_residual_stft(so, sp)

        # ── Write zoom STFT to L2 cache ────────────────────────────────────
        idx = self._current_stage_id
        stage_name = (
            self._available_stages[idx]
            if self._available_stages and idx < len(self._available_stages)
            else "final"
        )
        sig_id = id(self._raw_original)
        # zoom_key encodes the exact visible time/freq window so the same
        # viewport always hits the cache regardless of how it was reached.
        rect = self._compute_stft_view_rect()
        if rect is not None:
            t_min, t_max, f_min_mhz, f_max_mhz = rect
            zoom_key = f"t{t_min:.6f}_{t_max:.6f}_f{f_min_mhz:.3f}_{f_max_mhz:.3f}"
        else:
            zoom_key = None  # should not happen but safe fallback
        cache_entry = {
            "stage_name": stage_name,
            "signal_id": sig_id,
            "zoom_key": zoom_key,
            "stft_original": so,
            "stft_processed": sp,
            "freq_axis_stft": ff,
            "time_axis_stft": times_zoom,
            "residual_stft": self._data.residual_stft,
            "residual_psd": None,
            "time_axis": self._data.time_axis,
            "time_domain_original": self._data.time_domain_original,
            "time_domain_processed": self._data.time_domain_processed,
            "psd_original": self._data.psd_original,
            "psd_processed": self._data.psd_processed,
            "freq_axis_psd": self._data.freq_axis_psd,
        }
        _SPECTRAL_CACHE.put(stage_name, _VIEW_STFT_ZOOM, sig_id, cache_entry, zoom_key)

        self._redraw()

    def _draw_time_domain(self) -> None:
        """Draw time domain plot"""
        if self._data is None:
            return

        fig_orig = self._canvas_original.figure
        fig_proc = self._canvas_processed.figure

        # A: Original | B: Processed
        self._draw_time_canvas(fig_orig, self._data.time_domain_original, "Original Signal - Time Domain")
        self._draw_time_canvas(fig_proc, self._data.time_domain_processed, "Processed Signal - Time Domain")

        self._colorbar.hide()

    def _draw_time_canvas(self, fig, signal, title) -> None:
        """Draw time domain on a single canvas.

        Args:
            fig: Matplotlib Figure
            signal: Already-decimated abs envelope (same length as time_axis), NOT raw IQ
            title: Plot title
        """
        fig.clear()
        ax = fig.add_subplot(111)
        if signal is not None and self._data is not None and self._data.time_axis is not None:
            time_axis = self._data.time_axis
            n_signal = len(signal)
            n_time = len(time_axis)
            # Align: if lengths differ slightly due to integer division, use the shorter one
            n = min(n_signal, n_time)
            ax.plot(time_axis[:n], signal[:n], linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|x(t)|")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.canvas.draw()

    def _draw_psd(self) -> None:
        """Draw PSD plot"""
        if self._data is None:
            return

        fig_orig = self._canvas_original.figure
        fig_proc = self._canvas_processed.figure

        # Original signal PSD
        fig_orig.clear()
        ax_orig = fig_orig.add_subplot(111)
        if self._data.psd_original is not None and self._data.freq_axis_psd is not None:
            ax_orig.plot(self._data.freq_axis_psd / 1e6, self._data.psd_original, linewidth=0.8)
            ax_orig.set_xlabel("Frequency (MHz)")
            ax_orig.set_ylabel("Power Spectral Density (dB/Hz)")
            ax_orig.set_title("Original Signal - Power Spectral Density")
            ax_orig.grid(True, alpha=0.3)
            ax_orig.set_xlim([self._data.freq_axis_psd.min() / 1e6, self._data.freq_axis_psd.max() / 1e6])
            self._annotate_peaks(ax_orig, self._data.freq_axis_psd, self._data.psd_original, "#1a73e8")
        self._canvas_original.draw()

        # Processed signal PSD
        fig_proc.clear()
        ax_proc = fig_proc.add_subplot(111)
        if self._data.psd_processed is not None and self._data.freq_axis_psd is not None:
            ax_proc.plot(self._data.freq_axis_psd / 1e6, self._data.psd_processed, linewidth=0.8)
            ax_proc.set_xlabel("Frequency (MHz)")
            ax_proc.set_ylabel("Power Spectral Density (dB/Hz)")
            ax_proc.set_title("Processed Signal - Power Spectral Density")
            ax_proc.grid(True, alpha=0.3)
            ax_proc.set_xlim([self._data.freq_axis_psd.min() / 1e6, self._data.freq_axis_psd.max() / 1e6])
            self._annotate_peaks(ax_proc, self._data.freq_axis_psd, self._data.psd_processed, "#ea4335")
        self._canvas_processed.draw()

        self._colorbar.hide()

    def _draw_psd_canvas(self, fig, freqs, psd, title, color) -> None:
        """Draw PSD on a single canvas."""
        fig.clear()
        ax = fig.add_subplot(111)
        if psd is not None and freqs is not None:
            ax.plot(freqs / 1e6, psd, linewidth=0.8, color=color)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Power Spectral Density (dB/Hz)")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([freqs.min() / 1e6, freqs.max() / 1e6])
        ax.set_title(title)
        fig.canvas.draw()

    def _draw_stft(self) -> None:
        """Draw STFT plot"""
        if self._data is None:
            return

        cmap = get_spectrogram_cmap()

        # Normal mode: Original vs Processed
        im_orig = self._draw_stft_canvas(
            self._canvas_original.figure,
            self._data.stft_original,
            self._data.time_axis_stft,
            self._data.freq_axis_stft,
            "Original Signal - Spectrogram",
            cmap,
        )
        im_proc = self._draw_stft_canvas(
            self._canvas_processed.figure,
            self._data.stft_processed,
            self._data.time_axis_stft,
            self._data.freq_axis_stft,
            "Processed Signal - Spectrogram",
            cmap,
        )
        # Cache imshow refs + axes so slider drag can update extent without full redraw
        self._stft_im_orig = im_orig
        self._stft_im_proc = im_proc
        self._stft_ax_orig = self._canvas_original.figure.axes[0] if self._canvas_original.figure.axes else None
        self._stft_ax_proc = self._canvas_processed.figure.axes[0] if self._canvas_processed.figure.axes else None

        self._update_stft_nav_labels()

        # Show colorbar
        self._colorbar.show()
        if self._data.stft_original is not None:
            vo, vc = self._stft_color_limits(self._data.stft_original)
            self._colorbar.set_range(vo, vc)

    def _draw_stft_canvas(self, fig, stft, time_axis, freq_axis, title, cmap) -> "matplotlib.image.AxesImage | None":
        """Draw STFT spectrogram on a single canvas.

        imshow: X-axis = TIME (seconds), Y-axis = FREQUENCY (MHz).
        extent=[x_left, x_right, y_bottom, y_top].
        """
        fig.clear()
        ax = fig.add_subplot(111)
        if stft is not None and time_axis is not None and freq_axis is not None:
            # Determine visible extent: use current slider/view window if available
            rect = self._compute_stft_view_rect()
            if rect is not None and self._nav_t0 is not None:
                x_lo, x_hi, y_lo, y_hi = rect  # x=time(s), y=freq(MHz)
            else:
                x_lo = float(np.min(time_axis))
                x_hi = float(np.max(time_axis))
                y_lo = float(np.min(freq_axis)) / 1e6
                y_hi = float(np.max(freq_axis)) / 1e6

            vo, vc = self._stft_color_limits(stft)
            im = ax.imshow(
                stft,          # (nfreq, ntime) — matches imshow convention
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="nearest",  # fastest resampler; fine for spectrograms
                vmin=vo,
                vmax=vc,
                extent=[x_lo, x_hi, y_lo, y_hi],  # [left, right, bottom, top]
            )
            ax.set_xlim(x_lo, x_hi)   # time on x-axis
            ax.set_ylim(y_lo, y_hi)   # frequency on y-axis
            # Cache im ref on the canvas so slider drag can find it later
            fig.canvas._cached_im = im
            fig.canvas.draw()
            return im
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(title)
        fig.canvas.draw()
        return None

    def _stft_color_limits(self, stft: np.ndarray) -> tuple[float, float]:
        """Return (vmin, vmax) for the spectrogram.

        On the first processing run after a file load the scale is locked to
        the data and reused for every subsequent stacking run so the colourmap
        does not drift as algorithms are added/removed.
        """
        if self._stft_vmin_locked is not None and self._stft_vmax_locked is not None:
            return self._stft_vmin_locked, self._stft_vmax_locked

        a = np.asarray(stft, dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return -100.0, 0.0
        lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
        lo = max(lo, -120.0)
        hi = min(hi, 10.0)
        if hi - lo < 15:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 25.0, mid + 25.0
        return lo, hi

    def _draw_residual(self) -> None:
        """Draw residual spectrum"""
        if self._data is None:
            return

        fig_orig = self._canvas_original.figure

        # Residual PSD
        fig_orig.clear()
        ax_orig = fig_orig.add_subplot(111)
        if self._data.residual_psd is not None and self._data.freq_axis_psd is not None:
            ax_orig.plot(self._data.freq_axis_psd / 1e6, self._data.residual_psd, linewidth=0.8, color="#ea4335")
            ax_orig.set_xlabel("Frequency (MHz)")
            ax_orig.set_ylabel("Residual (dB)")
            ax_orig.set_title("Residual Power Spectrum")
            ax_orig.grid(True, alpha=0.3)
        self._canvas_original.draw()

        # Residual STFT: X-axis = time (s), Y-axis = frequency (MHz)
        fig_proc = self._canvas_processed.figure
        fig_proc.clear()
        ax_proc = fig_proc.add_subplot(111)
        if self._data.residual_stft is not None:
            rect = self._compute_stft_view_rect()
            if rect is not None and self._nav_t0 is not None:
                x_lo, x_hi, y_lo, y_hi = rect
            else:
                x_lo = float(self._data.time_axis_stft.min())
                x_hi = float(self._data.time_axis_stft.max())
                y_lo = float(self._data.freq_axis_stft.min()) / 1e6
                y_hi = float(self._data.freq_axis_stft.max()) / 1e6
            im = ax_proc.imshow(
                self._data.residual_stft,  # (nfreq, ntime)
                aspect="auto",
                origin="lower",
                cmap="hot",
                interpolation="nearest",  # fastest resampler; fine for spectrograms
                extent=[x_lo, x_hi, y_lo, y_hi],  # [left, right, bottom, top]
            )
            ax_proc.set_xlim(x_lo, x_hi)
            ax_proc.set_ylim(y_lo, y_hi)
            fig_proc.canvas._cached_im = im
            ax_proc.set_xlabel("Time (s)")
            ax_proc.set_ylabel("Frequency (MHz)")
            ax_proc.set_title("Residual Spectrogram")
        self._canvas_processed.draw()

        if self._has_stft_nav():
            self._update_stft_nav_labels()

        self._colorbar.show()
        if self._data.residual_stft is not None:
            vmin = np.nanmin(self._data.residual_stft)
            vmax = np.nanmax(self._data.residual_stft)
            self._colorbar.set_range(vmin, vmax)

    # =========================================================================
    # Public Interface
    # =========================================================================

    def update_data(
        self,
        original: np.ndarray | None = None,
        processed: np.ndarray | None = None,
        fs: float = 100_000_000,
        fc: float = 0.0,
        stage_signals: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Update spectrum data

        STFT / PSD are computed in the background thread and arrive via
        apply_spectrum_data (spectrum_ready signal).  This method only
        stores the raw IQ and triggers a redraw if spectrum data has not
        yet arrived.

        Args:
            original: Original IQ signal
            processed: Processed IQ signal
            fs: Sampling rate
            fc: Center frequency
            stage_signals: Intermediate stage signals dictionary
        """
        if original is None or processed is None:
            return

        self._calculator.fs = fs
        self._calculator.fc = fc
        new_sig_id = id(original)
        is_new_file = (self._loaded_signal_id != new_sig_id)
        self._loaded_signal_id = new_sig_id
        self._raw_original = original
        self._raw_processed = processed
        self._clear_saved_axis_limits()
        self._clear_stft_navigation()

        # Only reset the STFT scale lock when the file itself changes,
        # not when the same file is re-processed with different params.
        if is_new_file:
            self._is_first_processing = True
            self._stft_vmin_locked = None
            self._stft_vmax_locked = None
            _SPECTRAL_CACHE.clear()
        else:
            # Same file re-processed: keep the scale lock and navigation bounds
            # so the display zoom level is preserved.
            pass

        # Data placeholder — actual STFT/PSD will be injected via apply_spectrum_data
        # Use decimated time-domain for display to avoid 8-byte complex arrays retained
        self._data = SpectrumData(fs=fs, fc=fc)
        n = len(original)
        max_pts = 2_000_000
        step = max(1, n // max_pts)
        self._data.time_axis = np.arange(0, n, step, dtype=np.float64) / fs
        self._data.time_domain_original = np.abs(original[::step])
        self._data.time_domain_processed = np.abs(processed[::step])

        # Redraw in case spectrum data hasn't arrived yet
        self._redraw()

    def apply_spectrum_data(self, spectrum_data: dict) -> None:
        """Apply pre-computed STFT/PSD from the background ProcessingWorker.

        Called via ``spectrum_ready`` signal.  Must be thread-safe (Qt
        ensures the signal is delivered on the main thread).

        Data is already in L2 cache; this method just unpacks and applies it
        to _data so _redraw can use the high-resolution values.
        """
        if self._data is None:
            self._data = SpectrumData(fs=self._calculator.fs, fc=self._calculator.fc)

        did_stft = False

        if spectrum_data.get("stft_original") is not None:
            self._data.stft_original = spectrum_data.get("stft_original")
            self._data.stft_processed = spectrum_data.get("stft_processed")
            self._data.freq_axis_stft = spectrum_data.get("freq_axis_stft")
            self._data.time_axis_stft = spectrum_data.get("time_axis_stft")
            self._data.residual_stft = spectrum_data.get("residual_stft")
            self._data.residual_psd = spectrum_data.get("residual_psd")

            # Only sync nav bounds when the file itself changed (new sig_id).
            # For re-processing the same file, preserve the existing
            # navigation bounds so the zoom level / slider positions are kept.
            if self._is_first_processing:
                self._sync_stft_nav_bounds()

            did_stft = True

            # Lock the colour scale on the very first processing run after a file load.
            # Subsequent stacking runs will reuse it so the display does not jump.
            if self._is_first_processing and self._data.stft_original is not None:
                vo, vc = self._stft_color_limits(self._data.stft_original)
                self._stft_vmin_locked = vo
                self._stft_vmax_locked = vc
                self._is_first_processing = False

        if spectrum_data.get("psd_original") is not None:
            self._data.psd_original = spectrum_data.get("psd_original")
            self._data.psd_processed = spectrum_data.get("psd_processed")
            self._data.freq_axis_psd = spectrum_data.get("freq_axis_psd")

        # Re-draw with high-resolution data.
        # Only reset navigation when the file itself changed
        # (STFT bounds reflect original signal, not the processed variant).
        if did_stft and self._current_view_type in (2, 3):
            if self._is_first_processing:
                self._reset_stft_navigation()
            self._redraw()
        elif self._data.psd_processed is not None:
            self._redraw()

    def set_theme(self, dark: bool) -> None:
        """Set theme"""
        self._is_dark_theme = dark
        self._canvas_original._is_dark_theme = dark
        self._canvas_processed._is_dark_theme = dark
        self._ensure_data_for_current_view()
        self._redraw()

    def set_stage_signals(self, stage_signals: dict[str, np.ndarray]) -> None:
        """Store intermediate stage signals for the stage selector."""
        self._stage_signals = stage_signals
        # Build available stage list from stored signals.
        # Always include 'final' explicitly: ProcessingWorker.run() excludes it from
        # the returned dict so that L2 cache keys stay consistent ('final' is the
        # cache key for the last pipeline stage), but the selector needs it.
        self._available_stages = list(stage_signals.keys())
        if "final" not in self._available_stages:
            self._available_stages.append("final")

        # Set stage to "final" (processed signal) since that is the default view
        self._current_stage_id = len(self._available_stages) - 1
        self._current_stage_name = "final"

    def set_config(self, config: dict) -> None:
        """Update pipeline flow display with current config."""
        pass

    def clear_data(self) -> None:
        """Release all held signal and spectrum data to free memory.

        Call this when loading a new file or shutting down to prevent
        large arrays from being retained in GPU/CPU memory.
        """
        self._raw_original = None
        self._raw_processed = None
        self._stage_signals.clear()
        self._available_stages.clear()
        self._current_stage_name = "final"
        self._current_stage_id = 0
        self._current_view_type = 0
        self._data = None
        self._clear_saved_axis_limits()
        self._clear_stft_navigation()
        # Clear L2 spectrum cache
        _SPECTRAL_CACHE.clear()
        self._stft_im_orig = None
        self._stft_im_proc = None
        self._stft_ax_orig = None
        self._stft_ax_proc = None
        self._is_first_processing = True
        self._stft_vmin_locked = None
        self._stft_vmax_locked = None
        self._stft_worker.stop()
        self._zoom_recompute_timer.stop()
        self._slider_debounce_timer.stop()
        self._pan_debounce_timer.stop()
        self._loaded_signal_id = None
        self._is_first_processing = True

        # Release canvas figures
        for canvas in (self._canvas_original, self._canvas_processed):
            if canvas is not None and canvas.figure.axes:
                canvas.figure.clear()
                canvas.draw()

        # Clear stage selector
        if hasattr(self, '_stage_selector'):
            self._stage_selector.blockSignals(True)
            self._stage_selector.clear()
            self._stage_selector.addItems(["Original Signal", "Processed Signal"])
            self._stage_selector.blockSignals(False)

    def zoom_in(self) -> None:
        """Zoom in (narrow visible range; axis ticks reflect the window)."""
        if self._current_view_type in (2, 3) and self._nav_t0 is not None:
            self._slider_nav_block = True
            self._slider_time.setValue(max(0, self._slider_time.value() - 80))
            self._slider_freq.setValue(max(0, self._slider_freq.value() - 80))
            self._slider_nav_block = False
            self._on_stft_slider_moved()
            return
        self._zoom_axes(1 / 1.1)

    def zoom_out(self) -> None:
        """Zoom out."""
        if self._current_view_type in (2, 3) and self._nav_t0 is not None:
            self._slider_nav_block = True
            self._slider_time.setValue(min(1000, self._slider_time.value() + 80))
            self._slider_freq.setValue(min(1000, self._slider_freq.value() + 80))
            self._slider_nav_block = False
            self._on_stft_slider_moved()
            return
        self._zoom_axes(1.1)

    def _zoom_axes(self, scale: float) -> None:
        for canvas in (self._canvas_original, self._canvas_processed):
            if canvas is None or not canvas.figure.axes:
                continue
            ax = canvas.figure.axes[0]
            xl, xr = ax.get_xlim()
            yb, yt = ax.get_ylim()
            xc = 0.5 * (xl + xr)
            yc = 0.5 * (yb + yt)
            half_w = 0.5 * (xr - xl) * scale
            half_h = 0.5 * (yt - yb) * scale
            ax.set_xlim(xc - half_w, xc + half_w)
            ax.set_ylim(yc - half_h, yc + half_h)
            canvas.draw()
        self._schedule_zoom_recompute()

    def reset_zoom(self) -> None:
        """Restore axes to the full data extent for the current view."""
        self._zoom_recompute_timer.stop()
        self._slider_debounce_timer.stop()
        self._pan_debounce_timer.stop()
        self._stft_worker.stop()
        self._clear_saved_axis_limits()
        if self._nav_t0 is not None:
            self._reset_stft_navigation()
        if self._data is None:
            # No signal loaded yet (e.g. R / toolbar reset before any load).
            return
        self._data.stft_original = None
        self._data.stft_processed = None
        self._ensure_data_for_current_view()
        if self._data is not None and self._data.time_axis_stft is not None and self._data.freq_axis_stft is not None:
            t_ax = self._data.time_axis_stft
            f_ax = self._data.freq_axis_stft
            self._nav_t0 = float(np.min(t_ax))
            self._nav_t1 = float(np.max(t_ax))
            self._nav_f_lo_mhz = float(np.min(f_ax)) / 1e6
            self._nav_f_hi_mhz = float(np.max(f_ax)) / 1e6
            self._reset_stft_navigation()
        self._redraw()

    def _annotate_peaks(self, ax, freqs: np.ndarray, psd: np.ndarray, color: str) -> None:
        """Annotate top N peaks on the given axes."""
        peaks = self._calculator.find_peaks(psd, freqs, n_peaks=3)
        for peak in peaks:
            freq = peak.get("freq", 0)
            amp = peak.get("power", 0)
            ax.annotate(
                f"{freq / 1e6:.3f} MHz\n{amp:.1f} dB",
                xy=(freq / 1e6, amp),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=7,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor="none"),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
            )

    def get_current_view_type(self) -> int:
        """Get current view type"""
        return self._current_view_type

    def get_current_stage_id(self) -> int:
        """Get current stage ID"""
        return self._current_stage_id

    def export_figure(self, file_path: str, dpi: int = 300) -> None:
        """Export current spectrum

        Args:
            file_path: Export file path
            dpi: DPI setting
        """
        fig = self._canvas_original.figure
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")


class SpectrumCanvas(FigureCanvasQTAgg):
    """Spectrum Canvas

    Spectrum plotting component based on Matplotlib FigureCanvasQTAgg.
    Supports blit-accelerated rendering.

    All interactive events are forwarded to the parent SpectrumViewer via Qt
    signals so that zoom / pan / key-press logic lives in one place and avoids
    cross-reference attribute errors.
    """

    # Forwarded signals (pass ``self`` so parent can sync the other canvas)
    scroll_zoomed = Signal(object)  # SpectrumCanvas that was scrolled
    pan_dragged = Signal(float, float)  # dx, dy in data units
    reset_requested = Signal()      # R key / double-click
    cursor_moved = Signal(float, float)  # x, y in data units
    spectrogram_pan_step = Signal(float, float)  # dx, dy in data units (time, MHz)
    spectrogram_pan_end = Signal()

    def __init__(self, figure: Figure | None = None, parent=None):
        if figure is None:
            figure = Figure(dpi=150)

        super().__init__(figure)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.updateGeometry()

        # Background cache (for blit)
        self._background = None

        # Mouse state
        self._is_dragging = False
        self._last_pos = None
        self._interaction_mode: str = "default"  # "default" | "spectrogram"

        # Box selection state
        self._box_select_mode = False
        self._box_start: tuple[float, float] | None = None
        self._box_end: tuple[float, float] | None = None
        self._box_artist = None

        # Crosshair lines (persistent across redraws)
        self._cross_vline = None
        self._cross_hline = None
        self._cross_visible = True

        # Peak annotation
        self._peak_annotations = []

        self._is_dark_theme = False

        # Interaction events
        self._setup_events()

    def set_interaction_mode(self, mode: str) -> None:
        """default: wheel zoom. spectrogram: hand pan only, no wheel zoom."""
        self._interaction_mode = mode if mode in ("default", "spectrogram") else "default"
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def set_box_select_mode(self, enabled: bool) -> None:
        """Enable/disable box selection mode."""
        self._box_select_mode = enabled
        if not enabled:
            self._clear_box_selection()

    def _clear_box_selection(self) -> None:
        """Remove any box selection rectangle."""
        if self._box_artist is not None:
            self._box_artist.remove()
            self._box_artist = None
            self._box_start = None
            self._box_end = None
            self.draw_idle()

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _setup_events(self) -> None:
        """Setup interaction events"""
        self.mpl_connect("scroll_event", self._on_scroll)
        self.mpl_connect("motion_notify_event", self._on_motion)
        self.mpl_connect("button_press_event", self._on_button_press)
        self.mpl_connect("button_release_event", self._on_button_release)
        self.mpl_connect("key_press_event", self._on_key_press)

    def _on_scroll(self, event) -> None:
        """Mouse wheel: zoom (time/PSD). Spectrogram mode: disabled — use toolbar sliders."""
        if self._interaction_mode == "spectrogram":
            return
        if event.inaxes is None:
            return

        ax = event.inaxes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        scale = 1.1 if event.button == "up" else 1 / 1.1

        x = event.xdata
        y = event.ydata
        if x is not None and y is not None:
            new_xlim = [x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale]
            new_ylim = [y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale]
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

        self.draw_idle()
        self.scroll_zoomed.emit(self)

    def _on_motion(self, event) -> None:
        """Spectrogram: left-drag pans (hand). Box-select: draw rectangle. Otherwise: cursor readout with crosshair."""
        # Box selection preview
        if self._box_select_mode and self._is_dragging and self._box_start is not None:
            if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
                # Update box preview
                ax = event.inaxes
                x0 = min(self._box_start[0], event.xdata)
                x1 = max(self._box_start[0], event.xdata)
                y0 = min(self._box_start[1], event.ydata)
                y1 = max(self._box_start[1], event.ydata)

                # Remove old box
                if self._box_artist is not None:
                    self._box_artist.remove()

                edge_color = "#1a73e8" if not self._is_dark_theme else "#8ab4f8"
                fill_alpha = 0.15 if not self._is_dark_theme else 0.1
                self._box_artist = ax.axvspan(x0, x1, ymin=y0, ymax=y1,
                                              alpha=fill_alpha, color=edge_color,
                                              zorder=10)
                self.draw_idle()
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            return

        if self._interaction_mode == "spectrogram" and self._is_dragging and self._last_pos is not None:
            if event.inaxes is not None and event.xdata is not None and event.ydata is not None:
                dx = event.xdata - self._last_pos[0]
                dy = event.ydata - self._last_pos[1]
                self._last_pos = (event.xdata, event.ydata)
                self.spectrogram_pan_step.emit(float(dx), float(dy))
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            return
        if event.inaxes is None:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self._hide_crosshair()
            return

        if self._interaction_mode == "spectrogram":
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

        if event.xdata is not None and event.ydata is not None:
            self._update_crosshair(event.inaxes, event.xdata, event.ydata)
            self.cursor_moved.emit(event.xdata, event.ydata)

    def _update_crosshair(self, ax, x: float, y: float) -> None:
        """Update crosshair lines."""
        if not hasattr(ax, 'get_xlim'):
            return
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Remove old lines
        if self._cross_vline is not None:
            self._cross_vline.remove()
            self._cross_vline = None
        if self._cross_hline is not None:
            self._cross_hline.remove()
            self._cross_hline = None

        # Color adapts to theme
        line_color = "#aaaaaa" if self._is_dark_theme else "#666666"
        linestyle = "--"

        self._cross_vline = ax.axvline(x, color=line_color, linestyle=linestyle, linewidth=0.8, alpha=0.6)
        self._cross_hline = ax.axhline(y, color=line_color, linestyle=linestyle, linewidth=0.8, alpha=0.6)

        self._cross_dirty = True

    def _hide_crosshair(self) -> None:
        if self._cross_vline is not None:
            self._cross_vline.remove()
            self._cross_vline = None
        if self._cross_hline is not None:
            self._cross_hline.remove()
            self._cross_hline = None

    def _clear_crosshair(self) -> None:
        self._hide_crosshair()

    def _on_button_press(self, event) -> None:
        """Mouse button press: start drag-pan (spectrogram) or box selection."""
        if event.button == 1 and event.inaxes is not None:
            # Box selection mode
            if self._box_select_mode:
                self._box_start = (event.xdata, event.ydata)
                self._box_end = (event.xdata, event.ydata)
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                return

            self._is_dragging = True
            self._last_pos = (event.xdata, event.ydata)
            if self._interaction_mode == "spectrogram":
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _on_button_release(self, event) -> None:
        """Mouse button release: end drag-pan or complete box selection."""
        # Box selection: zoom to the selected region
        if self._box_select_mode and event.button == 1:
            if self._box_start is not None and event.inaxes is not None and event.xdata is not None:
                self._box_end = (event.xdata, event.ydata)
                self._apply_box_selection(event.inaxes)
            self._clear_box_selection()
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            return

        was_spectrogram_drag = self._interaction_mode == "spectrogram" and self._is_dragging
        self._is_dragging = False
        self._last_pos = None
        if was_spectrogram_drag and event.button == 1:
            self.spectrogram_pan_end.emit()
        self.setCursor(
            QCursor(
                Qt.CursorShape.OpenHandCursor
                if self._interaction_mode == "spectrogram" and self.underMouse()
                else Qt.CursorShape.ArrowCursor
            )
        )

    def _apply_box_selection(self, ax) -> None:
        """Apply box selection: zoom to the selected rectangle."""
        if self._box_start is None or self._box_end is None:
            return
        x0 = min(self._box_start[0], self._box_end[0])
        x1 = max(self._box_start[0], self._box_end[0])
        y0 = min(self._box_start[1], self._box_end[1])
        y1 = max(self._box_start[1], self._box_end[1])

        if x1 - x0 < 1e-10 and y1 - y0 < 1e-10:
            return  # Too small, ignore

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        self.draw_idle()
        self.scroll_zoomed.emit(self)

    def _on_key_press(self, event) -> None:
        """Forward key press to parent."""
        if event.key == "r":
            self.reset_requested.emit()
        elif event.key == " ":
            # Space: play/pause — handled elsewhere, skip
            pass

    def _reset_view(self) -> None:
        """Reset view (fallback, prefer parent via signal)."""
        for ax in self.figure.axes:
            ax.relim()
            ax.autoscale_view()
        self.draw_idle()

    def initial_draw(self) -> None:
        """Initial draw (cache background)."""
        self.draw()
        self._background = self.copy_from_bbox(self.figure.bbox)

    def update_plot(self) -> None:
        """Incremental update (using blit)."""
        if self._background is None:
            self.draw()
            self._background = self.copy_from_bbox(self.figure.bbox)
            return

        self.restore_region(self._background)
        for ax in self.figure.axes:
            for artist in ax.get_children():
                if artist.get_visible():
                    ax.draw_artist(artist)
        self.blit(self.figure.bbox)


class ColorBarWidget(QWidget):
    """Spectrum colorbar component"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vmin = -80
        self._vmax = 0

        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 4, 16, 4)
        layout.setSpacing(8)

        # Minimum value label
        self._label_min = QLabel(f"{self._vmin} dB")
        self._label_min.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        layout.addWidget(self._label_min)

        # Colorbar
        self._colorbar = ColorBarGradient()
        layout.addWidget(self._colorbar, stretch=1)

        # Maximum value label
        self._label_max = QLabel(f"{self._vmax} dB")
        self._label_max.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        layout.addWidget(self._label_max)

    def set_range(self, vmin: float, vmax: float) -> None:
        """Set color range"""
        self._vmin = vmin
        self._vmax = vmax
        self._label_min.setText(f"{vmin:.1f} dB")
        self._label_max.setText(f"{vmax:.1f} dB")


class ColorBarGradient(QWidget):
    """Color gradient bar"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(16)
        self.setMaximumHeight(20)

    def paintEvent(self, event) -> None:
        """Draw color gradient"""
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QLinearGradient, QPainter

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Create gradient
        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0.0, QApplication.instance().palette().window().color())
        gradient.setColorAt(1.0, QApplication.instance().palette().highlight().color())

        # Draw gradient rectangle
        rect = QRect(0, 0, self.width(), self.height())
        painter.fillRect(rect, gradient)

        # Draw border
        painter.setPen(QApplication.instance().palette().mid().color())
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
