"""
Spectrum Viewer
~~~~~~~~~~~~~~~

Matplotlib QtAgg canvas integration with blit-accelerated rendering.
Supports four view modes: time domain, PSD, STFT, and residual spectrum.
Supports left-right comparison of original vs processed signals.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
except ImportError:
    FigureCanvasQTAgg = object  # type: ignore
    Figure = object  # type: ignore

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QCursor
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
from ChannelSimulation.viewer.signal_proc.processing_worker import StftWorker
from ChannelSimulation.viewer.core.ui_style import get_color, get_spectrogram_cmap


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
        self._is_dark_theme = False

        # Data
        self._data: SpectrumData | None = None
        self._raw_original: np.ndarray | None = None
        self._raw_processed: np.ndarray | None = None
        self._full_lim_orig: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._full_lim_proc: tuple[tuple[float, float], tuple[float, float]] | None = None
        # Saved limits are only valid for the view mode that created them (x/y semantics differ).
        self._saved_limits_view_type: int | None = None

        # STFT navigation (full recording bounds; not overwritten by zoom-recompute STFT crops)
        self._nav_t0: float | None = None
        self._nav_t1: float | None = None
        self._nav_f_lo_mhz: float | None = None
        self._nav_f_hi_mhz: float | None = None
        self._stft_center_t: float = 0.0
        self._stft_center_f_mhz: float = 0.0
        self._slider_nav_block: bool = False

        # Playback state
        self._is_playing = False
        self._play_frame = 0

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

        # Stage selector
        self._stage_selector = QComboBox()
        self._stage_selector.addItems(["Original Signal", "Processed Signal"])
        self._stage_selector.currentIndexChanged.connect(self._on_stage_changed)
        toolbar_layout.addWidget(QLabel("Signal:"))
        toolbar_layout.addWidget(self._stage_selector)

        # Play button
        self._btn_play = QLabel("▶")
        self._btn_play.setStyleSheet(f"color: {get_color('accent')}; font-size: 16px; cursor: pointer;")
        self._btn_play.mousePressEvent = lambda e: self._toggle_play()
        self._btn_play.setToolTip("Play animation")
        toolbar_layout.addWidget(self._btn_play)

        layout.addWidget(toolbar)

        # Spectrogram: time / frequency span sliders (no scroll-wheel zoom)
        self._stft_nav_row = QWidget()
        nav_layout = QHBoxLayout(self._stft_nav_row)
        nav_layout.setContentsMargins(8, 0, 8, 4)
        nav_layout.setSpacing(12)
        nav_layout.addWidget(QLabel("时间窗口:"))
        self._slider_time = QSlider(Qt.Orientation.Horizontal)
        self._slider_time.setRange(0, 1000)
        self._slider_time.setValue(1000)
        self._slider_time.setToolTip("向左：放大时间轴（更短时段）；向右：显示更长时段")
        self._slider_time.valueChanged.connect(self._on_stft_time_slider_changed)
        nav_layout.addWidget(self._slider_time, stretch=1)
        self._label_time_span = QLabel("—")
        self._label_time_span.setMinimumWidth(120)
        nav_layout.addWidget(self._label_time_span)
        nav_layout.addWidget(QLabel("频率窗口:"))
        self._slider_freq = QSlider(Qt.Orientation.Horizontal)
        self._slider_freq.setRange(0, 1000)
        self._slider_freq.setValue(1000)
        self._slider_freq.setToolTip("向左：放大频率轴（更窄频带）；向右：更宽频带")
        self._slider_freq.valueChanged.connect(self._on_stft_freq_slider_changed)
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

        # Playback timer
        self._play_timer = QTimer()
        self._play_timer.timeout.connect(self._on_play_timeout)

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
        self._ensure_data_for_current_view()
        self._update_spectrogram_canvas_modes()
        self._redraw()

    def _on_stage_changed(self, index: int) -> None:
        """Handle stage change"""
        self._current_stage_id = index
        self._ensure_data_for_current_view()
        self._redraw()

    def _toggle_play(self) -> None:
        """Toggle playback state"""
        self._is_playing = not self._is_playing
        if self._is_playing:
            self._btn_play.setText("⏸")
            self._play_timer.start(50)  # 50ms interval, ~20 FPS
        else:
            self._btn_play.setText("▶")
            self._play_timer.stop()

    def _on_play_timeout(self) -> None:
        """Playback timeout (animation frame update)"""
        if self._data is None:
            return

        self._play_frame = (self._play_frame + 1) % 100
        # TODO: Implement STFT window sliding animation
        self._redraw()

    def _ensure_data_for_current_view(self) -> None:
        if self._raw_original is None or self._raw_processed is None:
            return
        if self._data is None:
            self._data = SpectrumData(fs=self._calculator.fs, fc=self._calculator.fc)
        self._calculator.ensure_for_view(
            self._data,
            self._raw_original,
            self._raw_processed,
            self._current_view_type,
        )

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
        self._slider_nav_block = True
        self._slider_time.setValue(1000)
        self._slider_freq.setValue(1000)
        self._slider_nav_block = False
        self._stft_center_t = 0.5 * (self._nav_t0 + self._nav_t1)
        self._stft_center_f_mhz = 0.5 * (self._nav_f_lo_mhz + self._nav_f_hi_mhz)
        self._clamp_stft_centers()
        self._update_stft_nav_labels()

    @staticmethod
    def _frac_from_slider(s: int) -> float:
        """Map 0..1000 to visible fraction of full axis (0.1% .. 100%)."""
        s = int(np.clip(s, 0, 1000))
        return 0.001 + 0.999 * (s / 1000.0)

    def _compute_stft_view_rect(self) -> tuple[float, float, float, float] | None:
        """Return (t_lo, t_hi, f_lo_mhz, f_hi_mhz) or None."""
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
        t_lo = ct - half_t
        t_hi = ct + half_t
        f_lo = cf - half_f
        f_hi = cf + half_f
        return (t_lo, t_hi, f_lo, f_hi)

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
        t_lo, t_hi, f_lo, f_hi = rect
        canvases = [self._canvas_processed] if right_only else [self._canvas_original, self._canvas_processed]
        for c in canvases:
            if not c.figure.axes:
                continue
            ax = c.figure.axes[0]
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(f_lo, f_hi)
            c.draw_idle()

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
        self._clamp_stft_centers()
        self._update_stft_nav_labels()
        if self._current_view_type == 2:
            self._apply_stft_view_limits_to_figures()
        elif self._current_view_type == 3:
            self._apply_stft_view_limits_to_figures(right_only=True)
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
        if self._current_view_type == 2:
            self._apply_stft_view_limits_to_figures()
        else:
            self._apply_stft_view_limits_to_figures(right_only=True)

    def _on_spectrogram_pan_end(self) -> None:
        if self._current_view_type == 2:
            if self._stft_is_full_overview():
                return
            self._schedule_zoom_recompute()

    def _on_residual_pan_end(self) -> None:
        if self._current_view_type == 3:
            if self._stft_is_full_overview():
                return
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
            c.draw_idle()

    def _on_canvas_cursor_moved(self, x: float, y: float) -> None:
        """Called whenever the cursor moves over any canvas.
        Subclass or connect this to update a status readout.
        """
        pass

    def _schedule_zoom_recompute(self) -> None:
        """Debounce: wait for zoom to settle, then recompute STFT for visible window."""
        if self._current_view_type not in (2, 3):
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
        """Called on the main thread when the StftWorker finishes."""
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

        self._redraw()

    def _draw_time_domain(self) -> None:
        """Draw time domain plot"""
        if self._data is None:
            return

        fig_orig = self._canvas_original.figure
        fig_proc = self._canvas_processed.figure

        # Clear and redraw original signal
        fig_orig.clear()
        ax_orig = fig_orig.add_subplot(111)
        if self._data.time_domain_original is not None:
            ax_orig.plot(self._data.time_axis, self._data.time_domain_original, linewidth=0.8)
        ax_orig.set_xlabel("Time (s)")
        ax_orig.set_ylabel("|x(t)|")
        ax_orig.set_title("Original Signal - Time Domain")
        ax_orig.grid(True, alpha=0.3)
        self._canvas_original.draw()

        # Clear and redraw processed signal
        fig_proc.clear()
        ax_proc = fig_proc.add_subplot(111)
        if self._data.time_domain_processed is not None:
            ax_proc.plot(self._data.time_axis, self._data.time_domain_processed, linewidth=0.8)
        ax_proc.set_xlabel("Time (s)")
        ax_proc.set_ylabel("|x(t)|")
        ax_proc.set_title("Processed Signal - Time Domain")
        ax_proc.grid(True, alpha=0.3)
        self._canvas_processed.draw()

        self._colorbar.hide()

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
        self._canvas_processed.draw()

        self._colorbar.hide()

    def _draw_stft(self) -> None:
        """Draw STFT plot"""
        if self._data is None:
            return

        cmap = get_spectrogram_cmap()
        fig_orig = self._canvas_original.figure
        fig_proc = self._canvas_processed.figure

        # Original signal STFT
        fig_orig.clear()
        ax_orig = fig_orig.add_subplot(111)
        if self._data.stft_original is not None:
            vo, vc = self._stft_color_limits(self._data.stft_original)
            im = ax_orig.imshow(
                self._data.stft_original,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                vmin=vo,
                vmax=vc,
                extent=[
                    self._data.time_axis_stft.min(),
                    self._data.time_axis_stft.max(),
                    self._data.freq_axis_stft.min() / 1e6,
                    self._data.freq_axis_stft.max() / 1e6,
                ],
            )
            ax_orig.set_xlabel("Time (s)")
            ax_orig.set_ylabel("Frequency (MHz)")
            ax_orig.set_title("Original Signal - Spectrogram")
            if (
                not self._has_stft_nav()
                and self._full_lim_orig is not None
                and self._saved_limits_view_type == self._current_view_type
            ):
                (xl, xr), (yb, yt) = self._full_lim_orig
                ax_orig.set_xlim(xl, xr)
                ax_orig.set_ylim(yb, yt)
        self._canvas_original.draw()

        # Processed signal STFT
        fig_proc.clear()
        ax_proc = fig_proc.add_subplot(111)
        if self._data.stft_processed is not None:
            vp, vc = self._stft_color_limits(self._data.stft_processed)
            im = ax_proc.imshow(
                self._data.stft_processed,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                interpolation="nearest",
                vmin=vp,
                vmax=vc,
                extent=[
                    self._data.time_axis_stft.min(),
                    self._data.time_axis_stft.max(),
                    self._data.freq_axis_stft.min() / 1e6,
                    self._data.freq_axis_stft.max() / 1e6,
                ],
            )
            ax_proc.set_xlabel("Time (s)")
            ax_proc.set_ylabel("Frequency (MHz)")
            ax_proc.set_title("Processed Signal - Spectrogram")
            if (
                not self._has_stft_nav()
                and self._full_lim_proc is not None
                and self._saved_limits_view_type == self._current_view_type
            ):
                (xl, xr), (yb, yt) = self._full_lim_proc
                ax_proc.set_xlim(xl, xr)
                ax_proc.set_ylim(yb, yt)
        self._canvas_processed.draw()

        if self._has_stft_nav():
            self._apply_stft_view_limits_to_figures()
            self._update_stft_nav_labels()

        # Show colorbar
        self._colorbar.show()
        if self._data.stft_original is not None:
            vo, vc = self._stft_color_limits(self._data.stft_original)
            self._colorbar.set_range(vo, vc)

    def _stft_color_limits(self, stft: np.ndarray) -> tuple[float, float]:
        """Typical spectrogram dynamic range (similar to MATLAB -110..0 dB style)."""
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

        # Clear processed canvas (residual only needs one plot)
        fig_proc = self._canvas_processed.figure
        fig_proc.clear()
        ax_proc = fig_proc.add_subplot(111)
        if self._data.residual_stft is not None:
            im = ax_proc.imshow(
                self._data.residual_stft,
                aspect="auto",
                origin="lower",
                cmap="hot",
                interpolation="nearest",
                extent=[
                    self._data.time_axis_stft.min(),
                    self._data.time_axis_stft.max(),
                    self._data.freq_axis_stft.min() / 1e6,
                    self._data.freq_axis_stft.max() / 1e6,
                ],
            )
            ax_proc.set_xlabel("Time (s)")
            ax_proc.set_ylabel("Frequency (MHz)")
            ax_proc.set_title("Residual Spectrogram")
        self._canvas_processed.draw()
        if self._has_stft_nav():
            self._apply_stft_view_limits_to_figures(right_only=True)
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
        self._raw_original = original
        self._raw_processed = processed
        self._clear_saved_axis_limits()
        self._clear_stft_navigation()

        # Data placeholder — actual STFT/PSD will be injected via apply_spectrum_data
        self._data = SpectrumData(fs=fs, fc=fc)
        self._data.time_axis = np.arange(len(original), dtype=np.float64) / fs
        self._data.time_domain_original = np.abs(original)
        self._data.time_domain_processed = np.abs(processed)

        # Redraw in case spectrum data hasn't arrived yet
        self._redraw()

    def apply_spectrum_data(self, spectrum_data: dict) -> None:
        """Apply pre-computed STFT/PSD from the background ProcessingWorker.

        Called via ``spectrum_ready`` signal.  Must be thread-safe (Qt
        ensures the signal is delivered on the main thread).
        """
        if self._data is None or self._raw_original is None:
            return

        if spectrum_data.get("stft_original") is not None:
            self._data.stft_original = spectrum_data.get("stft_original")
            self._data.stft_processed = spectrum_data.get("stft_processed")
            self._data.freq_axis_stft = spectrum_data.get("freq_axis_stft")
            self._data.time_axis_stft = spectrum_data.get("time_axis_stft")
            t_ax = self._data.time_axis_stft
            f_ax = self._data.freq_axis_stft
            if t_ax is not None and f_ax is not None and len(t_ax) and len(f_ax):
                self._nav_t0 = float(np.min(t_ax))
                self._nav_t1 = float(np.max(t_ax))
                self._nav_f_lo_mhz = float(np.min(f_ax)) / 1e6
                self._nav_f_hi_mhz = float(np.max(f_ax)) / 1e6
                self._reset_stft_navigation()

        if spectrum_data.get("psd_original") is not None:
            self._data.psd_original = spectrum_data.get("psd_original")
            self._data.psd_processed = spectrum_data.get("psd_processed")
            self._data.freq_axis_psd = spectrum_data.get("freq_axis_psd")

        if spectrum_data.get("residual_stft") is not None:
            self._data.residual_stft = spectrum_data.get("residual_stft")
            self._data.residual_psd = spectrum_data.get("residual_psd")

        if spectrum_data.get("stft_original") is not None or spectrum_data.get("psd_original") is not None:
            self._clear_saved_axis_limits()

        self._redraw()

    def set_theme(self, dark: bool) -> None:
        """Set theme"""
        self._is_dark_theme = dark
        self._ensure_data_for_current_view()
        self._redraw()

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
            if not canvas.figure.axes:
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
            canvas.draw_idle()
        self._schedule_zoom_recompute()

    def reset_zoom(self) -> None:
        """Restore axes to the full data extent for the current view."""
        self._zoom_recompute_timer.stop()
        self._stft_worker.stop()
        self._clear_saved_axis_limits()
        if self._nav_t0 is not None:
            self._reset_stft_navigation()
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

        # Interaction events
        self._setup_events()

    def set_interaction_mode(self, mode: str) -> None:
        """default: wheel zoom. spectrogram: hand pan only, no wheel zoom."""
        self._interaction_mode = mode if mode in ("default", "spectrogram") else "default"
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

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
        """Spectrogram: left-drag pans (hand). Otherwise: cursor readout."""
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
            return
        if self._interaction_mode == "spectrogram":
            self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        if event.xdata is not None and event.ydata is not None:
            self.cursor_moved.emit(event.xdata, event.ydata)

    def _on_button_press(self, event) -> None:
        """Mouse button press: start drag-pan (spectrogram mode)."""
        if event.button == 1 and event.inaxes is not None:
            self._is_dragging = True
            self._last_pos = (event.xdata, event.ydata)
            if self._interaction_mode == "spectrogram":
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def _on_button_release(self, event) -> None:
        """Mouse button release: end drag-pan."""
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
