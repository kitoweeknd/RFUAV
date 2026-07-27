"""
Spectrum Viewer (pyqtgraph)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-performance spectrum display built on pyqtgraph (Qt GraphicsScene, the
standard for real-time Python signal/SDR GUIs). Replaces the old matplotlib
FigureCanvas, whose ``fig.clear()`` + ``imshow`` + ``draw()`` teardown made every
STFT view-switch take ~700 ms.

Two side-by-side columns (Original | Processed), each a persistent ``PlotItem``
holding both a persistent ``ImageItem`` (spectrogram / residual) and a persistent
``PlotDataItem`` (time / PSD). Switching view or stage only calls
``setImage``/``setData`` + toggles visibility — no teardown, so switches are a few
milliseconds. A shared ``HistogramLUTItem`` is the real, draggable colorbar;
linked ``ViewBox`` es give native, synced pan/zoom of the two columns.

Four views: Time domain, PSD, Spectrogram (STFT), Residual.

Public interface preserved for MainWindow / CFO estimator:
``apply_processing_result``, ``apply_spectrum_data``, ``apply_stage_spectrum``,
``set_theme``, ``set_config``,
``clear_data``, ``zoom_in`` / ``zoom_out`` / ``reset_zoom``,
``get_current_view_type`` / ``get_current_stage_id``, ``export_figure``, and the
readable ``_raw_original`` / ``_raw_processed`` attributes.
"""

from __future__ import annotations

import numpy as np

import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumCalculator
from ChannelSimulation.viewer.core.ui_style import (
    get_color,
    get_pg_colormap,
    apply_pyqtgraph_theme,
)

# View types (kept stable for get_current_view_type compatibility)
_V_TIME, _V_PSD, _V_STFT, _V_RESID = 0, 1, 2, 3

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


class SpectrumViewer(QWidget):
    """pyqtgraph-based spectrum viewer with instant view/stage switching."""

    # Fixed initial time window shown right after a file loads: 1/8 of the full
    # span. Full frequency axis is always shown; pan/zoom reveals the rest.
    INITIAL_TIME_FRAC = 0.125

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calculator = SpectrumCalculator()

        # ── state ─────────────────────────────────────────────────────────
        self._current_view_type = _V_STFT
        self._current_stage_name = "final"
        self._current_stage_id = 0
        self._available_stages: list[str] = ["final"]

        # View-range control: the visible range is reset only on new data or a
        # view-type change — NOT on stage switch or theme change — so the user's
        # zoom/pan survives stage comparison. Persistent auto-range used to fight
        # every wheel/Zoom action; now ranging is one-shot and user-driven.
        self._needs_range_reset = True
        self._full_extent: tuple[float, float, float, float] | None = None

        self._raw_original: np.ndarray | None = None
        self._raw_processed: np.ndarray | None = None
        self._stage_signals: dict[str, np.ndarray] = {}

        # stage_name -> complete spectrum dict (all four views). The viewer owns
        # this map (fed by apply_spectrum_data / apply_stage_spectrum) instead of
        # reaching into the worker's global cache, so lookups never depend on the
        # worker's internal signal-id counter.
        self._stage_spectra: dict[str, dict] = {}

        # STFT colour-scale lock (survives stacking runs; reset on new file).
        self._levels: tuple[float, float] | None = None
        self._is_first_processing = True

        self._init_ui()

    # =====================================================================
    # UI construction
    # =====================================================================
    def _init_ui(self) -> None:
        apply_pyqtgraph_theme()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ── toolbar: view + stage selectors, cursor readout ────────────────
        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 4, 8, 0)
        bl.setSpacing(8)

        bl.addWidget(QLabel("View:"))
        self._view_combo = QComboBox()
        self._view_combo.addItems(["Time Domain", "PSD", "Spectrogram", "Residual"])
        self._view_combo.setCurrentIndex(self._current_view_type)
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        bl.addWidget(self._view_combo)

        bl.addSpacing(12)
        bl.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        self._stage_combo.setMinimumWidth(130)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        bl.addWidget(self._stage_combo)

        bl.addSpacing(16)
        hint = QLabel("Zoom: wheel = both · wheel/drag on an axis = that axis only")
        hint.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        bl.addWidget(hint)

        bl.addStretch()
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(
            f"color: {get_color('text_secondary')}; font-size: 11px;"
        )
        bl.addWidget(self._cursor_label)
        layout.addWidget(bar)

        # ── graphics layout: [Original] [Processed] [Histogram] ────────────
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setToolTip(
            "Wheel over a plot zooms both axes.\n"
            "Wheel or drag directly over the Time axis zooms time only "
            "(frequency scale fixed).\n"
            "Wheel or drag over the Frequency axis zooms frequency only "
            "(time scale fixed).\n"
            "Double-click / the [A] button fits everything."
        )
        layout.addWidget(self._glw, stretch=1)

        self._plot_orig = self._glw.addPlot(row=0, col=0)
        self._plot_proc = self._glw.addPlot(row=0, col=1)
        for p in (self._plot_orig, self._plot_proc):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setMenuEnabled(False)

        # Persistent artists per column (created once, updated in place). The
        # STFT is pre-decimated to <=~3000 columns, so pyqtgraph renders the
        # QImage ONCE in setImage and Qt just transforms that cached pixmap on
        # pan/zoom — no per-frame work. (autoDownsample was tried but recomputes
        # a mean-pool over the whole ~6M-px image on every pan/zoom/axis-drag
        # frame, which reintroduced lag and could fault at extreme zoom.)
        self._img_orig = pg.ImageItem()
        self._img_proc = pg.ImageItem()
        self._curve_orig = pg.PlotDataItem(pen=pg.mkPen(get_color("accent"), width=1))
        self._curve_proc = pg.PlotDataItem(pen=pg.mkPen(get_color("danger"), width=1))
        self._plot_orig.addItem(self._img_orig)
        self._plot_orig.addItem(self._curve_orig)
        self._plot_proc.addItem(self._img_proc)
        self._plot_proc.addItem(self._curve_proc)

        # Curves can hold up to ~2M points (time domain). Configure downsampling
        # + clip-to-view at the PlotItem level (AFTER addItem, which otherwise
        # re-applies the plot's off-by-default settings) so only the visible,
        # on-screen-resolution samples are drawn — snappy pan/zoom on line views.
        for p in (self._plot_orig, self._plot_proc):
            p.setDownsampling(auto=True, mode="peak")
            p.setClipToView(True)

        cmap = get_pg_colormap()
        for im in (self._img_orig, self._img_proc):
            im.setColorMap(cmap)

        # Shared colorbar / histogram (draggable levels = interactive contrast).
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._img_orig)
        self._hist.gradient.setColorMap(cmap)
        self._hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self._glw.addItem(self._hist, row=0, col=2)

        # Rate-limit the cursor readout: sigMouseMoved fires on every pixel of
        # motion, and updating a toolbar label that often makes the whole scene
        # feel laggy. SignalProxy coalesces to ~60 Hz.
        self._mouse_proxy = pg.SignalProxy(
            self._glw.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved
        )

        self._apply_view_layout()

    # =====================================================================
    # Selectors
    # =====================================================================
    def _on_view_changed(self, index: int) -> None:
        self._current_view_type = int(index)
        self._needs_range_reset = True  # new view semantics → refit once
        self._apply_view_layout()
        self._redraw()

    def _on_stage_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._available_stages):
            return
        self._current_stage_id = index
        self._current_stage_name = self._available_stages[index]
        self._redraw()

    def _rebuild_stage_combo(self) -> None:
        """Populate the stage dropdown from the available stages (Processed last)."""
        self._stage_combo.blockSignals(True)
        self._stage_combo.clear()
        for s in self._available_stages:
            self._stage_combo.addItem(STAGE_DISPLAY_LABELS.get(s, s))
        idx = self._available_stages.index(self._current_stage_name) \
            if self._current_stage_name in self._available_stages else len(self._available_stages) - 1
        self._stage_combo.setCurrentIndex(idx)
        self._current_stage_id = idx
        self._stage_combo.blockSignals(False)

    def _apply_view_layout(self) -> None:
        """Toggle artist visibility, axis labels, links and colorbar per view."""
        vt = self._current_view_type
        is_img_left = vt == _V_STFT
        is_img_right = vt in (_V_STFT, _V_RESID)

        self._img_orig.setVisible(is_img_left)
        self._curve_orig.setVisible(not is_img_left)
        self._img_proc.setVisible(is_img_right)
        self._curve_proc.setVisible(not is_img_right)

        # Colorbar only for image views; link it to the primary image.
        self._hist.setVisible(vt in (_V_STFT, _V_RESID))
        self._hist.setImageItem(self._img_orig if vt == _V_STFT else self._img_proc)

        # Axis labels
        if vt == _V_TIME:
            self._set_labels("Time (s)", "|x(t)|", "Time (s)", "|x(t)|")
        elif vt == _V_PSD:
            self._set_labels("Frequency (MHz)", "PSD (dB/Hz)", "Frequency (MHz)", "PSD (dB/Hz)")
        elif vt == _V_STFT:
            self._set_labels("Time (s)", "Frequency (MHz)", "Time (s)", "Frequency (MHz)")
        else:  # residual: left PSD line, right STFT image
            self._set_labels("Frequency (MHz)", "Residual (dB)", "Time (s)", "Frequency (MHz)")

        # Link the two columns only when both share the same axis semantics.
        link = vt in (_V_TIME, _V_PSD, _V_STFT)
        vb = self._plot_proc.getViewBox()
        vb.setXLink(self._plot_orig if link else None)
        vb.setYLink(self._plot_orig if link else None)

        self._plot_orig.setTitle(self._orig_title())
        self._plot_proc.setTitle(self._proc_title())

    def _set_labels(self, xo, yo, xp, yp) -> None:
        self._plot_orig.setLabel("bottom", xo)
        self._plot_orig.setLabel("left", yo)
        self._plot_proc.setLabel("bottom", xp)
        self._plot_proc.setLabel("left", yp)

    def _orig_title(self) -> str:
        return {"0": "Original — Time", "1": "Original — PSD",
                "2": "Original — Spectrogram", "3": "Residual PSD"}[str(self._current_view_type)]

    def _proc_title(self) -> str:
        label = STAGE_DISPLAY_LABELS.get(self._current_stage_name, self._current_stage_name)
        return {_V_TIME: f"{label} — Time", _V_PSD: f"{label} — PSD",
                _V_STFT: f"{label} — Spectrogram", _V_RESID: "Residual Spectrogram"}[self._current_view_type]

    # =====================================================================
    # Data intake (called by MainWindow)
    # =====================================================================
    def apply_processing_result(self, original, processed, fs, fc, stage_signals=None) -> None:
        """Store raw signals + stage list; spectra arrive via apply_spectrum_data."""
        self._raw_original = original
        self._raw_processed = processed
        self._calculator.fs = fs
        self._calculator.fc = fc

        stage_signals = dict(stage_signals) if stage_signals else {}
        stage_signals["original"] = original
        stage_signals["final"] = processed
        self._stage_signals = stage_signals

        # Ordered stage list for the selector: pipeline order, Processed (final) last.
        order = ["preprocess", "fading", "doppler", "multipath", "cfo",
                 "phase_noise", "iq_imbalance", "awgn", "crop_time_shift"]
        avail = [s for s in order if s in stage_signals]
        avail.append("final")
        self._available_stages = avail
        self._current_stage_name = "final"
        self._rebuild_stage_combo()

        # Reset the colour-scale lock so it re-derives for this run.
        self._is_first_processing = True
        self._levels = None
        self._needs_range_reset = True  # new file → apply fixed initial window
        self._redraw()

    def apply_spectrum_data(self, spectrum_data: dict) -> None:
        """Final-stage spectra (from spectrum_ready). Stores + redraws."""
        if not spectrum_data:
            return
        self._stage_spectra["final"] = spectrum_data
        # 'original' spectra live inside every stage dict (stft_original/psd_original).
        if self._current_stage_name == "final":
            self._redraw()

    def apply_stage_spectrum(self, stage_name: str, spectrum_data: dict) -> None:
        """Intermediate-stage spectra (from stage_spectrum_cached). Stores; redraws if shown."""
        if not spectrum_data:
            return
        self._stage_spectra[stage_name] = spectrum_data
        if stage_name == self._current_stage_name:
            self._redraw()

    # =====================================================================
    # Rendering
    # =====================================================================
    def _current_spectrum(self) -> dict | None:
        """Resolve the spectrum dict for the current stage (cache → fallback compute)."""
        d = self._stage_spectra.get(self._current_stage_name)
        if d is not None:
            return d
        # Fallback: the stage's spectra have not been precomputed yet (background
        # worker still running). Compute synchronously from the raw stage signal.
        sig = self._stage_signals.get(self._current_stage_name)
        base = self._stage_signals.get("original")
        if sig is None or base is None:
            return None
        stft_p, f_stft, t_stft = self._calculator.compute_stft(sig)
        psd_p, f_psd = self._calculator.compute_psd(sig)
        stft_o, _, _ = self._calculator.compute_stft(base)
        psd_o, _ = self._calculator.compute_psd(base)
        n = len(base)
        step = max(1, n // 2_000_000)
        d = {
            "time_axis": np.arange(0, n, step) / self._calculator.fs,
            "time_domain_original": np.abs(base[::step]),
            "time_domain_processed": np.abs(sig[::step]),
            "stft_original": stft_o, "stft_processed": stft_p,
            "freq_axis_stft": f_stft, "time_axis_stft": t_stft,
            "psd_original": psd_o, "psd_processed": psd_p, "freq_axis_psd": f_psd,
            "residual_stft": self._calculator.compute_residual_stft(stft_o, stft_p),
            "residual_psd": self._calculator.compute_residual_psd(psd_o, psd_p),
        }
        self._stage_spectra[self._current_stage_name] = d
        return d

    def _redraw(self) -> None:
        d = self._current_spectrum()
        if d is None:
            return
        vt = self._current_view_type
        if vt == _V_TIME:
            self._draw_time(d)
        elif vt == _V_PSD:
            self._draw_psd(d)
        elif vt == _V_STFT:
            self._draw_stft(d)
        else:
            self._draw_residual(d)
        self._plot_orig.setTitle(self._orig_title())
        self._plot_proc.setTitle(self._proc_title())
        # Range only once per new-data / view change; never on stage switch,
        # so a user's zoom persists while stepping through pipeline stages.
        if self._needs_range_reset:
            self._apply_initial_range()
            self._needs_range_reset = False

    def _draw_time(self, d: dict) -> None:
        t = d.get("time_axis")
        yo = d.get("time_domain_original")
        yp = d.get("time_domain_processed")
        if t is not None and yo is not None:
            n = min(len(t), len(yo)); self._curve_orig.setData(t[:n], yo[:n])
        if t is not None and yp is not None:
            n = min(len(t), len(yp)); self._curve_proc.setData(t[:n], yp[:n])

    def _draw_psd(self, d: dict) -> None:
        f = d.get("freq_axis_psd")
        po = d.get("psd_original")
        pp = d.get("psd_processed")
        if f is not None and po is not None:
            self._curve_orig.setData(f / 1e6, po)
        if f is not None and pp is not None:
            self._curve_proc.setData(f / 1e6, pp)

    def _draw_stft(self, d: dict) -> None:
        so = d.get("stft_original")
        sp = d.get("stft_processed")
        f = d.get("freq_axis_stft")
        t = d.get("time_axis_stft")
        if so is None or f is None or t is None:
            return
        levels = self._resolve_levels(so)
        self._set_image(self._img_orig, so, t, f, levels)
        if sp is not None:
            self._set_image(self._img_proc, sp, t, f, levels)
        # Refresh histogram (linked to orig) to the locked levels.
        self._hist.blockSignals(True)
        self._hist.setLevels(*levels)
        self._hist.blockSignals(False)

    def _draw_residual(self, d: dict) -> None:
        # left: residual PSD line
        rp = d.get("residual_psd")
        f = d.get("freq_axis_psd")
        if rp is not None and f is not None:
            self._curve_orig.setData(f / 1e6, rp)
        # right: residual STFT image
        rs = d.get("residual_stft")
        fs_ = d.get("freq_axis_stft")
        ts = d.get("time_axis_stft")
        if rs is not None and fs_ is not None and ts is not None:
            lo, hi = float(np.nanmin(rs)), float(np.nanmax(rs))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = -100.0, 0.0
            self._set_image(self._img_proc, rs, ts, fs_, (lo, hi))
            self._hist.blockSignals(True)
            self._hist.setLevels(lo, hi)
            self._hist.blockSignals(False)

    def _set_image(self, img: pg.ImageItem, arr, t_axis, f_axis, levels) -> None:
        """setImage + map to (time seconds, frequency MHz) via setRect. No teardown."""
        img.setImage(arr, autoLevels=False)
        img.setLevels(levels)
        t0, t1 = float(t_axis[0]), float(t_axis[-1])
        f0, f1 = float(f_axis[0]) / 1e6, float(f_axis[-1]) / 1e6
        img.setRect(QRectF(t0, f0, max(t1 - t0, 1e-12), max(f1 - f0, 1e-12)))
        # Remember the full data extent so ranging (initial window, limits,
        # Reset) can reference it without re-reading the arrays.
        self._full_extent = (t0, t1, f0, f1)

    def _resolve_levels(self, stft: np.ndarray) -> tuple[float, float]:
        """Percentile colour levels, locked on the first run after a file load."""
        if self._levels is not None:
            return self._levels
        a = np.asarray(stft, dtype=np.float32)
        a = a[np.isfinite(a)]
        if a.size == 0:
            lv = (-100.0, 0.0)
        else:
            lo = max(float(np.percentile(a, 2)), -120.0)
            hi = min(float(np.percentile(a, 98)), 10.0)
            if hi - lo < 15:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - 25.0, mid + 25.0
            lv = (lo, hi)
        if self._is_first_processing:
            self._levels = lv
            self._is_first_processing = False
        return lv

    def _on_hist_levels_changed(self) -> None:
        """User dragged the colorbar: mirror levels to the visible images.

        Only the STFT view owns the persistent colour lock (``self._levels``);
        a drag in the Residual view shows its own residual-dB range and must not
        overwrite the lock, or STFT contrast breaks when switching back.
        """
        lo, hi = self._hist.getLevels()
        if self._current_view_type == _V_STFT:
            self._levels = (lo, hi)
        for im in (self._img_orig, self._img_proc):
            if im.isVisible():
                im.setLevels((lo, hi))

    def _apply_initial_range(self) -> None:
        """One-shot view range after new data / view change.

        Image views (STFT, residual-right): full frequency axis + a fixed
        initial time window (INITIAL_TIME_FRAC of the span), with pan/zoom
        bounded to the data. For the STFT view both plots are axis-linked, so
        the SAME limits are set on both — mismatched limits make the link
        clamp-bounce during single-axis zoom. Line views (Time, PSD,
        residual-left): clear any image limits (their MHz bounds would wrongly
        clamp an amplitude axis) and fit all.
        """
        vt = self._current_view_type
        if vt == _V_STFT:
            self._set_image_limits(self._plot_orig.getViewBox())
            self._set_image_limits(self._plot_proc.getViewBox())  # identical → no bounce
            self._set_initial_window(self._plot_orig.getViewBox())  # proc follows via link
        elif vt == _V_RESID:
            vb_o = self._plot_orig.getViewBox()
            self._clear_limits(vb_o)
            vb_o.autoRange()                                       # residual PSD line
            self._set_image_limits(self._plot_proc.getViewBox())
            self._set_initial_window(self._plot_proc.getViewBox())  # residual STFT image
        else:  # time / psd — line views
            for p in (self._plot_orig, self._plot_proc):
                vb = p.getViewBox()
                self._clear_limits(vb)
                vb.autoRange()

    @staticmethod
    def _clear_limits(vb) -> None:
        vb.setLimits(xMin=None, xMax=None, yMin=None, yMax=None,
                     minXRange=None, maxXRange=None, minYRange=None, maxYRange=None)

    def _set_image_limits(self, vb) -> None:
        """Bound pan (data extent + 2% margin) and zoom (can't zoom out past the
        full image; can't zoom in to a degenerate sub-pixel view)."""
        if self._full_extent is None:
            self._clear_limits(vb)
            return
        t0, t1, f0, f1 = self._full_extent
        tspan, fspan = t1 - t0, f1 - f0
        if tspan <= 0 or fspan <= 0:
            self._clear_limits(vb)
            return
        mx, my = tspan * 0.02, fspan * 0.02
        vb.setLimits(
            xMin=t0 - mx, xMax=t1 + mx, yMin=f0 - my, yMax=f1 + my,
            maxXRange=tspan * 1.1, maxYRange=fspan * 1.1,      # not past full extent
            minXRange=tspan * 5e-4, minYRange=fspan * 5e-4,    # not into a degenerate view
        )

    def _set_initial_window(self, vb) -> None:
        """Full frequency axis + fixed initial time window."""
        if self._full_extent is None:
            vb.autoRange()
            return
        t0, t1, f0, f1 = self._full_extent
        tspan = t1 - t0
        x_hi = t0 + tspan * self.INITIAL_TIME_FRAC if tspan > 0 else t1
        vb.setRange(xRange=(t0, x_hi), yRange=(f0, f1), padding=0)

    def _on_mouse_moved(self, evt) -> None:
        # Via SignalProxy the payload is a tuple of the signal's args.
        pos = evt[0] if isinstance(evt, (tuple, list)) else evt
        for p, tag in ((self._plot_orig, "L"), (self._plot_proc, "R")):
            vb = p.getViewBox()
            if p.sceneBoundingRect().contains(pos):
                pt = vb.mapSceneToView(pos)
                self._cursor_label.setText(f"{tag}:  x={pt.x():.4g}   y={pt.y():.4g}")
                return

    # =====================================================================
    # Public interface (MainWindow / toolbar)
    # =====================================================================
    def set_theme(self, dark: bool) -> None:
        apply_pyqtgraph_theme()
        cmap = get_pg_colormap()
        for im in (self._img_orig, self._img_proc):
            im.setColorMap(cmap)
        self._hist.gradient.setColorMap(cmap)
        for p in (self._plot_orig, self._plot_proc):
            p.getViewBox().setBackgroundColor(get_color("spectrum_bg"))
        self._redraw()

    def set_config(self, config: dict) -> None:
        pass  # pipeline-flow indicator removed; kept for interface compatibility

    def get_current_view_type(self) -> int:
        return self._current_view_type

    def get_current_stage_id(self) -> int:
        return self._current_stage_id

    def _active_viewboxes(self) -> list:
        """Viewboxes to drive directly. In linked views the proc VB follows
        orig, so returning only orig avoids double-applying the transform."""
        if self._current_view_type == _V_RESID:
            return [self._plot_orig.getViewBox(), self._plot_proc.getViewBox()]
        return [self._plot_orig.getViewBox()]

    def zoom_in(self) -> None:
        for vb in self._active_viewboxes():
            vb.scaleBy((1 / 1.25, 1 / 1.25))

    def zoom_out(self) -> None:
        for vb in self._active_viewboxes():
            vb.scaleBy((1.25, 1.25))

    def reset_zoom(self) -> None:
        """Reset = fit the whole spectrogram (full extent), unlike the fixed
        window applied on load."""
        for vb in self._active_viewboxes():
            vb.autoRange()

    def export_figure(self, file_path: str, dpi: int = 300) -> None:
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self._glw.scene())
        exporter.export(file_path)

    def clear_data(self) -> None:
        self._raw_original = None
        self._raw_processed = None
        self._stage_signals = {}
        self._stage_spectra = {}
        self._available_stages = ["final"]
        self._current_stage_name = "final"
        self._current_stage_id = 0
        self._levels = None
        self._is_first_processing = True
        for it in (self._curve_orig, self._curve_proc):
            it.clear()
        for im in (self._img_orig, self._img_proc):
            im.clear()
        self._rebuild_stage_combo()
