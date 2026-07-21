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
``apply_config_b_data`` / ``_toggle_compare_mode``, ``set_theme``, ``set_config``,
``clear_data``, ``zoom_in`` / ``zoom_out`` / ``reset_zoom``,
``get_current_view_type`` / ``get_current_stage_id``, ``export_figure``, and the
readable ``_raw_original`` / ``_raw_processed`` attributes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF, Signal
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._calculator = SpectrumCalculator()

        # ── state ─────────────────────────────────────────────────────────
        self._current_view_type = _V_STFT
        self._current_stage_name = "final"
        self._current_stage_id = 0
        self._available_stages: list[str] = ["final"]
        self._is_dark_theme = False

        self._raw_original: np.ndarray | None = None
        self._raw_processed: np.ndarray | None = None
        self._stage_signals: dict[str, np.ndarray] = {}

        # stage_name -> complete spectrum dict (all four views). The viewer owns
        # this map (fed by apply_spectrum_data / apply_stage_spectrum) instead of
        # reaching into the worker's global cache, so lookups never depend on the
        # worker's internal signal-id counter.
        self._stage_spectra: dict[str, dict] = {}
        self._signal_id: int | None = None

        # STFT colour-scale lock (survives stacking runs; reset on new file).
        self._levels: tuple[float, float] | None = None
        self._is_first_processing = True

        # Config-B compare mode (secondary feature; overlay curves).
        self._compare_mode = False
        self._cfg_b: dict | None = None

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

        bl.addStretch()
        self._cursor_label = QLabel("")
        self._cursor_label.setStyleSheet(
            f"color: {get_color('text_secondary')}; font-size: 11px;"
        )
        bl.addWidget(self._cursor_label)
        layout.addWidget(bar)

        # ── graphics layout: [Original] [Processed] [Histogram] ────────────
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw, stretch=1)

        self._plot_orig = self._glw.addPlot(row=0, col=0)
        self._plot_proc = self._glw.addPlot(row=0, col=1)
        for p in (self._plot_orig, self._plot_proc):
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setMenuEnabled(False)

        # Persistent artists per column (created once, updated in place).
        self._img_orig = pg.ImageItem()
        self._img_proc = pg.ImageItem()
        self._curve_orig = pg.PlotDataItem(pen=pg.mkPen(get_color("accent"), width=1))
        self._curve_proc = pg.PlotDataItem(pen=pg.mkPen(get_color("danger"), width=1))
        self._curve_b = pg.PlotDataItem(  # config-B overlay (compare mode)
            pen=pg.mkPen(get_color("success"), width=1, style=Qt.PenStyle.DashLine)
        )
        self._curve_b.setVisible(False)
        self._plot_orig.addItem(self._img_orig)
        self._plot_orig.addItem(self._curve_orig)
        self._plot_proc.addItem(self._img_proc)
        self._plot_proc.addItem(self._curve_proc)
        self._plot_proc.addItem(self._curve_b)

        cmap = get_pg_colormap()
        for im in (self._img_orig, self._img_proc):
            im.setColorMap(cmap)

        # Shared colorbar / histogram (draggable levels = interactive contrast).
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._img_orig)
        self._hist.gradient.setColorMap(cmap)
        self._hist.sigLevelsChanged.connect(self._on_hist_levels_changed)
        self._glw.addItem(self._hist, row=0, col=2)

        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

        self._apply_view_layout()

    # =====================================================================
    # Selectors
    # =====================================================================
    def _on_view_changed(self, index: int) -> None:
        self._current_view_type = int(index)
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
        self._redraw()

    def apply_spectrum_data(self, spectrum_data: dict) -> None:
        """Final-stage spectra (from spectrum_ready). Stores + redraws."""
        if not spectrum_data:
            return
        self._signal_id = spectrum_data.get("signal_id", self._signal_id)
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

    def apply_config_b_data(self, original_b, processed_b, fs_b, label: str = "Config B") -> None:
        """Store Config-B processed signal for the compare overlay (PSD/time only)."""
        try:
            calc = SpectrumCalculator(fs=fs_b, fc=self._calculator.fc)
            psd_b, freq_b = calc.compute_psd(processed_b)
            n = len(processed_b)
            step = max(1, n // 2_000_000)
            self._cfg_b = {
                "label": label,
                "psd": psd_b, "freq_psd": freq_b,
                "time_axis": np.arange(0, n, step) / fs_b,
                "time": np.abs(processed_b[::step]),
            }
        except Exception:
            self._cfg_b = None
        if self._compare_mode:
            self._redraw()

    def _toggle_compare_mode(self) -> None:
        self._compare_mode = not self._compare_mode
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

    def _draw_time(self, d: dict) -> None:
        t = d.get("time_axis")
        yo = d.get("time_domain_original")
        yp = d.get("time_domain_processed")
        if t is not None and yo is not None:
            n = min(len(t), len(yo)); self._curve_orig.setData(t[:n], yo[:n])
        if t is not None and yp is not None:
            n = min(len(t), len(yp)); self._curve_proc.setData(t[:n], yp[:n])
        self._draw_compare_overlay(kind="time")
        self._autorange()

    def _draw_psd(self, d: dict) -> None:
        f = d.get("freq_axis_psd")
        po = d.get("psd_original")
        pp = d.get("psd_processed")
        if f is not None and po is not None:
            self._curve_orig.setData(f / 1e6, po)
        if f is not None and pp is not None:
            self._curve_proc.setData(f / 1e6, pp)
        self._draw_compare_overlay(kind="psd")
        self._autorange()

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
        self._autorange()

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
        self._autorange()

    def _set_image(self, img: pg.ImageItem, arr, t_axis, f_axis, levels) -> None:
        """setImage + map to (time seconds, frequency MHz) via setRect. No teardown."""
        img.setImage(arr, autoLevels=False)
        img.setLevels(levels)
        t0, t1 = float(t_axis[0]), float(t_axis[-1])
        f0, f1 = float(f_axis[0]) / 1e6, float(f_axis[-1]) / 1e6
        img.setRect(QRectF(t0, f0, max(t1 - t0, 1e-12), max(f1 - f0, 1e-12)))

    def _draw_compare_overlay(self, kind: str) -> None:
        if not self._compare_mode or self._cfg_b is None:
            self._curve_b.setVisible(False)
            return
        b = self._cfg_b
        if kind == "psd":
            self._curve_b.setData(b["freq_psd"] / 1e6, b["psd"])
        else:
            self._curve_b.setData(b["time_axis"], b["time"])
        self._curve_b.setVisible(True)

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
        """User dragged the colorbar: mirror levels to both images + lock them."""
        lo, hi = self._hist.getLevels()
        self._levels = (lo, hi)
        for im in (self._img_orig, self._img_proc):
            if im.isVisible():
                im.setLevels((lo, hi))

    def _autorange(self) -> None:
        self._plot_orig.enableAutoRange()
        self._plot_proc.enableAutoRange()

    def _on_mouse_moved(self, pos) -> None:
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
        self._is_dark_theme = dark
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

    def zoom_in(self) -> None:
        for p in (self._plot_orig, self._plot_proc):
            p.getViewBox().scaleBy((1 / 1.25, 1 / 1.25))

    def zoom_out(self) -> None:
        for p in (self._plot_orig, self._plot_proc):
            p.getViewBox().scaleBy((1.25, 1.25))

    def reset_zoom(self) -> None:
        for p in (self._plot_orig, self._plot_proc):
            p.getViewBox().autoRange()

    def export_figure(self, file_path: str, dpi: int = 300) -> None:
        from pyqtgraph.exporters import ImageExporter
        exporter = ImageExporter(self._glw.scene())
        exporter.export(file_path)

    def clear_data(self) -> None:
        self._raw_original = None
        self._raw_processed = None
        self._stage_signals = {}
        self._stage_spectra = {}
        self._signal_id = None
        self._available_stages = ["final"]
        self._current_stage_name = "final"
        self._current_stage_id = 0
        self._levels = None
        self._is_first_processing = True
        self._cfg_b = None
        self._compare_mode = False
        for it in (self._curve_orig, self._curve_proc, self._curve_b):
            it.clear()
        for im in (self._img_orig, self._img_proc):
            im.clear()
        self._rebuild_stage_combo()
