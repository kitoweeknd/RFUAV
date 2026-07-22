"""
SNR Estimation View
~~~~~~~~~~~~~~~~~~~

Dedicated view for OFDM UAV signal SNR estimation.
Shows:
  - Top panel: before/after SNR metrics + algorithm parameters
  - Bottom: three-panel PSD visualisation (raw → compensated → downsampled)
  - Markers showing detected signal band and noise band
"""

from __future__ import annotations

import numpy as np

import pyqtgraph as pg

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator, SNRResult
from ChannelSimulation.viewer.core.ui_style import get_color, apply_pyqtgraph_theme


class _SNRWorker(QThread):
    """Background thread for SNR estimation — processes ONE signal.

    Carries a ``gen`` (run-generation id) that is echoed back with the
    result so the view can discard results from superseded/cancelled runs.
    Supports cooperative cancellation: :meth:`stop` sets a flag that is
    checked at the coarse checkpoints available in ``run()``.
    """

    # (SNRResult or None, gen)
    finished = Signal(object, int)

    def __init__(self, signal, fs, bw, nfft, gen, parent=None):
        super().__init__(parent)
        self._signal = signal
        self._fs = fs
        self._bw = bw
        self._nfft = nfft
        self._gen = gen
        self._cancelled = False

    def run(self):
        # Bail out early if cancelled before we even started computing.
        if self._cancelled:
            return
        estimator = SNREstimator(fs=self._fs, bw=self._bw, nfft=self._nfft)
        result = None
        try:
            if self._signal is not None and len(self._signal) > 0:
                result = estimator.estimate(self._signal)
        except Exception:
            import traceback
            traceback.print_exc()
        # A cancel may have arrived while computing — drop the stale result
        # and stay silent so no handler runs for an abandoned run.
        if self._cancelled:
            return
        self.finished.emit(result, self._gen)

    def stop(self) -> None:
        """Request cooperative cancellation of this worker."""
        self._cancelled = True


class SNRView(QWidget):
    """SNR Estimation View.

    Displays the three-step OFDM signal detection and SNR estimation process
    with before/after SNR comparison.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result_orig: SNRResult | None = None
        self._result_proc: SNRResult | None = None
        self._fs = 100e6
        self._bw = 20e6
        self._nfft = 2048
        self._signal_orig: np.ndarray | None = None
        self._signal_proc: np.ndarray | None = None
        self._worker_orig: _SNRWorker | None = None
        self._worker_proc: _SNRWorker | None = None
        # Workers whose wait() has not yet returned — kept referenced so a
        # still-running QThread is never garbage-collected (which would raise
        # "QThread: Destroyed while thread is still running").
        self._old_workers: list[_SNRWorker] = []
        # Run-generation id: bumped on every _start_worker so results from a
        # superseded/cancelled run are ignored by the done-handlers.
        self._run_gen = 0
        # Completion flags for the current run (do NOT rely on isRunning()).
        self._orig_done = False
        self._proc_done = False
        self._pending_draw = False

        self._init_ui()
        self._setup_figures()

        # Sync params from the toolbar controls so the value used on the first
        # Estimate matches what is displayed (combo defaults to 16384, not 2048).
        self._bw = self._spin_bw.value()
        self._nfft = int(self._combo_nfft.currentText())

    # ─────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────

    def set_signals(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
        fs: float,
    ) -> None:
        """Update the signals and trigger re-estimation in background."""
        self._signal_orig = original
        self._signal_proc = processed
        self._fs = fs
        self._start_worker()

    # ─────────────────────────────────────────────────────────────────
    #  UI Init
    # ─────────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        apply_pyqtgraph_theme()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Toolbar
        toolbar = self._build_toolbar()
        layout.addWidget(toolbar)

        # Metrics + canvas area
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Metrics row
        metrics_widget = self._build_metrics_panel()
        splitter.addWidget(metrics_widget)

        # Visualisation row
        vis_widget = self._build_visualisation_panel()
        splitter.addWidget(vis_widget)

        splitter.setStretchFactor(0, 0)  # metrics: fixed height
        splitter.setStretchFactor(1, 1)  # canvas: takes remaining space
        splitter.setSizes([160, 600])

        layout.addWidget(splitter)

    def _build_toolbar(self) -> QWidget:
        toolbar = QWidget()
        h = QHBoxLayout(toolbar)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(16)

        h.addWidget(QLabel("SNR Estimation Mode"))

        h.addSpacing(8)
        h.addWidget(QLabel("BW (Hz):"))
        self._spin_bw = QDoubleSpinBox()
        self._spin_bw.setRange(1e6, 100e6)
        self._spin_bw.setValue(20e6)
        self._spin_bw.setSuffix(" Hz")
        self._spin_bw.setDecimals(0)
        self._spin_bw.valueChanged.connect(self._on_param_changed)
        h.addWidget(self._spin_bw)

        h.addWidget(QLabel("FFT:"))
        self._combo_nfft = QComboBox()
        self._combo_nfft.addItems(["512", "1024", "2048", "4096", "8192", "16384", "32768"])
        self._combo_nfft.setCurrentText("16384")
        self._combo_nfft.currentIndexChanged.connect(self._on_param_changed)
        h.addWidget(self._combo_nfft)

        h.addSpacing(8)
        self._btn_estimate = QPushButton("Estimate")
        self._btn_estimate.setStyleSheet(f"""
            QPushButton {{
                background-color: {get_color('accent')};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 16px;
                font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
        """)
        self._btn_estimate.clicked.connect(self._on_estimate_clicked)
        h.addWidget(self._btn_estimate)

        h.addStretch()

        # Status label
        self._label_status = QLabel("Load a signal and click Estimate")
        self._label_status.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        h.addWidget(self._label_status)

        return toolbar

    def _build_metrics_panel(self) -> QWidget:
        widget = QWidget()
        h = QHBoxLayout(widget)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(16)

        # Original SNR card
        orig_card = self._make_metric_card("Original Signal", "#1a73e8")
        self._label_orig_snr = orig_card["value"]
        self._label_orig_band = orig_card["title"]
        self._label_orig_band.setText("—")
        h.addWidget(orig_card["widget"], stretch=1)

        # Arrow
        arrow = QLabel("→")
        arrow.setStyleSheet(f"font-size: 24px; color: {get_color('text_secondary')};")
        arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(arrow)

        # Processed SNR card
        proc_card = self._make_metric_card("Processed Signal", "#ea4335")
        self._label_proc_snr = proc_card["value"]
        self._label_proc_band = proc_card["title"]
        self._label_proc_band.setText("—")
        h.addWidget(proc_card["widget"], stretch=1)

        # Delta indicator
        delta_frame = QFrame()
        delta_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {get_color('bg_secondary')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
                padding: 8px;
            }}
        """)
        delta_layout = QVBoxLayout(delta_frame)
        delta_layout.setContentsMargins(8, 4, 8, 4)
        delta_layout.setSpacing(2)
        lbl = QLabel("SNR Δ")
        lbl.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        delta_layout.addWidget(lbl)
        self._label_delta = QLabel("—")
        self._label_delta.setStyleSheet(f"""
            color: {get_color('accent')};
            font-size: 20px;
            font-weight: bold;
            font-family: Consolas, Courier New, monospace;
        """)
        self._label_delta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        delta_layout.addWidget(self._label_delta)
        h.addWidget(delta_frame)

        # Bandwidth info
        info_frame = QFrame()
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {get_color('bg_secondary')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
                padding: 8px;
                min-width: 160px;
            }}
        """)
        info_layout = QFormLayout(info_frame)
        info_layout.setContentsMargins(4, 2, 4, 2)
        info_layout.setHorizontalSpacing(12)
        info_layout.setVerticalSpacing(4)
        self._label_f1 = QLabel("—")
        self._label_f1.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self._label_f2 = QLabel("—")
        self._label_f2.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self._label_bw = QLabel("—")
        self._label_bw.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        info_layout.addRow("f1:", self._label_f1)
        info_layout.addRow("f2:", self._label_f2)
        info_layout.addRow("Bandwidth:", self._label_bw)
        h.addWidget(info_frame)

        return widget

    def _make_metric_card(self, title: str, accent: str) -> dict:
        """Build a styled metric card and return refs to its sub-widgets."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {get_color('bg_secondary')};
                border: 1px solid {get_color('border')};
                border-radius: 8px;
                padding: 8px;
                min-width: 160px;
            }}
        """)

        vlayout = QVBoxLayout(card)
        vlayout.setContentsMargins(8, 6, 8, 6)
        vlayout.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 10px;")

        lbl_value = QLabel("—")
        lbl_value.setStyleSheet(f"""
            color: {accent};
            font-size: 22px;
            font-weight: bold;
            font-family: Consolas, Courier New, monospace;
        """)

        lbl_sub = QLabel("")
        lbl_sub.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 9px;")

        vlayout.addWidget(lbl_title)
        vlayout.addWidget(lbl_value)
        vlayout.addWidget(lbl_sub)

        return {"widget": card, "title": lbl_sub, "value": lbl_value}

    def _build_visualisation_panel(self) -> QWidget:
        """Three PSD canvases side by side."""
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Canvas 1: Raw signal PSD
        splitter.addWidget(self._make_canvas_widget("① Raw Signal PSD", "raw"))

        # Canvas 2: Compensated PSD
        splitter.addWidget(self._make_canvas_widget("② Frequency Compensated", "comp"))

        # Canvas 3: Downsampled PSD (with SNR markers)
        splitter.addWidget(self._make_canvas_widget("③ Downsampled FFT (Signal vs Noise)", "ds"))

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

        return splitter

    def _make_canvas_widget(self, title: str, key: str) -> QWidget:
        """Wrap a pyqtgraph PlotWidget in a titled container."""
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"""
            color: {get_color('text_secondary')};
            font-size: 10px;
            padding: 2px 4px;
        """)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl)

        pw = pg.PlotWidget()
        pw.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.setMenuEnabled(False)
        pw.addLegend(offset=(-10, 10))
        v.addWidget(pw)

        setattr(self, f"_pw_{key}", pw)

        return container

    def _setup_figures(self) -> None:
        """Clear all three plot widgets to an empty state."""
        for key in ("raw", "comp", "ds"):
            pw: pg.PlotWidget = getattr(self, f"_pw_{key}")
            pw.clear()
            pw.setTitle("No data")

    # ─────────────────────────────────────────────────────────────────
    #  Estimation
    # ─────────────────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        """Launch parallel background SNR estimation threads."""
        if self._signal_orig is None:
            return
        if len(self._signal_orig) < 512:
            self._label_status.setText(f"Signal too short ({len(self._signal_orig)} samples) — need at least 512")
            self._btn_estimate.setEnabled(True)
            return

        sig_len = len(self._signal_orig)
        effective_nfft = min(self._nfft, sig_len)
        if effective_nfft < 64:
            self._label_status.setText(f"Signal too short ({sig_len} samples) — need at least 64")
            self._btn_estimate.setEnabled(True)
            return

        # Supersede any running workers before starting a new run.
        self._run_gen += 1
        gen = self._run_gen
        self._stop_workers()

        # Reset per-run state.
        self._result_orig = None
        self._result_proc = None
        self._orig_done = False
        self._proc_done = False

        self._label_status.setText("Estimating…")
        self._btn_estimate.setEnabled(False)

        # Launch two workers in parallel — one per signal
        self._worker_orig = _SNRWorker(
            self._signal_orig, self._fs, self._bw, effective_nfft, gen, self,
        )
        self._worker_orig.finished.connect(self._on_orig_done)

        if self._signal_proc is not None and len(self._signal_proc) > 0:
            proc_len = len(self._signal_proc)
            proc_nfft = min(self._nfft, proc_len)
            proc_nfft = max(64, proc_nfft)
            self._worker_proc = _SNRWorker(
                self._signal_proc, self._fs, self._bw, proc_nfft, gen, self,
            )
            self._worker_proc.finished.connect(self._on_proc_done)
            self._worker_proc.start()
        else:
            self._result_proc = None
            self._worker_proc = None
            # Nothing to wait for on the processed side.
            self._proc_done = True

        self._worker_orig.start()

    def _stop_workers(self) -> None:
        """Cooperatively cancel running workers and detach their signals.

        Does NOT hard-null a worker that may still be running; such workers
        are moved to ``self._old_workers`` so the QThread object stays alive
        until it actually finishes (avoids "QThread: Destroyed while thread
        is still running").
        """
        for w in (self._worker_orig, self._worker_proc):
            if w is None:
                continue
            try:
                w.finished.disconnect()
            except (RuntimeError, TypeError):
                pass  # was never connected / already disconnected
            w.stop()  # cooperative cancel flag
            if not w.wait(200):
                # Still computing — keep a reference so it isn't GC'd mid-run.
                self._old_workers.append(w)
        self._worker_orig = None
        self._worker_proc = None
        self._prune_old_workers()

    def _prune_old_workers(self) -> None:
        """Drop references to superseded workers that have finished running."""
        self._old_workers = [w for w in self._old_workers if w.isRunning()]

    def _on_orig_done(self, r_orig: SNRResult | None, gen: int) -> None:
        """Called when original signal estimation completes."""
        if gen != self._run_gen:
            return  # stale result from a superseded/cancelled run
        self._result_orig = r_orig
        self._orig_done = True
        self._maybe_finish()

    def _on_proc_done(self, r_proc: SNRResult | None, gen: int) -> None:
        """Called when processed signal estimation completes."""
        if gen != self._run_gen:
            return  # stale result from a superseded/cancelled run
        self._result_proc = r_proc
        self._proc_done = True
        self._maybe_finish()

    def _maybe_finish(self) -> None:
        """Update UI once the expected results for this run have arrived."""
        # Rely on explicit completion flags, NOT QThread.isRunning() (the
        # thread may still be technically running when its result arrives).
        if not (self._orig_done and self._proc_done):
            return

        # Both results are in; the workers have emitted, so let them be
        # cleaned up on the next _start_worker via _stop_workers.
        self._prune_old_workers()

        if self._result_orig is not None:
            self._update_metric_card(
                self._label_orig_snr,
                self._label_orig_band,
                self._result_orig.snr_db,
                f"f1={self._result_orig.f1/1e6:.3f}MHz f2={self._result_orig.f2/1e6:.3f}MHz",
            )
        else:
            self._label_orig_snr.setText("Err")
            self._label_orig_band.setText("Estimation failed")

        if self._result_proc is not None:
            self._update_metric_card(
                self._label_proc_snr,
                self._label_proc_band,
                self._result_proc.snr_db,
                f"f1={self._result_proc.f1/1e6:.3f}MHz f2={self._result_proc.f2/1e6:.3f}MHz",
            )
            if self._result_orig is not None:
                delta = self._result_proc.snr_db - self._result_orig.snr_db
                self._label_delta.setText(f"{delta:+.2f} dB")
                if delta < -1:
                    c = get_color("danger")
                elif delta > 1:
                    c = get_color("success")
                else:
                    c = get_color("text_secondary")
                self._label_delta.setStyleSheet(f"""
                    color: {c};
                    font-size: 20px;
                    font-weight: bold;
                    font-family: Consolas, Courier New, monospace;
                """)
        else:
            self._label_proc_snr.setText("—")
            self._label_proc_band.setText("No processed signal")
            self._label_delta.setText("—")

        if self._result_orig is not None:
            self._label_f1.setText(f"{self._result_orig.f1/1e6:.4f} MHz")
            self._label_f2.setText(f"{self._result_orig.f2/1e6:.4f} MHz")
            self._label_bw.setText(f"{(self._result_orig.f2 - self._result_orig.f1)/1e6:.4f} MHz")
            self._draw_all()
        else:
            self._setup_figures()

        self._label_status.setText("Estimation complete" if self._result_orig else "Estimation failed")
        self._btn_estimate.setEnabled(True)

    def _update_metric_card(self, value_lbl: QLabel, sub_lbl: QLabel, snr: float, sub: str) -> None:
        value_lbl.setText(f"{snr:.2f} dB")
        sub_lbl.setText(sub)

    # ─────────────────────────────────────────────────────────────────
    #  Drawing
    # ─────────────────────────────────────────────────────────────────

    def _draw_all(self) -> None:
        """Redraw all three PSD canvases."""
        self._draw_raw()
        self._draw_compensated()
        self._draw_downsampled()

    @staticmethod
    def _downsample_for_plot(x: np.ndarray, y: np.ndarray, max_pts: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """Downsample (x, y) to at most max_pts points for efficient rendering.

        Uses slice-and-decimate to preserve shape while reducing point count.
        """
        n = len(x)
        if n <= max_pts:
            return x, y
        step = max(1, n // max_pts)
        return x[::step], y[::step]

    @staticmethod
    def _vline(x: float, color: str, style=Qt.PenStyle.DashLine, name=None):
        pen = pg.mkPen(color, width=1.2, style=style)
        return pg.InfiniteLine(pos=float(x), angle=90, pen=pen, name=name)

    def _draw_raw(self) -> None:
        """Plot ①: Raw signal PSD with detected band markers."""
        pw: pg.PlotWidget = self._pw_raw
        pw.clear()
        if self._result_orig is None:
            pw.setTitle("No data")
            return
        r = self._result_orig
        accent = get_color("accent")
        freq_plot, psd_db = self._downsample_for_plot(r.freq_full / 1e6, r.psd_full)
        pw.plot(freq_plot, psd_db, pen=pg.mkPen(accent, width=1),
                fillLevel=float(np.min(psd_db)), brush=pg.mkBrush(26, 115, 232, 60), name="PSD")
        pw.addItem(self._vline(r.f1 / 1e6, "#ea4335"))
        pw.addItem(self._vline(r.f2 / 1e6, "#ea4335"))
        pw.setLabel("bottom", "Frequency (MHz)")
        pw.setLabel("left", "PSD (normalised dB)")
        pw.setTitle(f"① Signal Band  f1={r.f1/1e6:.3f}  f2={r.f2/1e6:.3f} MHz")

    def _draw_compensated(self) -> None:
        """Plot ②: Frequency-compensated signal PSD."""
        pw: pg.PlotWidget = self._pw_comp
        pw.clear()
        if self._result_orig is None:
            pw.setTitle("No data")
            return
        r = self._result_orig
        accent = get_color("success")
        freq_plot, psd_plot = self._downsample_for_plot(r.freq_comp / 1e6, r.psd_comp)
        pw.plot(freq_plot, psd_plot, pen=pg.mkPen(accent, width=1),
                fillLevel=float(np.min(psd_plot)), brush=pg.mkBrush(52, 168, 83, 60))
        pw.addItem(self._vline(-(r.f2 - r.f_center) / 1e6, "#ea4335"))
        pw.addItem(self._vline(-(r.f1 - r.f_center) / 1e6, "#ea4335"))
        pw.setLabel("bottom", "Frequency (MHz)")
        pw.setLabel("left", "Power (linear)")
        pw.setTitle("② Frequency Compensated — Band Shift Corrected")

    def _draw_downsampled(self) -> None:
        """Plot ③: Downsampled FFT with signal and noise band regions + SNR."""
        pw: pg.PlotWidget = self._pw_ds
        pw.clear()
        if self._result_orig is None:
            pw.setTitle("No data")
            return
        r = self._result_orig
        freq_ds_full = r.freq_ds / 1e6
        n_ds = len(freq_ds_full)
        freq_plot, psd_ds_plot = self._downsample_for_plot(freq_ds_full, r.psd_ds)
        pw.plot(freq_plot, psd_ds_plot, pen=pg.mkPen(get_color("accent"), width=1), name="Original")

        # Signal band (green) and noise band (yellow) as shaded regions.
        sig_band = pg.LinearRegionItem(
            values=[freq_ds_full[r.idx1], freq_ds_full[min(r.idx2, n_ds - 1)]],
            brush=pg.mkBrush(52, 168, 83, 45), movable=False)
        noise_band = pg.LinearRegionItem(
            values=[freq_ds_full[r.idx3], freq_ds_full[min(r.idx4, n_ds - 1)]],
            brush=pg.mkBrush(251, 188, 4, 55), movable=False)
        for band in (sig_band, noise_band):
            band.setZValue(-10)
            pw.addItem(band)

        if self._result_proc is not None:
            pf, pp = self._downsample_for_plot(freq_ds_full, self._result_proc.psd_ds)
            pw.plot(pf, pp, pen=pg.mkPen(get_color("danger"), width=1), name="Processed")
            delta = self._result_proc.snr_db - r.snr_db
            title = (f"③ SNR orig={r.snr_db:.2f} dB  proc={self._result_proc.snr_db:.2f} dB  "
                     f"Δ={delta:+.2f} dB")
        else:
            title = f"③ SNR orig={r.snr_db:.2f} dB"
        pw.setLabel("bottom", "Frequency (MHz)")
        pw.setLabel("left", "PSD")
        pw.setTitle(title)

    # ─────────────────────────────────────────────────────────────────
    #  Event handlers
    # ─────────────────────────────────────────────────────────────────

    def _on_param_changed(self) -> None:
        self._bw = self._spin_bw.value()
        self._nfft = int(self._combo_nfft.currentText())
        self._label_status.setText("Parameters changed — click Estimate to recalculate")

    def _on_estimate_clicked(self) -> None:
        self._start_worker()
