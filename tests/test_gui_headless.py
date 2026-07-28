"""Headless GUI-layer tests.

The Qt *viewer* splits cleanly into a Qt-free signal-processing layer
(`PipelineRunner`, `SNREstimator`) and the Qt widgets on top. The former must
work with no display at all; the latter we only instantiate best-effort under
the offscreen platform plugin (marked xfail so a missing system Qt lib doesn't
fail the suite).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

# Ensure a headless Qt platform for anything that does spin up a QApplication.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


# ---------------------------------------------------------------------------
# Qt-free signal-processing layer
# ---------------------------------------------------------------------------
def test_pipeline_runner_headless(iq_array):
    from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner

    runner = PipelineRunner(fs=100e6, seed=42)
    runner.set_iq_data(iq_array[:16384].copy())
    runner.set_config({"stage_configs": {"awgn": {"enabled": True, "snr_db": 10.0}}})

    results = runner.run(return_all_stages=True)
    assert "original" in results and "final" in results
    assert results["final"].dtype == np.complex64
    assert results["original"].shape == (16384,)


def test_snr_estimator_headless(iq_array):
    from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator

    res = SNREstimator(fs=100e6, bw=20e6).estimate(iq_array)
    assert np.isfinite(res.snr_db)


# ---------------------------------------------------------------------------
# Qt widget layer — best effort under offscreen
# ---------------------------------------------------------------------------
def test_qapplication_offscreen():
    pytest.importorskip("PySide6", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    assert app is not None


@pytest.mark.xfail(reason="MainWindow needs full Qt platform libs; offscreen best-effort", strict=False)
def test_main_window_instantiate():
    pytest.importorskip("PySide6", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    from ChannelSimulation.viewer.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    assert win is not None
    win.close()
