"""
Signal processing: pipeline, threading, spectrum analysis, metrics.

``ProcessingWorker`` depends on PySide6 (``QThread``) and is imported lazily
via :pep:`562` ``__getattr__`` so that the Qt-free compute layer
(``pipeline_runner``, ``spectrum_data``, ``metrics_calculator``,
``snr_estimator``) — and the ``SNREstimation`` CLI that builds on it — can be
imported in a headless environment without a GUI stack installed.
"""
from ChannelSimulation.viewer.signal_proc.pipeline_runner import PipelineRunner
from ChannelSimulation.viewer.signal_proc.spectrum_data import SpectrumData
from ChannelSimulation.viewer.signal_proc.metrics_calculator import MetricsCalculator
from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator, SNRResult

__all__ = [
    "PipelineRunner",
    "ProcessingWorker",
    "SpectrumData",
    "MetricsCalculator",
    "SNREstimator",
    "SNRResult",
]


def __getattr__(name):
    # Defer the Qt-dependent worker so importing this package (and the compute
    # layer / SNR CLI) does not pull in PySide6.
    if name == "ProcessingWorker":
        from ChannelSimulation.viewer.signal_proc.processing_worker import (
            ProcessingWorker,
        )

        return ProcessingWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
