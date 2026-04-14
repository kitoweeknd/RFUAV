"""
Performance Monitor Panel
~~~~~~~~~~~~~~~~~~~~~~~~~

Real-time monitoring of system resource usage: CPU, memory, GPU utilization.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ChannelSimulation.viewer.core.ui_style import get_color


class PerformanceMonitor(QObject):
    """Performance Monitor

    Periodically samples system resource usage.
    """

    updated = Signal(dict)  # Update signal

    def __init__(self, interval_ms: int = 1000):
        super().__init__()
        self._interval_ms = interval_ms
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)
        self._psutil_available = False

        # Try to import psutil
        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            self._psutil = None

    def start(self) -> None:
        """Start monitoring"""
        self._timer.start(self._interval_ms)

    def stop(self) -> None:
        """Stop monitoring"""
        self._timer.stop()

    def _update(self) -> None:
        """Update sampling"""
        stats = {}

        if self._psutil_available:
            # CPU utilization
            stats["cpu_percent"] = self._psutil.cpu_percent(interval=0.1)

            # Memory usage
            mem = self._psutil.virtual_memory()
            stats["memory_used"] = mem.used
            stats["memory_total"] = mem.total
            stats["memory_percent"] = mem.percent

            # GPU utilization (if CUDA is available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_percent"] = util.gpu
                stats["gpu_memory_percent"] = util.memory
                pynvml.nvmlShutdown()
            except Exception:
                stats["gpu_percent"] = None
                stats["gpu_memory_percent"] = None

        else:
            # Simulated data (no psutil)
            import random
            stats["cpu_percent"] = random.uniform(10, 30)
            stats["memory_used"] = 8 * 1024 ** 3
            stats["memory_total"] = 32 * 1024 ** 3
            stats["memory_percent"] = 25
            stats["gpu_percent"] = None
            stats["gpu_memory_percent"] = None

        self.updated.emit(stats)

    def get_stats(self) -> dict:
        """Get current statistics"""
        if self._psutil_available:
            stats = {
                "cpu_percent": self._psutil.cpu_percent(interval=0.1),
                "memory_used": self._psutil.virtual_memory().used,
                "memory_total": self._psutil.virtual_memory().total,
                "memory_percent": self._psutil.virtual_memory().percent,
            }
        else:
            import random
            stats = {
                "cpu_percent": random.uniform(10, 30),
                "memory_used": 8 * 1024 ** 3,
                "memory_total": 32 * 1024 ** 3,
                "memory_percent": 25,
            }
        return stats


class PerformancePanel(QWidget):
    """Performance Monitor Panel

    Widget displaying CPU, memory, and GPU utilization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._monitor = PerformanceMonitor()
        self._monitor.updated.connect(self._on_update)
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 4, 8, 4)

        # CPU
        cpu_label = QLabel("CPU:")
        cpu_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        layout.addWidget(cpu_label, 0, 0)

        self._cpu_progress = QProgressBar()
        self._cpu_progress.setMaximumWidth(100)
        self._cpu_progress.setMaximumHeight(8)
        self._cpu_progress.setTextVisible(False)
        layout.addWidget(self._cpu_progress, 0, 1)

        self._cpu_value = QLabel("0%")
        self._cpu_value.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        layout.addWidget(self._cpu_value, 0, 2)

        # Memory
        mem_label = QLabel("Memory:")
        mem_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        layout.addWidget(mem_label, 1, 0)

        self._mem_progress = QProgressBar()
        self._mem_progress.setMaximumWidth(100)
        self._mem_progress.setMaximumHeight(8)
        self._mem_progress.setTextVisible(False)
        layout.addWidget(self._mem_progress, 1, 1)

        self._mem_value = QLabel("0 MB / 0 GB")
        self._mem_value.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        layout.addWidget(self._mem_value, 1, 2)

        # GPU (optional)
        gpu_label = QLabel("GPU:")
        gpu_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        gpu_label.setVisible(False)
        layout.addWidget(gpu_label, 2, 0)

        self._gpu_progress = QProgressBar()
        self._gpu_progress.setMaximumWidth(100)
        self._gpu_progress.setMaximumHeight(8)
        self._gpu_progress.setTextVisible(False)
        self._gpu_progress.setVisible(False)
        layout.addWidget(self._gpu_progress, 2, 1)

        self._gpu_value = QLabel("--")
        self._gpu_value.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        self._gpu_value.setVisible(False)
        layout.addWidget(self._gpu_value, 2, 2)

        # Frame rate
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet(f"color: {get_color('text_secondary')}; font-size: 11px;")
        layout.addWidget(fps_label, 0, 3)

        self._fps_value = QLabel("--")
        self._fps_value.setStyleSheet(f"color: {get_color('accent')}; font-size: 11px; font-weight: bold;")
        layout.addWidget(self._fps_value, 0, 4)

    def _on_update(self, stats: dict) -> None:
        """Update display"""
        # CPU
        cpu = stats.get("cpu_percent", 0)
        self._cpu_progress.setValue(int(cpu))
        self._cpu_value.setText(f"{cpu:.1f}%")

        # Set color based on CPU usage
        if cpu > 80:
            color = get_color("danger")
        elif cpu > 50:
            color = get_color("warning")
        else:
            color = get_color("success")
        self._cpu_value.setStyleSheet(f"color: {color}; font-size: 11px;")

        # Memory
        mem_used = stats.get("memory_used", 0)
        mem_total = stats.get("memory_total", 1)
        mem_pct = stats.get("memory_percent", 0)
        self._mem_progress.setValue(int(mem_pct))
        self._mem_value.setText(
            f"{self._format_bytes(mem_used)} / {self._format_bytes(mem_total)}"
        )

        # GPU
        gpu = stats.get("gpu_percent")
        if gpu is not None:
            self._gpu_progress.setValue(int(gpu))
            self._gpu_value.setText(f"{gpu}%")
            self._gpu_progress.setVisible(True)
            self._gpu_value.setVisible(True)
            layout = self.layout()
            layout.itemAtPosition(2, 0).widget().setVisible(True)

    def _format_bytes(self, n: int) -> str:
        """Format byte count"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    def update_fps(self, fps: float) -> None:
        """Update frame rate display"""
        self._fps_value.setText(f"{fps:.1f}")
        if fps < 10:
            color = get_color("danger")
        elif fps < 30:
            color = get_color("warning")
        else:
            color = get_color("success")
        self._fps_value.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold;")

    def start(self) -> None:
        """Start monitoring"""
        self._monitor.start()

    def stop(self) -> None:
        """Stop monitoring"""
        self._monitor.stop()


# FPS Calculator
class FPSCounter:
    """Frame Rate Counter"""

    def __init__(self, window_size: int = 60):
        self._window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """Record a frame, return current FPS"""
        import time
        now = time.perf_counter()
        self._timestamps.append(now)

        # Clean up expired records
        cutoff = now - 1.0  # 1 second window
        self._timestamps = [t for t in self._timestamps if t >= cutoff]

        return len(self._timestamps)

    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self._timestamps) < 2:
            return 0.0

        # Use simple moving average
        return len(self._timestamps)
