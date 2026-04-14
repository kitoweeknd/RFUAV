"""
Advanced features: batch processing, custom scripts, help, logging, performance.
"""
from ChannelSimulation.viewer.advanced.batch_processor import BatchProcessor
from ChannelSimulation.viewer.advanced.custom_script_engine import CustomScriptEngine
from ChannelSimulation.viewer.advanced.help_panel import HelpPanel
from ChannelSimulation.viewer.advanced.log_manager import LogManager, LogViewerDialog
from ChannelSimulation.viewer.advanced.performance_monitor import PerformanceMonitor

__all__ = [
    "BatchProcessor",
    "CustomScriptEngine",
    "HelpPanel",
    "LogManager",
    "LogViewerDialog",
    "PerformanceMonitor",
]
