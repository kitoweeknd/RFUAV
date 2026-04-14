"""
Log Manager
~~~~~~~~~~

A complete logging system with log file rotation and built-in UI log viewer.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QDialog,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class LogLevel(Enum):
    """Log levels"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LogManager(QObject):
    """Log Manager

    Provides a unified logging interface, supporting:
    - File logging (rotated by date)
    - Console logging
    - Qt signals (for UI updates)
    """

    new_message = Signal(str, int)  # message text, log level

    def __init__(
        self,
        log_dir: str | Path | None = None,
        level: LogLevel = LogLevel.INFO,
    ):
        super().__init__()
        self._level = level
        self._log_dir = Path(log_dir) if log_dir else self._get_default_log_dir()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self._logger = logging.getLogger("ChannelSimViewer")
        self._logger.setLevel(level.value)
        self._logger.handlers.clear()

        # File handler
        self._file_handler = self._create_file_handler()
        self._logger.addHandler(self._file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level.value)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        # User action logger
        self._user_logger = self._create_user_logger()

    def _get_default_log_dir(self) -> Path:
        """Get default log directory"""
        try:
            from ChannelSimulation.viewer.core.utils import get_log_dir
            return get_log_dir()
        except ImportError:
            return Path.home() / ".channelsim_viewer" / "logs"

    def _create_file_handler(self) -> logging.FileHandler:
        """Create file handler"""
        # Generate filename by date
        today = datetime.now().strftime("%Y%m%d")
        log_file = self._log_dir / f"viewer_{today}.log"

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(self._level.value)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        return handler

    def _create_user_logger(self) -> logging.Logger:
        """Create user action logger"""
        today = datetime.now().strftime("%Y%m%d")
        log_file = self._log_dir / f"user_actions_{today}.log"

        logger = logging.getLogger("ChannelSimViewer.UserActions")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def set_level(self, level: LogLevel) -> None:
        """Set log level"""
        self._level = level
        self._logger.setLevel(level.value)
        for handler in self._logger.handlers:
            handler.setLevel(level.value)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self._logger.debug(message)
        self.new_message.emit(message, logging.DEBUG)

    def info(self, message: str) -> None:
        """Log general information"""
        self._logger.info(message)
        self.new_message.emit(message, logging.INFO)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self._logger.warning(message)
        self.new_message.emit(message, logging.WARNING)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message"""
        self._logger.error(message, exc_info=exc_info)
        self.new_message.emit(message, logging.ERROR)

    def critical(self, message: str, exc_info: bool = False) -> None:
        """Log critical error"""
        self._logger.critical(message, exc_info=exc_info)
        self.new_message.emit(message, logging.CRITICAL)

    def log_user_action(self, action: str, details: dict | None = None) -> None:
        """Log user action

        Args:
            action: Action name
            details: Action details
        """
        if details:
            import json
            detail_str = json.dumps(details, ensure_ascii=False)
            self._user_logger.info(f"{action}: {detail_str}")
        else:
            self._user_logger.info(action)

    def log_file_operation(self, operation: str, file_path: str) -> None:
        """Log file operation"""
        self.log_user_action(f"File: {operation}", {"file": file_path})

    def log_config_change(self, param_name: str, old_value, new_value) -> None:
        """Log configuration change"""
        self.log_user_action(
            "Config: Changed",
            {"param": param_name, "old": str(old_value), "new": str(new_value)},
        )

    def log_error(self, error_type: str, error_message: str) -> None:
        """Log error (for error log file)"""
        self._logger.error(f"[{error_type}] {error_message}")

    def get_recent_logs(self, n_lines: int = 100) -> list[str]:
        """Get recent log lines

        Args:
            n_lines: Number of lines

        Returns:
            List of log lines
        """
        log_file = list(self._log_dir.glob("viewer_*.log"))
        if not log_file:
            return []

        # Get latest log file
        latest_log = max(log_file, key=lambda p: p.stat().st_mtime)

        with open(latest_log, "r", encoding="utf-8") as f:
            lines = f.readlines()

        return [line.strip() for line in lines[-n_lines:]]


class LogViewerDialog(QDialog):
    """Log viewer dialog"""

    def __init__(self, log_manager: LogManager | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Viewer")
        self.setMinimumSize(700, 500)
        self._log_manager = log_manager or get_log_manager()
        self._init_ui()
        self._load_logs()

    def _init_ui(self) -> None:
        """Initialize UI"""
        layout = QVBoxLayout(self)

        # Log text area
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        layout.addWidget(self._log_text)

        # Button row
        btn_layout = QVBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_logs)
        btn_layout.addWidget(refresh_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_logs)
        btn_layout.addWidget(clear_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)

        layout.addLayout(btn_layout)

    def _load_logs(self) -> None:
        """Load logs"""
        logs = self._log_manager.get_recent_logs(500)
        self._log_text.clear()
        self._log_text.appendPlainText("\n".join(logs))
        # Scroll to bottom
        self._log_text.verticalScrollBar().setValue(
            self._log_text.verticalScrollBar().maximum()
        )

    def _clear_logs(self) -> None:
        """Clear display"""
        self._log_text.clear()


# Global singleton
_log_manager: Optional[LogManager] = None


def get_log_manager() -> LogManager:
    """Get log manager singleton"""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager()
    return _log_manager


def init_logging(level: LogLevel = LogLevel.INFO) -> LogManager:
    """Initialize logging system"""
    global _log_manager
    _log_manager = LogManager(level=level)
    return _log_manager