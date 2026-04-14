"""
RFUAV ChannelSim Viewer Application Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles application initialization, splash screen display, high DPI adaptation,
and theme setup. Uses lazy import strategy to speed up startup.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# ============================================================================
# High DPI Configuration (must be set before QApplication creation)
# ============================================================================
try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Enable high DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
except ImportError:
    print("[ERROR] PySide6 not installed. Please run: pip install PySide6>=6.5")
    sys.exit(1)


# ============================================================================
# Global Variables
# ============================================================================
_app_instance: QApplication | None = None
_main_window: "MainWindow | None" = None


def get_app() -> QApplication:
    """Get the global QApplication instance."""
    global _app_instance
    if _app_instance is None:
        _app_instance = QApplication(sys.argv)
        _app_instance.setApplicationName("RFUAV ChannelSim Viewer")
        _app_instance.setApplicationVersion("1.3.0")
        _app_instance.setOrganizationName("RFUAV")
        _app_instance.setOrganizationDomain("rfuav.com")
    return _app_instance


def show_splash_screen(message: str = "Initializing...") -> "QSplashScreen | None":
    """Display the splash screen.

    Args:
        message: Initial message to display.

    Returns:
        Splash screen object or None.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QFont, QPainter, QPixmap, QColor
        from PySide6.QtWidgets import QSplashScreen

        # Create a simple splash screen
        pixmap = QPixmap(400, 200)
        pixmap.fill(QColor("#1a73e8"))

        painter = QPainter(pixmap)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "RFUAV ChannelSim Viewer")
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(pixmap.rect().adjusted(0, 30, 0, 0), Qt.AlignCenter, "v1.3.0")
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(pixmap.rect().adjusted(0, 60, 0, 0), Qt.AlignCenter, message)
        painter.end()

        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.setWindowFlag(Qt.FramelessWindowHint)
        splash.show()

        return splash
    except Exception as e:
        print(f"[WARNING] Failed to create splash screen: {e}")
        return None


def update_splash_message(splash, message: str) -> None:
    """Update splash screen message.

    Args:
        splash: Splash screen object.
        message: New message to display.
    """
    if splash is not None:
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QFont, QPainter, QPixmap, QColor
            from PySide6.QtWidgets import QSplashScreen

            pixmap = QPixmap(400, 200)
            pixmap.fill(QColor("#1a73e8"))

            painter = QPainter(pixmap)
            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "RFUAV ChannelSim Viewer")
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(pixmap.rect().adjusted(0, 30, 0, 0), Qt.AlignCenter, "v1.3.0")
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(pixmap.rect().adjusted(0, 60, 0, 0), Qt.AlignCenter, message)
            painter.end()

            splash.setPixmap(pixmap)
            splash.repaint()
            get_app().processEvents()
        except Exception:
            pass


def setup_theme(app: QApplication, dark: bool = False) -> None:
    """Set application theme.

    Args:
        app: QApplication instance.
        dark: Whether to use dark theme.
    """
    # Lazy import for faster startup
    from ChannelSimulation.viewer.core.ui_style import apply_stylesheet, set_theme
    set_theme(dark)
    apply_stylesheet(app)


def load_fonts() -> None:
    """Load custom fonts if needed."""
    # Currently using system default fonts
    pass


def setup_unhandled_exception_handler() -> None:
    """Set up global unhandled exception handler."""
    from ChannelSimulation.viewer.core.utils import get_log_dir

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception
        log_dir = get_log_dir()
        from datetime import datetime
        log_file = log_dir / f"crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"RFUAV ChannelSim Viewer Crash Report\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Python: {sys.version}\n")
            f.write(f"\nException Type: {exc_type.__name__}\n")
            f.write(f"Exception Message: {exc_value}\n")
            f.write(f"\nStack Trace:\n")
            f.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))

        print(f"[CRITICAL] Unhandled exception occurred. Saved to: {log_file}")
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = exception_handler


def restore_window_state(window) -> bool:
    """Restore window state.

    Args:
        window: Main window object.

    Returns:
        Whether restore was successful.
    """
    try:
        from ChannelSimulation.viewer.core.utils import get_app_data_dir
        import json

        state_file = get_app_data_dir() / "window_state.json"
        if not state_file.exists():
            return False

        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        from PySide6.QtCore import QPoint, QSize

        if "geometry" in state:
            geo = state["geometry"]
            window.resize(QSize(geo.get("width", 1400), geo.get("height", 900)))
            window.move(QPoint(geo.get("x", 100), geo.get("y", 100)))

        if state.get("maximized"):
            window.showMaximized()

        return True
    except Exception as e:
        print(f"[WARNING] Failed to restore window state: {e}")
        return False


def save_window_state(window) -> None:
    """Save window state.

    Args:
        window: Main window object.
    """
    try:
        from ChannelSimulation.viewer.core.utils import get_app_data_dir
        import json

        state_file = get_app_data_dir() / "window_state.json"

        geo = window.geometry()
        state = {
            "geometry": {
                "x": geo.x(),
                "y": geo.y(),
                "width": geo.width(),
                "height": geo.height(),
            },
            "maximized": window.isMaximized(),
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save window state: {e}")


def main() -> int:
    """Main entry point."""
    global _main_window

    # Enable high DPI
    app = get_app()

    # Set up unhandled exception handler
    setup_unhandled_exception_handler()

    # Show splash screen
    splash = show_splash_screen("Loading...")

    try:
        # Lazy import core modules
        update_splash_message(splash, "Loading UI styles...")
        from ChannelSimulation.viewer.core.ui_style import get_all_stylesheet
        setup_theme(app)

        update_splash_message(splash, "Loading main window...")
        from ChannelSimulation.viewer.main_window import MainWindow
        window = MainWindow()

        # Restore window state
        restore_window_state(window)

        update_splash_message(splash, "Finalizing initialization...")
        _main_window = window

        # Close splash screen and show main window
        if splash is not None:
            splash.finish(window)

        window.show()

        return app.exec()

    except ImportError as e:
        print(f"[ERROR] Missing required dependency: {e}")
        print("Please run: pip install -r ChannelSimulation/viewer/requirements.txt")
        return 1

    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
