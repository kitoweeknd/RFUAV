"""
RFUAV ChannelSim Viewer Application Entry Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles application initialization, splash screen display, high DPI adaptation,
and theme setup. Uses lazy import strategy to speed up startup.
"""

from __future__ import annotations

import os
import stat
import sys
import traceback
from pathlib import Path


def _unhide_qt_plugins() -> None:
    """Clear the macOS 'hidden' (UF_HIDDEN) flag on PySide6's Qt plugin files.

    uv installs on macOS can mark the venv's files hidden (a chflags flag).
    Qt scans its plugin directories with QDir, which SKIPS hidden files, so the
    platform plugin (libqcocoa.dylib) becomes invisible and the app aborts with
    "Could not find the Qt platform plugin cocoa". The flag reappears after any
    uv (re)install, so we self-heal here at startup instead of relying on a
    one-off manual `chflags -R nohidden .venv`. Best-effort, macOS only; the
    Qt frameworks themselves load via dlopen (which does not skip hidden files),
    so only the scanned plugin directory needs clearing.
    """
    if sys.platform != "darwin" or not hasattr(os, "chflags"):
        return
    try:
        import PySide6
        plugins = Path(PySide6.__file__).parent / "Qt" / "plugins"
        if not plugins.is_dir():
            return
        for p in plugins.rglob("*"):
            try:
                flags = p.lstat().st_flags
                if flags & stat.UF_HIDDEN:
                    os.chflags(p, flags & ~stat.UF_HIDDEN)
            except OSError:
                pass
    except Exception:
        pass


# ============================================================================
# High DPI Configuration (must be set before QApplication creation)
# ============================================================================
try:
    from PySide6 import __version__ as _PYSIDE_VERSION
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    # Ensure Qt can see its platform plugins even if uv hid the venv files.
    _unhide_qt_plugins()

    # The AA_EnableHighDpiScaling / AA_UseHighDpiPixmaps attributes are only
    # needed on PySide6 < 6.5; 6.5+ handles HiDPI automatically and marks them
    # deprecated (setting them emits a DeprecationWarning). Gate on the cheap
    # __version__ attribute (no importlib.metadata scan) so we neither pay the
    # lookup cost nor trigger the warning on modern PySide6.
    if tuple(int(x) for x in _PYSIDE_VERSION.split(".")[:2]) < (6, 5):
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


def _create_splash_pixmap() -> "QPixmap":
    """Create the base splash screen pixmap (title + version only).

    The per-message status text is overlaid separately via
    ``QSplashScreen.showMessage`` so this pixmap only needs to be built once.
    """
    from PySide6.QtGui import QPixmap, QColor, QPainter, QFont
    pixmap = QPixmap(400, 200)
    pixmap.fill(QColor("#1a73e8"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 16, QFont.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "RFUAV ChannelSim Viewer")
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(pixmap.rect().adjusted(0, 30, 0, 0), Qt.AlignCenter, "v1.3.0")
    painter.end()
    return pixmap


def show_splash_screen(message: str = "Initializing...") -> "QSplashScreen | None":
    """Display the splash screen.

    Args:
        message: Initial message to display.

    Returns:
        Splash screen object or None.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QSplashScreen

        pixmap = _create_splash_pixmap()
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
        splash.setWindowFlag(Qt.FramelessWindowHint)
        splash.show()
        splash.showMessage(message, Qt.AlignBottom | Qt.AlignHCenter, QColor("#ffffff"))
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
            from PySide6.QtGui import QColor

            splash.showMessage(message, Qt.AlignBottom | Qt.AlignHCenter, QColor("#ffffff"))
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


def setup_unhandled_exception_handler() -> None:
    """Set up global unhandled exception handler."""
    from ChannelSimulation.viewer.core.utils import get_log_dir

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Log the exception to a crash file, but never let an I/O error here
        # mask the original traceback.
        try:
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
        except Exception as log_error:
            print(f"[CRITICAL] Unhandled exception occurred (failed to write crash log: {log_error})")

        # Always print the original traceback, regardless of logging success.
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

        # Restore dock/toolbar layout (movable panels) if present.
        dock_state = state.get("dock_state")
        if dock_state:
            from PySide6.QtCore import QByteArray
            window.restoreState(QByteArray.fromBase64(dock_state.encode("ascii")))

        return True
    except Exception as e:
        print(f"[WARNING] Failed to restore window state: {e}")
        return False


def is_first_launch() -> bool:
    """Check if this is the first launch (no app data exists yet)."""
    try:
        from ChannelSimulation.viewer.core.utils import get_app_data_dir
        marker = get_app_data_dir() / ".first_run_done"
        return not marker.exists()
    except Exception:
        return False


def mark_first_launch_done() -> None:
    """Mark that the first launch tutorial has been shown."""
    try:
        from ChannelSimulation.viewer.core.utils import get_app_data_dir
        marker = get_app_data_dir() / ".first_run_done"
        marker.touch()
    except Exception:
        pass


def show_first_launch_help(window) -> None:
    """Show the quick-start tutorial on first launch."""
    from ChannelSimulation.viewer.advanced.help_panel import HelpDialog
    dialog = HelpDialog(window)
    dialog.setWindowTitle("Quick Start Tutorial — RFUAV ChannelSim Viewer")
    dialog.exec()
    mark_first_launch_done()


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
            # Dock/toolbar arrangement (base64) so a rearranged layout persists.
            "dock_state": bytes(window.saveState().toBase64()).decode("ascii"),
        }

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Failed to save window state: {e}")


def _ensure_presets_exist() -> None:
    """Ensure default presets exist, create if missing."""
    presets_dir = Path(__file__).parent / "core" / "presets"
    if not presets_dir.exists() or not list(presets_dir.glob("*.yaml")):
        from ChannelSimulation.viewer.business.presets_manager import create_default_presets
        create_default_presets()


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
        # Ensure default presets exist
        _ensure_presets_exist()

        # Lazy import core modules (dark theme by default)
        update_splash_message(splash, "Loading UI styles...")
        setup_theme(app, dark=True)

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

        # First-launch tutorial
        if is_first_launch():
            show_first_launch_help(window)

        return app.exec()

    except ImportError as e:
        print(f"[ERROR] Missing required dependency: {e}")
        return 1

    except Exception as e:
        print(f"[ERROR] Startup failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
