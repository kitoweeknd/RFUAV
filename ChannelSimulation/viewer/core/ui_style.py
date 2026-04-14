"""
UI Style System
~~~~~~~~~~~~~~~

Defines QSS stylesheets in Qt Fusion Design Language style,
supporting Light Fusion and Dark Fusion theme switching.

Color specifications follow the PRD document:
- Primary color: #1a73e8
- Secondary color: #5f6368
- Accent color: #ea4335
- Success color: #34a853
- Background: #ffffff / #1e1e1e (dark)
- Card background: #f8f9fa / #2d2d2d (dark)
- Border color: #dadce0 / #404040 (dark)
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy

# ============================================================================
# Theme color definitions
# ============================================================================

# Light theme
THEME_LIGHT: Final[dict[str, str]] = {
    # Background colors
    "bg_primary": "#ffffff",
    "bg_secondary": "#f8f9fa",
    "bg_tertiary": "#f1f3f4",
    # Text colors
    "text_primary": "#202124",
    "text_secondary": "#5f6368",
    "text_disabled": "#9aa0a6",
    # Accent colors
    "accent": "#1a73e8",
    "accent_hover": "#1557b0",
    "accent_disabled": "#aec4eb",
    # Functional colors
    "danger": "#ea4335",
    "danger_hover": "#c5221f",
    "success": "#34a853",
    "warning": "#fbbc04",
    # Borders
    "border": "#dadce0",
    "border_focus": "#1a73e8",
    "border_error": "#ea4335",
    # Controls
    "control_bg": "#f1f3f4",
    "control_hover": "#e8eaed",
    "slider_track": "#dadce0",
    "slider_handle": "#1a73e8",
    # Channel card
    "card_bg": "#ffffff",
    "card_shadow": "0 3px 6px rgba(0,0,0,0.08)",
    # Spectrum
    "spectrum_bg": "#ffffff",
    "grid_line": "#e0e0e0",
}

# Dark theme
THEME_DARK: Final[dict[str, str]] = {
    # Background colors
    "bg_primary": "#1e1e1e",
    "bg_secondary": "#2d2d2d",
    "bg_tertiary": "#383838",
    # Text colors
    "text_primary": "#e0e0e0",
    "text_secondary": "#9aa0a6",
    "text_disabled": "#6b6b6b",
    # Accent colors
    "accent": "#4a90e2",
    "accent_hover": "#5da0f2",
    "accent_disabled": "#3a4a5a",
    # Functional colors
    "danger": "#f2574a",
    "danger_hover": "#ff6b63",
    "success": "#3dba5f",
    "warning": "#fcc934",
    # Borders
    "border": "#404040",
    "border_focus": "#4a90e2",
    "border_error": "#f2574a",
    # Controls
    "control_bg": "#383838",
    "control_hover": "#454545",
    "slider_track": "#404040",
    "slider_handle": "#4a90e2",
    # Channel card
    "card_bg": "#2d2d2d",
    "card_shadow": "0 3px 6px rgba(0,0,0,0.3)",
    # Spectrum
    "spectrum_bg": "#1e1e1e",
    "grid_line": "#404040",
}

# Current theme (default light)
_current_theme: dict[str, str] = THEME_LIGHT.copy()
_is_dark_theme: bool = False


def get_theme() -> dict[str, str]:
    """Get current theme color dictionary"""
    return _current_theme.copy()


def is_dark_theme() -> bool:
    """Check if dark theme"""
    return _is_dark_theme


def set_theme(dark: bool) -> None:
    """Set theme type"""
    global _current_theme, _is_dark_theme
    _current_theme = THEME_DARK.copy() if dark else THEME_LIGHT.copy()
    _is_dark_theme = dark


def get_color(key: str) -> str:
    """Get specified color value from current theme"""
    return _current_theme.get(key, THEME_LIGHT.get(key, "#000000"))


# ============================================================================
# QSS stylesheet generators
# ============================================================================

def _css_var(name: str) -> str:
    """Generate CSS variable reference"""
    return f"var(--{name})"


def _build_css_variables() -> str:
    """Generate CSS variable definition block"""
    vars_list = [f"    --{k}: {v};" for k, v in _current_theme.items()]
    return ":\root {\n" + "\n".join(vars_list) + "\n}"


def get_main_window_qss() -> str:
    """Main window stylesheet"""
    return f"""
    QMainWindow {{
        background-color: {get_color('bg_primary')};
        color: {get_color('text_primary')};
    }}

    QWidget {{
        background-color: {get_color('bg_primary')};
        color: {get_color('text_primary')};
        font-family: "Segoe UI", "SF Pro", "Ubuntu", sans-serif;
        font-size: 13px;
    }}

    QLabel {{
        background-color: transparent;
        color: {get_color('text_primary')};
    }}

    QLabel.QLabelSecondary {{
        color: {get_color('text_secondary')};
        font-size: 11px;
    }}

    QLabel.QLabelMono {{
        font-family: "Consolas", "Courier New", monospace;
        font-size: 12px;
    }}
    """


def get_menu_qss() -> str:
    """Menu bar stylesheet"""
    return f"""
    QMenuBar {{
        background-color: {get_color('bg_secondary')};
        color: {get_color('text_primary')};
        border-bottom: 1px solid {get_color('border')};
        padding: 4px;
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 6px 12px;
        border-radius: 4px;
    }}

    QMenuBar::item:selected {{
        background-color: {get_color('control_hover')};
    }}

    QMenuBar::item:pressed {{
        background-color: {get_color('control_bg')};
    }}

    QMenu {{
        background-color: {get_color('bg_secondary')};
        color: {get_color('text_primary')};
        border: 1px solid {get_color('border')};
        border-radius: 8px;
        padding: 4px;
    }}

    QMenu::item {{
        padding: 8px 24px 8px 12px;
        border-radius: 4px;
    }}

    QMenu::item:selected {{
        background-color: {get_color('accent')};
        color: #ffffff;
    }}

    QMenu::separator {{
        height: 1px;
        background-color: {get_color('border')};
        margin: 4px 8px;
    }}
    """


def get_toolbar_qss() -> str:
    """Toolbar stylesheet"""
    return f"""
    QToolBar {{
        background-color: {get_color('bg_secondary')};
        border: none;
        border-bottom: 1px solid {get_color('border')};
        spacing: 8px;
        padding: 4px 8px;
        height: 48px;
    }}

    QToolBar::separator {{
        width: 1px;
        background-color: {get_color('border')};
        margin: 8px 4px;
    }}

    QToolButton {{
        background-color: transparent;
        border: none;
        border-radius: 4px;
        padding: 8px 12px;
        color: {get_color('text_primary')};
    }}

    QToolButton:hover {{
        background-color: {get_color('control_hover')};
    }}

    QToolButton:pressed {{
        background-color: {get_color('control_bg')};
    }}

    QToolButton:disabled {{
        color: {get_color('text_disabled')};
    }}
    """


def get_button_qss() -> str:
    """Button stylesheet"""
    return f"""
    QPushButton {{
        background-color: {get_color('bg_secondary')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 8px 16px;
        color: {get_color('text_primary')};
        min-height: 20px;
    }}

    QPushButton:hover {{
        background-color: {get_color('control_hover')};
        border-color: {get_color('accent')};
    }}

    QPushButton:pressed {{
        background-color: {get_color('control_bg')};
    }}

    QPushButton:disabled {{
        background-color: {get_color('bg_tertiary')};
        color: {get_color('text_disabled')};
        border-color: {get_color('border')};
    }}

    QPushButton#btnPrimary {{
        background-color: {get_color('accent')};
        border-color: {get_color('accent')};
        color: #ffffff;
    }}

    QPushButton#btnPrimary:hover {{
        background-color: {get_color('accent_hover')};
        border-color: {get_color('accent_hover')};
    }}

    QPushButton#btnPrimary:disabled {{
        background-color: {get_color('accent_disabled')};
        border-color: {get_color('accent_disabled')};
        color: #ffffff;
    }}

    QPushButton#btnDanger {{
        background-color: {get_color('danger')};
        border-color: {get_color('danger')};
        color: #ffffff;
    }}

    QPushButton#btnDanger:hover {{
        background-color: {get_color('danger_hover')};
        border-color: {get_color('danger_hover')};
    }}
    """


def get_checkbox_qss() -> str:
    """Checkbox stylesheet (toggle style)"""
    return f"""
    QCheckBox {{
        spacing: 8px;
        color: {get_color('text_primary')};
    }}

    QCheckBox::indicator {{
        width: 44px;
        height: 24px;
        border-radius: 12px;
        background-color: {get_color('slider_track')};
        border: none;
    }}

    QCheckBox::indicator:checked {{
        background-color: {get_color('accent')};
    }}

    QCheckBox::indicator:disabled {{
        background-color: {get_color('border')};
    }}

    QCheckBox::indicator:hover {{
        border: 2px solid {get_color('accent')};
    }}

    QCheckBox:disabled {{
        color: {get_color('text_disabled')};
    }}
    """


def get_slider_qss() -> str:
    """Slider stylesheet"""
    return f"""
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background-color: {get_color('slider_track')};
        border-radius: 2px;
    }}

    QSlider::groove:horizontal:disabled {{
        background-color: {get_color('border')};
    }}

    QSlider::handle:horizontal {{
        background-color: {get_color('slider_handle')};
        border: none;
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }}

    QSlider::handle:horizontal:hover {{
        background-color: {get_color('accent_hover')};
    }}

    QSlider::handle:horizontal:disabled {{
        background-color: {get_color('text_disabled')};
    }}

    QSlider::sub-page:horizontal {{
        background-color: {get_color('accent')};
        border-radius: 2px;
    }}
    """


def get_lineedit_qss() -> str:
    """Input field stylesheet"""
    return f"""
    QLineEdit {{
        background-color: {get_color('control_bg')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 6px 10px;
        color: {get_color('text_primary')};
        selection-background-color: {get_color('accent')};
        selection-color: #ffffff;
    }}

    QLineEdit:hover {{
        border-color: {get_color('accent')};
    }}

    QLineEdit:focus {{
        border-color: {get_color('border_focus')};
        background-color: {get_color('bg_primary')};
    }}

    QLineEdit:disabled {{
        background-color: {get_color('bg_tertiary')};
        color: {get_color('text_disabled')};
    }}

    QLineEdit:read-only {{
        background-color: {get_color('bg_secondary')};
    }}
    """


def get_combobox_qss() -> str:
    """Dropdown stylesheet"""
    return f"""
    QComboBox {{
        background-color: {get_color('control_bg')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 6px 10px;
        color: {get_color('text_primary')};
        min-height: 20px;
    }}

    QComboBox:hover {{
        border-color: {get_color('accent')};
    }}

    QComboBox:focus {{
        border-color: {get_color('border_focus')};
    }}

    QComboBox:disabled {{
        background-color: {get_color('bg_tertiary')};
        color: {get_color('text_disabled')};
    }}

    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 6px solid {get_color('text_secondary')};
        margin-right: 8px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {get_color('bg_secondary')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        selection-background-color: {get_color('accent')};
        selection-color: #ffffff;
        padding: 4px;
        max-height: 240px;
    }}

    QComboBox QAbstractItemView::item {{
        padding: 8px 12px;
        border-radius: 4px;
    }}

    QComboBox QAbstractItemView::item:hover {{
        background-color: {get_color('control_hover')};
    }}
    """


def get_groupbox_qss() -> str:
    """Group box stylesheet"""
    return f"""
    QGroupBox {{
        background-color: {get_color('card_bg')};
        border: 1px solid {get_color('border')};
        border-radius: 8px;
        margin-top: 12px;
        padding: 16px 12px 12px 12px;
        font-weight: bold;
        color: {get_color('text_primary')};
    }}

    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 0 8px;
        background-color: {get_color('card_bg')};
    }}
    """


def get_scrollbar_qss() -> str:
    """Scrollbar stylesheet"""
    return f"""
    QScrollArea, QAbstractScrollArea {{
        background-color: {get_color('bg_primary')};
    }}

    QScrollBar:vertical {{
        background-color: transparent;
        width: 10px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background-color: {get_color('border')};
        border-radius: 5px;
        min-height: 30px;
        margin: 2px;
    }}

    QScrollBar::handle:vertical:hover {{
        background-color: {get_color('text_secondary')};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
        border: none;
    }}

    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
    }}

    QScrollBar:horizontal {{
        background-color: transparent;
        height: 10px;
        margin: 0;
    }}

    QScrollBar::handle:horizontal {{
        background-color: {get_color('border')};
        border-radius: 5px;
        min-width: 30px;
        margin: 2px;
    }}

    QScrollBar::handle:horizontal:hover {{
        background-color: {get_color('text_secondary')};
    }}

    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
        border: none;
    }}

    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
    }}
    """


def get_progressbar_qss() -> str:
    """Progress bar stylesheet"""
    return f"""
    QProgressBar {{
        background-color: {get_color('control_bg')};
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
    }}

    QProgressBar::chunk {{
        background-color: {get_color('accent')};
        border-radius: 4px;
    }}
    """


def get_statusbar_qss() -> str:
    """Status bar stylesheet"""
    return f"""
    QStatusBar {{
        background-color: {get_color('bg_secondary')};
        border-top: 1px solid {get_color('border')};
        color: {get_color('text_secondary')};
        font-size: 12px;
    }}

    QStatusBar::item {{
        border: none;
    }}
    """


def get_tabwidget_qss() -> str:
    """Tab widget stylesheet"""
    return f"""
    QTabWidget::pane {{
        border: 1px solid {get_color('border')};
        border-radius: 8px;
        background-color: {get_color('bg_primary')};
        top: -1px;
    }}

    QTabBar {{
        background-color: {get_color('bg_secondary')};
        border: none;
    }}

    QTabBar::tab {{
        background-color: transparent;
        color: {get_color('text_secondary')};
        padding: 10px 20px;
        margin-right: 4px;
        border: none;
        border-radius: 8px 8px 0 0;
    }}

    QTabBar::tab:selected {{
        background-color: {get_color('bg_primary')};
        color: {get_color('accent')};
        font-weight: bold;
    }}

    QTabBar::tab:hover:!selected {{
        background-color: {get_color('control_hover')};
        color: {get_color('text_primary')};
    }}
    """


def get_splitter_qss() -> str:
    """Splitter stylesheet"""
    return f"""
    QSplitter::handle {{
        background-color: {get_color('border')};
    }}

    QSplitter::handle:horizontal {{
        width: 1px;
        margin: 4px 0;
    }}

    QSplitter::handle:vertical {{
        height: 1px;
        margin: 0 4px;
    }}

    QSplitter::handle:hover {{
        background-color: {get_color('accent')};
    }}
    """


def get_card_qss() -> str:
    """Card stylesheet (channel configuration card)"""
    return f"""
    QWidget.card {{
        background-color: {get_color('card_bg')};
        border: 1px solid {get_color('border')};
        border-radius: 8px;
        padding: 0;
    }}

    QWidget.cardTitle {{
        background-color: transparent;
        border: none;
    }}

    QPushButton.cardToggle {{
        background-color: transparent;
        border: none;
        padding: 12px;
        color: {get_color('text_primary')};
        text-align: left;
        font-weight: bold;
    }}

    QPushButton.cardToggle:hover {{
        background-color: {get_color('control_hover')};
    }}
    """


def get_spinbox_qss() -> str:
    """Spinbox stylesheet"""
    return f"""
    QSpinBox, QDoubleSpinBox {{
        background-color: {get_color('control_bg')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 6px 10px;
        color: {get_color('text_primary')};
    }}

    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: {get_color('accent')};
    }}

    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {get_color('border_focus')};
        background-color: {get_color('bg_primary')};
    }}

    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        background-color: transparent;
        border: none;
        width: 16px;
    }}

    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        background-color: transparent;
        border: none;
        width: 16px;
    }}
    """


def get_tooltip_qss() -> str:
    """Tooltip stylesheet"""
    return f"""
    QToolTip {{
        background-color: {get_color('bg_secondary')};
        color: {get_color('text_primary')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 6px 10px;
        font-size: 12px;
    }}
    """


def get_dialog_qss() -> str:
    """Dialog stylesheet"""
    return f"""
    QDialog {{
        background-color: {get_color('bg_primary')};
        color: {get_color('text_primary')};
    }}

    QMessageBox {{
        background-color: {get_color('bg_primary')};
    }}
    """


def get_textedit_qss() -> str:
    """Text edit stylesheet (log viewer)"""
    return f"""
    QTextEdit, QPlainTextEdit {{
        background-color: {get_color('bg_primary')};
        color: {get_color('text_primary')};
        border: 1px solid {get_color('border')};
        border-radius: 4px;
        padding: 8px;
        selection-background-color: {get_color('accent')};
        selection-color: #ffffff;
    }}
    """


def get_all_stylesheet() -> str:
    """Get complete stylesheet (all component styles combined)"""
    parts = [
        get_main_window_qss(),
        get_menu_qss(),
        get_toolbar_qss(),
        get_button_qss(),
        get_checkbox_qss(),
        get_slider_qss(),
        get_lineedit_qss(),
        get_combobox_qss(),
        get_groupbox_qss(),
        get_scrollbar_qss(),
        get_progressbar_qss(),
        get_statusbar_qss(),
        get_tabwidget_qss(),
        get_splitter_qss(),
        get_card_qss(),
        get_spinbox_qss(),
        get_tooltip_qss(),
        get_dialog_qss(),
        get_textedit_qss(),
    ]
    return "\n".join(parts)


# ============================================================================
# Layout constants
# ============================================================================

class LayoutConstants:
    """Layout system constants (base spacing 8px)"""

    # Spacing
    SPACING_XS: Final[int] = 4
    SPACING_SM: Final[int] = 8
    SPACING_MD: Final[int] = 12
    SPACING_LG: Final[int] = 16
    SPACING_XL: Final[int] = 24

    # Padding
    PADDING_XS: Final[int] = 4
    PADDING_SM: Final[int] = 8
    PADDING_MD: Final[int] = 12
    PADDING_LG: Final[int] = 16
    PADDING_XL: Final[int] = 24

    # Control sizes
    BUTTON_HEIGHT: Final[int] = 36
    INPUT_HEIGHT: Final[int] = 32
    SLIDER_HEIGHT: Final[int] = 4
    SLIDER_HANDLE_SIZE: Final[int] = 18

    # Border radius
    RADIUS_XS: Final[int] = 2
    RADIUS_SM: Final[int] = 4
    RADIUS_MD: Final[int] = 8

    # Left panel width
    LEFT_PANEL_WIDTH: Final[int] = 320

    # Window minimum size
    WINDOW_MIN_WIDTH: Final[int] = 1200
    WINDOW_MIN_HEIGHT: Final[int] = 800

    # Toolbar height
    TOOLBAR_HEIGHT: Final[int] = 48

    # Status bar height
    STATUSBAR_HEIGHT: Final[int] = 24

    # DPI settings
    DPI_SCREEN: Final[int] = 150
    DPI_EXPORT: Final[int] = 300


# ============================================================================
# Spectrogram colormap scheme
# ============================================================================

SPECTROGRAM_CMAP_LIGHT: Final[str] = "turbo"
SPECTROGRAM_CMAP_DARK: Final[str] = "turbo"  # turbo has sufficient contrast


def get_spectrogram_cmap() -> str:
    """Get spectrogram colormap for current theme"""
    return SPECTROGRAM_CMAP_DARK if is_dark_theme() else SPECTROGRAM_CMAP_LIGHT


# ============================================================================
# Helper functions
# ============================================================================

def apply_stylesheet(app, stylesheet: str | None = None) -> None:
    """Apply stylesheet to application"""
    if stylesheet is None:
        stylesheet = get_all_stylesheet()
    app.setStyleSheet(stylesheet)


def reload_theme_stylesheet(app) -> None:
    """Reload theme stylesheet (for dynamic theme switching)"""
    apply_stylesheet(app)
