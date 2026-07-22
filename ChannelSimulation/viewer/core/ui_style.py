"""
UI Style System
~~~~~~~~~~~~~~~

Defines QSS stylesheets for a modern dark signal-analysis aesthetic
("Graphite + Cyan"), with a coherent light variant.

Dark palette (default):
- Base #0f1115 · Panel #171a21 · Raised #1e232c (layered surfaces = depth)
- Accent (cyan) #22d3ee · Success #34d399 · Danger #ff6b5e
- Text #e6e9ef / secondary #8b93a1
"""

from __future__ import annotations

from typing import Final

# ============================================================================
# Theme color definitions
# ============================================================================

# Light theme (cyan accent, coherent with dark)
THEME_LIGHT: Final[dict[str, str]] = {
    # Background colors (layered surfaces)
    "bg_primary": "#ffffff",
    "bg_secondary": "#f4f6f8",
    "bg_tertiary": "#eceff3",
    # Text colors
    "text_primary": "#1a1d23",
    "text_secondary": "#5b6472",
    "text_disabled": "#9aa3b0",
    # Accent colors
    "accent": "#0891b2",
    "accent_hover": "#0e7490",
    "accent_pressed": "#155e75",
    "accent_disabled": "#a5d8e4",
    "accent_soft": "rgba(8, 145, 178, 0.12)",
    "accent_soft_hover": "rgba(8, 145, 178, 0.20)",
    "text_on_accent": "#ffffff",
    # Functional colors
    "danger": "#e5484d",
    "danger_hover": "#c93b40",
    "success": "#12a150",
    "success_hover": "#0f8a45",
    "success_pressed": "#0c733a",
    "warning": "#f5a623",
    # Borders
    "border": "#dce1e7",
    "border_focus": "#0891b2",
    "border_error": "#e5484d",
    # Controls
    "control_bg": "#eceff3",
    "control_hover": "#e2e6ec",
    "slider_track": "#dce1e7",
    "slider_handle": "#0891b2",
    # Channel card
    "card_bg": "#ffffff",
    # Spectrum
    "spectrum_bg": "#ffffff",
    "grid_line": "#e3e7ec",
}

# Dark theme (Graphite + Cyan — default)
THEME_DARK: Final[dict[str, str]] = {
    # Background colors (near-black base → layered cool-graphite surfaces)
    "bg_primary": "#0f1115",
    "bg_secondary": "#171a21",
    "bg_tertiary": "#1e232c",
    # Text colors
    "text_primary": "#e6e9ef",
    "text_secondary": "#8b93a1",
    "text_disabled": "#565d69",
    # Accent colors (cyan)
    "accent": "#22d3ee",
    "accent_hover": "#38dced",
    "accent_pressed": "#0fb6d4",
    "accent_disabled": "#1c3a42",
    "accent_soft": "rgba(34, 211, 238, 0.14)",
    "accent_soft_hover": "rgba(34, 211, 238, 0.22)",
    "text_on_accent": "#062830",
    # Functional colors
    "danger": "#ff6b5e",
    "danger_hover": "#ff8478",
    "success": "#34d399",
    "success_hover": "#4ade9f",
    "success_pressed": "#22b883",
    "warning": "#fbbf24",
    # Borders
    "border": "#2a2f3a",
    "border_focus": "#22d3ee",
    "border_error": "#ff6b5e",
    # Controls
    "control_bg": "#1e232c",
    "control_hover": "#262c37",
    "slider_track": "#2a2f3a",
    "slider_handle": "#22d3ee",
    # Channel card
    "card_bg": "#171a21",
    # Spectrum
    "spectrum_bg": "#0f1115",
    "grid_line": "#232833",
}

# Current theme (default dark — the norm for signal-analysis / SDR tools).
_current_theme: dict[str, str] = THEME_DARK
_is_dark_theme: bool = True

# Cache of fully composed stylesheets keyed by the dark-theme flag.
# Themes are immutable Final constants, so a stylesheet built for a given
# flag never goes stale; both entries may coexist across theme switches.
_stylesheet_cache: dict[bool, str] = {}


def is_dark_theme() -> bool:
    """Check if dark theme"""
    return _is_dark_theme


def set_theme(dark: bool) -> None:
    """Set theme type"""
    global _current_theme, _is_dark_theme
    if dark == _is_dark_theme:
        return
    _current_theme = THEME_DARK if dark else THEME_LIGHT
    _is_dark_theme = dark


def get_color(key: str) -> str:
    """Get specified color value from current theme"""
    return _current_theme.get(key, THEME_LIGHT.get(key, "#000000"))


# ============================================================================
# QSS stylesheet generators
# ============================================================================

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
        padding: 4px 4px 0px 4px;
    }}

    QMenuBar::item {{
        background-color: transparent;
        padding: 8px 16px;
        border-radius: 4px;
        margin-bottom: 4px;
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

    QToolButton:checked {{
        background-color: {get_color('accent_soft')};
        color: {get_color('accent')};
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
        color: {get_color('text_on_accent')};
        font-weight: 600;
    }}

    QPushButton#btnPrimary:hover {{
        background-color: {get_color('accent_hover')};
        border-color: {get_color('accent_hover')};
    }}

    QPushButton#btnPrimary:pressed {{
        background-color: {get_color('accent_pressed')};
        border-color: {get_color('accent_pressed')};
    }}

    QPushButton#btnPrimary:disabled {{
        background-color: {get_color('accent_disabled')};
        border-color: {get_color('accent_disabled')};
        color: {get_color('text_disabled')};
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

    QPushButton#btnScenario {{
        background-color: {get_color('accent_soft')};
        border: 1px solid {get_color('accent')};
        border-radius: 6px;
        color: {get_color('accent')};
        font-weight: 600;
        padding: 6px 14px;
        min-height: 24px;
    }}

    QPushButton#btnScenario:hover {{
        background-color: {get_color('accent_soft_hover')};
        border-color: {get_color('accent_hover')};
        color: {get_color('accent_hover')};
    }}

    QPushButton#btnScenario:pressed {{
        background-color: {get_color('accent_soft_hover')};
        border-color: {get_color('accent_pressed')};
    }}

    QPushButton#btnScenario:disabled {{
        background-color: {get_color('bg_tertiary')};
        border-color: {get_color('border')};
        color: {get_color('text_disabled')};
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


def get_dockwidget_qss() -> str:
    """Dock widget title-bar stylesheet (matches the themed surfaces)."""
    return f"""
    QDockWidget {{
        color: {get_color('text_primary')};
        font-weight: 600;
    }}

    QDockWidget::title {{
        background-color: {get_color('bg_secondary')};
        padding: 8px 12px;
        border-bottom: 1px solid {get_color('border')};
        text-align: left;
    }}

    QDockWidget::close-button, QDockWidget::float-button {{
        background: transparent;
        border: none;
        padding: 2px;
        icon-size: 14px;
        right: 6px;
    }}

    QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
        background-color: {get_color('control_hover')};
        border-radius: 4px;
    }}
    """


def get_all_stylesheet() -> str:
    """Get complete stylesheet (all component styles combined)"""
    cached = _stylesheet_cache.get(_is_dark_theme)
    if cached is not None:
        return cached
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
        get_spinbox_qss(),
        get_tooltip_qss(),
        get_dialog_qss(),
        get_textedit_qss(),
        get_dockwidget_qss(),
    ]
    stylesheet = "\n".join(parts)
    _stylesheet_cache[_is_dark_theme] = stylesheet
    return stylesheet


# ============================================================================
# Spectrogram colormap scheme
# ============================================================================

SPECTROGRAM_CMAP_LIGHT: Final[str] = "turbo"
SPECTROGRAM_CMAP_DARK: Final[str] = "turbo"  # turbo has sufficient contrast


def get_spectrogram_cmap() -> str:
    """Get spectrogram colormap for current theme"""
    return SPECTROGRAM_CMAP_DARK if is_dark_theme() else SPECTROGRAM_CMAP_LIGHT


def get_pg_colormap():
    """Return a pyqtgraph ColorMap for the spectrogram (matplotlib 'turbo').

    Bridges the theme's colormap name to a pyqtgraph ColorMap so ImageItem /
    HistogramLUTItem render with the same palette the old matplotlib canvas used.
    """
    import pyqtgraph as pg
    return pg.colormap.getFromMatplotlib(get_spectrogram_cmap())


def apply_pyqtgraph_theme() -> None:
    """Set pyqtgraph's global background/foreground from the active theme.

    Called once at startup and again on theme switch so every pyqtgraph plot
    (spectrogram, PSD, time, SNR) picks up the dark/light palette.
    """
    import pyqtgraph as pg
    pg.setConfigOptions(
        background=get_color("spectrum_bg"),
        foreground=get_color("text_primary"),
        antialias=False,  # snappier interaction; curves are downsampled/clipped anyway
        imageAxisOrder="row-major",  # image arrays are (row=freq, col=time)
    )


# ============================================================================
# Helper functions
# ============================================================================

def apply_stylesheet(app, stylesheet: str | None = None) -> None:
    """Apply stylesheet to application"""
    if stylesheet is None:
        stylesheet = get_all_stylesheet()
    app.setStyleSheet(stylesheet)
