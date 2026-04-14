"""
Keyboard Shortcuts Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines all keyboard shortcuts used in the application.
Uses standardized naming and grouping for easy management and maintenance.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence


class ShortcutCategory(Enum):
    """Shortcut category"""
    FILE = auto()      # File operations
    VIEW = auto()      # View operations
    EDIT = auto()      # Edit operations
    PROCESS = auto()   # Processing operations
    TOOLS = auto()     # Tools operations
    HELP = auto()      # Help


class Shortcut:
    """Single shortcut definition"""

    def __init__(
        self,
        id: str,
        category: ShortcutCategory,
        label: str,
        key_sequence: str,
        description: str,
    ):
        self.id = id
        self.category = category
        self.label = label
        self.key_sequence = key_sequence
        self.description = description
        self.qkey_sequence = QKeySequence(key_sequence)

    def __repr__(self) -> str:
        return f"Shortcut({self.id}, {self.key_sequence})"


# All shortcut definitions
SHORTCUTS: list[Shortcut] = [
    # File operations
    Shortcut(
        id="file.load_iq",
        category=ShortcutCategory.FILE,
        label="Load IQ File",
        key_sequence="Ctrl+O",
        description="Open file dialog to load IQ binary file",
    ),
    Shortcut(
        id="file.save_config",
        category=ShortcutCategory.FILE,
        label="Save Configuration",
        key_sequence="Ctrl+Shift+S",
        description="Save current configuration as YAML file",
    ),
    Shortcut(
        id="file.load_config",
        category=ShortcutCategory.FILE,
        label="Load Configuration",
        key_sequence="Ctrl+Shift+O",
        description="Load configuration from YAML file",
    ),
    Shortcut(
        id="file.export_iq",
        category=ShortcutCategory.FILE,
        label="Export IQ",
        key_sequence="Ctrl+S",
        description="Export processed IQ data",
    ),
    Shortcut(
        id="file.export_fig",
        category=ShortcutCategory.FILE,
        label="Export Spectrum",
        key_sequence="Ctrl+E",
        description="Export current spectrum as PNG image",
    ),
    Shortcut(
        id="file.quit",
        category=ShortcutCategory.FILE,
        label="Exit Program",
        key_sequence="Ctrl+Q",
        description="Close the application",
    ),

    # View operations
    Shortcut(
        id="view.toggle_theme",
        category=ShortcutCategory.VIEW,
        label="Toggle Theme",
        key_sequence="Ctrl+T",
        description="Toggle between light/dark theme",
    ),
    Shortcut(
        id="view.zoom_in",
        category=ShortcutCategory.VIEW,
        label="Zoom In",
        key_sequence="Ctrl++",
        description="Zoom in the spectrum view",
    ),
    Shortcut(
        id="view.zoom_out",
        category=ShortcutCategory.VIEW,
        label="Zoom Out",
        key_sequence="Ctrl+-",
        description="Zoom out the spectrum view",
    ),
    Shortcut(
        id="view.reset_zoom",
        category=ShortcutCategory.VIEW,
        label="Reset Zoom",
        key_sequence="Ctrl+0",
        description="Reset spectrum view to default zoom level",
    ),
    Shortcut(
        id="view.toggle_fullscreen",
        category=ShortcutCategory.VIEW,
        label="Fullscreen",
        key_sequence="F11",
        description="Toggle fullscreen mode",
    ),

    # Edit operations
    Shortcut(
        id="edit.undo",
        category=ShortcutCategory.EDIT,
        label="Undo",
        key_sequence="Ctrl+Z",
        description="Undo recent parameter changes",
    ),
    Shortcut(
        id="edit.redo",
        category=ShortcutCategory.EDIT,
        label="Redo",
        key_sequence="Ctrl+Y",
        description="Redo previous undo operation",
    ),
    Shortcut(
        id="edit.reset",
        category=ShortcutCategory.EDIT,
        label="Reset Parameters",
        key_sequence="Ctrl+R",
        description="Restore all parameters to default values",
    ),

    # Processing operations
    Shortcut(
        id="process.preview",
        category=ShortcutCategory.PROCESS,
        label="Preview",
        key_sequence="F5",
        description="Apply current parameters and refresh spectrum preview",
    ),
    Shortcut(
        id="process.stop",
        category=ShortcutCategory.PROCESS,
        label="Stop Processing",
        key_sequence="Escape",
        description="Stop the currently running processing task",
    ),
    Shortcut(
        id="process.play_animation",
        category=ShortcutCategory.PROCESS,
        label="Play Animation",
        key_sequence="Space",
        description="Play/pause STFT animation",
    ),

    # Tools operations
    Shortcut(
        id="tools.batch",
        category=ShortcutCategory.TOOLS,
        label="Batch Processing",
        key_sequence="Ctrl+B",
        description="Open batch processing dialog",
    ),
    Shortcut(
        id="tools.help",
        category=ShortcutCategory.TOOLS,
        label="Usage Guide",
        key_sequence="F1",
        description="Show help documentation",
    ),
    Shortcut(
        id="tools.about",
        category=ShortcutCategory.TOOLS,
        label="About",
        key_sequence="",
        description="Show about dialog",
    ),

    # Help
    Shortcut(
        id="help.shortcuts",
        category=ShortcutCategory.HELP,
        label="Shortcut List",
        key_sequence="Ctrl+/",
        description="Show keyboard shortcuts help dialog",
    ),
]


def get_shortcut(shortcut_id: str) -> Shortcut | None:
    """Get shortcut definition by ID

    Args:
        shortcut_id: Shortcut ID

    Returns:
        Shortcut definition or None
    """
    for shortcut in SHORTCUTS:
        if shortcut.id == shortcut_id:
            return shortcut
    return None


def get_shortcuts_by_category(category: ShortcutCategory) -> list[Shortcut]:
    """Get all shortcuts in the specified category

    Args:
        category: Category

    Returns:
        List of shortcuts
    """
    return [s for s in SHORTCUTS if s.category == category]


def get_all_categories() -> list[ShortcutCategory]:
    """Get all shortcut categories

    Returns:
        List of categories
    """
    return list(ShortcutCategory)


def format_shortcut_list() -> str:
    """Format shortcut list as text

    Returns:
        Formatted text
    """
    lines = ["Keyboard Shortcuts", "=" * 40, ""]

    current_category = None
    for shortcut in SHORTCUTS:
        if shortcut.category != current_category:
            current_category = shortcut.category
            lines.append(f"\n{current_category.name}:")
            lines.append("-" * 20)

        if shortcut.key_sequence:
            lines.append(f"  {shortcut.label}: {shortcut.key_sequence}")
        else:
            lines.append(f"  {shortcut.label}: (None)")

    return "\n".join(lines)


def get_shortcut_table_html() -> str:
    """Generate HTML table for shortcuts

    Returns:
        HTML table
    """
    rows = []
    for shortcut in SHORTCUTS:
        if shortcut.key_sequence:
            keys = shortcut.key_sequence.replace("+", "</kbd>+<kbd>")
            shortcut_str = f"<kbd>{keys}</kbd>"
        else:
            shortcut_str = "-"
        rows.append(
            f"<tr><td>{shortcut.label}</td>"
            f"<td>{shortcut.description}</td>"
            f"<td>{shortcut_str}</td></tr>"
        )

    return f"""
    <table>
        <thead>
            <tr><th>Action</th><th>Description</th><th>Shortcut</th></tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """
