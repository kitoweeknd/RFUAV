"""
Collapsible section (accordion row) for sidebar panels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget

from ChannelSimulation.viewer.core.ui_style import get_color


class CollapsibleSection(QWidget):
    """A titled header that expands/collapses its content (dropdown-style)."""

    def __init__(
        self,
        title: str,
        content: QWidget,
        parent: QWidget | None = None,
        *,
        expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self._content = content

        self._btn = QToolButton(self)
        self._btn.setText(title)
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._btn.setAutoRaise(True)
        self._btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn.setStyleSheet(
            f"""
            QToolButton {{
                border: none;
                border-radius: 5px;
                font-weight: 600;
                font-size: 12px;
                letter-spacing: 0.3px;
                padding: 6px 6px;
                color: {get_color('text_primary')};
                text-align: left;
            }}
            QToolButton:hover {{
                background-color: {get_color('control_hover')};
            }}
            """
        )
        self._btn.toggled.connect(self._on_toggled)

        # Clean 1px divider (flat, theme-coloured — not the dated sunken bevel).
        self._line = QFrame()
        self._line.setFixedHeight(1)
        self._line.setStyleSheet(
            f"background-color: {get_color('border')}; border: none;"
        )

        self._content.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._btn)
        outer.addWidget(self._line)
        outer.addWidget(self._content)

    def _on_toggled(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self._btn.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )
        self._line.setVisible(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self._btn.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self._btn.isChecked()
