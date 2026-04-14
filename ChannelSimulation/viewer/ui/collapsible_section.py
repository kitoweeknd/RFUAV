"""
Collapsible section (accordion row) for sidebar panels.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QToolButton, QVBoxLayout, QWidget


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
        self._btn.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; padding: 4px 2px; text-align: left; }"
        )
        self._btn.toggled.connect(self._on_toggled)

        self._line = QFrame()
        self._line.setFrameShape(QFrame.Shape.HLine)
        self._line.setFrameShadow(QFrame.Shadow.Sunken)

        self._content.setVisible(expanded)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)
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
