"""Heatmap viewer: embeds a matplotlib heatmap figure."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class HeatmapViewer(QWidget):
    """Displays a heatmap :class:`matplotlib.figure.Figure` with a nav toolbar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._placeholder = QLabel("No heatmap yet.")
        self._canvas: Optional[FigureCanvasQTAgg] = None
        self._toolbar: Optional[NavigationToolbar2QT] = None
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._placeholder)

    def set_figure(self, fig: Optional[Figure]) -> None:
        # Clear previous canvas/toolbar.
        for w in (self._toolbar, self._canvas):
            if w is not None:
                self._layout.removeWidget(w)
                w.setParent(None)
        self._canvas = None
        self._toolbar = None

        if fig is None:
            self._placeholder.setVisible(True)
            return
        self._placeholder.setVisible(False)
        self._canvas = FigureCanvasQTAgg(fig)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        self._canvas.draw_idle()
