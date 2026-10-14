"""Image viewer with zoom / pan (QGraphicsView based)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QVBoxLayout,
    QWidget,
)


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert an HxW(xC) uint8 numpy array to a QPixmap."""
    if arr.ndim == 2:
        h, w = arr.shape
        img = QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, c = arr.shape
        if c == 4:
            img = QImage(arr.tobytes(), w, h, 4 * w, QImage.Format_RGBA8888)
        else:
            arr3 = np.ascontiguousarray(arr[:, :, :3])
            img = QImage(arr3.tobytes(), w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class _ZoomView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)


class ImageViewer(QWidget):
    """Displays an SRH image array with mouse-wheel zoom and drag pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._view = _ZoomView(self._scene)
        self._item: Optional[QGraphicsPixmapItem] = None
        self._caption = QLabel("No image loaded.")

        layout = QVBoxLayout(self)
        layout.addWidget(self._view)
        layout.addWidget(self._caption)

    def set_image(self, arr: Optional[np.ndarray], caption: str = "") -> None:
        self._scene.clear()
        self._item = None
        if arr is None:
            self._caption.setText(caption or "No image data (metadata only).")
            return
        pix = numpy_to_qpixmap(np.ascontiguousarray(arr))
        self._item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._item.boundingRect())
        self._view.fitInView(self._item, Qt.KeepAspectRatio)
        self._caption.setText(caption)

    def reset_view(self) -> None:
        if self._item is not None:
            self._view.fitInView(self._item, Qt.KeepAspectRatio)
