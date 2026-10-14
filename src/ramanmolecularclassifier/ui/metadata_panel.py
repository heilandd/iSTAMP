"""A key/value metadata table panel."""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MetadataPanel(QWidget):
    """Displays a flat dict of metadata + optional proprietary-tag list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = QLabel("Metadata")
        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Field", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        layout.addWidget(self._table)

    def set_metadata(self, data: Dict[str, object], proprietary: List[str] | None = None) -> None:
        rows = list(data.items())
        if proprietary:
            rows.append(("proprietary/private tags", f"{len(proprietary)} found"))
            for i, tag in enumerate(proprietary[:25]):
                rows.append((f"  private[{i}]", tag))
        self._table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self._table.setItem(r, 0, QTableWidgetItem(str(k)))
            self._table.setItem(r, 1, QTableWidgetItem(str(v)))
