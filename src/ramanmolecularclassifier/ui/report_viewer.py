"""Report viewer: shows the generated HTML report and links to artefacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..reporting.report_engine import ReportOutputs


class ReportViewer(QWidget):
    """Renders the HTML report inline and offers buttons to open PDF/xlsx."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._browser = QTextBrowser(self)
        self._browser.setOpenExternalLinks(True)
        self._status = QLabel("No report generated yet.")

        self._btn_pdf = QPushButton("Open PDF")
        self._btn_xlsx = QPushButton("Open Excel")
        self._btn_dir = QPushButton("Open output folder")
        for b in (self._btn_pdf, self._btn_xlsx, self._btn_dir):
            b.setEnabled(False)

        self._btn_pdf.clicked.connect(lambda: self._open(self._pdf))
        self._btn_xlsx.clicked.connect(lambda: self._open(self._xlsx))
        self._btn_dir.clicked.connect(lambda: self._open(self._dir))

        btns = QHBoxLayout()
        btns.addWidget(self._btn_pdf)
        btns.addWidget(self._btn_xlsx)
        btns.addWidget(self._btn_dir)
        btns.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._status)
        layout.addLayout(btns)
        layout.addWidget(self._browser)

        self._pdf: Optional[Path] = None
        self._xlsx: Optional[Path] = None
        self._dir: Optional[Path] = None

    def set_reports(self, reports: Optional[ReportOutputs]) -> None:
        if reports is None:
            self._status.setText("No report generated yet.")
            return
        self._pdf = reports.pdf
        self._xlsx = reports.xlsx
        self._dir = reports.html.parent if reports.html else (
            reports.pdf.parent if reports.pdf else None)
        self._btn_pdf.setEnabled(self._pdf is not None)
        self._btn_xlsx.setEnabled(self._xlsx is not None)
        self._btn_dir.setEnabled(self._dir is not None)

        if reports.html and reports.html.exists():
            self._browser.setHtml(reports.html.read_text(encoding="utf-8"))
            self._status.setText(f"Report: {reports.html.name}")
        else:
            self._status.setText("Report generated (no HTML preview).")

    @staticmethod
    def _open(path: Optional[Path]) -> None:
        if path is not None and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
