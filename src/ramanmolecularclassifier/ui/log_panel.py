"""A simple append-only log panel with a Python ``logging`` handler."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class _QtLogHandler(logging.Handler):
    """Forwards ``logging`` records to the panel via a Qt signal."""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn

    def emit(self, record):  # noqa: D401
        self._emit_fn(self.format(record))


class LogPanel(QWidget):
    """Thread-safe log view (use :meth:`append` from any thread)."""

    _message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._view = QPlainTextEdit(self)
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(5000)
        self._view.setFont(QFont("Menlo", 10))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        # Signal hop ensures appends happen on the GUI thread.
        self._message.connect(self._do_append, Qt.QueuedConnection)

    def append(self, text: str) -> None:
        """Append a line (safe to call from worker threads)."""
        self._message.emit(text)

    def _do_append(self, text: str) -> None:
        self._view.appendPlainText(text)

    def attach_logger(self, logger: logging.Logger) -> None:
        handler = _QtLogHandler(self.append)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                                "%H:%M:%S"))
        logger.addHandler(handler)
