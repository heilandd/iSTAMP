"""The main application window (QMainWindow) with tabbed panels.

The window wires the head-less :mod:`ramanmolecularclassifier.core` pipeline to
the Qt UI.  Heavy work (model load + inference + reports) runs on a background
``QThread`` worker so the GUI stays responsive.  All UI logic here is optional
for head-less use - the test-suite and example generator drive the core layer
directly.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import APP_NAME, APP_SUBTITLE, __version__
from .core import conch
from .core.image_io import load_image
from .core.pipeline import AnalysisResult, analyse_graph_file
from .ui.heatmap_viewer import HeatmapViewer
from .ui.image_viewer import ImageViewer
from .ui.log_panel import LogPanel
from .ui.metadata_panel import MetadataPanel
from .ui.prediction_panel import PredictionPanel
from .ui.report_viewer import ReportViewer

logger = logging.getLogger("ramanmolecularclassifier")


class _AnalysisWorker(QObject):
    """Runs :func:`analyse_graph_file` off the GUI thread."""

    finished = Signal(object)     # AnalysisResult
    failed = Signal(str)
    message = Signal(str)

    def __init__(self, graph_path: str, out_dir: str):
        super().__init__()
        self._graph_path = graph_path
        self._out_dir = out_dir

    def run(self):
        try:
            result = analyse_graph_file(
                self._graph_path,
                out_dir=self._out_dir,
                log=self.message.emit,
            )
            self.finished.emit(result)
        except Exception:  # noqa: BLE001
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - {APP_SUBTITLE}  v{__version__}")
        self.resize(1280, 840)

        self._log = LogPanel()
        self._log.attach_logger(logger)
        logger.setLevel(logging.INFO)

        self._image_viewer = ImageViewer()
        self._metadata = MetadataPanel()
        self._prediction = PredictionPanel()
        self._heatmap = HeatmapViewer()
        self._report = ReportViewer()

        self._thread: Optional[QThread] = None
        self._worker: Optional[_AnalysisWorker] = None
        self._last_result: Optional[AnalysisResult] = None

        self._build_ui()
        logger.info("%s", conch.conch_status_message())

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        toolbar = QWidget()
        tl = QHBoxLayout(toolbar)
        btn_open_graph = QPushButton("Open graph (.pt) and classify")
        btn_open_image = QPushButton("Open raw SRH image (viewer + metadata)")
        btn_open_graph.clicked.connect(self._on_open_graph)
        btn_open_image.clicked.connect(self._on_open_image)
        tl.addWidget(btn_open_graph)
        tl.addWidget(btn_open_image)
        tl.addStretch(1)
        tl.addWidget(QLabel(conch.conch_status_message()))

        tabs = QTabWidget()
        tabs.addTab(self._image_viewer, "Image")
        tabs.addTab(self._metadata, "Metadata")
        tabs.addTab(self._prediction, "Prediction")
        tabs.addTab(self._heatmap, "Heatmap")
        tabs.addTab(self._report, "Report")
        self._tabs = tabs

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tabs)
        splitter.addWidget(self._log)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(toolbar)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

    # ------------------------------------------------------------------ #
    def _on_open_graph(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open precomputed graph", "", "PyG graphs (*.pt);;All files (*)")
        if not path:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            out_dir = str(Path(path).parent / "analysis")
        self._start_analysis(path, out_dir)

    def _on_open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SRH image", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.dcm);;All files (*)")
        if not path:
            return
        try:
            img = load_image(path, downsample=8)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", str(exc))
            return
        self._image_viewer.set_image(img.display_array(),
                                     f"{img.fmt} | {img.width}x{img.height}")
        self._metadata.set_metadata(img.metadata, img.proprietary_tags)
        for w in img.warnings:
            logger.warning(w)
        logger.info("Raw image loaded. %s", conch.conch_status_message())
        QMessageBox.information(
            self, "Raw image loaded",
            "Image and metadata loaded.\n\n"
            "Classification of raw images requires the CONCH foundation model "
            "(not available in this build) to embed tiles into 512-d features.\n\n"
            "Use a precomputed Img_*_pyg.pt graph for full classification.")
        self._tabs.setCurrentWidget(self._image_viewer)

    # ------------------------------------------------------------------ #
    def _start_analysis(self, graph_path: str, out_dir: str):
        self._thread = QThread()
        self._worker = _AnalysisWorker(graph_path, out_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._log.append)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.failed.connect(self._on_analysis_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()
        logger.info("Started analysis of %s", Path(graph_path).name)

    def _on_analysis_done(self, result: AnalysisResult):
        self._last_result = result
        self._metadata.set_metadata(result.bundle.metadata())
        self._prediction.set_prediction(result.prediction)
        if result.background_image is not None:
            self._image_viewer.set_image(
                result.background_image.display_array(),
                f"SRH image for {result.bundle.image_id}")
        # Rebuild a live heatmap figure for the viewer.
        try:
            from .core.heatmap import build_heatmap
            bg = (result.background_image.display_array()
                  if result.background_image else None)
            ds = result.background_image.downsample if result.background_image else 1
            hm = build_heatmap(result.prediction, scalar="predicted_class_prob",
                               background_image=bg, image_downsample=ds)
            self._heatmap.set_figure(hm.figure)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Heatmap render failed: %s", exc)
        self._report.set_reports(result.reports)
        self._tabs.setCurrentWidget(self._prediction)
        logger.info("Analysis complete: %s (p=%.3f)",
                    result.prediction.predicted_label,
                    result.prediction.predicted_probability)

    def _on_analysis_failed(self, tb: str):
        logger.error("Analysis failed:\n%s", tb)
        QMessageBox.critical(self, "Analysis failed", tb)
