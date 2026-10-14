"""Prediction panel: predicted class, confidence and a probability bar chart."""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..core.inference import ImagePrediction
from ..reporting.report_engine import CLASS_COLORS


class PredictionPanel(QWidget):
    """Shows the image-level prediction plus a 5-class probability bar chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._headline = QLabel("No prediction yet.")
        self._headline.setAlignment(Qt.AlignCenter)
        self._headline.setStyleSheet("font-size: 15px; font-weight: bold; color:#1f3b5c;")
        self._sub = QLabel("")
        self._sub.setAlignment(Qt.AlignCenter)

        self._fig = Figure(figsize=(5, 3))
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self._headline)
        layout.addWidget(self._sub)
        layout.addWidget(self._canvas)

    def set_prediction(self, pred: Optional[ImagePrediction]) -> None:
        self._fig.clear()
        if pred is None:
            self._headline.setText("No prediction yet.")
            self._sub.setText("")
            self._canvas.draw_idle()
            return

        self._headline.setText(
            f"{pred.predicted_label}   (p = {pred.predicted_probability:.3f})"
        )
        verdict = "high" if pred.top_probability >= 0.6 else (
            "moderate" if pred.top_probability >= 0.4 else "low")
        ref = f" | reference: {pred.true_label}" if pred.true_label else ""
        self._sub.setText(
            f"margin {pred.margin:.3f} | entropy {pred.entropy:.3f} | "
            f"{pred.n_subgraphs} subgraphs | confidence: {verdict}{ref}"
        )

        ax = self._fig.add_subplot(111)
        x = np.arange(len(pred.labels))
        colors = [CLASS_COLORS.get(l, "#888888") for l in pred.labels]
        bars = ax.bar(x, pred.mean_probabilities, color=colors, edgecolor="black", linewidth=0.5)
        bars[pred.predicted_index].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(pred.labels, rotation=20, ha="right", fontsize=7)
        ax.set_ylim(0, min(1.0, float(pred.mean_probabilities.max()) + 0.12))
        ax.set_ylabel("Probability", fontsize=8)
        self._fig.tight_layout()
        self._canvas.draw_idle()
