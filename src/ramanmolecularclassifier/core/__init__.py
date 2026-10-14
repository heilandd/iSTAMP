"""Core numerical / IO layer for RamanMolecularClassifier.

Everything in this sub-package is UI-free and importable head-less.  Matplotlib
is forced onto the non-interactive ``Agg`` backend on import so that heatmap and
report rendering works without a display (e.g. under ``QT_QPA_PLATFORM=offscreen``
in CI).
"""

from __future__ import annotations

import matplotlib

# Force a non-interactive backend for every head-less code path.  The Qt UI
# embeds figures via FigureCanvasQTAgg, which works fine on top of Agg.
matplotlib.use("Agg", force=True)
