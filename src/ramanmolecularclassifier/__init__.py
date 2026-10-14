"""RamanMolecularClassifier - iSTAMP Desktop.

A cross-platform (macOS / Windows / Ubuntu) desktop application that runs a
trained GATv2 graph neural network classifier for Raman / SRH glioblastoma
molecular subtyping, packaged for a Nature Communications revision.

The package is organised as:

* ``core``       - model loading, inference, graph/image IO, heatmaps, QC.
* ``ui``         - PySide6 widgets (image viewer, panels, log).
* ``reporting``  - PDF / HTML / Excel report engine.
* ``main_window``- the :class:`~ramanmolecularclassifier.main_window.MainWindow`.
* ``app``        - the entry point (``python -m ramanmolecularclassifier``).

All heavy numerical work (model loading, inference, heatmap rendering, report
generation) is decoupled from the Qt UI so it can be driven head-less, which is
how the test-suite and the example-output generator exercise it.
"""

from __future__ import annotations

__version__ = "1.0.0"

APP_NAME = "RamanMolecularClassifier"
APP_SUBTITLE = "iSTAMP Desktop"

#: Canonical molecular-subtype class order expected by the trained model.
CLASS_LABELS = [
    "GBM_MES_TYP",
    "GBM_MES_ATYP",
    "GBM_RTK1",
    "GBM_RTK2",
    "GBM_PNC",
]

__all__ = [
    "__version__",
    "APP_NAME",
    "APP_SUBTITLE",
    "CLASS_LABELS",
]
