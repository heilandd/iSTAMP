"""Application entry point.

Run with either::

    python -m ramanmolecularclassifier
    ramanmolecularclassifier            # if installed as a console script

A small CLI is also provided for head-less batch analysis (no display needed),
which is what CI / the example-output generator use::

    python -m ramanmolecularclassifier --analyse Img_1000_pyg.pt --out ./out
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _run_cli(graph_path: str, out_dir: str, checkpoint: str | None) -> int:
    """Head-less analysis path (no Qt window)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S")
    from .core.pipeline import analyse_graph_file

    result = analyse_graph_file(
        graph_path, checkpoint_path=checkpoint, out_dir=out_dir,
        log=lambda m: logging.getLogger("rmc").info(m),
    )
    print(f"Predicted: {result.prediction.predicted_label} "
          f"(p={result.prediction.predicted_probability:.3f})")
    if result.reports:
        for name, p in vars(result.reports).items():
            if p:
                print(f"  {name}: {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ramanmolecularclassifier",
        description="RamanMolecularClassifier - iSTAMP Desktop "
                    "(GBM Raman/SRH molecular subtype classifier).",
    )
    parser.add_argument("--analyse", "--analyze", dest="analyse", metavar="GRAPH.pt",
                        help="Head-less: analyse a precomputed graph file and exit.")
    parser.add_argument("--out", default="./rmc_output",
                        help="Output directory for head-less analysis.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to best_model_full.pt (overrides default).")
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__
        print(__version__)
        return 0

    if args.analyse:
        return _run_cli(args.analyse, args.out, args.checkpoint)

    # GUI path.
    from PySide6.QtWidgets import QApplication

    from . import APP_NAME
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
