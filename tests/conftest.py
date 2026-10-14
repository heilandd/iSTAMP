"""Shared pytest fixtures and head-less environment setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Force off-screen Qt and Agg matplotlib for all tests.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib

matplotlib.use("Agg", force=True)

# Make the src/ package importable without installation.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REVISION_DATA = Path("/Users/henrikheiland/Desktop/Project_Raman/Revision/data")
CHECKPOINT = Path("/Users/henrikheiland/Desktop/Project_Raman/Revision/best_model_full.pt")
JPEG_DIR = Path("/Users/henrikheiland/Desktop/Project_Raman/JPEG_I")


def _first_graph_file() -> Path | None:
    if not REVISION_DATA.exists():
        return None
    for p in sorted(REVISION_DATA.glob("Img_*_pyg.pt")):
        if "largNN" not in p.name:
            return p
    return None


@pytest.fixture(scope="session")
def graph_file() -> Path:
    p = _first_graph_file()
    if p is None:
        pytest.skip("No revision graph files available.")
    return p


@pytest.fixture(scope="session")
def checkpoint() -> Path:
    if not CHECKPOINT.exists():
        pytest.skip("Model checkpoint not available.")
    return CHECKPOINT


@pytest.fixture(scope="session")
def jpeg_file() -> Path:
    if not JPEG_DIR.exists():
        pytest.skip("JPEG directory not available.")
    files = sorted(JPEG_DIR.glob("*.jpeg"))
    if not files:
        pytest.skip("No JPEG files available.")
    return files[0]
