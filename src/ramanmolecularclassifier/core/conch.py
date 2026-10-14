"""CONCH foundation-model encoder interface (STUB).

The trained classifier consumes **512-d CONCH node features** for every SRH
tile.  Turning a *raw* new SRH image into a graph therefore requires embedding
each tile through the CONCH foundation model.

The CONCH model weights are **not available** in this environment (they are
gated/licensed and not redistributable in the revision package).  Rather than
fabricate embeddings, this module exposes a clear, well-documented stub.  The
application detects the missing encoder and gracefully restricts raw-image
handling to *viewer + metadata + QC*, while full inference remains available on
the pre-computed ``Img_*_pyg.pt`` graphs (which already contain CONCH features).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class CONCHUnavailableError(NotImplementedError):
    """Raised when raw-image embedding is attempted without CONCH weights."""


class CONCHEncoder:
    """Interface for embedding SRH tiles into 512-d CONCH node features.

    This is a deliberately non-functional stub.  Calling :meth:`encode_tiles`
    (or :meth:`is_available` returning ``True``) requires the CONCH weights,
    which are absent here.  The signature documents what a future, weights-
    enabled implementation must provide so that raw-image inference can be wired
    in without changing the rest of the app.
    """

    EMBED_DIM = 512

    def __init__(self, weights_path: Optional[str | Path] = None):
        self.weights_path = Path(weights_path) if weights_path else None

    def is_available(self) -> bool:
        """Whether usable CONCH weights are present. Always ``False`` here."""
        return bool(self.weights_path and self.weights_path.exists())

    def encode_tiles(self, tiles: np.ndarray) -> np.ndarray:  # noqa: D401
        """Embed ``tiles`` ``[n, H, W, 3]`` into ``[n, 512]`` features.

        Raises
        ------
        CONCHUnavailableError
            Always, in this build - the weights are not bundled.
        """
        raise CONCHUnavailableError(
            "Raw-image inference requires the CONCH foundation model to embed "
            "SRH tiles into 512-d node features, but the CONCH weights are not "
            "available in this environment.\n\n"
            "What works instead:\n"
            "  * Full classification + heatmaps + reports on the pre-computed "
            "per-image graph files (Img_*_pyg.pt), which already contain CONCH "
            "features.\n"
            "  * Raw SRH images can still be viewed and their metadata / QC "
            "inspected.\n\n"
            "To enable raw-image inference, provide CONCH weights and implement "
            "CONCHEncoder.encode_tiles() to return L2-normalised 512-d vectors "
            "per tile, then build a Delaunay graph over tile centroids "
            "(matching the training pipeline)."
        )


def conch_status_message() -> str:
    """Human-readable one-liner about CONCH availability for the UI/log."""
    return (
        "CONCH foundation model: NOT available. Raw-image -> graph embedding is "
        "disabled; use pre-computed Img_*_pyg.pt graphs for inference."
    )
