"""Loading and inspection of pre-computed per-image PyG graph files.

The application's *primary working input format* is the precomputed graph dict
produced during the revision::

    {
      'pyg_dataset_by_hop': {hop:int -> list[Data]},
      'patient_id', 'sample_id', 'image_id', 'image_path',
      'graph_method', 'available_hops', 'prune_threshold'
    }

Each ``Data`` carries ``x[N,512]``, ``edge_index``, ``edge_attr[E,1]``,
``pos[N,2]`` and (for the revision data) a soft target ``y[5]``.  This module
provides a thin, well-typed wrapper so the rest of the app never has to know the
on-disk layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class GraphBundle:
    """A loaded per-image graph file with convenient accessors."""

    path: Path
    patient_id: Optional[str]
    sample_id: Optional[str]
    image_id: Optional[str]
    image_path: Optional[str]
    graph_method: Optional[str]
    available_hops: List[int]
    prune_threshold: Optional[float]
    datasets_by_hop: Dict[int, List]

    # ----- hop / subgraph access ----------------------------------------- #
    def hops(self) -> List[int]:
        """Sorted list of available hop levels."""
        return sorted(self.datasets_by_hop.keys())

    def subgraphs(self, hop: Optional[int] = None) -> List:
        """Return the list of subgraph ``Data`` objects for a given hop.

        If ``hop`` is ``None`` the largest available hop is used (the most
        context-rich neighbourhood).
        """
        if hop is None:
            hop = self.default_hop()
        if hop not in self.datasets_by_hop:
            raise KeyError(
                f"Hop {hop} not available; choose from {self.hops()}."
            )
        return self.datasets_by_hop[hop]

    def default_hop(self) -> int:
        """The hop used when the caller does not specify one (largest hop)."""
        return max(self.hops())

    def n_subgraphs(self, hop: Optional[int] = None) -> int:
        return len(self.subgraphs(hop))

    # ----- spatial positions --------------------------------------------- #
    def seed_positions(self, hop: Optional[int] = None) -> np.ndarray:
        """Return one representative ``(x, y)`` per subgraph.

        Every subgraph is a seed-centred neighbourhood.  The seed node is, by
        construction, node index 0 of the subgraph (the first ``node_id`` equals
        the ``seed_id``).  We use that node's ``pos`` as the subgraph's spatial
        anchor for heatmap rendering.
        """
        out = []
        for g in self.subgraphs(hop):
            pos = _to_numpy(g.pos)
            out.append(pos[0])  # seed node anchors the neighbourhood
        return np.asarray(out, dtype=float)

    def all_node_positions(self, hop: Optional[int] = None) -> np.ndarray:
        """Concatenated node positions across all subgraphs (may repeat)."""
        chunks = [_to_numpy(g.pos) for g in self.subgraphs(hop)]
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 2))

    def true_target(self, hop: Optional[int] = None) -> Optional[np.ndarray]:
        """Mean soft target ``y`` across subgraphs, if present (else None)."""
        ys = []
        for g in self.subgraphs(hop):
            y = getattr(g, "y", None)
            if y is not None:
                ys.append(_to_numpy(y).reshape(-1))
        if not ys:
            return None
        return np.mean(np.stack(ys, axis=0), axis=0)

    def metadata(self) -> Dict[str, object]:
        """Flat metadata dict suitable for display / reports."""
        return {
            "patient_id": self.patient_id,
            "sample_id": self.sample_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "graph_method": self.graph_method,
            "available_hops": self.available_hops,
            "prune_threshold": self.prune_threshold,
            "source_file": str(self.path),
        }


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def load_graph_bundle(path: os.PathLike | str) -> GraphBundle:  # type: ignore[name-defined]
    """Load a ``Img_*_pyg.pt`` / ``Img_*_pyg_largNN.pt`` file.

    Parameters
    ----------
    path:
        Path to the ``.pt`` graph dictionary.

    Returns
    -------
    GraphBundle
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graph file not found: {p}")

    obj = torch.load(p, weights_only=False, map_location="cpu")
    if not isinstance(obj, dict) or "pyg_dataset_by_hop" not in obj:
        raise ValueError(
            f"'{p}' is not a recognised graph bundle "
            "(missing 'pyg_dataset_by_hop')."
        )

    datasets = {int(k): v for k, v in obj["pyg_dataset_by_hop"].items()}
    available = obj.get("available_hops") or sorted(datasets.keys())

    return GraphBundle(
        path=p,
        patient_id=obj.get("patient_id"),
        sample_id=obj.get("sample_id"),
        image_id=obj.get("image_id"),
        image_path=obj.get("image_path"),
        graph_method=obj.get("graph_method"),
        available_hops=list(available),
        prune_threshold=obj.get("prune_threshold"),
        datasets_by_hop=datasets,
    )


# Late import to keep the public signature clean while still type-friendly.
import os  # noqa: E402
