"""Quality-control metrics for a graph bundle + its prediction.

QC metrics flag whether an image-level prediction is trustworthy: how much
spatial evidence supports it (number of nodes / subgraphs), how confident the
per-subgraph votes are, and how many regions fall below a confidence threshold
(and are therefore non-diagnostic).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from .graph_io import GraphBundle
from .inference import ImagePrediction

#: Below this top-probability a subgraph is considered low-confidence / non-diagnostic.
DEFAULT_LOW_CONF_THRESHOLD = 0.40


@dataclass
class QCReport:
    """Container of QC metrics (serialisable via :meth:`as_dict`)."""

    hop: Optional[int]
    n_subgraphs: int
    n_nodes_total: int
    mean_nodes_per_subgraph: float
    mean_edges_per_subgraph: float
    graph_density: float                 # mean edges / possible directed edges
    mean_confidence: float
    median_confidence: float
    p10_confidence: float
    p90_confidence: float
    low_confidence_threshold: float
    fraction_low_confidence: float
    n_low_confidence: int
    prediction_entropy: float
    prediction_margin: float
    diagnostic: bool                     # overall pass/fail flag
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def compute_qc(
    bundle: GraphBundle,
    prediction: ImagePrediction,
    hop: Optional[int] = None,
    low_conf_threshold: float = DEFAULT_LOW_CONF_THRESHOLD,
    min_subgraphs: int = 5,
    max_low_fraction: float = 0.75,
) -> QCReport:
    """Compute QC metrics for one image.

    Parameters
    ----------
    bundle:
        The loaded :class:`GraphBundle` (provides graph topology).
    prediction:
        The :class:`ImagePrediction` returned by :func:`predict_image`.
    low_conf_threshold:
        Per-subgraph top-probability below which a region is non-diagnostic.
    min_subgraphs / max_low_fraction:
        Thresholds for the overall ``diagnostic`` pass/fail flag.
    """
    hop = hop if hop is not None else prediction.hop
    subs = bundle.subgraphs(hop)

    node_counts = []
    edge_counts = []
    densities = []
    for g in subs:
        n = int(g.x.shape[0])
        e = int(g.edge_index.shape[1]) if getattr(g, "edge_index", None) is not None else 0
        node_counts.append(n)
        edge_counts.append(e)
        possible = n * (n - 1)
        densities.append(e / possible if possible > 0 else 0.0)

    node_counts = np.asarray(node_counts, dtype=float)
    edge_counts = np.asarray(edge_counts, dtype=float)

    conf = prediction.per_subgraph_confidence()
    n_low = int((conf < low_conf_threshold).sum())
    frac_low = float(n_low / len(conf)) if len(conf) else 1.0

    diagnostic = (
        len(subs) >= min_subgraphs
        and frac_low <= max_low_fraction
        and prediction.top_probability >= low_conf_threshold
    )

    notes = []
    if len(subs) < min_subgraphs:
        notes.append(f"Few subgraphs ({len(subs)} < {min_subgraphs}).")
    if frac_low > max_low_fraction:
        notes.append(
            f"High fraction of low-confidence regions ({frac_low:.0%})."
        )
    if prediction.top_probability < low_conf_threshold:
        notes.append("Image-level top probability below diagnostic threshold.")
    if not notes:
        notes.append("All QC checks passed.")

    return QCReport(
        hop=hop,
        n_subgraphs=len(subs),
        n_nodes_total=int(node_counts.sum()) if len(node_counts) else 0,
        mean_nodes_per_subgraph=float(node_counts.mean()) if len(node_counts) else 0.0,
        mean_edges_per_subgraph=float(edge_counts.mean()) if len(edge_counts) else 0.0,
        graph_density=float(np.mean(densities)) if densities else 0.0,
        mean_confidence=float(conf.mean()) if len(conf) else 0.0,
        median_confidence=float(np.median(conf)) if len(conf) else 0.0,
        p10_confidence=float(np.percentile(conf, 10)) if len(conf) else 0.0,
        p90_confidence=float(np.percentile(conf, 90)) if len(conf) else 0.0,
        low_confidence_threshold=low_conf_threshold,
        fraction_low_confidence=frac_low,
        n_low_confidence=n_low,
        prediction_entropy=prediction.entropy,
        prediction_margin=prediction.margin,
        diagnostic=bool(diagnostic),
        notes=" ".join(notes),
    )
