"""Run inference on graphs and aggregate subgraph -> image predictions.

The classifier operates on *seed-centred subgraphs*.  An image is represented by
many such subgraphs (one per seed point).  To obtain an image-level prediction
we:

1. run the model on every subgraph (batched for speed),
2. soft-max each subgraph's 5 raw scores into probabilities,
3. average the probabilities across subgraphs (mean-pool over space),
4. take the arg-max as the image-level predicted class.

We also expose the *per-subgraph* probabilities (needed by the spatial heatmap)
and several confidence summaries (top probability, margin, entropy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from .. import CLASS_LABELS


@dataclass
class SubgraphPrediction:
    """Per-subgraph result used for spatial heatmaps and QC."""

    index: int
    seed_id: Optional[str]
    pos: np.ndarray              # (2,) spatial anchor of the seed node
    probabilities: np.ndarray    # (5,)
    predicted_index: int
    confidence: float            # top probability of this subgraph


@dataclass
class ImagePrediction:
    """Aggregated image-level prediction plus per-subgraph detail."""

    labels: List[str]
    mean_probabilities: np.ndarray         # (5,)
    predicted_index: int
    predicted_label: str
    predicted_probability: float
    top_probability: float
    second_probability: float
    margin: float
    entropy: float
    n_subgraphs: int
    subgraph_predictions: List[SubgraphPrediction] = field(default_factory=list)
    true_target: Optional[np.ndarray] = None   # mean soft target y if available
    hop: Optional[int] = None

    # --- convenience ----------------------------------------------------- #
    @property
    def true_label(self) -> Optional[str]:
        if self.true_target is None:
            return None
        return self.labels[int(np.argmax(self.true_target))]

    def probabilities_dict(self) -> dict:
        return {lab: float(p) for lab, p in zip(self.labels, self.mean_probabilities)}

    def per_subgraph_confidence(self) -> np.ndarray:
        return np.asarray([s.confidence for s in self.subgraph_predictions], dtype=float)

    def per_subgraph_class_prob(self, class_index: int) -> np.ndarray:
        """Probability of ``class_index`` for every subgraph (for heatmaps)."""
        return np.asarray(
            [s.probabilities[class_index] for s in self.subgraph_predictions],
            dtype=float,
        )

    def per_subgraph_positions(self) -> np.ndarray:
        if not self.subgraph_predictions:
            return np.empty((0, 2))
        return np.stack([s.pos for s in self.subgraph_predictions], axis=0)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


@torch.no_grad()
def run_subgraph_scores(
    model: torch.nn.Module,
    subgraphs: List,
    device: str = "cpu",
    batch_size: int = 16,
) -> np.ndarray:
    """Return raw ``[n_subgraphs, 5]`` class scores for a list of ``Data``.

    Uses ``torch_geometric.data.Batch`` to batch subgraphs.  Each subgraph keeps
    its own ``batch`` index inside the combined batch so ``global_*_pool`` yields
    one pooled vector per subgraph.
    """
    model.eval()
    scores: List[np.ndarray] = []
    for start in range(0, len(subgraphs), batch_size):
        chunk = subgraphs[start : start + batch_size]
        batch = Batch.from_data_list(chunk).to(device)
        out = model(batch)
        scores.append(out["pred"].detach().cpu().numpy())
    if not scores:
        return np.empty((0, len(CLASS_LABELS)))
    return np.vstack(scores)


def predict_image(
    model: torch.nn.Module,
    subgraphs: List,
    device: str = "cpu",
    labels: Optional[List[str]] = None,
    batch_size: int = 16,
    true_target: Optional[np.ndarray] = None,
    hop: Optional[int] = None,
) -> ImagePrediction:
    """Run + aggregate inference over all subgraphs of one image.

    Parameters
    ----------
    model:
        A loaded :class:`RamanGATv2MultiBias`.
    subgraphs:
        List of ``torch_geometric.data.Data`` seed-centred subgraphs.
    true_target:
        Optional mean soft-target ``y`` (5,) to record alongside the prediction
        (never used to influence the prediction itself).

    Returns
    -------
    ImagePrediction
    """
    labels = labels or CLASS_LABELS
    if len(subgraphs) == 0:
        raise ValueError("Cannot predict on an empty subgraph list.")

    raw = run_subgraph_scores(model, subgraphs, device=device, batch_size=batch_size)
    probs = _softmax_np(raw)                       # (n, 5)
    mean_probs = probs.mean(axis=0)                # (5,)

    pred_idx = int(np.argmax(mean_probs))
    sorted_p = np.sort(mean_probs)[::-1]
    top_p = float(sorted_p[0])
    second_p = float(sorted_p[1]) if len(sorted_p) > 1 else 0.0
    entropy = float(-(mean_probs * np.log(mean_probs + 1e-12)).sum())

    sub_preds: List[SubgraphPrediction] = []
    for i, (g, p) in enumerate(zip(subgraphs, probs)):
        pos = getattr(g, "pos", None)
        if pos is not None:
            pos_np = pos.detach().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
            anchor = pos_np[0]
        else:
            anchor = np.array([np.nan, np.nan])
        sub_preds.append(
            SubgraphPrediction(
                index=i,
                seed_id=str(getattr(g, "seed_id", i)),
                pos=np.asarray(anchor, dtype=float),
                probabilities=p.astype(float),
                predicted_index=int(np.argmax(p)),
                confidence=float(p.max()),
            )
        )

    return ImagePrediction(
        labels=list(labels),
        mean_probabilities=mean_probs.astype(float),
        predicted_index=pred_idx,
        predicted_label=labels[pred_idx],
        predicted_probability=float(mean_probs[pred_idx]),
        top_probability=top_p,
        second_probability=second_p,
        margin=top_p - second_p,
        entropy=entropy,
        n_subgraphs=len(subgraphs),
        subgraph_predictions=sub_preds,
        true_target=None if true_target is None else np.asarray(true_target, dtype=float),
        hop=hop,
    )
