"""Spatial probability / confidence heatmaps over the SRH image.

Each subgraph is anchored at a seed-node ``(x, y)`` position and carries a
5-class probability vector.  We rasterise a chosen scalar (probability of a
class, or per-subgraph confidence) onto a regular grid by interpolating between
the scattered seed positions, then overlay it - semi-transparently - on the
(optionally provided) SRH image.

Interpolation strategy
----------------------
* If SciPy is available we use :func:`scipy.interpolate.griddata` (linear with a
  nearest-neighbour fill for the convex-hull exterior).  With >=3 well-spread
  points and ``rbf=True`` we instead use a thin-plate-spline RBF for smoother
  fields.
* Otherwise we fall back to a Gaussian-weighted scatter (always available).

Low-confidence / non-diagnostic regions (per-subgraph confidence below a
threshold) are flagged: their seed markers are drawn with a distinct ring and
optionally hatched, so reviewers can see where the model is unsure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .inference import ImagePrediction


@dataclass
class HeatmapResult:
    figure: Figure
    grid: np.ndarray
    extent: Tuple[float, float, float, float]
    scalar_name: str
    n_points: int
    n_low_confidence: int


def _interpolate_grid(
    xy: np.ndarray,
    values: np.ndarray,
    extent: Tuple[float, float, float, float],
    grid_res: int = 200,
    rbf: bool = False,
) -> np.ndarray:
    xmin, xmax, ymin, ymax = extent
    gx = np.linspace(xmin, xmax, grid_res)
    gy = np.linspace(ymin, ymax, grid_res)
    GX, GY = np.meshgrid(gx, gy)

    if len(values) < 3:
        # Not enough points to interpolate a surface - nearest fill.
        grid = np.full(GX.shape, float(np.mean(values)) if len(values) else np.nan)
        return grid

    try:
        if rbf:
            from scipy.interpolate import RBFInterpolator

            interp = RBFInterpolator(
                xy, values, kernel="thin_plate_spline", smoothing=1e-3
            )
            grid = interp(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
            return grid
        from scipy.interpolate import griddata

        lin = griddata(xy, values, (GX, GY), method="linear")
        near = griddata(xy, values, (GX, GY), method="nearest")
        grid = np.where(np.isnan(lin), near, lin)
        return grid
    except Exception:  # noqa: BLE001 - SciPy missing or singular config
        # Gaussian-weighted scatter fallback (inverse-distance-ish).
        sigma = 0.06 * max(xmax - xmin, ymax - ymin)
        grid = np.zeros(GX.shape)
        wsum = np.zeros(GX.shape)
        for (px, py), v in zip(xy, values):
            d2 = (GX - px) ** 2 + (GY - py) ** 2
            w = np.exp(-d2 / (2 * sigma**2))
            grid += w * v
            wsum += w
        return grid / np.clip(wsum, 1e-9, None)


def build_heatmap(
    prediction: ImagePrediction,
    scalar: str = "predicted_class_prob",
    class_index: Optional[int] = None,
    background_image: Optional[np.ndarray] = None,
    image_downsample: int = 1,
    low_conf_threshold: float = 0.40,
    grid_res: int = 220,
    rbf: bool = False,
    cmap: str = "magma",
    alpha: float = 0.55,
    figsize: Tuple[float, float] = (8.5, 7.5),
    flag_low_confidence: bool = True,
) -> HeatmapResult:
    """Render a spatial heatmap figure.

    Parameters
    ----------
    prediction:
        Image-level prediction (provides per-subgraph positions / probabilities).
    scalar:
        ``"predicted_class_prob"`` (probability of the image's predicted class),
        ``"confidence"`` (per-subgraph top probability), or ``"class_prob"``
        (probability of ``class_index``).
    background_image:
        Optional SRH image array to overlay on (same coordinate system as the
        graph ``pos``).  If supplied at a down-sampled resolution, pass
        ``image_downsample`` so positions are scaled to match.
    rbf:
        Use thin-plate-spline RBF interpolation instead of linear griddata.
    flag_low_confidence:
        Mark subgraphs below ``low_conf_threshold`` with a distinct ring.
    """
    pos = prediction.per_subgraph_positions()
    if pos.shape[0] == 0:
        raise ValueError("Prediction has no subgraph positions for a heatmap.")

    if scalar == "confidence":
        values = prediction.per_subgraph_confidence()
        scalar_name = "Per-region confidence"
    elif scalar == "class_prob":
        ci = class_index if class_index is not None else prediction.predicted_index
        values = prediction.per_subgraph_class_prob(ci)
        scalar_name = f"P({prediction.labels[ci]})"
    else:  # predicted_class_prob
        ci = prediction.predicted_index
        values = prediction.per_subgraph_class_prob(ci)
        scalar_name = f"P({prediction.labels[ci]}) [predicted]"

    # Scale graph positions to the (possibly down-sampled) background image.
    plot_pos = pos / float(image_downsample)
    xs, ys = plot_pos[:, 0], plot_pos[:, 1]

    pad = 0.04 * max(xs.ptp() or 1.0, ys.ptp() or 1.0)
    extent = (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)

    grid = _interpolate_grid(plot_pos, values, extent, grid_res=grid_res, rbf=rbf)

    fig = Figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.08, 0.74, 0.86])
    cax = fig.add_axes([0.83, 0.18, 0.03, 0.6])

    if background_image is not None:
        ax.imshow(background_image, zorder=0)

    norm = Normalize(vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    im = ax.imshow(
        grid,
        extent=extent,
        origin="upper",
        cmap=cmap,
        alpha=alpha,
        norm=norm,
        zorder=1,
        aspect="auto" if background_image is None else "equal",
    )

    low_mask = values < low_conf_threshold
    n_low = int(low_mask.sum())

    # Confident seed markers.
    ax.scatter(
        xs[~low_mask], ys[~low_mask],
        c="white", s=14, edgecolors="black", linewidths=0.4, zorder=3,
        label="diagnostic region",
    )
    if flag_low_confidence and n_low:
        ax.scatter(
            xs[low_mask], ys[low_mask],
            facecolors="none", edgecolors="red", s=46, linewidths=1.3, zorder=4,
            label=f"low-confidence (<{low_conf_threshold:.2f})",
        )
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)

    ax.set_title(
        f"Spatial heatmap - {scalar_name}\n"
        f"image-level: {prediction.predicted_label} "
        f"(p={prediction.predicted_probability:.2f})",
        fontsize=10,
    )
    ax.axis("off")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(scalar_name, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    return HeatmapResult(
        figure=fig,
        grid=grid,
        extent=extent,
        scalar_name=scalar_name,
        n_points=len(values),
        n_low_confidence=n_low,
    )


def save_heatmap(result: HeatmapResult, out_path: os.PathLike | str, dpi: int = 200) -> Path:  # type: ignore[name-defined]
    """Save a heatmap figure to PNG or PDF (by extension)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.figure.savefig(out, dpi=dpi, bbox_inches="tight")
    return out


import os  # noqa: E402
