"""
Visualisation helpers for the iSTAMP app / report.

- preprocessing_panel : the pipeline "how the data were produced" (original SRH ->
  retained tissue patches -> spatial graph).
- xai_overlay         : the rich explainability figure — semi-transparent patch tiles
  coloured by integrated gradients + graph edges coloured by attention + nodes,
  with dual colourbars (matches the requested reference style).

All drawing works on the downscaled SRH image when available, else a white canvas.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# selectable colour schemes (user can switch in the app sidebar)
IG_CMAPS = {"magma": "magma", "inferno": "inferno", "viridis": "viridis",
            "plasma": "plasma", "cividis": "cividis"}
ATTN_CMAPS = {"viridis": "viridis", "turbo": "turbo", "cool": "cool", "YlGnBu": "YlGnBu"}


def _tile_size(pos):
    """Estimate a patch tile edge length from the median nearest-neighbour spacing."""
    if pos is None or len(pos) < 2:
        return 40.0
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = np.median(d.min(1))
    return float(nn * 0.92)


def _prep_axes(ax, bg_img, pos, scale, zoom=False):
    if bg_img is not None:
        ax.imshow(bg_img)
        if zoom and pos is not None:
            P = pos * scale
            pad = _tile_size(pos) * scale * 1.1
            ax.set_xlim(P[:, 0].min() - pad, P[:, 0].max() + pad)
            ax.set_ylim(P[:, 1].max() + pad, P[:, 1].min() - pad)  # image y is inverted
        return scale
    # white canvas sized to node bounds
    if pos is not None:
        pad = _tile_size(pos)
        ax.set_xlim(pos[:, 0].min() - pad, pos[:, 0].max() + pad)
        ax.set_ylim(pos[:, 1].max() + pad, pos[:, 1].min() - pad)  # invert y like an image
    ax.set_facecolor("white")
    return 1.0


def preprocessing_panel(bg_img, pos, edge_index, scale=1.0, figsize=(12, 4.2)):
    """3-panel pipeline figure: original -> tissue patches -> spatial graph."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    ts = _tile_size(pos) * (scale if bg_img is not None else 1.0)
    P = pos * (scale if bg_img is not None else 1.0) if pos is not None else None

    # 1) original — full section, with the analysed subgraph region marked
    s = _prep_axes(axes[0], bg_img, pos, scale, zoom=False)
    if P is not None:
        x0, y0 = P[:, 0].min() - ts, P[:, 1].min() - ts
        w, h = P[:, 0].ptp() + 2 * ts, P[:, 1].ptp() + 2 * ts
        axes[0].add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="#E32219", lw=1.5))
    axes[0].set_title("1 · SRH image (analysed region boxed)", fontsize=10); axes[0].axis("off")

    # 2) retained tissue patches (graph nodes = tiles kept after tissue gating) — zoomed
    s = _prep_axes(axes[1], bg_img, pos, scale, zoom=True)
    if P is not None:
        for (x, y) in P:
            axes[1].add_patch(Rectangle((x - ts / 2, y - ts / 2), ts, ts,
                                        facecolor="#0072B2", alpha=0.35, edgecolor="white", lw=0.4))
    axes[1].set_title("2 · Tissue patches (nodes)", fontsize=10); axes[1].axis("off")

    # 3) spatial graph (nodes + edges) — zoomed
    s = _prep_axes(axes[2], bg_img, pos, scale, zoom=True)
    if P is not None and edge_index is not None:
        segs = []
        ei = np.asarray(edge_index)
        for a, b in zip(ei[0], ei[1]):
            if a == b:
                continue
            segs.append([P[a], P[b]])
        axes[2].add_collection(LineCollection(segs, colors="#222222", linewidths=0.8, alpha=0.7))
        axes[2].scatter(P[:, 0], P[:, 1], s=22, c="#D55E00", edgecolors="white", linewidths=0.4, zorder=3)
    axes[2].set_title("3 · Spatial ego-graph", fontsize=10); axes[2].axis("off")

    fig.suptitle("Preprocessing pipeline — how the prediction is produced", fontsize=11, y=1.02)
    fig.tight_layout()
    return fig


def xai_overlay(bg_img, pos, node_ig, edge_index, edge_attn, scale=1.0,
                ig_cmap="magma", attn_cmap="viridis", target_name="", zoom=True, figsize=(9, 9.6)):
    """
    Tiles = integrated gradients (per patch), edges = attention, nodes = markers.
    node_ig  : [N] signed IG importance (magnitude used for tile colour)
    edge_index: [2,E]; edge_attn: [E] attention weights aligned to edge_index columns.
    zoom=True crops the view to the (localized) subgraph so tiles fill the frame.
    """
    fig, ax = plt.subplots(figsize=figsize)
    s = _prep_axes(ax, bg_img, pos, scale, zoom=zoom)
    sc = (scale if bg_img is not None else 1.0)
    P = pos * sc
    ts = _tile_size(pos) * sc

    # --- tiles coloured by |integrated gradients| ---
    ig_mag = np.abs(np.asarray(node_ig, float))
    ig_norm = mcolors.Normalize(vmin=0.0, vmax=(ig_mag.max() or 1.0))
    ig_sm = cm.ScalarMappable(norm=ig_norm, cmap=ig_cmap)
    for (x, y), v in zip(P, ig_mag):
        ax.add_patch(Rectangle((x - ts / 2, y - ts / 2), ts, ts,
                               facecolor=ig_sm.to_rgba(v), alpha=0.55,
                               edgecolor="white", lw=0.3, zorder=2))

    # --- edges coloured by attention ---
    if edge_index is not None and edge_attn is not None and len(edge_attn):
        ei = np.asarray(edge_index)
        attn = np.asarray(edge_attn, float)
        segs, cols = [], []
        a_lo, a_hi = np.percentile(attn, 5), np.percentile(attn, 95)
        a_hi = a_hi if a_hi > a_lo else a_lo + 1e-6
        attn_norm = mcolors.Normalize(vmin=a_lo, vmax=a_hi)
        attn_sm = cm.ScalarMappable(norm=attn_norm, cmap=attn_cmap)
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if a == b or a >= len(P) or b >= len(P):
                continue
            segs.append([P[a], P[b]]); cols.append(attn_sm.to_rgba(attn[k]))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.6, zorder=3))
    else:
        attn_sm = None

    # --- nodes ---
    ax.scatter(P[:, 0], P[:, 1], s=32, c="#E32219", edgecolors="black",
               linewidths=0.4, zorder=4)

    ax.set_title(f"XAI overlay — tiles = integrated gradients, edges = attention"
                 + (f"  ({target_name})" if target_name else ""), fontsize=11)
    ax.axis("off")

    # colourbars: IG (right, vertical) + attention (bottom, horizontal)
    cb1 = fig.colorbar(ig_sm, ax=ax, fraction=0.046, pad=0.02)
    cb1.set_label("Integrated gradients", fontsize=9)
    if attn_sm is not None:
        cax = fig.add_axes([0.15, 0.03, 0.7, 0.02])
        cb2 = fig.colorbar(attn_sm, cax=cax, orientation="horizontal")
        cb2.set_label("Attention", fontsize=9)
        fig.subplots_adjust(bottom=0.10)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.subplots_adjust(left=0.02, right=0.9, top=0.95, bottom=0.08)
    return fig


def fig_to_png(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
