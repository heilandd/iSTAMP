import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable



def get_image_dataarray(image_element):
    """
    Return a single-scale DataArray from either:
    - a single-scale SpatialData image (DataArray), or
    - a multiscale SpatialData image (DataTree -> scale0 -> image)
    """
    # multiscale image
    if hasattr(image_element, "keys") and "scale0" in image_element.keys():
        return image_element["scale0"]["image"]

    # single-scale image
    return image_element

def plotImageScatter(adata, s=20, color="red"):
  import matplotlib.pyplot as plt
  img0 = get_image_dataarray(sdata.images["raw_image"])
  img_np = img0.transpose("y", "x", "c").values
  pts = sdata.points["random_points"].compute()
  
  plt.figure(figsize=(8, 8))
  plt.imshow(img_np)
  plt.scatter(pts["x"], pts["y"], s=s, c=color)
  plt.axis("off")
  plt.show()


def get_image_scale0(image_element):
    """
    Return highest-resolution DataArray from a SpatialData image element.
    Works for both single-scale DataArray and multiscale DataTree.
    """
    if hasattr(image_element, "keys") and "scale0" in image_element.keys():
        return image_element["scale0"]["image"]
    return image_element



def plot_tiles_with_qc(
    sdata,
    qc_df: pd.DataFrame,
    image_name: str = "raw_image",
    points_name: str = "random_points",
    qc_column: str = "gray_entropy",
    tile_size: int = 448,
    point_size: int = 8,
    point_color: str = "red",
    alpha: float = 0.5,
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 10),
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
):
    """
    Plot:
    - background image
    - red point for each tile center
    - tile rectangles colored by a QC metric

    Parameters
    ----------
    sdata
        SpatialData object.
    qc_df
        DataFrame returned by compute_qc_for_all_tiles().
        Must contain columns: x, y, and qc_column.
    image_name
        Name of the image in sdata.images.
    points_name
        Name of the points element in sdata.points.
    qc_column
        QC feature column to color the tiles by.
    tile_size
        Tile width/height in pixels.
    point_size
        Size of red center dots.
    point_color
        Color of point markers.
    alpha
        Transparency of tile overlays.
    cmap
        Colormap for QC values.
    figsize
        Figure size.
    vmin, vmax
        Optional fixed color scaling.
    show_colorbar
        Whether to show the QC colorbar.
    """
    if qc_column not in qc_df.columns:
        raise ValueError(f"'{qc_column}' not found in qc_df columns.")

    # image
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

    # points
    pts = sdata.points[points_name]
    if hasattr(pts, "compute"):
        pts = pts.compute()

    # merge qc onto points if needed
    plot_df = qc_df.copy()

    # color normalization
    values = plot_df[qc_column].to_numpy(dtype=float)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)

    half = tile_size / 2.0

    # draw tiles
    for row in plot_df.itertuples(index=False):
        x = float(row.x)
        y = float(row.y)
        qc_value = float(getattr(row, qc_column))

        color = cmap_obj(norm(qc_value))

        rect = Rectangle(
            (x - half, y - half),
            tile_size,
            tile_size,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
            alpha=alpha,
        )
        ax.add_patch(rect)

    # draw points on top
    ax.scatter(
        pts["x"],
        pts["y"],
        c=point_color,
        s=point_size,
        linewidths=0,
    )

    ax.set_title(f"Tiles colored by: {qc_column}")
    ax.axis("off")

    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(qc_column)

    plt.tight_layout()
    plt.show()


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_graph_and_subgraphs_on_image(
    sdata,
    qc_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    subgraphs: list[dict],
    image_name: str = "raw_image",
    qc_column: str = "diagnostic_score",
    tile_size: int = 448,
    alpha_tiles: float = 0.35,
    cmap_tiles: str = "viridis",
    point_size: int = 10,
    point_color: str = "red",
    edge_color: str = "white",
    edge_alpha: float = 0.15,
    edge_linewidth: float = 0.6,
    subgraph_linewidth: float = 2.0,
    subgraph_node_size: int = 18,
    max_subgraphs_to_plot: int | None = 15,
    random_seed: int = 42,
    figsize: tuple[int, int] = (12, 12),
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Plot image + QC-colored tiles + all graph edges + colored subgraphs.

    Parameters
    ----------
    qc_df
        Must contain at least: point_id, x, y, and qc_column.
    edge_df
        Must contain: from_id, to_id
    subgraphs
        List of dicts, each with at least:
            - 'seed_id'
            - 'node_ids'
    """
    if qc_column not in qc_df.columns:
        raise ValueError(f"Column '{qc_column}' not found in qc_df.")

    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

    plot_df = qc_df.copy()
    plot_df["point_id"] = plot_df["point_id"].astype(str)

    coord_map = plot_df.set_index("point_id")[["x", "y"]].to_dict(orient="index")

    # tile colors
    values = plot_df[qc_column].to_numpy(dtype=float)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)

    norm = Normalize(vmin=vmin, vmax=vmax)
    tile_cmap = plt.get_cmap(cmap_tiles)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)

    # 1. draw tiles
    half = tile_size / 2.0
    for row in plot_df.itertuples(index=False):
        x = float(row.x)
        y = float(row.y)
        val = float(getattr(row, qc_column))

        rect = Rectangle(
            (x - half, y - half),
            tile_size,
            tile_size,
            facecolor=tile_cmap(norm(val)),
            edgecolor=None,
            linewidth=0,
            alpha=alpha_tiles,
        )
        ax.add_patch(rect)

    # 2. draw all graph edges faintly
    for row in edge_df.itertuples(index=False):
        from_id = str(row.from_id)
        to_id = str(row.to_id)

        if from_id not in coord_map or to_id not in coord_map:
            continue

        x1, y1 = coord_map[from_id]["x"], coord_map[from_id]["y"]
        x2, y2 = coord_map[to_id]["x"], coord_map[to_id]["y"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=edge_color,
            alpha=edge_alpha,
            linewidth=edge_linewidth,
            zorder=3,
        )

    # 3. choose subgraphs to plot
    rng = random.Random(random_seed)
    sg_to_plot = subgraphs.copy()

    if max_subgraphs_to_plot is not None and len(sg_to_plot) > max_subgraphs_to_plot:
        sg_to_plot = rng.sample(sg_to_plot, max_subgraphs_to_plot)

    # generate distinct colors
    subgraph_cmap = plt.get_cmap("tab20")
    sg_colors = [subgraph_cmap(i % 20) for i in range(len(sg_to_plot))]

    # precompute edge set lookup for fast membership checks
    edge_lookup = set()
    for row in edge_df.itertuples(index=False):
        a = str(row.from_id)
        b = str(row.to_id)
        edge_lookup.add(tuple(sorted((a, b))))

    # 4. overlay colored subgraphs
    for i, sg in enumerate(sg_to_plot):
        color = sg_colors[i]
        node_ids = [str(n) for n in sg["node_ids"] if str(n) in coord_map]

        if len(node_ids) == 0:
            continue

        # draw edges within the subgraph
        node_set = set(node_ids)
        for a in node_ids:
            for b in node_ids:
                if a >= b:
                    continue
                if tuple(sorted((a, b))) in edge_lookup:
                    x1, y1 = coord_map[a]["x"], coord_map[a]["y"]
                    x2, y2 = coord_map[b]["x"], coord_map[b]["y"]

                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color=color,
                        alpha=0.95,
                        linewidth=subgraph_linewidth,
                        zorder=4,
                    )

        # draw subgraph nodes
        xs = [coord_map[n]["x"] for n in node_ids]
        ys = [coord_map[n]["y"] for n in node_ids]
        ax.scatter(
            xs,
            ys,
            s=subgraph_node_size,
            color=color,
            edgecolors="black",
            linewidths=0.2,
            zorder=5,
        )

        # highlight seed
        seed_id = str(sg["seed_id"])
        if seed_id in coord_map:
            sx, sy = coord_map[seed_id]["x"], coord_map[seed_id]["y"]
            ax.scatter(
                [sx],
                [sy],
                s=subgraph_node_size * 3,
                color=color,
                edgecolors="white",
                linewidths=1.0,
                marker="o",
                zorder=6,
            )

    # 5. draw all points on top
    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        c=point_color,
        s=point_size,
        linewidths=0,
        zorder=7,
    )

    ax.set_title(f"Tiles colored by {qc_column} with graph + subgraphs")
    ax.axis("off")

    if show_colorbar:
        sm = ScalarMappable(norm=norm, cmap=tile_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(qc_column)

    plt.tight_layout()
    plt.show()

def plot_pyg_result_on_image(
    sdata,
    qc_df,
    result: dict,
    hop: int = 3,
    image_name: str = "raw_image",
    qc_column: str = "diagnostic_score",
    tile_size: int = 448,
    alpha_tiles: float = 0.15,
    cmap_tiles: str = "viridis",
    point_size: int = 10,
    point_color: str = "red",
    edge_color: str = "white",
    edge_alpha: float = 0.15,
    edge_linewidth: float = 0.6,
    subgraph_linewidth: float = 2.0,
    subgraph_node_size: int = 18,
    max_subgraphs_to_plot: int | None = 15,
    random_seed: int = 42,
    figsize: tuple[int, int] = (12, 12),
    show_colorbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Wrapper to plot graph/subgraphs from the PyG result dict for one hop size.
    """
    if hop not in result["subgraphs_by_hop"]:
        raise ValueError(f"Hop={hop} not found in result['subgraphs_by_hop']. Available: {list(result['subgraphs_by_hop'].keys())}")

    subgraphs = result["subgraphs_by_hop"][hop]
    edge_df = result["edge_df_pruned"]

    plot_graph_and_subgraphs_on_image(
        sdata=sdata,
        qc_df=qc_df,
        edge_df=edge_df,
        subgraphs=subgraphs,
        image_name=image_name,
        qc_column=qc_column,
        tile_size=tile_size,
        alpha_tiles=alpha_tiles,
        cmap_tiles=cmap_tiles,
        point_size=point_size,
        point_color=point_color,
        edge_color=edge_color,
        edge_alpha=edge_alpha,
        edge_linewidth=edge_linewidth,
        subgraph_linewidth=subgraph_linewidth,
        subgraph_node_size=subgraph_node_size,
        max_subgraphs_to_plot=max_subgraphs_to_plot,
        random_seed=random_seed,
        figsize=figsize,
        show_colorbar=show_colorbar,
        vmin=vmin,
        vmax=vmax,
    )

import matplotlib.pyplot as plt


def plot_confidence_threshold_curves(threshold_df, confidence_col="top_prob"):
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(threshold_df["threshold"], threshold_df["f1"], label="F1")
    ax1.plot(threshold_df["threshold"], threshold_df["accuracy"], label="Accuracy")
    ax1.plot(threshold_df["threshold"], threshold_df["balanced_accuracy"], label="Balanced Accuracy")
    ax1.set_xlabel(f"Confidence threshold ({confidence_col})")
    ax1.set_ylabel("Performance")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(threshold_df["threshold"], threshold_df["coverage"], linestyle="--", label="Coverage")
    ax2.set_ylabel("Coverage")

    plt.title("Performance vs confidence threshold")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    cm,
    labels,
    normalize="col",   # "col", "row", or None
    vmin=0.2,
    vmax=0.8,
    figsize=(8, 6),
    show_counts=True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    # ---- Convert to DataFrame ----
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # ---- Normalize ----
    if normalize == "col":
        norm_df = cm_df / cm_df.sum(axis=0).replace(0, np.nan)
    elif normalize == "row":
        norm_df = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0)
    else:
        norm_df = cm_df.copy()

    norm_df = norm_df.fillna(0)

    # ---- Custom color palette (white -> Reds) ----
    colors = ["#FFFFFF"] + sns.color_palette("Reds", 9)
    cmap = LinearSegmentedColormap.from_list("custom_reds", colors, N=50)

    # ---- Plot ----
    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        norm_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="black",
        cbar_kws={
            "shrink": 0.8,
            "aspect": 30
        }
    )

    # ---- Overlay counts ----
    if show_counts:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=10
                )

    # ---- Styling (ggplot-like) ----
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    ax.set_xlabel("")
    ax.set_ylabel("")

    # remove grid
    ax.grid(False)

    # square tiles (coord_fixed)
    ax.set_aspect("equal")

    # border
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.show()

def plot_f1_scores(
    pred_df,
    labels,
    threshold,
    use_high_confidence=False,
    figsize=(8, 5)
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import f1_score

    df = pred_df.copy()

    # ---- Optional filtering ----
    if use_high_confidence:
        df = df[df["high_confidence"]]

    y_true = df["true_class"].values
    y_pred = df["pred_class"].values

    # ---- Compute per-class F1 ----
    f1_per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=np.arange(len(labels)),
        zero_division=0
    )

    f1_df = pd.DataFrame({
        "class": labels,
        "f1": f1_per_class
    })

    # ---- Sort descending ----
    f1_df = f1_df.sort_values("f1", ascending=False)

    # ---- Plot ----
    plt.figure(figsize=figsize)

    ax = sns.barplot(
        data=f1_df,
        x="class",
        y="f1",
        palette="Reds_r"
    )

    # ---- Threshold line ----
    ax.axhline(
        y=threshold,
        linestyle="--",
        color="black",
        linewidth=1
    )

    # ---- Styling ----
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1 score")
    plt.xlabel("")
    plt.title("Per-class F1 score (sorted)")

    # annotate values
    for i, row in f1_df.iterrows():
        ax.text(
            x=list(f1_df.index).index(i),
            y=row["f1"] + 0.01,
            s=f"{row['f1']:.2f}",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()
    plt.show()

    return f1_df


def plot_waterfall_by_top_score(
    pred_df,
    labels,
    score_prefix="prob_",
    class_colors=None,
    sort_desc=True,
    figsize=(14, 5),
    use_high_confidence=False,
    confidence_col="high_confidence",
):
    import numpy as np
    import matplotlib.pyplot as plt

    df = pred_df.copy()

    if use_high_confidence and confidence_col in df.columns:
        df = df[df[confidence_col]].copy()

    score_cols = [f"{score_prefix}{lab}" for lab in labels]
    scores = df[score_cols].to_numpy()

    df["top_score"] = scores.max(axis=1)
    df["top_class_idx"] = scores.argmax(axis=1)
    df["top_class"] = [labels[i] for i in df["top_class_idx"]]

    df = df.sort_values("top_score", ascending=not sort_desc).reset_index(drop=True)

    # ---- use fixed color scheme ----
    if class_colors is None:
        class_colors = {lab: plt.get_cmap("tab10")(i) for i, lab in enumerate(labels)}

    bar_colors = [class_colors[c] for c in df["top_class"]]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    ax.bar(x, df["top_score"], color=bar_colors, width=0.95)

    ax.set_ylabel("Highest class score")
    ax.set_xlabel("Images")
    ax.set_title("Waterfall plot (consistent class colors)")

    if len(df) > 100:
        ax.set_xticks([])

    # ---- legend ----
    handles = [
        plt.Line2D([0], [0], color=class_colors[lab], lw=6, label=lab)
        for lab in labels
    ]
    ax.legend(handles=handles, title="Top class", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

    return df

def plot_class_waterfalls(
    pred_df,
    labels,
    threshold,
    score_prefix="prob_",
    class_colors=None,
    use_high_confidence=False,
    confidence_col="high_confidence",
    figsize_per_panel=(12, 3),
    sharey=True,
):
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pred_df.copy()

    if use_high_confidence and confidence_col in df.columns:
        df = df[df[confidence_col]].copy()

    if class_colors is None:
        class_colors = {
            "GBM_MES_TYP": "#1f77b4",
            "GBM_MES_ATYP": "#ff7f0e",
            "GBM_RTK1": "#2ca02c",
            "GBM_RTK2": "#d62728",
            "GBM_PNC": "#9467bd",
        }

    score_cols = [f"{score_prefix}{lab}" for lab in labels]
    missing = [c for c in score_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing score columns: {missing}")

    n_classes = len(labels)
    fig_h = figsize_per_panel[1] * n_classes
    fig, axes = plt.subplots(
        nrows=n_classes,
        ncols=1,
        figsize=(figsize_per_panel[0], fig_h),
        sharex=False,
        sharey=sharey,
    )

    if n_classes == 1:
        axes = [axes]

    for ax, lab in zip(axes, labels):
        col = f"{score_prefix}{lab}"

        sub = df[[col]].copy()
        sub = sub.rename(columns={col: "score"})
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)

        x = np.arange(len(sub))
        ax.bar(
            x,
            sub["score"].values,
            color=class_colors.get(lab, "gray"),
            width=0.95,
        )

        ax.axhline(
            y=threshold,
            linestyle="--",
            color="black",
            linewidth=1.2,
        )

        ax.set_title(lab)
        ax.set_ylabel("Score")
        ax.set_xlim(-0.5, len(sub) - 0.5)

        if len(sub) > 100:
            ax.set_xticks([])
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(x + 1, rotation=90, fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Images ranked from highest to lowest score")
    plt.tight_layout()
    plt.show()


def plot_combined_class_waterfall(
    pred_df,
    labels,
    threshold,
    score_prefix="prob_",
    class_colors=None,
    use_high_confidence=False,
    confidence_col="high_confidence",
    figsize=(18, 5),
    title=None,
    ylabel="Score",
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pred_df.copy()

    if use_high_confidence and confidence_col in df.columns:
        df = df[df[confidence_col]].copy()

    if class_colors is None:
        class_colors = {
            "GBM_MES_TYP": "#1f77b4",
            "GBM_MES_ATYP": "#ff7f0e",
            "GBM_RTK1": "#2ca02c",
            "GBM_RTK2": "#d62728",
            "GBM_PNC": "#9467bd",
        }

    grouped_frames = []
    block_ranges = []
    start = 0

    for lab in labels:
        col = f"{score_prefix}{lab}"
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

        sub = df[[col]].copy()
        sub = sub.rename(columns={col: "score"})
        sub["class"] = lab
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)

        n = len(sub)
        if n == 0:
            continue

        grouped_frames.append(sub)
        block_ranges.append((lab, start, start + n - 1))
        start += n

    if len(grouped_frames) == 0:
        raise ValueError("No data available for plotting.")

    plot_df = pd.concat(grouped_frames, axis=0).reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        x,
        plot_df["score"].values,
        color=[class_colors.get(c, "gray") for c in plot_df["class"]],
        width=0.95,
        edgecolor=None,
    )

    # threshold line
    ax.axhline(
        y=threshold,
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )

    # separators + class labels
    for i, (lab, start_idx, end_idx) in enumerate(block_ranges):
        center = (start_idx + end_idx) / 2
        ax.text(
            center,
            1.02,
            lab.replace("GBM_", "").replace("_", " "),
            ha="center",
            va="bottom",
            fontsize=12,
            transform=ax.get_xaxis_transform(),
        )

        if i > 0:
            ax.axvline(start_idx - 0.5, color="white", linewidth=3)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_xlim(-0.5, len(plot_df) - 0.5)
    ax.set_ylim(0, max(1.05, float(plot_df["score"].max()) * 1.05))

    if title is not None:
        ax.set_title(title)

    ax.set_xticks([])

    # cleaner styling
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    plt.tight_layout()
    plt.show()

    return plot_df, block_ranges


