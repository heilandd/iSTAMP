# =============================================================================
# MsClaude optimized copy — original unchanged at
#   /Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/Evaluate.py
#
# Purpose: prediction collection, ROC/AUC + confusion metrics, confidence
# thresholding, integrated-gradients / attention XAI overlays.
#
# Light improvements only (NO scientific-logic changes):
#   - Added module docstring.
#   - `plot_xai_on_image` is DEFINED TWICE (the 2nd shadows the 1st). Kept both
#     for byte-for-byte reproducibility; flagged here — remove the first.
#   - Added `bootstrap_auc_ci`, a NEW helper to report 95% CIs for AUC, which
#     the manuscript currently lacks (point estimates only).
#   - Caveat documented (not changed): metrics are computed on the VALIDATION
#     loader. If that loader was built with the leaky graph-level random_split
#     (see GNN_prepare.make_train_val_loaders), these numbers are optimistic.
#     Recompute on a patient-grouped held-out cohort.
# =============================================================================
"""Evaluation, metrics, confidence analysis and XAI for the iSTAMP model."""

import numpy as np
import torch


def bootstrap_auc_ci(y_true_binary, scores, n_boot: int = 2000, seed: int = 42):
    """Return (auc, lo95, hi95) via stratified bootstrap. NEW (MsClaude).

    Adds the confidence interval that the manuscript reports are missing.
    """
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y_true_binary = np.asarray(y_true_binary)
    scores = np.asarray(scores)
    n = len(y_true_binary)
    point = roc_auc_score(y_true_binary, scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true_binary[idx]
        if len(np.unique(yb)) < 2:
            continue  # skip degenerate resamples
        boots.append(roc_auc_score(yb, scores[idx]))
    if len(boots) == 0:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def collect_predictions(model, loader, device):
    model.eval()
    model.to(device)

    all_scores = []
    all_targets = []
    all_graphs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data, return_attention=False)

            pred = out["pred"]
            y = data.y.float().view_as(pred)

            all_scores.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_graphs.append(data.cpu())

    all_scores = np.vstack(all_scores)
    all_targets = np.vstack(all_targets)

    return all_scores, all_targets, all_graphs


import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(
    model,
    val_loader,
    device,
    labels=None,
    save_prefix=None,
):
    """
    Evaluate model using dominant-class interpretation of the subtype score vector.

    This assumes:
    - model outputs shape [n_samples, 5]
    - targets are subtype score vectors of shape [n_samples, 5]
    - dominant subtype is defined via argmax(target)

    Outputs:
    - one-vs-rest ROC/AUC for each subtype
    - accuracy / balanced accuracy
    - classification report
    - confusion matrix
    """
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    # collect predictions
    model.eval()
    model.to(device)

    all_scores = []
    all_targets = []

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data, return_attention=False)

            pred = out["pred"]
            y = data.y.float().view_as(pred)

            all_scores.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_scores = np.vstack(all_scores)
    all_targets = np.vstack(all_targets)

    # dominant class labels
    y_true = np.argmax(all_targets, axis=1)
    y_pred = np.argmax(all_scores, axis=1)

    # One-vs-rest ROC
    auc_scores = {}
    plt.figure(figsize=(10, 6))

    for i in range(len(labels)):
        y_true_binary = (y_true == i).astype(int)

        # ROC AUC only works if both classes are present
        if len(np.unique(y_true_binary)) < 2:
            print(f"Skipping {labels[i]}: only one class present in validation set.")
            continue

        try:
            auc_value = roc_auc_score(y_true_binary, all_scores[:, i])
            auc_scores[labels[i]] = auc_value

            fpr, tpr, _ = roc_curve(y_true_binary, all_scores[:, i])
            plt.plot(fpr, tpr, label=f"{labels[i]} (AUC = {auc_value:.3f})")
        except Exception as e:
            print(f"Skipping {labels[i]} due to error: {e}")

    # Optional "max-score-group" metric
    try:
        correct_mask = (y_true == y_pred).astype(int)
        winning_score = all_scores[np.arange(len(y_pred)), y_pred]

        if len(np.unique(correct_mask)) > 1:
            auc_max = roc_auc_score(correct_mask, winning_score)
            auc_scores["Max_Score_Group"] = auc_max

            fpr, tpr, _ = roc_curve(correct_mask, winning_score)
            plt.plot(
                fpr,
                tpr,
                label=f"Max_Score_Group (AUC = {auc_max:.3f})",
                linestyle="--",
                linewidth=2,
            )
    except Exception as e:
        print(f"Skipping Max_Score_Group due to error: {e}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for GBM Subtypes")
    plt.legend()
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}.svg", format="svg", transparent=True)
        plt.savefig(f"{save_prefix}.pdf", format="pdf", transparent=True)

    plt.show()

    # Accuracy metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print("AUC scores per class:", auc_scores)
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"Balanced Accuracy: {balanced_acc:.3f}")

    # Force report to always use all 5 classes
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            digits=3,
            zero_division=0,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    return {
        "auc_scores": auc_scores,
        "all_scores": all_scores,
        "all_targets": all_targets,
        "y_true": y_true,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": cm,
    }



from sklearn.metrics import roc_auc_score, roc_curve


def compute_auc_multiclass(y_true, all_scores, class_names, save_prefix=None):
    auc_scores = {}

    plt.figure(figsize=(10, 6))

    for i in range(len(class_names)):
        y_true_binary = (y_true == i).astype(int)

        try:
            auc_value = roc_auc_score(y_true_binary, all_scores[:, i])
            auc_scores[class_names[i]] = auc_value

            fpr, tpr, _ = roc_curve(y_true_binary, all_scores[:, i])
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {auc_value:.3f})")
        except ValueError:
            print(f"Skipping {class_names[i]}: only one class present in y_true.")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend()
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}.svg", format="svg", transparent=True)
        plt.savefig(f"{save_prefix}.pdf", format="pdf", transparent=True)

    plt.show()

    return auc_scores



### XAI

def extract_attention_for_graph(model, graph_data, device):
    model.eval()
    graph_data = graph_data.to(device)

    with torch.no_grad():
        out = model(graph_data, return_attention=True)

    pred = out["pred"].detach().cpu().numpy()
    attention_weights = out["attention_weights"]

    # convert attention outputs to CPU numpy
    att_list = []
    for edge_idx, alpha in attention_weights:
        att_list.append({
            "edge_index": edge_idx.detach().cpu().numpy(),
            "alpha": alpha.detach().cpu().numpy(),
        })

    return {
        "pred": pred,
        "attention_weights": att_list,
        "node_embeddings": out["node_embeddings"].detach().cpu().numpy(),
        "graph_embedding": out["graph_embedding"].detach().cpu().numpy(),
    }

from captum.attr import IntegratedGradients


class GraphForwardWrapper(torch.nn.Module):
    def __init__(self, model, data_template, target_idx: int):
        super().__init__()
        self.model = model
        self.data_template = data_template
        self.target_idx = target_idx

    def forward(self, x):
        data = self.data_template.clone()
        data.x = x
        out = self.model(data, return_attention=False)["pred"]
        return out[:, self.target_idx]


def compute_integrated_gradients_for_graph(
    model,
    graph_data,
    target_idx: int,
    device,
    baseline: str = "zeros",
    n_steps: int = 50,
):
    model.eval()
    graph_data = graph_data.to(device)

    x = graph_data.x.clone().detach().requires_grad_(True)

    if baseline == "zeros":
        x_baseline = torch.zeros_like(x)
    elif baseline == "mean":
        x_baseline = x.mean(dim=0, keepdim=True).repeat(x.shape[0], 1)
    else:
        raise ValueError("baseline must be 'zeros' or 'mean'")

    wrapped_model = GraphForwardWrapper(model, graph_data, target_idx=target_idx).to(device)
    ig = IntegratedGradients(wrapped_model)

    attributions, delta = ig.attribute(
        inputs=x,
        baselines=x_baseline,
        return_convergence_delta=True,
        n_steps=n_steps,
    )

    return {
        "attributions": attributions.detach().cpu().numpy(),   # [n_nodes, n_features]
        "delta": delta.detach().cpu().numpy(),
    }


def aggregate_node_importance_from_ig(attributions, mode="abs_mean"):
    if mode == "abs_mean":
        return np.mean(np.abs(attributions), axis=1)
    elif mode == "abs_sum":
        return np.sum(np.abs(attributions), axis=1)
    elif mode == "signed_mean":
        return np.mean(attributions, axis=1)
    else:
        raise ValueError("Unsupported mode")
def aggregate_attention_last_layer(attention_weights):
    """
    Use last layer attention and average across heads.
    """
    last = attention_weights[-1]
    edge_index = last["edge_index"]      # shape [2, n_edges]
    alpha = last["alpha"]                # shape [n_edges, n_heads] or [n_edges]

    if alpha.ndim == 2:
        alpha_mean = alpha.mean(axis=1)
    else:
        alpha_mean = alpha

    return edge_index, alpha_mean


from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize

def get_image_scale0(image_element):
    """
    Return highest-resolution DataArray from a SpatialData image element.
    Works for both single-scale DataArray and multiscale DataTree.
    """
    if hasattr(image_element, "keys") and "scale0" in image_element.keys():
        return image_element["scale0"]["image"]
    return image_element
def plot_xai_on_image(
    sdata,
    qc_df,
    graph_data,
    node_importance,
    edge_index,
    edge_attention,
    image_name="raw_image",
    tile_size=448,
    figsize=(14, 14),
    tile_alpha=0.45,
    edge_linewidth=2.0,
    point_size=20,
    node_cmap="magma",
    edge_cmap="viridis",
):
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

    qc_map = qc_df.copy()
    qc_map["point_id"] = qc_map["point_id"].astype(str)
    coord_map = qc_map.set_index("point_id")[["x", "y"]].to_dict(orient="index")

    node_ids = [str(n) for n in graph_data.node_ids]

    # node importance colormap
    node_norm = Normalize(vmin=float(np.min(node_importance)), vmax=float(np.max(node_importance)))
    node_cmap_obj = cm.get_cmap(node_cmap)

    # edge attention colormap
    edge_norm = Normalize(vmin=float(np.min(edge_attention)), vmax=float(np.max(edge_attention)))
    edge_cmap_obj = cm.get_cmap(edge_cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)

    half = tile_size / 2.0

    # tiles by IG importance
    for i, pid in enumerate(node_ids):
        if pid not in coord_map:
            continue
        x = coord_map[pid]["x"]
        y = coord_map[pid]["y"]
        color = node_cmap_obj(node_norm(node_importance[i]))

        rect = Rectangle(
            (x - half, y - half),
            tile_size,
            tile_size,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
            alpha=tile_alpha,
            zorder=2,
        )
        ax.add_patch(rect)

    # edges by attention
    for e in range(edge_index.shape[1]):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])

        pid1 = node_ids[src]
        pid2 = node_ids[dst]

        if pid1 not in coord_map or pid2 not in coord_map:
            continue

        x1, y1 = coord_map[pid1]["x"], coord_map[pid1]["y"]
        x2, y2 = coord_map[pid2]["x"], coord_map[pid2]["y"]

        color = edge_cmap_obj(edge_norm(edge_attention[e]))

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=edge_linewidth,
            alpha=0.9,
            zorder=3,
        )

    # points
    xs = [coord_map[pid]["x"] for pid in node_ids if pid in coord_map]
    ys = [coord_map[pid]["y"] for pid in node_ids if pid in coord_map]
    ax.scatter(xs, ys, c="red", s=point_size, zorder=4)

    ax.axis("off")
    ax.set_title("XAI overlay: tiles = integrated gradients, edges = attention")

    # colorbars
    sm1 = cm.ScalarMappable(norm=node_norm, cmap=node_cmap_obj)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=ax, fraction=0.03, pad=0.02)
    cbar1.set_label("Integrated gradients")

    sm2 = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap_obj)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.08)
    cbar2.set_label("Attention")

    plt.tight_layout()
    plt.show()


def plot_xai_on_image(
    sdata,
    qc_df,
    graph_data,
    node_importance,
    edge_index,
    edge_attention,
    image_name="raw_image",
    tile_size=448,
    crop_margin=150,
    figsize=(14, 14),
    tile_alpha=0.45,
    edge_alpha=0.9,
    edge_linewidth=2.0,
    point_size=20,
    node_cmap="magma",
    edge_cmap="viridis",
    node_vmin=None,
    node_vmax=None,
    edge_vmin=None,
    edge_vmax=None,
    shrink_tiles=0.9,
    margin = 4,
    show_points=True,
    title="XAI overlay: tiles = integrated gradients, edges = attention",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # -----------------------------
    # Load image
    # -----------------------------
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values
    img_h, img_w = img_np.shape[:2]

    # -----------------------------
    # Coordinate map
    # -----------------------------
    qc_map = qc_df.copy()
    qc_map["point_id"] = qc_map["point_id"].astype(str)
    coord_map = qc_map.set_index("point_id")[["x", "y"]].to_dict(orient="index")

    node_ids = [str(n) for n in graph_data.node_ids]

    valid_pids = [pid for pid in node_ids if pid in coord_map]
    if len(valid_pids) == 0:
        raise ValueError("No graph node_ids were found in qc_df['point_id'].")

    xs_all = np.array([coord_map[pid]["x"] for pid in valid_pids], dtype=float)
    ys_all = np.array([coord_map[pid]["y"] for pid in valid_pids], dtype=float)

    # -----------------------------
    # Crop image around used tiles
    # -----------------------------
    half = tile_size / 2.0
    xmin = max(0, int(np.floor(xs_all.min() - half - crop_margin)))
    xmax = min(img_w, int(np.ceil(xs_all.max() + half + crop_margin)))
    ymin = max(0, int(np.floor(ys_all.min() - half - crop_margin)))
    ymax = min(img_h, int(np.ceil(ys_all.max() + half + crop_margin)))

    img_crop = img_np[ymin:ymax, xmin:xmax]

    local_coord_map = {
        pid: {
            "x": coord_map[pid]["x"] - xmin,
            "y": coord_map[pid]["y"] - ymin,
        }
        for pid in valid_pids
    }

    # -----------------------------
    # Norms and colormaps
    # -----------------------------
    node_vals = np.asarray(node_importance, dtype=float)
    edge_vals = np.asarray(edge_attention, dtype=float)

    if node_vmin is None:
        node_vmin = float(np.nanmin(node_vals))
    if node_vmax is None:
        node_vmax = float(np.nanmax(node_vals))

    if edge_vmin is None:
        edge_vmin = float(np.nanmin(edge_vals))
    if edge_vmax is None:
        edge_vmax = float(np.nanmax(edge_vals))

    node_norm = Normalize(vmin=node_vmin, vmax=node_vmax)
    edge_norm = Normalize(vmin=edge_vmin, vmax=edge_vmax)

    node_cmap_obj = cm.get_cmap(node_cmap)
    edge_cmap_obj = cm.get_cmap(edge_cmap)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_crop, zorder=0)

    # slightly shrink tiles to reduce overlap
    tile_side = tile_size * shrink_tiles
    half_shrunk = tile_side / 2.0

    # tiles by integrated gradients
    for i, pid in enumerate(node_ids):
        if pid not in local_coord_map:
            continue

        x = local_coord_map[pid]["x"]
        y = local_coord_map[pid]["y"]
        color = node_cmap_obj(node_norm(node_vals[i]))

        rect = Rectangle(
            (x - half_shrunk, y - half_shrunk),
            tile_side,
            tile_side,
            facecolor=color,
            edgecolor=None,
            linewidth=0,
            alpha=tile_alpha,
            zorder=2,
        )
        ax.add_patch(rect)

    # edges by attention
    for e in range(edge_index.shape[1]):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])

        pid1 = node_ids[src]
        pid2 = node_ids[dst]

        if pid1 not in local_coord_map or pid2 not in local_coord_map:
            continue

        x1, y1 = local_coord_map[pid1]["x"], local_coord_map[pid1]["y"]
        x2, y2 = local_coord_map[pid2]["x"], local_coord_map[pid2]["y"]

        color = edge_cmap_obj(edge_norm(edge_vals[e]))

        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=edge_linewidth,
            alpha=edge_alpha,
            zorder=3,
        )

    # points
    if show_points:
        xs = [local_coord_map[pid]["x"] for pid in node_ids if pid in local_coord_map]
        ys = [local_coord_map[pid]["y"] for pid in node_ids if pid in local_coord_map]
        ax.scatter(xs, ys, c="red", s=point_size, zorder=4)

    ax.axis("off")
    ax.set_title(title)

    # -----------------------------
    # Separate colorbars
    # right = integrated gradients
    # bottom = attention
    # -----------------------------
    divider = make_axes_locatable(ax)

    cax_right = divider.append_axes("right", size="3%", pad=0.05)
    sm1 = cm.ScalarMappable(norm=node_norm, cmap=node_cmap_obj)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, cax=cax_right)
    cbar1.set_label("Integrated gradients")

    cax_bottom = divider.append_axes("bottom", size="3%", pad=0.4)
    sm2 = cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap_obj)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, cax=cax_bottom, orientation="horizontal")
    cbar2.set_label("Attention")
    cbar2.ax.xaxis.set_ticks_position("bottom")

    plt.tight_layout(pad=margin)
    plt.show()

    return {
        "crop_box": (xmin, xmax, ymin, ymax),
        "node_vmin": node_vmin,
        "node_vmax": node_vmax,
        "edge_vmin": edge_vmin,
        "edge_vmax": edge_vmax,
    }











import numpy as np
import pandas as pd
import torch


def evaluate_model_with_metadata(
    model,
    val_loader,
    device,
    labels=None,
):
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    model.eval()
    model.to(device)

    rows = []
    graph_objects = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch, return_attention=False)

            pred = out["pred"].detach().cpu().numpy()
            true = batch.y.float().view_as(out["pred"]).detach().cpu().numpy()

            batch_cpu = batch.cpu()

            # PyG batch can contain multiple graphs
            num_graphs = batch_cpu.num_graphs

            for i in range(num_graphs):
                pred_i = pred[i]
                true_i = true[i]

                pred_class = int(np.argmax(pred_i))
                true_class = int(np.argmax(true_i))

                winning_score = float(pred_i[pred_class])
                true_score = float(true_i[true_class])

                correct = int(pred_class == true_class)

                row = {
                    "graph_idx": len(graph_objects),
                    "pred_class": pred_class,
                    "true_class": true_class,
                    "pred_label": labels[pred_class],
                    "true_label": labels[true_class],
                    "winning_score": winning_score,
                    "true_score": true_score,
                    "correct": correct,
                }

                # optional metadata if present
                for attr in ["image_id", "sample_id", "patient_id", "seed_id", "image_path"]:
                    if hasattr(batch_cpu, attr):
                        val = getattr(batch_cpu, attr)
                        try:
                            row[attr] = val[i]
                        except Exception:
                            row[attr] = val

                # store all subtype predictions
                for j, lab in enumerate(labels):
                    row[f"pred_{lab}"] = float(pred_i[j])
                    row[f"true_{lab}"] = float(true_i[j])

                rows.append(row)

            # important: save each graph object separately
            graphs_list = batch_cpu.to_data_list()
            graph_objects.extend(graphs_list)

    eval_df = pd.DataFrame(rows)
    return eval_df, graph_objects

def get_graph_by_seed_id(graphs, seed_id):
    for g in graphs:
        if str(g.seed_id) == str(seed_id):
            return g
    raise ValueError(f"seed_id {seed_id} not found in graphs")


## F1 Score

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score


def collect_predictions_with_confidence(model, loader, device, labels=None):
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    model.eval()
    model.to(device)

    rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, return_attention=False)

            logits = out["pred"]
            probs = torch.softmax(logits, dim=1)

            y_true_scores = batch.y.float().view_as(logits)

            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()
            y_true_np = y_true_scores.cpu().numpy()

            batch_cpu = batch.cpu()
            graphs = batch_cpu.to_data_list()

            for i, g in enumerate(graphs):
                pred_scores = logits_np[i]
                prob_scores = probs_np[i]
                true_scores = y_true_np[i]

                pred_class = int(np.argmax(pred_scores))
                true_class = int(np.argmax(true_scores))

                sorted_probs = np.sort(prob_scores)[::-1]
                top_prob = float(sorted_probs[0])
                second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
                margin = top_prob - second_prob

                entropy = float(-(prob_scores * np.log(prob_scores + 1e-12)).sum())

                row = {
                    "pred_class": pred_class,
                    "true_class": true_class,
                    "pred_label": labels[pred_class],
                    "true_label": labels[true_class],
                    "correct": int(pred_class == true_class),
                    "top_prob": top_prob,
                    "margin": margin,
                    "entropy": entropy,
                }

                for j, lab in enumerate(labels):
                    row[f"logit_{lab}"] = float(pred_scores[j])
                    row[f"prob_{lab}"] = float(prob_scores[j])
                    row[f"true_{lab}"] = float(true_scores[j])

                for attr in ["seed_id", "image_id", "sample_id", "patient_id", "image_path"]:
                    if hasattr(g, attr):
                        row[attr] = getattr(g, attr)

                rows.append(row)

    return pd.DataFrame(rows)


def evaluate_confidence_thresholds(
    pred_df: pd.DataFrame,
    confidence_col: str = "top_prob",
    threshold_grid=None,
    average: str = "weighted",
):
    if threshold_grid is None:
        threshold_grid = np.linspace(0.0, 1.0, 101)

    results = []

    for thr in threshold_grid:
        keep = pred_df[pred_df[confidence_col] >= thr].copy()

        coverage = len(keep) / len(pred_df) if len(pred_df) > 0 else 0.0

        if len(keep) == 0:
            results.append({
                "threshold": thr,
                "coverage": 0.0,
                "n_kept": 0,
                "f1": np.nan,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
            })
            continue

        y_true = keep["true_class"].to_numpy()
        y_pred = keep["pred_class"].to_numpy()

        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        results.append({
            "threshold": thr,
            "coverage": coverage,
            "n_kept": len(keep),
            "f1": f1,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
        })

    return pd.DataFrame(results)

def select_best_confidence_threshold(
    threshold_df: pd.DataFrame,
    min_coverage: float = 0.2,
):
    df = threshold_df.copy()
    df = df[df["coverage"] >= min_coverage].copy()

    if len(df) == 0:
        raise ValueError("No threshold satisfies the required minimum coverage.")

    best_row = df.sort_values(["f1", "coverage"], ascending=[False, False]).iloc[0]
    return best_row


def summarize_confidence_filtered_predictions(pred_df, confidence_col, threshold):
    keep = pred_df[pred_df[confidence_col] >= threshold].copy()

    y_true = keep["true_class"].to_numpy()
    y_pred = keep["pred_class"].to_numpy()

    out = {
        "threshold": threshold,
        "confidence_col": confidence_col,
        "n_total": len(pred_df),
        "n_kept": len(keep),
        "coverage": len(keep) / len(pred_df) if len(pred_df) > 0 else np.nan,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(keep) > 0 else np.nan,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if len(keep) > 0 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred) if len(keep) > 0 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(keep) > 0 else np.nan,
    }
    return out


import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def evaluate_per_patient(pred_df, labels, patient_col="patient_id"):
    results = {}

    for patient, df_p in pred_df.groupby(patient_col):
        print(f"\nProcessing patient: {patient}")

        # --- filter high confidence ---
        df_p = df_p[df_p["high_confidence"]]

        if len(df_p) < 10:
            print("Skipping (too few samples)")
            continue

        y_true = df_p["true_class"].values
        y_pred = df_p["pred_class"].values

        # --- metrics ---
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        # --- ROC (one-vs-rest) ---
        # expects probabilities per class
        prob_cols = [f"prob_{l}" for l in labels]
        y_score = df_p[prob_cols].values

        # binarize labels
        y_true_bin = label_binarize(y_true, classes=np.arange(len(labels)))

        # check if ROC is valid
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(
                y_true_bin,
                y_score,
                average="weighted",
                multi_class="ovr"
            )
        else:
            auc = np.nan
            print("Warning: single class → ROC undefined")

        # --- confusion matrix ---
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))

        # --- plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # confusion matrix
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=axes[0], colorbar=False)
        axes[0].set_title(f"{patient} - Confusion")

        # ROC curves (per class)
        if len(np.unique(y_true)) > 1:
            for i, label in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                axes[1].plot(fpr, tpr, label=label)
            axes[1].plot([0, 1], [0, 1], "k--")
            axes[1].set_title(f"{patient} - ROC")
            axes[1].legend(fontsize=6)
        else:
            axes[1].set_title("ROC not defined")

        plt.tight_layout()
        plt.show()

        # --- store results ---
        results[patient] = {
            "n_samples": len(df_p),
            "accuracy": acc,
            "f1_weighted": f1,
            "auc_weighted_ovr": auc
        }

    return pd.DataFrame(results).T

