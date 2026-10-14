
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from reportlab.lib import colors
from reportlab.lib.units import cm

def draw_header(canvas, doc, header_path):
    header_width = 17.0 * cm
    header_height = 2.0 * cm

    x = doc.leftMargin
    y = doc.pagesize[1] - header_height - 0.45 * cm

    canvas.drawImage(
        str(header_path),
        x,
        y,
        width=header_width,
        height=header_height,
        preserveAspectRatio=True,
        mask="auto",
    )

def draw_footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(
        doc.leftMargin,
        0.75 * cm,
        "For research use only – not for clinical decision making",
    )

def _first_page(canvas, doc, header_path):
    draw_header(canvas, doc, header_path)
    draw_footer(canvas, doc)

def _later_pages(canvas, doc, header_path):
    draw_header(canvas, doc, header_path)
    draw_footer(canvas, doc)

def _collapse_major_subtypes(mean_probs, labels, mes_mode: str = "max"):
    prob_map = {lab: float(mean_probs[i]) for i, lab in enumerate(labels)}

    mes_typ = prob_map.get("GBM_MES_TYP", 0.0)
    mes_atyp = prob_map.get("GBM_MES_ATYP", 0.0)
    rtk1 = prob_map.get("GBM_RTK1", 0.0)
    rtk2 = prob_map.get("GBM_RTK2", 0.0)

    if mes_mode == "mean":
        mes = (mes_typ + mes_atyp) / 2.0
    elif mes_mode == "sum":
        mes = mes_typ + mes_atyp
    else:
        mes = max(mes_typ, mes_atyp)

    major_scores = {
        "MES": float(mes),
        "RTK1": float(rtk1),
        "RTK2": float(rtk2),
    }
    return major_scores


def _traffic_light_from_score(score: float, green_thr: float = 0.35, yellow_thr: float = 0.20):
    if score >= green_thr:
        return "high", "#2ca25f"
    elif score >= yellow_thr:
        return "intermediate", "#f0ad00"
    else:
        return "low", "#d62728"


def summarize_sample_prediction(
    model,
    graphs,
    device,
    labels=None,
    confidence_threshold: float = 0.35,
    mes_mode: str = "max",
    green_thr: float = 0.35,
    yellow_thr: float = 0.20,
):
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    loader = DataLoader(graphs, batch_size=8, shuffle=False)

    model.eval()
    model.to(device)

    all_logits = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, return_attention=False)

            logits = out["pred"]
            probs = torch.softmax(logits, dim=1)

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)

    mean_logits = all_logits.mean(axis=0)
    mean_probs = all_probs.mean(axis=0)

    pred_idx = int(np.argmax(mean_probs))
    pred_label = labels[pred_idx]
    pred_prob = float(mean_probs[pred_idx])

    sorted_probs = np.sort(mean_probs)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob
    entropy = float(-(mean_probs * np.log(mean_probs + 1e-12)).sum())

    major_scores = _collapse_major_subtypes(mean_probs, labels, mes_mode=mes_mode)
    major_pred_label = max(major_scores, key=major_scores.get)
    major_pred_score = float(major_scores[major_pred_label])

    confidence_label, confidence_color = _traffic_light_from_score(
        major_pred_score,
        green_thr=green_thr,
        yellow_thr=yellow_thr,
    )
    valid_prediction = major_pred_score >= confidence_threshold

    g0 = graphs[0]

    summary = {
        "predicted_class": pred_label,
        "predicted_class_index": pred_idx,
        "predicted_probability": pred_prob,
        "top_probability": top_prob,
        "margin": margin,
        "entropy": entropy,
        "valid_prediction": valid_prediction,
        "confidence_threshold": confidence_threshold,
        "green_threshold": green_thr,
        "yellow_threshold": yellow_thr,
        "mean_logits": mean_logits,
        "mean_probabilities": mean_probs,
        "score_table": pd.DataFrame({
            "Class": labels,
            "Mean_Probability": mean_probs,
        }).sort_values("Mean_Probability", ascending=False).reset_index(drop=True),
        "major_score_table": pd.DataFrame({
            "Class": list(major_scores.keys()),
            "Score": list(major_scores.values()),
        }).sort_values("Score", ascending=False).reset_index(drop=True),
        "major_predicted_class": major_pred_label,
        "major_predicted_score": major_pred_score,
        "confidence_label": confidence_label,
        "confidence_color": confidence_color,
        "n_graphs": len(graphs),
        "sample_id": getattr(g0, "sample_id", None),
        "patient_id": getattr(g0, "patient_id", None),
        "image_id": getattr(g0, "image_id", None),
        "image_path": getattr(g0, "image_path", None),
        "center": getattr(g0, "Center", None) if hasattr(g0, "Center") else None,
        "sample_type": getattr(g0, "sample_type", None) if hasattr(g0, "sample_type") else None,
    }

    return summary


from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def make_major_subtype_barplot(
    summary: dict,
    out_path: str | Path,
    class_colors: dict | None = None,
    figsize=(7.2, 3.8),
):
    if class_colors is None:
        class_colors = {
            "MES": "#1f77b4",
            "RTK1": "#2ca02c",
            "RTK2": "#d62728",
        }

    out_path = Path(out_path)
    major_df = summary["major_score_table"].copy()

    order = ["MES", "RTK1", "RTK2"]
    major_df["Class"] = pd.Categorical(major_df["Class"], categories=order, ordered=True)
    major_df = major_df.sort_values("Class").reset_index(drop=True)

    scores = major_df["Score"].to_numpy()
    labels = major_df["Class"].tolist()
    colors = [class_colors.get(x, "#888888") for x in labels]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.08, 0.18, 0.63, 0.70])
    ax_light = fig.add_axes([0.78, 0.20, 0.18, 0.66])

    x = np.arange(len(labels))
    bars = ax.bar(x, scores, color=colors, width=0.65, edgecolor="black", linewidth=0.6)

    ax.axhline(summary["yellow_threshold"], linestyle="--", color="#777777", linewidth=1.0)
    ax.axhline(summary["green_threshold"], linestyle="--", color="#555555", linewidth=1.0)

    for xi, yi in zip(x, scores):
        ax.text(xi, yi + 0.015, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, max(0.50, float(np.max(scores)) + 0.08))
    ax.set_ylabel("Mean probability", fontsize=10)
    ax.set_title("Major subtype score overview", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Traffic light panel
    ax_light.axis("off")
    ax_light.set_xlim(0, 1)
    ax_light.set_ylim(0, 1)

    active = summary["confidence_label"]
    light_specs = [
        ("high", 0.78, "#2ca25f", "#b8e3c7"),
        ("intermediate", 0.50, "#f0ad00", "#f7e3a3"),
        ("low", 0.22, "#d62728", "#efb2b2"),
    ]

    for name, y, active_col, inactive_col in light_specs:
        color = active_col if active == name else inactive_col
        circ = Circle((0.40, y), 0.12, facecolor=color, edgecolor="black", linewidth=0.8)
        ax_light.add_patch(circ)

    ax_light.text(0.72, 0.78, "≥ 0.35", va="center", fontsize=9)
    ax_light.text(0.72, 0.50, "0.20–0.35", va="center", fontsize=9)
    ax_light.text(0.72, 0.22, "< 0.20", va="center", fontsize=9)
    ax_light.text(0.02, 0.94, "Confidence", fontsize=10, fontweight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize


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
    figsize=(7.0, 7.0),
    tile_alpha=0.30,
    edge_linewidth=2.0,
    point_size=14,
    node_cmap="magma",
    edge_cmap="viridis",
    node_vmin=None,
    node_vmax=None,
    edge_vmin=None,
    edge_vmax=None,
    return_figure=False,
):
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values
    img_h, img_w = img_np.shape[:2]

    qc_map = qc_df.copy()

    if "point_id" in qc_map.columns and qc_map.index.name == "point_id":
        qc_map = qc_map.reset_index(drop=True)
    elif qc_map.index.name == "point_id" and "point_id" not in qc_map.columns:
        qc_map = qc_map.reset_index()
    elif "point_id" not in qc_map.columns:
        qc_map["point_id"] = qc_map.index.astype(str)

    qc_map["point_id"] = qc_map["point_id"].astype(str)
    coord_map = qc_map.set_index("point_id")[["x", "y"]].to_dict(orient="index")

    node_ids = [str(n) for n in graph_data.node_ids]
    valid_pids = [pid for pid in node_ids if pid in coord_map]
    if len(valid_pids) == 0:
        raise ValueError("No graph node_ids found in qc_df['point_id'].")

    xs_all = np.array([coord_map[pid]["x"] for pid in valid_pids], dtype=float)
    ys_all = np.array([coord_map[pid]["y"] for pid in valid_pids], dtype=float)

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

    node_importance = np.asarray(node_importance, dtype=float)
    edge_attention = np.asarray(edge_attention, dtype=float)

    if node_vmin is None:
        node_vmin = float(np.min(node_importance))
    if node_vmax is None:
        node_vmax = float(np.max(node_importance) + 1e-12)
    if edge_vmin is None:
        edge_vmin = float(np.min(edge_attention))
    if edge_vmax is None:
        edge_vmax = float(np.max(edge_attention) + 1e-12)

    node_norm = Normalize(vmin=node_vmin, vmax=node_vmax)
    edge_norm = Normalize(vmin=edge_vmin, vmax=edge_vmax)

    node_cmap_obj = mpl_cm.get_cmap(node_cmap)
    edge_cmap_obj = mpl_cm.get_cmap(edge_cmap)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.10, 0.70, 0.82])
    cax_edge = fig.add_axes([0.80, 0.16, 0.025, 0.30])
    cax_node = fig.add_axes([0.86, 0.16, 0.025, 0.30])

    ax.imshow(img_crop)

    for i, pid in enumerate(node_ids):
        if pid not in local_coord_map:
            continue

        x = local_coord_map[pid]["x"]
        y = local_coord_map[pid]["y"]
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

    for e in range(edge_index.shape[1]):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])

        pid1 = node_ids[src]
        pid2 = node_ids[dst]

        if pid1 not in local_coord_map or pid2 not in local_coord_map:
            continue

        x1, y1 = local_coord_map[pid1]["x"], local_coord_map[pid1]["y"]
        x2, y2 = local_coord_map[pid2]["x"], local_coord_map[pid2]["y"]

        color = edge_cmap_obj(edge_norm(edge_attention[e]))
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=edge_linewidth, alpha=0.95, zorder=3)

    xs = [local_coord_map[pid]["x"] for pid in node_ids if pid in local_coord_map]
    ys = [local_coord_map[pid]["y"] for pid in node_ids if pid in local_coord_map]
    ax.scatter(xs, ys, c="white", s=point_size, zorder=4)

    ax.axis("off")
    ax.set_title("Tiles = integrated gradients, edges = attention", fontsize=9, pad=6)

    sm_edge = mpl_cm.ScalarMappable(norm=edge_norm, cmap=edge_cmap_obj)
    sm_edge.set_array([])
    cbar_edge = fig.colorbar(sm_edge, cax=cax_edge)
    cbar_edge.set_label("Attention", fontsize=8)
    cbar_edge.ax.tick_params(labelsize=7)

    sm_node = mpl_cm.ScalarMappable(norm=node_norm, cmap=node_cmap_obj)
    sm_node.set_array([])
    cbar_node = fig.colorbar(sm_node, cax=cax_node)
    cbar_node.set_label("Integrated gradients", fontsize=8)
    cbar_node.ax.tick_params(labelsize=7)

    if return_figure:
        return fig

    plt.show()

def build_eval_df_for_graphs(
    model,
    graphs,
    device,
    labels=None,
    batch_size=8,
):
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.loader import DataLoader

    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    rows = []
    graph_idx = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, return_attention=False)

            logits = out["pred"]
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            batch_size_now = probs.shape[0]

            for i in range(batch_size_now):
                g = graphs[graph_idx]

                prob_vec = probs[i]
                pred_class = int(np.argmax(prob_vec))
                winning_score = float(np.max(prob_vec))

                true_class = None
                true_label = None
                correct = None

                if hasattr(g, "y") and g.y is not None:
                    y_np = g.y.detach().cpu().numpy().reshape(-1)
                    if len(y_np) == len(labels):
                        true_class = int(np.argmax(y_np))
                        true_label = labels[true_class]
                        correct = int(pred_class == true_class)

                rows.append({
                    "graph_index": graph_idx,
                    "seed_id": str(getattr(g, "seed_id", "")),
                    "sample_id": str(getattr(g, "sample_id", "")),
                    "patient_id": str(getattr(g, "patient_id", "")),
                    "image_id": str(getattr(g, "image_id", "")),
                    "pred_class": pred_class,
                    "pred_label": labels[pred_class],
                    "winning_score": winning_score,
                    "true_class": true_class,
                    "true_label": true_label,
                    "correct": correct,
                })

                for j, lab in enumerate(labels):
                    rows[-1][f"prob_{lab}"] = float(prob_vec[j])

                graph_idx += 1

    eval_df = pd.DataFrame(rows)
    return eval_df

def select_top_graph_rows(
    eval_df: pd.DataFrame,
    top_k: int = 2,
    only_correct: bool = False,
    level: int = 0,
):
    df = eval_df.copy()

    if only_correct and "correct" in df.columns:
        df = df[df["correct"] == 1].copy()

    if len(df) == 0:
        raise ValueError("No usable graphs available after filtering.")

    df = df.sort_values("winning_score", ascending=False).reset_index(drop=True)

    if level >= len(df):
        raise IndexError(f"Requested level={level}, but only {len(df)} graphs available.")

    return [df.iloc[i] for i in range(level, min(level + top_k, len(df)))]

def make_sample_xai_panels(
    model,
    sdata,
    qc_df,
    graphs,
    device,
    out_dir,
    labels=None,
    image_name="raw_image",
    tile_size=200,
    crop_margin=150,
    level=0,
    only_correct=False,
    top_k=2,
):
    from pathlib import Path

    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df = build_eval_df_for_graphs(
        model=model,
        graphs=graphs,
        device=device,
        labels=labels,
    )

    top_rows = select_top_graph_rows(
        eval_df=eval_df,
        only_correct=only_correct,
        level=level,
        top_k=top_k,
    )

    panel_info = []
    for i, row in enumerate(top_rows, start=1):
        graph = get_graph_by_seed_id(graphs, row["seed_id"])

        out_png = save_xai_plot_for_graph(
            model=model,
            sdata=sdata,
            qc_df=qc_df,
            graph_data=graph,
            device=device,
            target_idx=int(row["pred_class"]),
            out_path=out_dir / f"xai_top_{i}.png",
            image_name=image_name,
            tile_size=tile_size,
            crop_margin=crop_margin,
        )

        panel_info.append({
            "row": row,
            "png": out_png,
        })

    return {
        "panels": panel_info,
        "eval_df": eval_df,
    }

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_xai_plot_for_graph(
    model,
    sdata,
    qc_df,
    graph_data,
    device,
    target_idx: int,
    out_path: str | Path,
    image_name: str = "raw_image",
    tile_size: int = 200,
    crop_margin: int = 150,
    tile_alpha: float = 0.2,
    figsize=(7.5, 7.5),
    ig_mode: str = "abs_mean",
    ig_steps: int = 50,
    baseline: str = "zeros",
):
    att_res = extract_attention_for_graph(model, graph_data, device)
    edge_index_att, edge_attention = aggregate_attention_last_layer(att_res["attention_weights"])

    ig_res = compute_integrated_gradients_for_graph(
        model=model,
        graph_data=graph_data,
        target_idx=target_idx,
        device=device,
        baseline=baseline,
        n_steps=ig_steps,
    )

    node_importance = aggregate_node_importance_from_ig(
        ig_res["attributions"],
        mode=ig_mode,
    )

    node_importance = np.asarray(node_importance, dtype=float)
    if np.max(node_importance) > np.min(node_importance):
        node_importance = (node_importance - np.min(node_importance)) / (
            np.max(node_importance) - np.min(node_importance) + 1e-12
        )

    fig = plot_xai_on_image(
        sdata=sdata,
        qc_df=qc_df,
        graph_data=graph_data,
        node_importance=node_importance,
        edge_index=edge_index_att,
        edge_attention=edge_attention,
        image_name=image_name,
        tile_size=tile_size,
        crop_margin=crop_margin,
        tile_alpha=tile_alpha,
        figsize=figsize,
        return_figure=True,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out_path


def generate_sample_report(
    sdata,
    qc_df,
    graphs,
    model,
    device,
    out_dir,
    image_name="raw_image",
    labels=None,
    confidence_threshold=0.35,
    tile_size=448,
    include_xai=True,
    xai_level=0,
    xai_only_correct=False,
    xai_tile_size=200,
    xai_crop_margin=150,
    header_path="/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/Header.png",
):
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay_png, summary = make_sample_prediction_overlay(
        sdata=sdata,
        qc_df=qc_df,
        graphs=graphs,
        model=model,
        device=device,
        image_name=image_name,
        labels=labels,
        out_path=out_dir / "prediction_overlay.png",
        tile_size=tile_size,
    )

    major_barplot_png = make_major_subtype_barplot(
        summary=summary,
        out_path=out_dir / "major_subtype_barplot.png",
    )

    xai_info = None
    if include_xai:
        try:
            xai_info = make_sample_xai_panels(
                model=model,
                sdata=sdata,
                qc_df=qc_df,
                graphs=graphs,
                device=device,
                out_dir=out_dir,
                labels=labels,
                image_name=image_name,
                tile_size=xai_tile_size,
                crop_margin=xai_crop_margin,
                level=xai_level,
                only_correct=xai_only_correct,
                top_k=2,
            )
        except Exception as e:
            print(f"Warning: XAI panel generation failed and will be skipped: {e}")
            xai_info = None

    pdf_path = out_dir / f"{summary['sample_id']}_research_report.pdf"

    build_sample_pdf_report(
        summary=summary,
        overlay_png=overlay_png,
        major_barplot_png=major_barplot_png,
        out_pdf=pdf_path,
        xai_info=xai_info,
        header_path=header_path,
    )

    return pdf_path, summary

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)


def build_sample_pdf_report(
    summary: dict,
    overlay_png: str | Path,
    major_barplot_png: str | Path,
    out_pdf: str | Path,
    xai_info: dict | None = None,
    header_path: str | Path = "/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/Header.png",
    title: str = "Raman Histology Research Classifier Report",
    subtitle: str = "For research use only - not for clinical decision making",
):
    out_pdf = Path(out_pdf)
    header_path = Path(header_path)

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.2 * cm,
        bottomMargin=1.3 * cm,
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TitleLarge",
        parent=styles["Title"],
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#1f3b5c"),
        alignment=TA_LEFT,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="Subtle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#666666"),
        leading=11,
    ))

    styles.add(ParagraphStyle(
        name="HeadingSmall",
        parent=styles["Heading2"],
        fontSize=11.5,
        leading=13,
        textColor=colors.HexColor("#1f3b5c"),
        spaceAfter=5,
    ))

    styles.add(ParagraphStyle(
        name="BodySmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
        textColor=colors.black,
    ))

    story = []

    story.append(Spacer(1, 2.55 * cm))
    story.append(Paragraph(title, styles["TitleLarge"]))
    story.append(Paragraph(subtitle, styles["Subtle"]))
    story.append(Spacer(1, 0.30 * cm))

    validity = {
        "high": "HIGH CONFIDENCE",
        "intermediate": "INTERMEDIATE CONFIDENCE",
        "low": "LOW CONFIDENCE / REVIEW REQUIRED",
    }[summary["confidence_label"]]

    summary_data = [
        ["Major predicted class", summary["major_predicted_class"]],
        ["Major predicted score", f"{summary['major_predicted_score']:.3f}"],
        ["Top 5-class probability", f"{summary['top_probability']:.3f}"],
        ["Margin", f"{summary['margin']:.3f}"],
        ["Confidence category", validity],
        ["Green threshold", f"{summary['green_threshold']:.2f}"],
        ["Yellow threshold", f"{summary['yellow_threshold']:.2f}"],
    ]

    t = Table(summary_data, colWidths=[5.0 * cm, 10.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f9fbfc")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c7d6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6e1ea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.28 * cm))

    story.append(Paragraph("Sample metadata", styles["HeadingSmall"]))
    meta_data = [
        ["Sample ID", str(summary.get("sample_id", ""))],
        ["Patient ID", str(summary.get("patient_id", ""))],
        ["Image ID", str(summary.get("image_id", ""))],
        ["Center", str(summary.get("center", ""))],
        ["Sample type", str(summary.get("sample_type", ""))],
        ["Graphs analyzed", str(summary.get("n_graphs", ""))],
    ]

    mt = Table(meta_data, colWidths=[4.5 * cm, 10.5 * cm])
    mt.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d0d7de")),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#e2e8ee")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Major subtype score overview", styles["HeadingSmall"]))
    story.append(RLImage(str(major_barplot_png), width=15.2 * cm, height=7.4 * cm))
    story.append(Spacer(1, 0.20 * cm))
    
    story.append(PageBreak())
    story.append(Spacer(1, 2.70 * cm))
    story.append(Paragraph("Spatial prediction overview", styles["HeadingSmall"]))
    story.append(RLImage(str(overlay_png), width=12.8 * cm, height=12.8 * cm))
    story.append(Spacer(1, 0.18 * cm))
    
    if xai_info is not None and len(xai_info["panels"]) > 0:
        story.append(PageBreak())
        story.append(Spacer(1, 2.60 * cm))
        story.append(Paragraph("Selected XAI hotspot panels", styles["TitleLarge"]))
        story.append(Paragraph(
            "The two panels below show selected high-scoring subgraphs, cropped to the local tile region. "
            "Tiles are colored by integrated gradients and edges by graph attention.",
            styles["Subtle"],
        ))
        story.append(Spacer(1, 0.25 * cm))

        rows = [["", "Top panel 1", "Top panel 2"]]
        meta_rows = [["Seed ID"], ["Predicted label"], ["Winning score"]]

        for i, panel in enumerate(xai_info["panels"][:2], start=1):
            row = panel["row"]
            if len(rows[0]) < i + 1:
                rows[0].append(f"Top panel {i}")

        xai_meta = [
            ["", "Top panel 1", "Top panel 2"],
            ["Seed ID"] + [str(p["row"]["seed_id"]) for p in xai_info["panels"][:2]],
            ["Predicted label"] + [str(p["row"]["pred_label"]) for p in xai_info["panels"][:2]],
            ["Winning score"] + [f"{float(p['row']['winning_score']):.3f}" for p in xai_info["panels"][:2]],
        ]

        xt = Table(xai_meta, colWidths=[3.7 * cm, 5.6 * cm, 5.6 * cm])
        xt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c7d6")),
            ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#d6e1ea")),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        story.append(xt)
        story.append(Spacer(1, 0.20 * cm))

        img_table = Table([[
            RLImage(str(xai_info["panels"][0]["png"]), width=7.2 * cm, height=7.2 * cm),
            RLImage(str(xai_info["panels"][1]["png"]), width=7.2 * cm, height=7.2 * cm),
        ]], colWidths=[7.4 * cm, 7.4 * cm])

        img_table.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(img_table)
    
    story.append(PageBreak())
    story.append(Spacer(1, 2.70 * cm))
    disclaimer = (
        "This report is generated automatically for research purposes only. "
        "It is not validated for clinical diagnosis, treatment decision-making, "
        "or patient management."
    )
    story.append(Paragraph(disclaimer, styles["Subtle"]))
    
    doc.build(
        story,
        onFirstPage=lambda c, d: _first_page(c, d, header_path),
        onLaterPages=lambda c, d: _later_pages(c, d, header_path),
    )

    return out_pdf

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from torch_geometric.loader import DataLoader


def make_sample_prediction_overlay(
    sdata,
    qc_df: pd.DataFrame,
    graphs,
    model,
    device,
    image_name: str = "raw_image",
    labels=None,
    out_path: str | Path = "sample_prediction_overlay.png",
    tile_size: int = 448,
    alpha: float = 0.45,
    cmap: str = "magma",
):
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    summary = summarize_sample_prediction(model, graphs, device, labels=labels)
    pred_idx = summary["predicted_class_index"]

    loader = DataLoader(graphs, batch_size=8, shuffle=False)
    all_scores = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, return_attention=False)
            pred = out["pred"].detach().cpu().numpy()
            all_scores.append(pred)

    all_scores = np.vstack(all_scores)

    rows = []
    for g, score_vec in zip(graphs, all_scores):
        rows.append({
            "point_id": str(g.seed_id),
            "score": float(score_vec[pred_idx]),
        })

    graph_score_df = pd.DataFrame(rows)

    qc_df = qc_df.copy()

    # make point_id unambiguous
    if "point_id" in qc_df.columns and qc_df.index.name == "point_id":
        qc_df = qc_df.reset_index(drop=True)
    elif qc_df.index.name == "point_id" and "point_id" not in qc_df.columns:
        qc_df = qc_df.reset_index()
    elif "point_id" not in qc_df.columns:
        qc_df["point_id"] = qc_df.index.astype(str)

    qc_df["point_id"] = qc_df["point_id"].astype(str)

    overlay_df = qc_df.merge(
        graph_score_df,
        on="point_id",
        how="inner",
    )

    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

    values = overlay_df["score"].to_numpy(dtype=float)
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_np)

    half = tile_size / 2.0

    for row in overlay_df.itertuples(index=False):
        x = float(row.x)
        y = float(row.y)
        color = cmap_obj(norm(float(row.score)))

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

    ax.scatter(overlay_df["x"], overlay_df["y"], c="white", s=8, linewidths=0)
    ax.axis("off")
    ax.set_title(f"Predicted class score overlay: {summary['predicted_class']}")

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{summary['predicted_class']} score")

    plt.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path, summary

