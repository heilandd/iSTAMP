from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

def summarize_sample_prediction(
    model,
    graphs,
    device,
    labels=None,
    confidence_threshold: float = 0.70,
):
    """
    Summarize predictions for one sample from a list of PyG graphs.

    Returns
    -------
    summary : dict
    """
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

    pred_idx = int(np.argmax(mean_logits))
    pred_label = labels[pred_idx]
    pred_score = float(mean_logits[pred_idx])

    sorted_probs = np.sort(mean_probs)[::-1]
    top_prob = float(sorted_probs[0])
    second_prob = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_prob - second_prob
    entropy = float(-(mean_probs * np.log(mean_probs + 1e-12)).sum())

    valid_prediction = top_prob >= confidence_threshold

    g0 = graphs[0]

    summary = {
        "predicted_class": pred_label,
        "predicted_class_index": pred_idx,
        "predicted_score": pred_score,
        "top_probability": top_prob,
        "margin": margin,
        "entropy": entropy,
        "valid_prediction": valid_prediction,
        "confidence_threshold": confidence_threshold,
        "mean_logits": mean_logits,
        "mean_probabilities": mean_probs,
        "score_table": pd.DataFrame({
            "Class": labels,
            "Mean_Logit": mean_logits,
            "Mean_Probability": mean_probs,
        }).sort_values("Mean_Probability", ascending=False).reset_index(drop=True),
        "n_graphs": len(graphs),
        "sample_id": getattr(g0, "sample_id", None),
        "patient_id": getattr(g0, "patient_id", None),
        "image_id": getattr(g0, "image_id", None),
        "image_path": getattr(g0, "image_path", None),
        "center": getattr(g0, "Center", None) if hasattr(g0, "Center") else None,
        "sample_type": getattr(g0, "sample_type", None) if hasattr(g0, "sample_type") else None,
    }

    return summary
def get_graphs_for_sample(dataset, sample_id: str):
    return [g for g in dataset if str(getattr(g, "sample_id", "")) == str(sample_id)]


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


########### Full report:

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


def build_sample_pdf_report(
    summary: dict,
    overlay_png: str | Path,
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

    # PAGE 1
    story.append(Spacer(1, 2.55 * cm))
    story.append(Paragraph(title, styles["TitleLarge"]))
    story.append(Paragraph(subtitle, styles["Subtle"]))
    story.append(Spacer(1, 0.30 * cm))

    validity = (
        "VALID / HIGHER CONFIDENCE"
        if summary["valid_prediction"]
        else "LOW CONFIDENCE / REVIEW REQUIRED"
    )

    summary_data = [
        ["Predicted class", summary["predicted_class"]],
        ["Predicted score", f"{summary['predicted_score']:.3f}"],
        ["Top probability", f"{summary['top_probability']:.3f}"],
        ["Margin", f"{summary['margin']:.3f}"],
        ["Entropy", f"{summary['entropy']:.3f}"],
        ["Confidence threshold", f"{summary['confidence_threshold']:.2f}"],
        ["Interpretation", validity],
    ]

    t = Table(summary_data, colWidths=[5.0 * cm, 10.0 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf3f8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
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
    story.append(Spacer(1, 0.30 * cm))

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
    story.append(Spacer(1, 0.30 * cm))

    story.append(Paragraph("Subtype score overview", styles["HeadingSmall"]))
    score_df = summary["score_table"].copy()
    score_rows = [["Class", "Mean logit", "Mean probability"]]
    for row in score_df.itertuples(index=False):
        score_rows.append([
            str(row.Class),
            f"{row.Mean_Logit:.3f}",
            f"{row.Mean_Probability:.3f}",
        ])

    st = Table(score_rows, colWidths=[6.0 * cm, 4.0 * cm, 4.0 * cm])
    st.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7f2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7c7d6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#d6e1ea")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(st)
    story.append(Spacer(1, 0.30 * cm))

    story.append(Paragraph("Subtype interpretation", styles["HeadingSmall"]))
    subtype_text = (
        "<b>GBM_RTK1</b> and <b>GBM_RTK2</b> refer to receptor tyrosine kinase-associated glioblastoma states, "
        "which are often linked to signaling-driven tumor programs. "
        "<br/><br/>"
        "<b>GBM_MES_TYP</b> and <b>GBM_MES_ATYP</b> refer to mesenchymal-like programs, which are frequently associated "
        "with inflammatory, stromal, or treatment-associated adaptation patterns. "
        "<br/><br/>"
        "<b>GBM_PNC</b> refers to a proneural-like state, typically associated with more neural-developmental "
        "or lineage-related transcriptional features. "
        "<br/><br/>"
        "These categories are shown here as research-oriented molecular classifier outputs and should not be interpreted "
        "as stand-alone clinical diagnoses."
    )
    story.append(Paragraph(subtype_text, styles["BodySmall"]))
    story.append(Spacer(1, 0.28 * cm))

    story.append(Paragraph("Spatial prediction overview", styles["HeadingSmall"]))
    story.append(RLImage(str(overlay_png), width=13.2 * cm, height=13.2 * cm))
    story.append(Spacer(1, 0.20 * cm))

    disclaimer = (
        "This report is generated automatically for research purposes only. "
        "It is not validated for clinical diagnosis, treatment decision-making, "
        "or patient management. Predictions are based on image-derived graph "
        "representations and should be interpreted together with molecular, histological, "
        "and neuropathological evidence."
    )
    story.append(Paragraph(disclaimer, styles["Subtle"]))

    # PAGE 2: XAI
    if xai_info is not None:
        story.append(PageBreak())
        story.append(Spacer(1, 2.70 * cm))
        story.append(Paragraph("Explainable AI overview", styles["TitleLarge"]))
        story.append(Paragraph(
            "Tiles are colored by integrated gradients and graph edges are colored by attention weights. "
            "This visualization highlights which image regions and graph relations contributed most strongly "
            "to the selected subtype prediction.",
            styles["Subtle"],
        ))
        story.append(Spacer(1, 0.20 * cm))

        best_row = xai_info["best_row"]
        worst_row = xai_info["worst_row"]

        xai_meta = [
            ["", "Highest-confidence subgraph", "Lowest-confidence subgraph"],
            ["Seed ID", str(best_row["seed_id"]), str(worst_row["seed_id"])],
            ["Predicted label", str(best_row["pred_label"]), str(worst_row["pred_label"])],
            ["True label", str(best_row["true_label"]), str(worst_row["true_label"])],
            ["Winning score", f"{best_row['winning_score']:.3f}", f"{worst_row['winning_score']:.3f}"],
        ]

        xt = Table(xai_meta, colWidths=[4.0 * cm, 5.9 * cm, 5.9 * cm])
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
        story.append(Spacer(1, 0.22 * cm))

        story.append(Paragraph("Highest-confidence subgraph", styles["HeadingSmall"]))
        story.append(RLImage(str(xai_info["best_png"]), width=12.6 * cm, height=12.0 * cm))
        story.append(Spacer(1, 0.18 * cm))
        
        story.append(PageBreak())
        story.append(Spacer(1, 2.70 * cm))
        story.append(Paragraph("Lowest-confidence subgraph", styles["HeadingSmall"]))
        story.append(RLImage(str(xai_info["worst_png"]), width=12.6 * cm, height=12.0 * cm))

    doc.build(
        story,
        onFirstPage=lambda c, d: _first_page(c, d, header_path),
        onLaterPages=lambda c, d: _later_pages(c, d, header_path),
    )

    return out_pdf


def generate_sample_report(
    sdata,
    qc_df,
    graphs,
    model,
    device,
    out_dir,
    image_name="raw_image",
    labels=None,
    confidence_threshold=0.70,
    tile_size=448,
    eval_df=None,
    include_xai=True,
    xai_level=0,
    xai_only_correct=True,
    xai_tile_size=200,
    header_path="/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/Header.png",
):
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

    xai_info = None
    if include_xai and eval_df is not None:
        image_id = summary["image_id"]

        xai_info = make_sample_xai_panels(
            model=model,
            sdata=sdata,
            qc_df=qc_df,
            graphs=graphs,
            eval_df=eval_df,
            image_id=image_id,
            out_dir=out_dir,
            labels=labels,
            image_name=image_name,
            tile_size=xai_tile_size,
            level=xai_level,
            only_correct=xai_only_correct,
        )

    pdf_path = out_dir / f"{summary['sample_id']}_research_report.pdf"

    build_sample_pdf_report(
        summary=summary,
        overlay_png=overlay_png,
        out_pdf=pdf_path,
        xai_info=xai_info,
        header_path=header_path,
    )

    return pdf_path, summary


###### XAI

def make_sample_xai_panels(
    model,
    sdata,
    qc_df,
    graphs,
    eval_df,
    image_id,
    out_dir,
    labels=None,
    image_name="raw_image",
    tile_size=200,
    level=0,
    only_correct=True,
):
    if labels is None:
        labels = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_row, worst_row = select_best_worst_graph_rows_for_image(
        eval_df=eval_df,
        image_id=image_id,
        only_correct=only_correct,
        level=level,
    )

    best_graph = get_graph_by_seed_id(graphs, best_row["seed_id"])
    worst_graph = get_graph_by_seed_id(graphs, worst_row["seed_id"])

    best_png = save_xai_plot_for_graph(
        model=model,
        sdata=sdata,
        qc_df=qc_df,
        graph_data=best_graph,
        device=device,
        target_idx=int(best_row["pred_class"]),
        out_path=out_dir / "xai_best.png",
        image_name=image_name,
        tile_size=tile_size,
    )

    worst_png = save_xai_plot_for_graph(
        model=model,
        sdata=sdata,
        qc_df=qc_df,
        graph_data=worst_graph,
        device=device,
        target_idx=int(worst_row["pred_class"]),
        out_path=out_dir / "xai_worst.png",
        image_name=image_name,
        tile_size=tile_size,
    )

    return {
        "best_row": best_row,
        "worst_row": worst_row,
        "best_png": best_png,
        "worst_png": worst_png,
    }


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
    figsize=(9.2, 8.8),
    tile_alpha=0.35,
    edge_linewidth=2.0,
    point_size=16,
    node_cmap="magma",
    edge_cmap="viridis",
    return_figure=False,
):
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

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

    node_importance = np.asarray(node_importance, dtype=float)
    edge_attention = np.asarray(edge_attention, dtype=float)

    if np.max(node_importance) == np.min(node_importance):
        node_importance = np.zeros_like(node_importance)

    if np.max(edge_attention) == np.min(edge_attention):
        edge_attention = np.zeros_like(edge_attention)

    node_norm = Normalize(vmin=float(np.min(node_importance)), vmax=float(np.max(node_importance) + 1e-12))
    edge_norm = Normalize(vmin=float(np.min(edge_attention)), vmax=float(np.max(edge_attention) + 1e-12))

    node_cmap_obj = mpl_cm.get_cmap(node_cmap)
    edge_cmap_obj = mpl_cm.get_cmap(edge_cmap)

    # Explicit layout: main image + two separated colorbars
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.06, 0.08, 0.72, 0.84])     # main image
    cax_edge = fig.add_axes([0.81, 0.22, 0.022, 0.42])
    cax_node = fig.add_axes([0.86, 0.22, 0.022, 0.42])

    ax.imshow(img_np)

    half = tile_size / 2.0

    # tiles = integrated gradients
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

    # edges = attention
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
            alpha=0.95,
            zorder=3,
        )

    xs = [coord_map[pid]["x"] for pid in node_ids if pid in coord_map]
    ys = [coord_map[pid]["y"] for pid in node_ids if pid in coord_map]
    ax.scatter(xs, ys, c="red", s=point_size, zorder=4)

    ax.axis("off")
    ax.set_title(
        "XAI overlay: tiles = integrated gradients, edges = attention",
        fontsize=9,
        pad=6
    )

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

from pathlib import Path
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
    tile_alpha: float = 0.2,
    figsize=(10, 10),
    ig_mode: str = "abs_mean",
    ig_steps: int = 50,
    baseline: str = "zeros",
):
    # attention
    att_res = extract_attention_for_graph(model, graph_data, device)
    edge_index_att, edge_attention = aggregate_attention_last_layer(att_res["attention_weights"])

    # integrated gradients
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

    # normalize node importance for cleaner display
    node_importance = np.asarray(node_importance, dtype=float)
    if np.max(node_importance) > np.min(node_importance):
        node_importance = (node_importance - np.min(node_importance)) / (np.max(node_importance) - np.min(node_importance) + 1e-12)

    # create figure
    fig = plot_xai_on_image(
        sdata=sdata,
        qc_df=qc_df,
        graph_data=graph_data,
        node_importance=node_importance,
        edge_index=edge_index_att,
        edge_attention=edge_attention,
        image_name=image_name,
        tile_size=tile_size,
        tile_alpha=tile_alpha,
        figsize=figsize,
        return_figure=True,   # important, see updated plotting function below
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path

import pandas as pd


def select_best_worst_graph_rows_for_image(
    eval_df: pd.DataFrame,
    image_id: str,
    only_correct: bool = True,
    level: int = 0,
):
    df = eval_df.copy()
    df = df[df["image_id"].astype(str) == str(image_id)].copy()

    if only_correct:
        df = df[df["correct"] == 1].copy()

    if len(df) == 0:
        raise ValueError(f"No graphs found for image_id={image_id} with only_correct={only_correct}")

    df_desc = df.sort_values("winning_score", ascending=False).reset_index(drop=True)
    df_asc = df.sort_values("winning_score", ascending=True).reset_index(drop=True)

    if level >= len(df_desc):
        raise IndexError(f"Requested level={level}, but only {len(df_desc)} graphs available.")

    best_row = df_desc.iloc[level]
    worst_row = df_asc.iloc[level]

    return best_row, worst_row

