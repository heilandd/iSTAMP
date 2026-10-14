"""Report engine - PDF, HTML and Excel exports for one image analysis.

A single :func:`generate_reports` call takes the inputs of one image analysis
(metadata, prediction, QC, heatmap PNGs, runtime) and emits:

* a multi-section **PDF** (reportlab),
* a self-contained **HTML** report (images base64-embedded),
* an **Excel** workbook (probabilities, per-subgraph table, QC sheet),
* a **per-subgraph prediction CSV**,
* a **class-probability bar chart** PNG.

It models its structure on the original ``Report_v2.py`` research report but is a
clean, dependency-light re-implementation that works purely from the precomputed
graph data (no spatialdata/zarr dependency) and never fabricates results.
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .. import APP_NAME, APP_SUBTITLE, __version__
from ..core.inference import ImagePrediction
from ..core.qc import QCReport

DISCLAIMER = (
    "This report is generated automatically for research purposes only. "
    "It is NOT validated for clinical diagnosis, treatment decision-making, "
    "or patient management."
)

CLASS_COLORS = {
    "GBM_MES_TYP": "#1f77b4",
    "GBM_MES_ATYP": "#17becf",
    "GBM_RTK1": "#2ca02c",
    "GBM_RTK2": "#d62728",
    "GBM_PNC": "#9467bd",
}


@dataclass
class ReportInputs:
    """Everything needed to render the reports for one image."""

    image_id: str
    prediction: ImagePrediction
    qc: QCReport
    metadata: Dict[str, object]
    model_version: str
    graph_config: Dict[str, object]
    preprocessing: Dict[str, object]
    runtime_seconds: float
    heatmap_paths: List[Path] = field(default_factory=list)
    interpretation: str = ""
    out_dir: Path = field(default_factory=lambda: Path("."))


@dataclass
class ReportOutputs:
    pdf: Optional[Path] = None
    html: Optional[Path] = None
    xlsx: Optional[Path] = None
    csv: Optional[Path] = None
    barplot: Optional[Path] = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_interpretation(pred: ImagePrediction, qc: QCReport) -> str:
    """Generate a factual, non-fabricated interpretation paragraph."""
    conf_word = (
        "high" if pred.top_probability >= 0.6
        else "moderate" if pred.top_probability >= 0.4
        else "low"
    )
    lines = [
        f"The image-level prediction aggregates {pred.n_subgraphs} seed-centred "
        f"subgraph(s). The mean-pooled probability identifies "
        f"{pred.predicted_label} as the most likely molecular subtype "
        f"(p = {pred.predicted_probability:.3f}, margin to runner-up = "
        f"{pred.margin:.3f}, {conf_word} confidence).",
    ]
    if pred.true_label is not None:
        agree = "agrees with" if pred.true_label == pred.predicted_label else "differs from"
        lines.append(
            f"The reference soft-label for this image indicates "
            f"{pred.true_label}; the prediction {agree} the reference."
        )
    if qc.diagnostic:
        lines.append(
            "QC checks passed: spatial coverage and per-region confidence are "
            "adequate to consider this a diagnostic-quality result (research use)."
        )
    else:
        lines.append(
            "QC flagged this result as below diagnostic quality: " + qc.notes
        )
    return " ".join(lines)


def make_probability_barplot(pred: ImagePrediction, out_path: Path, figsize=(7.2, 3.6)) -> Path:
    """Bar chart of the 5 class probabilities (predicted class highlighted)."""
    out_path = Path(out_path)
    labels = pred.labels
    probs = pred.mean_probabilities
    colors_list = [CLASS_COLORS.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    bars = ax.bar(x, probs, color=colors_list, edgecolor="black", linewidth=0.6, width=0.7)
    bars[pred.predicted_index].set_edgecolor("black")
    bars[pred.predicted_index].set_linewidth(2.2)

    for xi, yi in zip(x, probs):
        ax.text(xi, yi + 0.012, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, min(1.0, float(probs.max()) + 0.12))
    ax.set_ylabel("Mean probability", fontsize=9)
    ax.set_title(
        f"Class probabilities - predicted: {pred.predicted_label} "
        f"({pred.predicted_probability:.2f})",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _per_subgraph_dataframe(pred: ImagePrediction) -> pd.DataFrame:
    rows = []
    for s in pred.subgraph_predictions:
        row = {
            "subgraph_index": s.index,
            "seed_id": s.seed_id,
            "pos_x": float(s.pos[0]),
            "pos_y": float(s.pos[1]),
            "predicted_label": pred.labels[s.predicted_index],
            "confidence": s.confidence,
        }
        for j, lab in enumerate(pred.labels):
            row[f"prob_{lab}"] = float(s.probabilities[j])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CSV + Excel
# --------------------------------------------------------------------------- #
def write_csv(inp: ReportInputs) -> Path:
    out = inp.out_dir / f"{inp.image_id}_per_subgraph_predictions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    _per_subgraph_dataframe(inp.prediction).to_csv(out, index=False)
    return out


def write_xlsx(inp: ReportInputs) -> Path:
    out = inp.out_dir / f"{inp.image_id}_report.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    pred = inp.prediction

    summary_df = pd.DataFrame(
        {
            "field": [
                "image_id", "predicted_label", "predicted_probability",
                "top_probability", "second_probability", "margin", "entropy",
                "n_subgraphs", "hop", "true_label", "model_version",
                "runtime_seconds", "generated_at",
            ],
            "value": [
                inp.image_id, pred.predicted_label,
                round(pred.predicted_probability, 5), round(pred.top_probability, 5),
                round(pred.second_probability, 5), round(pred.margin, 5),
                round(pred.entropy, 5), pred.n_subgraphs, pred.hop,
                pred.true_label, inp.model_version,
                round(inp.runtime_seconds, 3), _timestamp(),
            ],
        }
    )
    probs_df = pd.DataFrame(
        {"class": pred.labels, "mean_probability": pred.mean_probabilities}
    )
    if pred.true_target is not None:
        probs_df["reference_soft_label"] = pred.true_target
    qc_df = pd.DataFrame(
        {"metric": list(inp.qc.as_dict().keys()), "value": list(inp.qc.as_dict().values())}
    )
    meta_df = pd.DataFrame(
        {"field": list(inp.metadata.keys()), "value": [str(v) for v in inp.metadata.values()]}
    )
    per_sub_df = _per_subgraph_dataframe(pred)

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        probs_df.to_excel(writer, sheet_name="Probabilities", index=False)
        qc_df.to_excel(writer, sheet_name="QC", index=False)
        meta_df.to_excel(writer, sheet_name="Metadata", index=False)
        per_sub_df.to_excel(writer, sheet_name="PerSubgraph", index=False)
    return out


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _img_b64(path: Path) -> str:
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = Path(path).suffix.lstrip(".").lower()
    mime = "png" if ext == "png" else ("jpeg" if ext in ("jpg", "jpeg") else ext)
    return f"data:image/{mime};base64,{b64}"


def _html_table(rows: List[List[str]], header: Optional[List[str]] = None) -> str:
    out = ["<table>"]
    if header:
        out.append("<tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in header) + "</tr>")
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def write_html(inp: ReportInputs, barplot_png: Path) -> Path:
    out = inp.out_dir / f"{inp.image_id}_report.html"
    pred = inp.prediction
    qc = inp.qc

    prob_rows = [
        [lab, f"{p:.4f}"] for lab, p in zip(pred.labels, pred.mean_probabilities)
    ]
    meta_rows = [[k, str(v)] for k, v in inp.metadata.items()]
    qc_rows = [[k, str(v)] for k, v in qc.as_dict().items()]
    graph_rows = [[k, str(v)] for k, v in inp.graph_config.items()]
    prep_rows = [[k, str(v)] for k, v in inp.preprocessing.items()]

    heat_imgs = "\n".join(
        f'<figure><img src="{_img_b64(p)}" alt="heatmap"/>'
        f"<figcaption>{html.escape(Path(p).name)}</figcaption></figure>"
        for p in inp.heatmap_paths
        if Path(p).suffix.lower() == ".png"
    )

    conf_badge = "diagnostic" if qc.diagnostic else "review-required"
    badge_color = "#2ca25f" if qc.diagnostic else "#d62728"

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{html.escape(APP_NAME)} - {html.escape(inp.image_id)}</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; color:#222;
        max-width: 980px; margin: 24px auto; padding: 0 18px; }}
 h1 {{ color:#1f3b5c; margin-bottom:2px; }}
 h2 {{ color:#1f3b5c; border-bottom:2px solid #e2e8ee; padding-bottom:3px;
       margin-top:28px; }}
 .sub {{ color:#666; margin-top:0; }}
 table {{ border-collapse: collapse; width:100%; margin:8px 0; font-size:13px; }}
 th,td {{ border:1px solid #d6e1ea; padding:5px 8px; text-align:left; }}
 th {{ background:#edf3f8; }}
 .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
           color:#fff; font-weight:bold; background:{badge_color}; }}
 figure {{ margin:10px 0; }} img {{ max-width:100%; border:1px solid #ddd; }}
 figcaption {{ font-size:11px; color:#666; }}
 .disclaimer {{ background:#fff8e6; border:1px solid #f0ad00; padding:8px 12px;
                border-radius:6px; font-size:12px; }}
 footer {{ color:#888; font-size:11px; margin-top:30px; }}
</style></head><body>
<h1>{html.escape(APP_NAME)}</h1>
<p class="sub">{html.escape(APP_SUBTITLE)} &middot; v{__version__} &middot; generated {_timestamp()}</p>

<h2>Result</h2>
<p>Predicted molecular subtype:
   <strong>{html.escape(pred.predicted_label)}</strong>
   (p = {pred.predicted_probability:.3f}, margin = {pred.margin:.3f})
   &nbsp; <span class="badge">{conf_badge}</span></p>
<figure><img src="{_img_b64(barplot_png)}" alt="probabilities"/></figure>
{_html_table(prob_rows, ["Class", "Mean probability"])}

<h2>Interpretation</h2>
<p>{html.escape(inp.interpretation)}</p>

<h2>Spatial heatmaps</h2>
{heat_imgs or "<p>No heatmap available.</p>"}

<h2>Quality control</h2>
{_html_table(qc_rows, ["Metric", "Value"])}

<h2>Sample metadata</h2>
{_html_table(meta_rows, ["Field", "Value"])}

<h2>Model &amp; graph configuration</h2>
<p><strong>Model:</strong> {html.escape(inp.model_version)}</p>
{_html_table(graph_rows, ["Graph parameter", "Value"])}

<h2>Preprocessing</h2>
{_html_table(prep_rows, ["Step", "Value"])}
<p><strong>Runtime:</strong> {inp.runtime_seconds:.3f} s</p>

<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
<footer>{html.escape(APP_NAME)} {html.escape(APP_SUBTITLE)} -
 host {html.escape(platform.platform())}</footer>
</body></html>"""

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleLarge", parent=styles["Title"], fontSize=17, leading=21,
        textColor=colors.HexColor("#1f3b5c"), alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle(
        name="Subtle", parent=styles["Normal"], fontSize=9,
        textColor=colors.HexColor("#666666"), leading=11))
    styles.add(ParagraphStyle(
        name="HeadingSmall", parent=styles["Heading2"], fontSize=12, leading=14,
        textColor=colors.HexColor("#1f3b5c"), spaceAfter=5))
    styles.add(ParagraphStyle(
        name="BodySmall", parent=styles["Normal"], fontSize=9, leading=12))
    return styles


def _kv_table(data, col_widths=(5.2 * cm, 10.0 * cm)):
    t = Table([[str(k), str(v)] for k, v in data], colWidths=list(col_widths))
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#b7c7d6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d6e1ea")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(doc.leftMargin, 0.75 * cm,
                      "For research use only - not for clinical decision making")
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.75 * cm,
                           f"{APP_NAME} {APP_SUBTITLE}")


def write_pdf(inp: ReportInputs, barplot_png: Path) -> Path:
    out = inp.out_dir / f"{inp.image_id}_research_report.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    pred = inp.prediction
    qc = inp.qc
    styles = _styles()

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        rightMargin=1.6 * cm, leftMargin=1.6 * cm,
        topMargin=1.4 * cm, bottomMargin=1.3 * cm,
    )
    story = []
    story.append(Paragraph(f"{APP_NAME} - Research Classifier Report", styles["TitleLarge"]))
    story.append(Paragraph(
        f"{APP_SUBTITLE} &middot; v{__version__} &middot; generated {_timestamp()}",
        styles["Subtle"]))
    story.append(Spacer(1, 0.3 * cm))

    conf_cat = "DIAGNOSTIC QUALITY" if qc.diagnostic else "REVIEW REQUIRED / LOW QUALITY"
    story.append(Paragraph("Result summary", styles["HeadingSmall"]))
    story.append(_kv_table([
        ("Image ID", inp.image_id),
        ("Predicted molecular subtype", pred.predicted_label),
        ("Predicted probability", f"{pred.predicted_probability:.3f}"),
        ("Top probability", f"{pred.top_probability:.3f}"),
        ("Margin to runner-up", f"{pred.margin:.3f}"),
        ("Prediction entropy", f"{pred.entropy:.3f}"),
        ("Reference soft-label", pred.true_label or "n/a"),
        ("QC verdict", conf_cat),
        ("Subgraphs analysed", pred.n_subgraphs),
        ("Hop level", pred.hop),
    ]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Class probabilities", styles["HeadingSmall"]))
    story.append(RLImage(str(barplot_png), width=15.0 * cm, height=7.5 * cm))
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("Interpretation", styles["HeadingSmall"]))
    story.append(Paragraph(html.escape(inp.interpretation), styles["BodySmall"]))

    story.append(PageBreak())
    story.append(Paragraph("Quality control", styles["HeadingSmall"]))
    story.append(_kv_table(list(qc.as_dict().items())))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Sample metadata", styles["HeadingSmall"]))
    story.append(_kv_table([(k, v) for k, v in inp.metadata.items()]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Model &amp; graph configuration", styles["HeadingSmall"]))
    story.append(Paragraph(html.escape(inp.model_version), styles["BodySmall"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(_kv_table([(k, v) for k, v in inp.graph_config.items()]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph("Preprocessing &amp; runtime", styles["HeadingSmall"]))
    prep = list(inp.preprocessing.items()) + [("runtime_seconds", f"{inp.runtime_seconds:.3f}")]
    story.append(_kv_table(prep))

    png_heatmaps = [p for p in inp.heatmap_paths if Path(p).suffix.lower() == ".png"]
    if png_heatmaps:
        story.append(PageBreak())
        story.append(Paragraph("Spatial heatmaps", styles["HeadingSmall"]))
        for p in png_heatmaps:
            story.append(RLImage(str(p), width=14.5 * cm, height=12.8 * cm))
            story.append(Spacer(1, 0.2 * cm))

    story.append(PageBreak())
    story.append(Paragraph(DISCLAIMER, styles["Subtle"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def generate_reports(
    inp: ReportInputs,
    make_pdf: bool = True,
    make_html: bool = True,
    make_xlsx: bool = True,
    make_csv: bool = True,
) -> ReportOutputs:
    """Produce all requested report artefacts for one image.

    The bar chart is always rendered (it is embedded by both the PDF and HTML).
    """
    inp.out_dir = Path(inp.out_dir)
    inp.out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.interpretation:
        inp.interpretation = default_interpretation(inp.prediction, inp.qc)

    barplot = make_probability_barplot(
        inp.prediction, inp.out_dir / f"{inp.image_id}_class_probabilities.png"
    )

    outputs = ReportOutputs(barplot=barplot)
    if make_csv:
        outputs.csv = write_csv(inp)
    if make_xlsx:
        outputs.xlsx = write_xlsx(inp)
    if make_html:
        outputs.html = write_html(inp, barplot)
    if make_pdf:
        outputs.pdf = write_pdf(inp, barplot)
    return outputs
