"""Report generation for the iSTAMP Streamlit app: HTML, Excel (xlsx) and PDF bytes."""
import io
import datetime
import numpy as np
import pandas as pd

from istamp_backend import CLASSES, SHORT

MODEL_VERSION = "iSTAMP RamanGATv2MultiBias v1.0 (leakage-corrected release)"
DISCLAIMER = ("RESEARCH USE ONLY — not a clinical or diagnostic device. Predictions are "
              "leakage-corrected model estimates; confidence is uncalibrated softmax.")


def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


def biopsy_table(biopsies):
    rows = []
    for b in biopsies:
        r = {"biopsy": b["name"], "image_id": b.get("image_id"), "status": b["status"],
             "n_subgraphs": b.get("n_subgraphs"), "predicted": b.get("pred_label"),
             "confidence": b.get("conf")}
        for c in CLASSES:
            r[f"prob_{SHORT[c]}"] = b["probs"][CLASSES.index(c)] if b.get("probs") is not None else None
        rows.append(r)
    return pd.DataFrame(rows)


def build_excel(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame([{"patient_id": patient_id, "predicted_subtype": patient_label,
                       "confidence": patient_conf, "n_biopsies": len(biopsies),
                       "model": MODEL_VERSION, "generated": _now()}]).to_excel(xl, sheet_name="summary", index=False)
        if patient_probs is not None:
            pd.DataFrame({"subtype": [SHORT[c] for c in CLASSES],
                          "probability": patient_probs}).to_excel(xl, sheet_name="patient_probabilities", index=False)
        biopsy_table(biopsies).to_excel(xl, sheet_name="per_biopsy", index=False)
        if qc:
            pd.DataFrame([qc]).to_excel(xl, sheet_name="qc", index=False)
    buf.seek(0)
    return buf.getvalue()


def _b64(png):
    import base64
    return "data:image/png;base64," + base64.b64encode(png).decode() if png else ""


def build_html(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc,
               interpretation="", imgs=None):
    imgs = imgs or {}
    def bar(p):
        return (f'<div style="background:#eee;border-radius:3px;width:180px;display:inline-block;'
                f'vertical-align:middle"><div style="width:{p*100:.0f}%;background:#0072B2;'
                f'height:12px;border-radius:3px"></div></div> {p*100:.1f}%')
    prob_rows = ""
    if patient_probs is not None:
        for c, p in zip(CLASSES, patient_probs):
            prob_rows += f"<tr><td>{SHORT[c]}</td><td>{bar(p)}</td></tr>"
    bt = biopsy_table(biopsies).to_html(index=False, float_format=lambda x: f"{x:.3f}")
    qc_html = "".join(f"<li>{k}: {v}</li>" for k, v in (qc or {}).items())
    # per-biopsy image gallery (preprocessing + XAI) so the report shows how predictions were made
    gallery = ""
    for b in biopsies:
        im = imgs.get(b["name"], {})
        if not (im.get("prep") or im.get("xai")):
            continue
        gallery += f'<h3>{b["name"]} — {b.get("pred_label","")} ({(b.get("conf") or 0)*100:.0f}%)</h3>'
        if im.get("prep"):
            gallery += (f'<p><i>Preprocessing &amp; graph construction</i><br>'
                        f'<img src="{_b64(im["prep"])}" style="max-width:100%"></p>')
        if im.get("xai"):
            gallery += (f'<p><i>Interpretability — integrated gradients (tiles) + attention (edges)</i><br>'
                        f'<img src="{_b64(im["xai"])}" style="max-width:70%"></p>')
    gallery_section = f"<h2>Images — preprocessing &amp; interpretability</h2>{gallery}" if gallery else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>iSTAMP report {patient_id}</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;color:#222;max-width:820px;margin:24px auto;padding:0 16px}}
h1{{font-size:20px}} h2{{font-size:15px;border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:22px}}
table{{border-collapse:collapse;font-size:12px}} td,th{{padding:3px 8px;border:1px solid #e3e3e3}}
.badge{{display:inline-block;background:#0072B2;color:#fff;padding:4px 10px;border-radius:4px;font-weight:bold}}
.disc{{color:#a33;font-size:11px;margin-top:24px}}</style></head><body>
<h1>iSTAMP molecular subtype report</h1>
<p><b>Patient:</b> {patient_id} &nbsp; <b>Generated:</b> {_now()} &nbsp; <b>Model:</b> {MODEL_VERSION}</p>
<h2>Patient-level prediction</h2>
<p><span class="badge">{patient_label or 'n/a'}</span> &nbsp; confidence {('%.1f%%'%(patient_conf*100)) if patient_conf else 'n/a'}
&nbsp;(mean over {len(biopsies)} biopsies)</p>
<table><tr><th>Subtype</th><th>Probability</th></tr>{prob_rows}</table>
<h2>Per-biopsy results</h2>{bt}
<h2>Quality control</h2><ul>{qc_html}</ul>
<h2>Interpretation</h2><p>{interpretation}</p>
{gallery_section}
<p class="disc">{DISCLAIMER}</p></body></html>"""


def build_pdf(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc,
              interpretation="", imgs=None):
    imgs = imgs or {}
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    Image as RLImage)
    from reportlab.lib.styles import getSampleStyleSheet
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=18 * mm, bottomMargin=16 * mm)
    st = getSampleStyleSheet()
    e = [Paragraph("iSTAMP molecular subtype report", st["Title"]),
         Paragraph(f"Patient: <b>{patient_id}</b> &nbsp; Generated: {_now()}", st["Normal"]),
         Paragraph(f"Model: {MODEL_VERSION}", st["Normal"]), Spacer(1, 8)]
    e.append(Paragraph(f"<b>Patient-level prediction:</b> {patient_label or 'n/a'} "
                       f"(confidence {('%.1f%%' % (patient_conf * 100)) if patient_conf else 'n/a'}, "
                       f"mean over {len(biopsies)} biopsies)", st["Normal"]))
    e.append(Spacer(1, 6))
    if patient_probs is not None:
        data = [["Subtype", "Probability"]] + [[SHORT[c], f"{p*100:.1f}%"]
                                               for c, p in zip(CLASSES, patient_probs)]
        t = Table(data, colWidths=[80, 80])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0072B2")),
                               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        e.append(t); e.append(Spacer(1, 10))
    bt = biopsy_table(biopsies)
    bdata = [list(bt.columns)] + bt.astype(object).where(pd.notnull(bt), "").values.tolist()
    bdata = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row] for row in bdata]
    tb = Table(bdata)
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
                            ("FONTSIZE", (0, 0), (-1, -1), 6.5)]))
    e.append(Paragraph("<b>Per-biopsy results</b>", st["Heading3"])); e.append(tb)
    # per-biopsy image gallery: preprocessing + XAI (how the prediction was made)
    for b in biopsies:
        im = imgs.get(b["name"], {})
        if not (im.get("prep") or im.get("xai")):
            continue
        e.append(Spacer(1, 8))
        e.append(Paragraph(f"<b>{b['name']} — {b.get('pred_label','')} "
                           f"({(b.get('conf') or 0)*100:.0f}%)</b>", st["Heading3"]))
        if im.get("prep"):
            e.append(Paragraph("Preprocessing &amp; graph construction", st["Normal"]))
            try:
                e.append(RLImage(io.BytesIO(im["prep"]), width=165 * mm, height=57 * mm))
            except Exception:
                pass
        if im.get("xai"):
            e.append(Paragraph("Interpretability — integrated gradients (tiles) + attention (edges)", st["Normal"]))
            try:
                e.append(RLImage(io.BytesIO(im["xai"]), width=115 * mm, height=122 * mm))
            except Exception:
                pass
    if qc:
        e.append(Spacer(1, 8)); e.append(Paragraph("<b>Quality control</b>", st["Heading3"]))
        e.append(Paragraph("; ".join(f"{k}: {v}" for k, v in qc.items()), st["Normal"]))
    if interpretation:
        e.append(Spacer(1, 8)); e.append(Paragraph("<b>Interpretation</b>", st["Heading3"]))
        e.append(Paragraph(interpretation, st["Normal"]))
    e.append(Spacer(1, 14)); e.append(Paragraph(DISCLAIMER, st["Italic"]))
    doc.build(e)
    buf.seek(0)
    return buf.getvalue()
