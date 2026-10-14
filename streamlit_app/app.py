"""
iSTAMP — Streamlit app for intraoperative glioblastoma epigenetic-subtype prediction
from stimulated Raman histology (SRH).

Upload one or more biopsy images (single or multiple biopsies per patient) or precomputed
PyG graphs. For every biopsy you get, all accessible in the UI and in the report:
  • the SRH image,
  • the preprocessing / graph-construction steps (how the data are produced),
  • per-biopsy prediction + probabilities + confidence,
  • an integrated-gradients + attention XAI overlay (how the prediction is made),
and a patient-level prediction aggregated across biopsies. Reports export to PDF/HTML/Excel.

Run:  streamlit run app.py
"""
import io
import os
import numpy as np
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import istamp_backend as B
import istamp_viz as V
import report_utils as R

HEADER_LOGO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "header.png")
try:
    from PIL import Image as _PILImage
    _PAGE_ICON = _PILImage.open(HEADER_LOGO) if os.path.exists(HEADER_LOGO) else None
except Exception:
    _PAGE_ICON = None

st.set_page_config(page_title="iSTAMP — SRH molecular classifier",
                   page_icon=_PAGE_ICON, layout="wide")

SUBTYPE_COLORS = {"GBM_MES_TYP": "#D55E00", "GBM_MES_ATYP": "#CC79A7", "GBM_RTK1": "#0072B2",
                  "GBM_RTK2": "#009E73", "GBM_PNC": "#E69F00"}


@st.cache_resource(show_spinner="Loading iSTAMP model…")
def get_model():
    return B.load_model("cpu")


@st.cache_data(show_spinner=False)
def load_display_image(path, max_px=1400):
    if not path or not os.path.exists(path):
        return None, None
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path).convert("RGB")
        scale = min(1.0, max_px / max(im.size))
        if scale < 1.0:
            im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
        return np.asarray(im), scale
    except Exception:
        return None, None


def image_from_bytes(raw, max_px=1400):
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        scale = min(1.0, max_px / max(im.size))
        if scale < 1.0:
            im = im.resize((int(im.size[0] * scale), int(im.size[1] * scale)))
        return np.asarray(im), scale
    except Exception:
        return None, None


def prob_bar(probs, title=""):
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    ax.barh([B.SHORT[c] for c in B.CLASSES], probs,
            color=[SUBTYPE_COLORS[c] for c in B.CLASSES])
    ax.set_xlim(0, 1); ax.set_xlabel("probability"); ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=9)
    for i, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.9), i, f"{p*100:.0f}%", va="center", fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
if os.path.exists(HEADER_LOGO):
    st.image(HEADER_LOGO, width=560)
else:
    st.title("iSTAMP — SRH molecular subtype classifier")
st.caption("Intraoperative Spatially-informed Tumor Architecture Mapping and Profiling · "
           "**RESEARCH USE ONLY — not for clinical/diagnostic use**")

with st.sidebar:
    st.header("Settings")
    patient_id = st.text_input("Patient ID", value="patient_001")
    hop = st.select_slider("Graph density (hop)", options=[1, 2, 3, 4, 5, 6, 7], value=3,
                           help="Neighbourhood size of the ego-graph. hop 3 ≈ 20–40 patches (default).")
    conf_thr = st.slider("High-confidence threshold", 0.20, 0.60, 0.35, 0.05)
    ig_steps = st.slider("Integrated-gradients steps", 8, 64, 32, 8)
    st.markdown("**Colour schemes**")
    ig_cmap = st.selectbox("Integrated gradients (tiles)", list(V.IG_CMAPS), index=0)
    attn_cmap = st.selectbox("Attention (edges)", list(V.ATTN_CMAPS), index=0)
    st.markdown("---")
    st.caption("Cohort SRH images auto-resolve to their precomputed CONCH graphs. Brand-new raw "
               "images need CONCH embeddings (not redistributed) → shown viewer/QC-only.")

st.subheader("1 · Upload biopsies")
st.write("Upload **one or more** biopsies for this patient — SRH images "
         "(`.jpg/.jpeg/.png/.tif/.tiff/.dcm`) or precomputed graphs (`.pt`).")
uploads = st.file_uploader("Drop biopsy files here", accept_multiple_files=True,
                           type=["jpg", "jpeg", "png", "tif", "tiff", "dcm", "pt", "pth"])
if not uploads:
    st.info("⬆️ Upload at least one biopsy to run the classifier.")
    st.stop()

model = get_model()

# ---- process every biopsy: predict + preprocessing + XAI (once) ------------
biopsies = []
prog = st.progress(0.0, text="Running iSTAMP…")
for k, up in enumerate(uploads):
    raw = up.getvalue()
    res = B.resolve_upload(up.name, raw, hop=hop)
    b = {"name": up.name, "status": res["status"], "image_id": res["image_id"],
         "patient_id": res.get("patient_id"), "image_path": res.get("image_path"),
         "note": res["note"], "probs": None, "pred_label": None, "conf": None,
         "n_subgraphs": None, "prep_png": None, "xai_png": None, "top": None, "bottom": None,
         "disp_img": None}
    # background image for display/overlays
    bg, scale = load_display_image(res.get("image_path"))
    if bg is None and up.name.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
        bg, scale = image_from_bytes(raw)
    b["disp_img"] = bg
    if res.get("graphs"):
        probs, mean_logit, _ = B.predict_graphs(model, res["graphs"], "cpu")
        tgt = int(probs.argmax())
        b.update(probs=probs, mean_logit=mean_logit, pred_label=B.SHORT[B.CLASSES[tgt]],
                 conf=float(probs.max()), n_subgraphs=len(res["graphs"]), hop=res["hop"])
        g0 = res["graphs"][0]
        pos, eig = B.graph_geometry(g0)
        # preprocessing panel
        b["prep_png"] = V.fig_to_png(V.preprocessing_panel(bg, pos, eig, scale=scale or 1.0))
        # XAI overlay (IG tiles + attention edges)
        node_ig, _ = B.integrated_gradients(model, g0, tgt, steps=ig_steps)
        ei, attn = B.get_attention(model, g0)
        b["xai_png"] = V.fig_to_png(V.xai_overlay(
            bg, pos, node_ig, ei, attn, scale=scale or 1.0,
            ig_cmap=V.IG_CMAPS[ig_cmap], attn_cmap=V.ATTN_CMAPS[attn_cmap],
            target_name=b["pred_label"], zoom=True))
        b["top"], b["bottom"] = B.top_bottom_nodes(node_ig)
    biopsies.append(b)
    prog.progress((k + 1) / len(uploads), text=f"Processed {k+1}/{len(uploads)} biopsies")
prog.empty()

predictable = [b for b in biopsies if b["probs"] is not None]

# ---- patient-level prediction ----------------------------------------------
patient_probs = patient_label = patient_conf = None
if predictable:
    st.subheader("2 · Patient-level prediction")
    patient_probs, _ = B.aggregate_patient([b["mean_logit"] for b in predictable])
    patient_label = B.SHORT[B.CLASSES[int(patient_probs.argmax())]]
    patient_conf = float(patient_probs.max())
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.metric(f"Predicted subtype · {len(predictable)} biopsies", patient_label,
                  f"{patient_conf*100:.0f}% confidence")
        (st.success if patient_conf >= conf_thr else st.warning)(
            "High-confidence patient-level call." if patient_conf >= conf_thr
            else "Below high-confidence threshold — interpret with caution / resample.")
        if len(predictable) > 1:
            het = np.array([b["probs"] for b in predictable])
            agree = float((het.argmax(1) == patient_probs.argmax()).mean())
            st.caption(f"Intra-patient concordance: {agree*100:.0f}% of biopsies agree "
                       f"with the patient-level call.")
    with c2:
        st.pyplot(prob_bar(patient_probs, "Patient-level probabilities"), clear_figure=True)

# ---- per-biopsy detail (tabs scale to many samples) ------------------------
st.subheader("3 · Per-biopsy details")
st.caption("Each tab shows the image, the preprocessing/graph steps, the prediction and the XAI overlay.")
tabs = st.tabs([f"{'✅' if b['probs'] is not None and b['conf']>=conf_thr else ('⚠️' if b['probs'] is not None else '🚫')} "
                f"{b['name'][:22]}" for b in biopsies])
for tab, b in zip(tabs, biopsies):
    with tab:
        st.markdown(f"**{b['name']}** — `{b['status']}` · image `{b.get('image_id')}`"
                    + (f" · patient `{b['patient_id']}`" if b.get('patient_id') else ""))
        if b["probs"] is None:
            if b["disp_img"] is not None:
                st.image(b["disp_img"], caption="Uploaded image (viewer only)", use_container_width=True)
            st.warning(b["note"])
            continue
        top = st.columns([1.1, 1])
        with top[0]:
            if b["disp_img"] is not None:
                st.image(b["disp_img"], caption=f"SRH image · {b['image_id']}", use_container_width=True)
        with top[1]:
            flag = "✅ high-confidence" if b["conf"] >= conf_thr else "⚠️ low-confidence"
            st.metric(b["pred_label"], f"{b['conf']*100:.0f}%", help=flag)
            st.caption(f"{b['n_subgraphs']} subgraphs · hop {b['hop']} · {flag}")
            st.pyplot(prob_bar(b["probs"]), clear_figure=True)
        with st.expander("🔬 Preprocessing & graph construction (how the data are produced)", expanded=False):
            if b["prep_png"]:
                st.image(b["prep_png"], use_container_width=True)
                st.caption("1) SRH input with the analysed region boxed · 2) tissue patches retained "
                           "after the tissue/diagnostic-score gate (background excluded) · "
                           "3) the spatial ego-graph (nodes = patches, edges = spatial proximity).")
        with st.expander("🧩 Interpretability — integrated gradients + attention (how the prediction is made)",
                         expanded=True):
            if b["xai_png"]:
                st.image(b["xai_png"], use_container_width=True)
                st.caption(f"Tiles coloured by **integrated gradients** (patch contribution to "
                           f"{b['pred_label']}); edges coloured by **attention**; red = graph nodes. "
                           f"Most influential patches: nodes {b['top']}; least: {b['bottom']}.")

# ---- report -----------------------------------------------------------------
st.subheader("4 · Report")
qc = {"biopsies_uploaded": len(biopsies), "biopsies_predicted": len(predictable),
      "biopsies_image_only": sum(1 for b in biopsies if b["probs"] is None),
      "mean_confidence": round(float(np.mean([b["conf"] for b in predictable])), 3) if predictable else None,
      "low_confidence_biopsies": sum(1 for b in predictable if b["conf"] < conf_thr),
      "graph_hop": hop}
interp = ("Patient-level subtype = mean subtype score across all biopsies (spatially stable across "
          "regions per the manuscript). Preprocessing panels show tissue-gated patch extraction and "
          "graph construction; integrated gradients + attention localise the patches driving each "
          f"call. Confidence is uncalibrated softmax; calls below {conf_thr:.2f} are flagged for review.")

if predictable:
    imgs = {b["name"]: {"prep": b["prep_png"], "xai": b["xai_png"]} for b in predictable}
    html = R.build_html(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc, interp, imgs)
    xlsx = R.build_excel(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc)
    try:
        pdf = R.build_pdf(patient_id, biopsies, patient_probs, patient_label, patient_conf, qc, interp, imgs)
    except Exception as ex:
        pdf = None; st.caption(f"(PDF export unavailable: {ex})")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ PDF report", pdf or b"", file_name=f"iSTAMP_{patient_id}.pdf",
                       mime="application/pdf", disabled=pdf is None)
    d2.download_button("⬇️ HTML report", html, file_name=f"iSTAMP_{patient_id}.html", mime="text/html")
    d3.download_button("⬇️ Excel", xlsx, file_name=f"iSTAMP_{patient_id}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("Preview HTML report (includes all images)"):
        st.components.v1.html(html, height=560, scrolling=True)
else:
    st.info("No predictable biopsies (image-only). Upload cohort images or `.pt` graphs to generate a report.")

st.markdown("---")
st.caption(R.DISCLAIMER)
