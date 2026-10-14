"""
Backend for the iSTAMP Streamlit app.

Handles: model loading, resolving uploaded SRH images to their precomputed CONCH
graphs (cohort demo), direct PyG `.pt` graph upload, inference, patient-level
aggregation, and integrated-gradients (IG) interpretability.

Paths are configurable via environment variables so the app can run anywhere:
    RAMAN_MODEL   -> best_model_full.pt
    RAMAN_DATA    -> directory of Img_*_pyg.pt graphs
    RAMAN_META    -> meta.csv (image_id <-> image filename, for image->graph resolution)

CONCH foundation-model weights are not redistributed; brand-new raw images cannot be
embedded here. Uploading a cohort image resolves to its exact precomputed graph, and
uploading a `.pt` graph always works. Nothing is fabricated — all outputs are real model runs.
"""
import io
import os
import sys
import glob
import numpy as np
import torch

# ---- configurable paths -----------------------------------------------------
_DEF_ROOT = "/Users/henrikheiland/Desktop/Project_Raman"
MODEL_PATH = os.environ.get("RAMAN_MODEL", os.path.join(_DEF_ROOT, "Revision", "best_model_full.pt"))
DATA_DIR = os.environ.get("RAMAN_DATA", os.path.join(_DEF_ROOT, "Revision", "data"))
META_PATH = os.environ.get("RAMAN_META", os.path.join(_DEF_ROOT, "Revision", "meta.csv"))

# locate models.py (shipped in the iSTAMP package)
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in [
    os.path.join(_HERE, "..", "src", "istamp_analysis"),
    os.path.join(_HERE, "..", "notebooks", "lib"),
    os.path.join(_DEF_ROOT, "Raman_Project", "Package"),
]:
    if os.path.isdir(_cand):
        sys.path.insert(0, os.path.abspath(_cand))

CLASSES = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]
SHORT = {"GBM_MES_TYP": "MES", "GBM_MES_ATYP": "MES-ATYP", "GBM_RTK1": "RTK1",
         "GBM_RTK2": "RTK2", "GBM_PNC": "PNC"}
BIAS_CLASSES = {"Center": 4, "sample_type": 2, "image_id": 1274, "patient_id": 233}


def softmax(z):
    z = np.asarray(z, float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ---- model ------------------------------------------------------------------
def load_model(device="cpu"):
    from models import RamanGATv2MultiBias  # noqa
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    p = ckpt["params"]
    hidden = p["hidden_multiplier"] * p["num_heads"]
    model = RamanGATv2MultiBias(
        in_channels=512, hidden_channels=hidden, out_channels=5, edge_dim=1,
        num_heads=p["num_heads"], num_layers=p["num_layers"], dropout=p["dropout"],
        use_batch_norm=p["use_batch_norm"], bias_classes=BIAS_CLASSES,
        bias_hidden_dim=p["bias_hidden_dim"], grl_alpha=p["grl_alpha"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model


# ---- image <-> graph resolution --------------------------------------------
def build_image_index():
    """basename(jpeg) -> image_id, using meta.csv. Empty dict if meta absent."""
    idx = {}
    if os.path.exists(META_PATH):
        import pandas as pd
        m = pd.read_csv(META_PATH)
        if {"path", "image_id"} <= set(m.columns):
            for _, r in m.iterrows():
                if isinstance(r["path"], str):
                    idx[os.path.basename(r["path"]).lower()] = r["image_id"]
    return idx


def _prep_data(g, device="cpu"):
    if not hasattr(g, "batch") or g.batch is None:
        g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
    return g.to(device)


def graph_from_dict(d, hop=3):
    hops = d.get("pyg_dataset_by_hop", {})
    if hop not in hops or not hops[hop]:
        # fall back to any available hop closest to requested
        avail = sorted(hops.keys())
        hop = min(avail, key=lambda h: abs(h - hop)) if avail else None
    if hop is None:
        return None, None
    return hops[hop], hop


def resolve_upload(name, raw_bytes, hop=3):
    """
    Return dict describing the resolved biopsy:
      {status, image_id, patient_id, graphs (list of Data), hop, graph_path, image_path, note}
    status in {'graph','resolved_image','image_only'}
    """
    lname = name.lower()
    # (1) direct .pt graph upload
    if lname.endswith((".pt", ".pth")):
        d = torch.load(io.BytesIO(raw_bytes), weights_only=False, map_location="cpu")
        graphs, used = graph_from_dict(d, hop)
        return {"status": "graph", "image_id": d.get("image_id", name),
                "patient_id": d.get("patient_id"), "graphs": graphs, "hop": used,
                "graph_path": None, "image_path": d.get("image_path"),
                "note": "Loaded precomputed PyG graph."}
    # (2) image upload -> resolve to cohort graph by filename
    idx = build_image_index()
    image_id = idx.get(os.path.basename(lname))
    if image_id is None:
        # maybe user named it Img_123.jpg
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem.startswith("Img_") and os.path.exists(os.path.join(DATA_DIR, f"{stem}_pyg.pt")):
            image_id = stem
    if image_id:
        gp = os.path.join(DATA_DIR, f"{image_id}_pyg.pt")
        if os.path.exists(gp):
            d = torch.load(gp, weights_only=False, map_location="cpu")
            graphs, used = graph_from_dict(d, hop)
            return {"status": "resolved_image", "image_id": image_id,
                    "patient_id": d.get("patient_id"), "graphs": graphs, "hop": used,
                    "graph_path": gp, "image_path": d.get("image_path"),
                    "note": "Matched to precomputed CONCH graph for this cohort image."}
    # (3) truly new image — cannot embed without CONCH
    return {"status": "image_only", "image_id": os.path.basename(name),
            "patient_id": None, "graphs": None, "hop": None, "graph_path": None,
            "image_path": None,
            "note": ("New image: CONCH embedding required to build node features. "
                     "Viewer/QC only. Upload the precomputed .pt graph for prediction.")}


# ---- inference --------------------------------------------------------------
@torch.no_grad()
def predict_graphs(model, graphs, device="cpu"):
    """Mean logits across a biopsy's subgraphs -> (probs[5], mean_logits[5], per_subgraph)."""
    logits = []
    for g in graphs:
        out = model(_prep_data(g.clone(), device))
        logits.append(out["pred"].cpu().numpy().reshape(-1))
    logits = np.vstack(logits)
    mean_logit = logits.mean(0)
    return softmax(mean_logit), mean_logit, logits


def aggregate_patient(biopsy_logits):
    """biopsy_logits: list of mean_logit[5] per biopsy -> patient probs/logits."""
    M = np.vstack(biopsy_logits)
    ml = M.mean(0)
    return softmax(ml), ml


# ---- interpretability: integrated gradients on node features ---------------
def integrated_gradients(model, g, target_class, steps=32, device="cpu"):
    """Return per-node signed IG attribution and node positions."""
    g = g.clone()
    x = g.x.float().to(device)
    baseline = torch.zeros_like(x)
    edge_index = g.edge_index.to(device)
    edge_attr = g.edge_attr.float().to(device) if getattr(g, "edge_attr", None) is not None else None
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    total = torch.zeros_like(x)
    for a in torch.linspace(0.0, 1.0, steps):
        xi = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        d = g.clone()
        d.x = xi; d.edge_index = edge_index; d.edge_attr = edge_attr; d.batch = batch
        score = model(d)["pred"][0, target_class]
        grad = torch.autograd.grad(score, xi, retain_graph=False)[0]
        total = total + grad.detach()
    ig = (x - baseline) * total / steps            # [N,512]
    node_attr = ig.sum(dim=1).cpu().numpy()        # signed per-node importance
    pos = g.pos.cpu().numpy() if getattr(g, "pos", None) is not None else None
    return node_attr, pos


def top_bottom_nodes(node_attr, k=4):
    order = np.argsort(node_attr)
    bottom = order[:k].tolist()
    top = order[::-1][:k].tolist()
    return top, bottom


@torch.no_grad()
def get_attention(model, g, device="cpu"):
    """Last-layer GATv2 attention averaged over heads -> (edge_index[2,E], attn[E])."""
    d = _prep_data(g.clone(), device)
    out = model(d, return_attention=True)
    aw = out.get("attention_weights")
    if not aw:
        return None, None
    ei, alpha = aw[-1]                      # last attention layer
    attn = alpha.mean(dim=1).detach().cpu().numpy().reshape(-1)  # mean over heads
    ei = ei.detach().cpu().numpy()
    return ei, attn


def graph_geometry(g):
    """Return (pos[N,2], edge_index[2,E]) for drawing; edge_index without self loops kept as-is."""
    pos = g.pos.cpu().numpy() if getattr(g, "pos", None) is not None else None
    ei = g.edge_index.cpu().numpy() if getattr(g, "edge_index", None) is not None else None
    return pos, ei
