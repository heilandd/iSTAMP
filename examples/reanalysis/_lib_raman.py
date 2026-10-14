"""
Reusable library for the MsClaude Raman revision reanalysis.

Loads the trained RamanGATv2MultiBias model and the per-image PyG graph caches,
runs inference, and returns 5-class methylation-subtype score predictions WITH
identifiers (patient_id, image_id, hop, n_nodes). All model outputs are exact
re-runs of the trained network; no values are fabricated.

Class index order (from Analysis package Evaluate.py defaults):
    0 GBM_MES_TYP, 1 GBM_MES_ATYP, 2 GBM_RTK1, 3 GBM_RTK2, 4 GBM_PNC
"""
import os, sys, glob, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

ROOT = "/Users/henrikheiland/Desktop/Project_Raman"
PKG = os.path.join(ROOT, "Raman_Project", "Package")
DATA = os.path.join(ROOT, "Revision", "data")
MODEL_PATH = os.path.join(ROOT, "Revision", "best_model_full.pt")
META_PATH = os.path.join(ROOT, "Revision", "meta.csv")

CLASSES = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]
BIAS_CLASSES = {"Center": 4, "sample_type": 2, "image_id": 1274, "patient_id": 233}

if PKG not in sys.path:
    sys.path.insert(0, PKG)


def softmax(z, axis=-1):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def load_model(device="cpu"):
    from models import RamanGATv2MultiBias
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    p = ckpt["params"]
    hidden = p["hidden_multiplier"] * p["num_heads"]
    model = RamanGATv2MultiBias(
        in_channels=512,
        hidden_channels=hidden,
        out_channels=5,
        edge_dim=1,
        num_heads=p["num_heads"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
        use_batch_norm=p["use_batch_norm"],
        bias_classes=BIAS_CLASSES,
        bias_hidden_dim=p["bias_hidden_dim"],
        grl_alpha=p["grl_alpha"],
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval().to(device)
    return model, ckpt, missing, unexpected


def _add_batch(data, device):
    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    return data.to(device)


@torch.no_grad()
def infer_graph(model, data, device="cpu"):
    """Return (pred_scores[5], n_nodes) for a single PyG Data graph."""
    data = _add_batch(data, device)
    out = model(data, return_attention=False)
    pred = out["pred"].cpu().numpy().reshape(-1)
    return pred, int(data.x.shape[0])


def iter_graph_files(kind="both"):
    """Yield (image_id, path, suffix) for cached graph files."""
    pats = []
    if kind in ("small", "both"):
        pats += sorted(glob.glob(os.path.join(DATA, "Img_*_pyg.pt")))
    if kind in ("large", "both"):
        pats += sorted(glob.glob(os.path.join(DATA, "Img_*_pyg_largNN.pt")))
    for p in pats:
        base = os.path.basename(p)
        suffix = "largNN" if "largNN" in base else "small"
        img = base.replace("_pyg_largNN.pt", "").replace("_pyg.pt", "")
        yield img, p, suffix


def load_graph_dict(path):
    return torch.load(path, weights_only=False, map_location="cpu")


def run_full_inference(device="cpu", limit=None, verbose=True):
    """
    Run inference over every cached graph, every hop, every subgraph.
    Returns a list of dict rows.
    """
    model, ckpt, missing, unexpected = load_model(device)
    rows = []
    files = list(iter_graph_files("both"))
    if limit:
        files = files[:limit]
    n = len(files)
    for fi, (img, path, suffix) in enumerate(files):
        try:
            d = load_graph_dict(path)
        except Exception as e:
            if verbose:
                print(f"  SKIP {img} {suffix}: {e}")
            continue
        patient_id = d.get("patient_id")
        image_id = d.get("image_id", img)
        by_hop = d.get("pyg_dataset_by_hop", {})
        for hop, graphs in by_hop.items():
            if graphs is None:
                continue
            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            for gi, g in enumerate(graphs):
                try:
                    pred, nn = infer_graph(model, g, device)
                except Exception as e:
                    continue
                # true target stored on graph y (5-dim score vector)
                y = getattr(g, "y", None)
                if y is not None:
                    y = np.asarray(y).reshape(-1)[:5]
                    y_true = int(np.argmax(y))
                else:
                    y_true = -1
                sm = softmax(pred)
                row = {
                    "image_id": image_id,
                    "patient_id": patient_id,
                    "graph_set": suffix,
                    "hop": int(hop),
                    "subgraph_idx": gi,
                    "n_nodes": nn,
                    "y_true": y_true,
                    "y_pred": int(np.argmax(pred)),
                    "conf": float(sm.max()),
                }
                for ci, c in enumerate(CLASSES):
                    row[f"logit_{c}"] = float(pred[ci])
                    row[f"prob_{c}"] = float(sm[ci])
                    if y is not None:
                        row[f"target_{c}"] = float(y[ci])
                rows.append(row)
        if verbose and (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{n} files, {len(rows)} rows")
    return rows, (missing, unexpected)


if __name__ == "__main__":
    m, ckpt, miss, unexp = load_model()
    print("Model loaded. missing:", len(miss), "unexpected:", len(unexp))
    print("Params:", ckpt["params"])
