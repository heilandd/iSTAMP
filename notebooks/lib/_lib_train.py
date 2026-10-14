"""
Retraining + honest evaluation library for the MsClaude revision.

Resolves the CRITICAL data-leakage finding: the original pipeline split graphs
with torch.random_split at the SUBGRAPH level (GNN_prepare.make_train_val_loaders),
so near-duplicate subgraphs from the same patient straddled train/val. The
Methods text claims patient-level splitting. Here we implement BOTH split modes
so the optimism gap can be quantified, and we report honest patient-disjoint
held-out metrics with bootstrap CIs and calibration.

All model outputs are exact; nothing is fabricated.
"""
import os, sys, glob, json, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lib_raman as L
from torch_geometric.loader import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score, confusion_matrix, roc_curve,
                             precision_recall_curve)

CLASSES = L.CLASSES
META = pd.read_csv(L.META_PATH)
# per-image -> patient, center, round
_img2meta = META.drop_duplicates("image_id").set_index("image_id")

CENTERS = sorted(META["Klinik"].dropna().unique().tolist())
CEN2I = {c: i for i, c in enumerate(CENTERS)}


def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assemble_graphs(hop=3, graph_set="small", limit=None):
    """Load all subgraphs at a given hop; attach patient/center/round/image labels."""
    if graph_set == "small":
        files = sorted(glob.glob(os.path.join(L.DATA, "Img_*_pyg.pt")))
    else:
        files = sorted(glob.glob(os.path.join(L.DATA, "Img_*_pyg_largNN.pt")))
    if limit:
        files = files[:limit]
    graphs = []
    pat_list = sorted(META["Sample.Number.internal"].dropna().unique().tolist())
    pat2i = {p: i for i, p in enumerate(pat_list)}
    for f in files:
        d = L.load_graph_dict(f)
        pid = d.get("patient_id")
        img = d.get("image_id")
        gl = d.get("pyg_dataset_by_hop", {}).get(hop, None)
        if not gl:
            continue
        row = _img2meta.loc[img] if img in _img2meta.index else None
        center = row["Klinik"] if row is not None else None
        rnd = row["Info"] if row is not None else None
        for g in gl:
            g = g.clone()
            g.patient_id = pid
            g.image_id = img
            g.center = center
            g.center_idx = torch.tensor(CEN2I.get(center, 0), dtype=torch.long)
            # sample_type: ROUND_I (primary-ish) vs ROUND_II -> 0/1
            g.sample_type_idx = torch.tensor(0 if rnd == "ROUND_I" else 1, dtype=torch.long)
            graphs.append(g)
    return graphs


def stratified_patient_split(graphs, test_frac=0.30, seed=42):
    """Patient-disjoint split, stratified by dominant true class per patient."""
    rng = np.random.RandomState(seed)
    # dominant class per patient
    pat_cls = {}
    for g in graphs:
        y = np.asarray(g.y).reshape(-1)[:5]
        pat_cls.setdefault(g.patient_id, []).append(int(np.argmax(y)))
    pat_dom = {p: int(np.bincount(v).argmax()) for p, v in pat_cls.items()}
    test_pats = set()
    by_cls = {}
    for p, c in pat_dom.items():
        by_cls.setdefault(c, []).append(p)
    for c, ps in by_cls.items():
        ps = np.array(sorted(ps)); rng.shuffle(ps)
        k = int(round(test_frac * len(ps)))
        test_pats.update(ps[:k].tolist())
    train = [g for g in graphs if g.patient_id not in test_pats]
    test = [g for g in graphs if g.patient_id in test_pats]
    return train, test, test_pats


def random_subgraph_split(graphs, test_frac=0.30, seed=42):
    """LEAKY split used by the original code (random_split at subgraph level)."""
    g = torch.Generator().manual_seed(seed)
    n = len(graphs); n_test = int(round(test_frac * n))
    perm = torch.randperm(n, generator=g).tolist()
    test_idx = set(perm[:n_test])
    train = [graphs[i] for i in range(n) if i not in test_idx]
    test = [graphs[i] for i in range(n) if i in test_idx]
    return train, test, None


def build_model(device, bias=True):
    from models import RamanGATv2MultiBias, RamanGATv2
    p = torch.load(L.MODEL_PATH, weights_only=False, map_location="cpu")["params"]
    hidden = p["hidden_multiplier"] * p["num_heads"]
    if bias:
        m = RamanGATv2MultiBias(
            in_channels=512, hidden_channels=hidden, out_channels=5, edge_dim=1,
            num_heads=p["num_heads"], num_layers=p["num_layers"], dropout=p["dropout"],
            use_batch_norm=p["use_batch_norm"],
            bias_classes={"Center": len(CENTERS), "sample_type": 2},
            bias_hidden_dim=p["bias_hidden_dim"], grl_alpha=p["grl_alpha"])
    else:
        m = RamanGATv2(in_channels=512, hidden_channels=hidden, out_channels=5, edge_dim=1,
                       num_heads=p["num_heads"], num_layers=p["num_layers"], dropout=p["dropout"],
                       use_batch_norm=p["use_batch_norm"])
    return m.to(device), p


def _attach_bias(batch):
    # map graph-level center_idx/sample_type_idx to attributes the model expects
    batch.bias_Center = batch.center_idx
    batch.bias_sample_type = batch.sample_type_idx


def train_model(train_graphs, val_graphs, device, params, epochs=120, patience=15,
                lambda_bias=None, bias=True, lr=None, batch_size=32, seed=42, log=print):
    set_seed(seed)
    model, _ = build_model(device, bias=bias)
    lr = lr or params["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    crit = nn.MSELoss(); bias_crit = nn.CrossEntropyLoss()
    if lambda_bias is None:
        lambda_bias = {"Center": 0.02, "sample_type": 0.09}
    tl = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_graphs, batch_size=64, shuffle=False)
    best = float("inf"); best_state = None; pc = 0; hist = []
    for ep in range(epochs):
        model.train(); tr = 0.0
        for b in tl:
            b = b.to(device); opt.zero_grad()
            out = model(b)
            y = b.y.float().view_as(out["pred"])
            loss = crit(out["pred"], y)
            if bias and "bias_logits" in out:
                _attach_bias(b)
                for bn, lg in out["bias_logits"].items():
                    lab = getattr(b, f"bias_{bn}").long()
                    loss = loss + lambda_bias.get(bn, 0.05) * bias_crit(lg, lab)
            loss.backward(); opt.step(); tr += loss.item()
        tr /= max(1, len(tl))
        model.eval(); vloss = 0.0
        with torch.no_grad():
            for b in vl:
                b = b.to(device)
                out = model(b); y = b.y.float().view_as(out["pred"])
                vloss += crit(out["pred"], y).item()
        vloss /= max(1, len(vl))
        sched.step(vloss); hist.append({"epoch": ep + 1, "train_mse": tr, "val_mse": vloss})
        if vloss < best - 1e-5:
            best = vloss; best_state = copy.deepcopy(model.state_dict()); pc = 0
        else:
            pc += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  epoch {ep+1}: train_mse={tr:.4f} val_mse={vloss:.4f} best={best:.4f}")
        if pc >= patience:
            log(f"  early stop at epoch {ep+1}")
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, graphs, device, batch_size=64):
    model.eval()
    dl = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    preds = []; rows = []
    ptr = 0
    for b in dl:
        b = b.to(device)
        out = model(b)
        p = out["pred"].cpu().numpy()
        preds.append(p)
    preds = np.vstack(preds)
    # identifiers in original order
    for i, g in enumerate(graphs):
        y = np.asarray(g.y).reshape(-1)[:5]
        rows.append({"patient_id": g.patient_id, "image_id": g.image_id,
                     "center": getattr(g, "center", None),
                     "y_true": int(np.argmax(y))})
    df = pd.DataFrame(rows)
    sm = L.softmax(preds)
    for ci, c in enumerate(CLASSES):
        df[f"score_{c}"] = preds[:, ci]
        df[f"prob_{c}"] = sm[:, ci]
    df["y_pred"] = preds.argmax(1)
    df["conf"] = sm.max(1)
    return df


def bootstrap_auc(y_bin, scores, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    N = len(y_bin); aucs = []
    if len(np.unique(y_bin)) < 2:
        return (np.nan, np.nan, np.nan)
    for _ in range(n):
        idx = rng.randint(0, N, N)
        if len(np.unique(y_bin[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_bin[idx], scores[idx]))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    return (float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def per_class_metrics(df, level="subgraph"):
    """Compute OvR AUC/PR-AUC + bootstrap CI per class on a predictions dataframe."""
    out = []
    yt = df["y_true"].values
    for ci, c in enumerate(CLASSES):
        ybin = (yt == ci).astype(int)
        sc = df[f"score_{c}"].values
        n_pos = int(ybin.sum())
        if len(np.unique(ybin)) < 2:
            out.append({"class": c, "level": level, "n_pos": n_pos, "auc": np.nan,
                        "auc_lo": np.nan, "auc_hi": np.nan,
                        "pr_auc": np.nan})
            continue
        auc = roc_auc_score(ybin, sc)
        m, lo, hi = bootstrap_auc(ybin, sc)
        pr = average_precision_score(ybin, sc)
        out.append({"class": c, "level": level, "n_pos": n_pos, "auc": float(auc),
                    "auc_boot_mean": m, "auc_lo": lo, "auc_hi": hi, "pr_auc": float(pr)})
    return pd.DataFrame(out)


def aggregate_patient(df):
    """Mean score per patient -> patient-level predictions dataframe."""
    score_cols = [f"score_{c}" for c in CLASSES]
    agg = df.groupby("patient_id").agg({**{sc: "mean" for sc in score_cols},
                                        "y_true": "first", "center": "first"}).reset_index()
    sm = L.softmax(agg[score_cols].values)
    for ci, c in enumerate(CLASSES):
        agg[f"prob_{c}"] = sm[:, ci]
    agg["y_pred"] = agg[score_cols].values.argmax(1)
    agg["conf"] = sm.max(1)
    return agg


def calibration(df, n_bins=10):
    """Reliability of max-prob confidence vs accuracy; returns bins + ECE."""
    conf = df["conf"].values
    correct = (df["y_pred"].values == df["y_true"].values).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []; ece = 0.0; N = len(conf)
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum() == 0:
            rows.append({"bin_lo": bins[i], "bin_hi": bins[i + 1], "n": 0,
                         "mean_conf": np.nan, "acc": np.nan}); continue
        mc = conf[m].mean(); ac = correct[m].mean()
        ece += (m.sum() / N) * abs(ac - mc)
        rows.append({"bin_lo": bins[i], "bin_hi": bins[i + 1], "n": int(m.sum()),
                     "mean_conf": float(mc), "acc": float(ac)})
    return pd.DataFrame(rows), float(ece)


def overall_metrics(df):
    yt, yp = df["y_true"].values, df["y_pred"].values
    return {"accuracy": float(accuracy_score(yt, yp)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(yt, yp, average="weighted", zero_division=0)),
            "n": int(len(df))}
