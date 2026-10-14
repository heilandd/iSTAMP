"""
Reviewer-driven reanalyses from the master prediction table (existing published
iSTAMP model re-run across all images/hops):

  STEP 4 / Reviewer 1.6 : graph density (hop / NN size) sweep
  Reviewer 1.3          : images-per-patient saturation curve
  Reviewer 1.3          : F1-optimal confidence threshold (high-confidence calls)
  Reviewer 3.5 (descr.) : per-center performance breakdown

Honest evaluation restricted to the held-out validation patients
(split == 'val', patient-disjoint reproduced split, seed 42). Absolute values
carry the documented leakage caveat of the published model; the comparative
trends across hop / #images are the scientific question the reviewers asked.

Outputs under 03_reanalysis_outputs/graph_density_testing/ and /saturation/.
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _lib_raman as L
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             balanced_accuracy_score, f1_score)

CLASSES = L.CLASSES
GD = os.path.join(HERE, "graph_density_testing"); os.makedirs(GD, exist_ok=True)
SAT = os.path.join(HERE, "saturation"); os.makedirs(SAT, exist_ok=True)

df = pd.read_csv(os.path.join(HERE, "master_predictions_meta.csv"))
val = df[df["split"] == "val"].copy()
print(f"Total rows {len(df)}, held-out val rows {len(val)}, val patients {val.patient_id.nunique()}")

score_cols = [f"logit_{c}" for c in CLASSES]


def per_class_auc(d):
    yt = d["y_true"].values; out = {}
    for ci, c in enumerate(CLASSES):
        yb = (yt == ci).astype(int)
        if len(np.unique(yb)) < 2:
            out[c] = np.nan; continue
        out[c] = roc_auc_score(yb, d[f"logit_{c}"].values)
    return out


def overall(d):
    yt, yp = d["y_true"].values, d["y_pred"].values
    return dict(accuracy=accuracy_score(yt, yp),
                balanced_accuracy=balanced_accuracy_score(yt, yp),
                f1_macro=f1_score(yt, yp, average="macro", zero_division=0),
                mean_conf=float(d["conf"].mean()))

# ============ STEP 4: GRAPH DENSITY / HOP SWEEP ============
print("\n=== Graph density / hop sweep ===")
rows = []
for (gs, hop), g in val.groupby(["graph_set", "hop"]):
    aucs = per_class_auc(g); ov = overall(g)
    rows.append({"graph_set": gs, "hop": hop, "n_subgraphs": len(g),
                 "median_nodes": float(g["n_nodes"].median()),
                 "mean_nodes": float(g["n_nodes"].mean()),
                 "mean_auc": float(np.nanmean(list(aucs.values()))),
                 **{f"auc_{c}": aucs[c] for c in CLASSES}, **ov})
dens = pd.DataFrame(rows).sort_values(["graph_set", "hop"])
dens.to_csv(os.path.join(GD, "performance_by_hop.csv"), index=False)
dens.to_excel(os.path.join(GD, "performance_by_hop.xlsx"), index=False)
print(dens[["graph_set", "hop", "median_nodes", "n_subgraphs", "mean_auc",
            "balanced_accuracy", "mean_conf"]].to_string(index=False))

# runtime proxy: time inference per hop on a small sample (real measurement)
import torch
sys.path.insert(0, L.PKG)
model, ckpt, *_ = L.load_model()
rt_rows = []
for gs, suffix in [("small", "_pyg.pt"), ("largNN", "_pyg_largNN.pt")]:
    import glob
    fs = sorted(glob.glob(os.path.join(L.DATA, f"Img_*{suffix}")))[:60]
    for f in fs:
        d = L.load_graph_dict(f)
        for hop, gl in d.get("pyg_dataset_by_hop", {}).items():
            if not gl:
                continue
            g0 = gl[0]
            t0 = time.perf_counter()
            for _ in range(3):
                L.infer_graph(model, g0)
            dt = (time.perf_counter() - t0) / 3 * 1000
            rt_rows.append({"graph_set": gs, "hop": int(hop),
                            "n_nodes": int(g0.x.shape[0]), "ms_per_graph": dt})
rt = pd.DataFrame(rt_rows).groupby(["graph_set", "hop"]).agg(
    n_nodes=("n_nodes", "mean"), ms_per_graph=("ms_per_graph", "mean")).reset_index()
rt.to_csv(os.path.join(GD, "runtime_by_hop.csv"), index=False)

# stability: per-image variance of predicted class probability across subgraphs, by hop
stab_rows = []
for (gs, hop), g in val.groupby(["graph_set", "hop"]):
    # std of top-class prob within each image, averaged
    s = g.groupby("image_id")["conf"].std().mean()
    # agreement: fraction of images where all subgraphs agree on argmax class
    agree = g.groupby("image_id")["y_pred"].apply(lambda x: (x.value_counts().iloc[0] / len(x))).mean()
    stab_rows.append({"graph_set": gs, "hop": hop, "mean_within_image_conf_std": float(s),
                      "mean_subgraph_agreement": float(agree)})
stab = pd.DataFrame(stab_rows).sort_values(["graph_set", "hop"])
stab.to_csv(os.path.join(GD, "stability_by_hop.csv"), index=False)

# recommendation
best_perf = dens.loc[dens["mean_auc"].idxmax()]
best_robust = stab.loc[stab["mean_subgraph_agreement"].idxmax()]
fastest_good = rt.merge(dens, on=["graph_set", "hop"]).sort_values("ms_per_graph").iloc[0]
rec = {
    "best_performing": {"graph_set": best_perf["graph_set"], "hop": int(best_perf["hop"]),
                        "mean_auc": float(best_perf["mean_auc"]),
                        "median_nodes": float(best_perf["median_nodes"])},
    "most_robust": {"graph_set": best_robust["graph_set"], "hop": int(best_robust["hop"]),
                    "subgraph_agreement": float(best_robust["mean_subgraph_agreement"])},
    "best_deployment_fast": {"graph_set": fastest_good["graph_set"], "hop": int(fastest_good["hop"]),
                             "ms_per_graph": float(fastest_good["ms_per_graph"])},
}
with open(os.path.join(GD, "recommendation.json"), "w") as f:
    json.dump(rec, f, indent=2)
print("Recommendation:", json.dumps(rec, indent=2))

# ============ Reviewer 1.3: IMAGES-PER-PATIENT SATURATION ============
print("\n=== Images-per-patient saturation (hop 3, small graphs) ===")
# use small-graph hop 3 (primary config); aggregate subgraphs->image->patient
v3 = val[(val.graph_set == "small") & (val.hop == 3)].copy()
# per-image mean logits
img = v3.groupby(["patient_id", "image_id"]).agg(
    {**{c: "mean" for c in score_cols}, "y_true": "first"}).reset_index()
rng = np.random.RandomState(42)
sat_rows = []
maxk = 8
for k in range(1, maxk + 1):
    accs = []; corrects = []
    for rep in range(30):
        preds = []; trues = []
        for pid, grp in img.groupby("patient_id"):
            n = len(grp)
            take = grp.sample(min(k, n), random_state=rng.randint(1e9))
            mean_logit = take[score_cols].mean().values
            preds.append(int(np.argmax(mean_logit))); trues.append(int(take["y_true"].iloc[0]))
        accs.append(accuracy_score(trues, preds))
        corrects.append(balanced_accuracy_score(trues, preds))
    sat_rows.append({"n_images": k, "patient_accuracy_mean": float(np.mean(accs)),
                     "patient_accuracy_std": float(np.std(accs)),
                     "balanced_accuracy_mean": float(np.mean(corrects))})
sat = pd.DataFrame(sat_rows)
sat.to_csv(os.path.join(SAT, "images_per_patient_saturation.csv"), index=False)
sat.to_excel(os.path.join(SAT, "images_per_patient_saturation.xlsx"), index=False)
print(sat.to_string(index=False))

# ============ Reviewer 1.3: F1-OPTIMAL CONFIDENCE THRESHOLD ============
print("\n=== F1-optimal high-confidence threshold (hop3 small) ===")
thr_rows = []
for thr in np.round(np.arange(0.30, 0.96, 0.05), 2):
    hc = v3[v3["conf"] >= thr]
    cov = len(hc) / len(v3)
    if len(hc) < 5:
        continue
    thr_rows.append({"threshold": float(thr), "coverage": float(cov),
                     "accuracy": float(accuracy_score(hc["y_true"], hc["y_pred"])),
                     "f1_macro": float(f1_score(hc["y_true"], hc["y_pred"], average="macro", zero_division=0)),
                     "balanced_accuracy": float(balanced_accuracy_score(hc["y_true"], hc["y_pred"]))})
thr = pd.DataFrame(thr_rows)
thr.to_csv(os.path.join(SAT, "confidence_threshold_tradeoff.csv"), index=False)
print(thr.to_string(index=False))

# ============ Reviewer 3.5: per-center breakdown ============
print("\n=== Per-center performance (hop3 small, held-out) ===")
cen_rows = []
for cen, g in v3.groupby("center"):
    aucs = per_class_auc(g); ov = overall(g)
    cen_rows.append({"center": cen, "n_subgraphs": len(g),
                     "n_patients": g.patient_id.nunique(),
                     "mean_auc": float(np.nanmean(list(aucs.values()))), **ov})
cen = pd.DataFrame(cen_rows)
cen.to_csv(os.path.join(GD, "performance_by_center.csv"), index=False)
print(cen.to_string(index=False))
print("\nDONE 03_graph_density_saturation.py")
