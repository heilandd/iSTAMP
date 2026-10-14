"""
Reviewer 3.5 (Critical): leave-one-centre-out (LOCO) EXTERNAL validation.
Reviewer 4.4: exploratory intraoperative MGMT prediction from SRH.

LOCO: for each of the 4 centres, train on the other 3 centres (patient-grouped,
identical architecture/hyperparameters, hop-3) and evaluate on the held-out
centre -> genuine external generalisation estimate. All real model outputs.

MGMT: aggregate the published model's per-image scores to patient level and test
whether the learned SRH representation carries MGMT signal (logistic probe on the
5 subtype scores -> MGMT status), reported as exploratory with honest CV AUC.
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _lib_train as T
import _lib_raman as L
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(HERE, "external_validation"); os.makedirs(OUT, exist_ok=True)
DEV = "cpu"
CLASSES = T.CLASSES
params = torch.load(L.MODEL_PATH, weights_only=False, map_location="cpu")["params"]
log = print

# ============ LOCO EXTERNAL VALIDATION ============
print("=== Assembling hop-3 graphs for LOCO ===")
graphs = T.assemble_graphs(hop=3, graph_set="small")
centers = sorted(set(g.center for g in graphs if g.center is not None))
print("centres:", centers)

loco_rows = []
for held in centers:
    tr = [g for g in graphs if g.center != held and g.center is not None]
    te = [g for g in graphs if g.center == held]
    if len(te) < 20:
        continue
    print(f"\n--- LOCO: hold out {held}  (train {len(tr)} sg / test {len(te)} sg) ---")
    t0 = time.time()
    model, hist = T.train_model(tr, te, DEV, params, epochs=120, patience=15,
                                bias=True, log=lambda *a: None)
    df = T.predict(model, te, DEV)
    pat = T.aggregate_patient(df)
    # per-class AUC on held-out centre (subgraph + patient)
    for level, dd in [("subgraph", df), ("patient", pat)]:
        yt = dd["y_true"].values
        aucs = {}
        for ci, c in enumerate(CLASSES):
            yb = (yt == ci).astype(int)
            aucs[c] = roc_auc_score(yb, dd[f"score_{c}"].values) if len(np.unique(yb)) > 1 else np.nan
        loco_rows.append({"held_out_center": held, "level": level,
                          "n": len(dd), "n_patients": dd.patient_id.nunique(),
                          "mean_auc": float(np.nanmean(list(aucs.values()))),
                          **{f"auc_{c}": aucs[c] for c in CLASSES}})
    df.to_csv(os.path.join(OUT, f"loco_pred_{held}_subgraph.csv"), index=False)
    print(f"   {held}: subgraph meanAUC={loco_rows[-2]['mean_auc']:.3f} "
          f"patient meanAUC={loco_rows[-1]['mean_auc']:.3f}  ({(time.time()-t0)/60:.1f} min)")

loco = pd.DataFrame(loco_rows)
loco.to_csv(os.path.join(OUT, "loco_external_validation.csv"), index=False)
loco.to_excel(os.path.join(OUT, "loco_external_validation.xlsx"), index=False)
print("\nLOCO summary:")
print(loco[["held_out_center", "level", "n_patients", "mean_auc"]].to_string(index=False))

# ============ MGMT exploratory probe ============
print("\n=== MGMT prediction (exploratory probe on published model scores) ===")
mp = pd.read_csv(os.path.join(HERE, "master_predictions_meta.csv"))
v = mp[(mp.graph_set == "small") & (mp.hop == 3)].copy()
score_cols = [f"logit_{c}" for c in CLASSES]
# patient-level aggregation of subtype scores + MGMT label
pat = v.groupby("patient_id").agg({**{c: "mean" for c in score_cols},
                                   "MGMTStatus": "first", "MGMTScore": "first"}).reset_index()
pat = pat.dropna(subset=["MGMTStatus"])
pat["mgmt_bin"] = (pat["MGMTStatus"].str.lower() == "methylated").astype(int)
X = StandardScaler().fit_transform(pat[score_cols].values)
y = pat["mgmt_bin"].values
print(f"patients with MGMT label: {len(y)}  methylated={y.sum()} unmeth={len(y)-y.sum()}")
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tri, tei in skf.split(X, y):
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(X[tri], y[tri])
    oof[tei] = clf.predict_proba(X[tei])[:, 1]
mgmt_auc = roc_auc_score(y, oof)
# bootstrap CI
rng = np.random.RandomState(42); bs = []
for _ in range(2000):
    idx = rng.randint(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2:
        continue
    bs.append(roc_auc_score(y[idx], oof[idx]))
mgmt = {"n_patients": int(len(y)), "n_methylated": int(y.sum()),
        "cv_auc": float(mgmt_auc), "ci_lo": float(np.percentile(bs, 2.5)),
        "ci_hi": float(np.percentile(bs, 97.5)),
        "note": "Exploratory: logistic probe on the 5 subtype scores from the published model; "
                "NOT a dedicated MGMT model. Patient-level 5-fold CV."}
pd.DataFrame([mgmt]).to_csv(os.path.join(OUT, "mgmt_prediction.csv"), index=False)
with open(os.path.join(OUT, "mgmt_prediction.json"), "w") as f:
    json.dump(mgmt, f, indent=2)
pat.assign(mgmt_oof=oof).to_csv(os.path.join(OUT, "mgmt_patient_predictions.csv"), index=False)
print(json.dumps(mgmt, indent=2))
print("DONE 04_loco_and_mgmt.py")
