"""
Step: build master prediction table by re-running the trained iSTAMP model
across every cached image graph, every hop (graph density), every subgraph.

Outputs:
  03_reanalysis_outputs/master_predictions.csv         (raw, with identifiers)
  03_reanalysis_outputs/master_predictions_meta.csv    (merged w/ clinical meta + split tag)

Reproducible: seed 42 patient-level stratified 70:30 split (documented A1).
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lib_raman as L

OUT = HERE
np.random.seed(42)

print("Running full inference (all images, all hops)...")
rows, (miss, unexp) = L.run_full_inference(device="cpu", verbose=True)
df = pd.DataFrame(rows)
print("Total prediction rows:", len(df))
df.to_csv(os.path.join(OUT, "master_predictions.csv"), index=False)

# ---- merge clinical metadata ----
meta = pd.read_csv(L.META_PATH)
meta_cols = ["image_id", "Sample.Number.internal", "Klinik", "Info", "sex",
             "Surgery.Type", "RANO.Resect", "thisClass", "thisScore",
             "MGMTStatus", "MGMTScore", "KPS_preOP"]
meta_cols = [c for c in meta_cols if c in meta.columns]
meta_s = meta[meta_cols].drop_duplicates("image_id").rename(
    columns={"Sample.Number.internal": "patient_meta", "Klinik": "center", "Info": "round"})
m = df.merge(meta_s, on="image_id", how="left")

# ---- reproducible patient-level 70:30 stratified split (seed 42) ----
# stratify by dominant methylation class at patient level
pat = (meta.groupby("Sample.Number.internal")
       .agg(center=("Klinik", "first"), cls=("thisClass", "first"))
       .reset_index().rename(columns={"Sample.Number.internal": "patient_id"}))
rng = np.random.RandomState(42)
val_ids = []
for cls, grp in pat.groupby("cls"):
    ids = grp["patient_id"].values.copy()
    rng.shuffle(ids)
    k = int(round(0.30 * len(ids)))
    val_ids.extend(ids[:k].tolist())
val_ids = set(val_ids)
m["split"] = np.where(m["patient_meta"].isin(val_ids), "val", "train")
pat["split"] = np.where(pat["patient_id"].isin(val_ids), "val", "train")

m.to_csv(os.path.join(OUT, "master_predictions_meta.csv"), index=False)
pat.to_csv(os.path.join(OUT, "patient_split.csv"), index=False)

summary = {
    "n_rows": int(len(m)),
    "n_images": int(m["image_id"].nunique()),
    "n_patients": int(m["patient_id"].nunique()),
    "hops_small": sorted(m.loc[m.graph_set == "small", "hop"].unique().tolist()),
    "hops_large": sorted(m.loc[m.graph_set == "largNN", "hop"].unique().tolist()),
    "n_val_patients": int(pat["split"].eq("val").sum()),
    "n_train_patients": int(pat["split"].eq("train").sum()),
    "centers": m["center"].value_counts().to_dict(),
    "missing_model_keys": len(miss),
}
with open(os.path.join(OUT, "inference_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str))
print("DONE.")
