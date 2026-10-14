"""
Headline reanalysis: resolve the data-leakage finding (Reviewer 1 & 3).

Trains two models on hop-3 subgraphs with identical architecture/hyperparameters:
  (A) patient-grouped 70:30 split (HONEST, no leakage) -> held-out patient metrics
  (B) random subgraph split (LEAKY, reproduces original pipeline) -> optimistic metrics

Reports per-class AUC (+bootstrap 95% CI), PR-AUC, accuracy/balanced-acc/F1,
calibration (ECE), at BOTH subgraph and patient level. Quantifies the optimism gap.

Outputs under 03_reanalysis_outputs/leakage_reanalysis/.
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _lib_train as T

OUT = os.path.join(HERE, "leakage_reanalysis"); os.makedirs(OUT, exist_ok=True)
MODELS = os.path.join(HERE, "models"); os.makedirs(MODELS, exist_ok=True)
LOG = open(os.path.join(OUT, "training_log.txt"), "w")
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.write(s + "\n"); LOG.flush()

DEV = "cpu"
HOP = 3
params = torch.load(T.L.MODEL_PATH, weights_only=False, map_location="cpu")["params"]
log("=== Leakage reanalysis ===", time.strftime("%Y-%m-%d %H:%M"))
log("Assembling hop-3 graphs...")
graphs = T.assemble_graphs(hop=HOP, graph_set="small")
log(f"{len(graphs)} subgraphs, {len(set(g.patient_id for g in graphs))} patients")

results = {}

def evaluate(model, test_graphs, tag):
    df = T.predict(model, test_graphs, DEV)
    df.to_csv(os.path.join(OUT, f"pred_{tag}_subgraph.csv"), index=False)
    pat = T.aggregate_patient(df)
    pat.to_csv(os.path.join(OUT, f"pred_{tag}_patient.csv"), index=False)
    sg_m = T.per_class_metrics(df, "subgraph")
    pt_m = T.per_class_metrics(pat, "patient")
    sg_m.to_csv(os.path.join(OUT, f"metrics_{tag}_subgraph.csv"), index=False)
    pt_m.to_csv(os.path.join(OUT, f"metrics_{tag}_patient.csv"), index=False)
    cal_sg, ece_sg = T.calibration(df)
    cal_pt, ece_pt = T.calibration(pat)
    cal_sg.to_csv(os.path.join(OUT, f"calibration_{tag}_subgraph.csv"), index=False)
    cal_pt.to_csv(os.path.join(OUT, f"calibration_{tag}_patient.csv"), index=False)
    ov_sg = T.overall_metrics(df); ov_pt = T.overall_metrics(pat)
    res = {"subgraph": {**ov_sg, "ece": ece_sg,
                        "mean_auc": float(np.nanmean(sg_m["auc"]))},
           "patient": {**ov_pt, "ece": ece_pt,
                       "mean_auc": float(np.nanmean(pt_m["auc"]))}}
    log(f"[{tag}] subgraph acc={ov_sg['accuracy']:.3f} bal={ov_sg['balanced_accuracy']:.3f} "
        f"meanAUC={res['subgraph']['mean_auc']:.3f} ECE={ece_sg:.3f}")
    log(f"[{tag}] patient   acc={ov_pt['accuracy']:.3f} bal={ov_pt['balanced_accuracy']:.3f} "
        f"meanAUC={res['patient']['mean_auc']:.3f} ECE={ece_pt:.3f}")
    log(f"[{tag}] per-class AUC (subgraph):")
    for _, r in sg_m.iterrows():
        log(f"     {r['class']:>14}  n_pos={int(r['n_pos']):4d}  AUC={r['auc']:.3f}  "
            f"CI=[{r.get('auc_lo', float('nan')):.3f},{r.get('auc_hi', float('nan')):.3f}]")
    return res

# ---------- (A) HONEST: patient-grouped ----------
log("\n--- (A) HONEST patient-grouped split ---")
tr, te, test_pats = T.stratified_patient_split(graphs, test_frac=0.30, seed=42)
log(f"train {len(tr)} sg / test {len(te)} sg / {len(test_pats)} held-out patients")
t0 = time.time()
modelA, histA = T.train_model(tr, te, DEV, params, epochs=150, patience=18, bias=True, log=log)
log(f"trained in {(time.time()-t0)/60:.1f} min")
pd.DataFrame(histA).to_csv(os.path.join(OUT, "history_patientCV.csv"), index=False)
torch.save({"model_state_dict": modelA.state_dict(), "params": params, "hop": HOP,
            "split": "patient_grouped", "test_patients": sorted(test_pats)},
           os.path.join(MODELS, "model_patientCV_hop3.pt"))
results["patient_grouped_honest"] = evaluate(modelA, te, "honest")

# ---------- (B) LEAKY: random subgraph split ----------
log("\n--- (B) LEAKY random subgraph split (reproduces original) ---")
trL, teL, _ = T.random_subgraph_split(graphs, test_frac=0.30, seed=42)
log(f"train {len(trL)} sg / test {len(teL)} sg (subgraph-level random)")
t0 = time.time()
modelB, histB = T.train_model(trL, teL, DEV, params, epochs=150, patience=18, bias=True, log=log)
log(f"trained in {(time.time()-t0)/60:.1f} min")
pd.DataFrame(histB).to_csv(os.path.join(OUT, "history_leaky.csv"), index=False)
results["random_subgraph_leaky"] = evaluate(modelB, teL, "leaky")

# ---------- optimism gap ----------
gap = {
    "subgraph_meanAUC": {
        "honest": results["patient_grouped_honest"]["subgraph"]["mean_auc"],
        "leaky": results["random_subgraph_leaky"]["subgraph"]["mean_auc"],
    },
    "subgraph_balanced_accuracy": {
        "honest": results["patient_grouped_honest"]["subgraph"]["balanced_accuracy"],
        "leaky": results["random_subgraph_leaky"]["subgraph"]["balanced_accuracy"],
    },
}
results["optimism_gap"] = gap
with open(os.path.join(OUT, "leakage_results.json"), "w") as f:
    json.dump(results, f, indent=2)
log("\n=== OPTIMISM GAP ===")
log(json.dumps(gap, indent=2))
log("DONE.")
LOG.close()
