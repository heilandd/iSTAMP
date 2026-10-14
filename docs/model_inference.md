# Model inference
Reconstruct `RamanGATv2MultiBias` (in=512, hidden=256, heads=8, layers=4, out=5, dropout=0.436,
domain-adversarial heads), load `best_model_full.pt['model_state_dict']` (strict=False → 0 missing),
build a PyG `Data` with `.x[N,512]`, `.edge_index`, `.edge_attr[E,1]`, `.batch=zeros(N)`, and read
`out["pred"]` (5 scores → softmax → class). Class order:
`[GBM_MES_TYP, GBM_MES_ATYP, GBM_RTK1, GBM_RTK2, GBM_PNC]`.
Aggregate subgraph scores by mean for image/patient-level calls.
**Evaluation must use patient-disjoint splits.** Reported headline metrics are leakage-corrected.
