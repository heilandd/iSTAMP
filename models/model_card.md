# Model card — iSTAMP RamanGATv2MultiBias
- **Architecture:** GATv2 (in=512 CONCH features, hidden=256, 8 heads, 4 layers, mean+max pooling,
  5-output regression head) with domain-adversarial heads (gradient reversal) for centre, sample
  type, image and patient confounders.
- **Task:** regress 5 calibrated methylation-subtype scores (argmax = class).
- **Training data:** 1,295 SRH images / 236 IDH-wildtype glioblastoma patients / 4 European centres
  (UKF, UKH, MUV, UKK); methylation labels from the Heidelberg classifier (v12.8).
- **Intended use:** RESEARCH ONLY — methodology demonstration; not a clinical/diagnostic device.
- **Metrics:** reported under **patient-disjoint** evaluation (leakage-corrected). See manuscript.
- **Limitations:** rare subtypes (PNC, MES-ATYP; n≤5 patients) are unreliable; performance is
  lower than earlier (leaky) estimates; requires CONCH embeddings for raw new images; not
  prospectively validated.
