# Changelog

## [1.0.0] — 2026-06-17
Initial public release accompanying the Nature Communications revision.
- iSTAMP analysis/training code (RamanGATv2 graph-attention model, domain-adversarial training).
- RamanMolecularClassifier cross-platform desktop application (PySide6).
- **Leakage-corrected evaluation**: patient-disjoint train/validation splitting; controlled
  optimism-gap experiment (random subgraph split vs patient split).
- Graph-density optimisation, sampling-saturation, confidence-threshold and per-centre analyses.
- Leave-one-centre-out external validation.
- PDF/HTML/Excel reporting engine; spatial heatmap overlays.
