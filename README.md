# iSTAMP — intraoperative Spatially-informed Tumor Architecture Mapping and Profiling

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-research--only-orange)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **Research use only — not for clinical or diagnostic use.**

iSTAMP predicts glioblastoma DNA-methylation **epigenetic subtypes** in near real time from
**stimulated Raman histology (SRH)** images, using a spatially informed **graph attention
network** over patch-level foundation-model (CONCH) embeddings. It combines local tissue
morphology with neighbourhood spatial context and provides explainable (attention + integrated
gradients) attributions.

This repository contains both:
- **Analysis & training code** (`src/istamp_analysis/`, audited copies in `src/istamp_analysis_audited/`)
- **RamanMolecularClassifier** — a cross-platform (macOS/Windows/Ubuntu) **desktop application**
  (`src/ramanmolecularclassifier/`) with image/metadata/prediction/heatmap/report panels.

### Evaluation note (important)
The public release reports **patient-disjoint (leakage-corrected)** evaluation. An earlier
internal pipeline split data at the subgraph level, which inflated metrics; see
`docs/model_inference.md` and the manuscript revision for the controlled optimism-gap analysis.

## Install
```bash
conda env create -f environment.yml && conda activate istamp
# or
pip install -r requirements.txt
```

## Quickstart
```bash
# Desktop app
python -m ramanmolecularclassifier            # (from src/, or after `pip install -e .`)
# Headless inference on a precomputed graph
python examples/quickstart.py
```
See `examples/reanalysis/` to reproduce the revision analyses (inference, leakage correction,
graph-density sweep, saturation, leave-one-centre-out external validation).



## Web app (Streamlit)
An upload-first web app is in [`streamlit_app/`](streamlit_app/): upload one or more biopsy images (single or multiple biopsies per patient), get per-biopsy and patient-level subtype predictions, integrated-gradients interpretability, and a downloadable PDF/HTML/Excel report.
```bash
cd streamlit_app && pip install -r requirements.txt && streamlit run app.py
```

## Reproducible analysis notebooks
Step-by-step Jupyter notebooks that reproduce the full revision analysis (inference, leakage-corrected training, graph-density/saturation, external validation, figures) are in
[`notebooks/`](notebooks/). Create the environment with
`conda env create -f environment_Raman_Analysis.yml` (env **`Raman_Analysis`**), then
`jupyter lab` and run `notebooks/00_environment_and_data_overview.ipynb` onward.

## Documentation
`docs/`: [installation](docs/installation.md) · [usage](docs/usage.md) ·
[input formats](docs/input_formats.md) · [model inference](docs/model_inference.md) ·
[report generation](docs/report_generation.md) · [desktop app](docs/desktop_app.md) ·
[troubleshooting](docs/troubleshooting.md)

## Model
Trained weights (`RamanGATv2MultiBias`) are included in this repository at
`models/best_model_full.pt` (~14 MB). See `models/model_card.md`.

## Citation
Straehle, Neidert, Villa, … Heiland *et al.* *Intraoperative classification of glioblastoma
through near real-time stimulated Raman scattering microscopy.* (under revision, Nature Communications, 2026).

## License
MIT (see `LICENSE`). Research use only.
