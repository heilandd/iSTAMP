# iSTAMP — Streamlit app

Upload SRH biopsy images (single or multiple biopsies per patient) and get molecular
epigenetic-subtype predictions, integrated-gradients interpretability, and a downloadable
report (PDF / HTML / Excel). **Research use only — not for clinical/diagnostic use.**

## Run
```bash
pip install -r requirements.txt
# point to your data/model (defaults target the local project):
export RAMAN_MODEL=/path/to/best_model_full.pt
export RAMAN_DATA=/path/to/graphs_dir           # directory of Img_*_pyg.pt
export RAMAN_META=/path/to/meta.csv             # enables image->graph resolution
streamlit run app.py
```
Then open http://localhost:8501.

## Inputs
- **SRH images** (`.jpg/.jpeg/.png/.tif/.tiff/.dcm`): cohort images are auto-resolved to their
  precomputed CONCH graphs and fully classified. A brand-new raw image cannot be embedded here
  (CONCH weights are not redistributed) — it is shown viewer/QC-only with a clear message.
- **Precomputed graphs** (`.pt`): always classified directly.

## Workflow
1. Enter a patient ID and upload one or more biopsies.
2. Per-biopsy predictions (subtype, probabilities, confidence).
3. Patient-level prediction = mean subtype scores across biopsies (+ intra-patient concordance).
4. Integrated-gradients interpretability overlay for a selected biopsy.
5. Download the report (PDF / HTML / Excel).

## What every biopsy shows
- **SRH image** (viewer).
- **Preprocessing panels** — original (analysed region boxed) → tissue patches retained after the tissue/diagnostic gate → the spatial ego-graph, so you can see *how the data are produced*.
- **Prediction** — subtype, per-class probabilities, confidence (low-confidence flagged).
- **XAI overlay** — semi-transparent patch tiles coloured by **integrated gradients** + graph edges coloured by **attention** (colour schemes selectable in the sidebar).

Multiple biopsies are laid out in tabs (scales to many samples); all images are embedded in the PDF/HTML report.

## Notes
All predictions are real model outputs (leakage-corrected `RamanGATv2MultiBias`); confidence is
uncalibrated softmax. Calls below the sidebar threshold are flagged for review.
