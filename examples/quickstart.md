# Quickstart examples

- **`example_inference.ipynb`** — end-to-end notebook: load the trained GATv2 model, run subgraph →
  patient-level subtype prediction, and compute integrated-gradient node attribution. Runs on a fresh
  clone using the bundled synthetic example.
- `quickstart.py` — minimal script: load one graph and print subtype probabilities.
- `data/synthetic_example_pyg.pt` — a **synthetic** format-faithful example graph (random features, no
  patient data) so the notebook/app run immediately after cloning. Predictions on it are meaningless.
- `reanalysis/` — full revision analyses (paths point to the project data; adapt `_lib_raman.DATA`).

## Running on real data
The real cohort graphs (`Img_*_pyg.pt`) and the Visium data are patient-derived and are distributed via
the Zenodo record (see `models/model_card.md`), not committed to this repository. After downloading
them, point the notebook/app at the folder:

```bash
export RAMAN_MODEL=models/best_model_full.pt
export RAMAN_DATA=/path/to/zenodo/graphs      # folder of Img_*_pyg.pt
jupyter lab examples/example_inference.ipynb
# or the interactive app:
streamlit run streamlit_app/app.py
```
