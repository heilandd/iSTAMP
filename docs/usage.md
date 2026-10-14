# Usage
## Desktop application
`python -m ramanmolecularclassifier` opens the GUI. Load a precomputed `Img_*_pyg.pt` graph (or a
TIFF/DICOM/JPEG for viewing), then inspect the prediction, heatmap, QC and report panels and export
PDF/HTML/Excel reports.
Headless: `python -m ramanmolecularclassifier --analyse <graph.pt>`.
## Analysis / reproduction
See `examples/reanalysis/`: `01_run_inference.py` (master predictions), `02_retrain_leakage.py`
(patient-disjoint vs leaky), `03_graph_density_saturation.py`, `04_loco_and_mgmt.py`.
