# optimized_scripts — change log

These are **copies** of the five key Python scripts with light, clearly-commented
improvements. **No original file was modified.** Each copy carries a header block
pointing to the unchanged original. No scientific logic was altered; existing
functions remain byte-equivalent in behaviour. New helpers are **opt-in** and
caveats are documented inline.

| File | Original |
|------|----------|
| `models.py` | `/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/models.py` |
| `Evaluate.py` | `/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/Evaluate.py` |
| `GNN_prepare.py` | `/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/GNN_prepare.py` |
| `preprocessing.py` | `/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/preprocessing.py` |
| `hyperparameter.py` | `/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/hyperparameter.py` |

All five pass `python -m py_compile`.

## GNN_prepare.py
- Added module docstring, type hints, `import random`.
- Added `set_global_seed(seed)` for deterministic splits/training.
- **Added `make_patient_grouped_loaders(...)`** — a NEW, opt-in alternative to
  `make_train_val_loaders`. It partitions graphs by `patient_id` so that all
  subgraphs of a patient stay entirely in train OR val (never both), and returns
  a `split_info` dict with an `assert` that the patient sets are disjoint. This
  directly fixes the data-leakage path (see `leakage_assessment.md`). The
  original `make_train_val_loaders` is left untouched for reproducibility.

## models.py
- Added module docstring and `set_model_seed(seed)`.
- Documented (did **not** change) two caveats to preserve trained-weight repro:
  (1) `GradientReversalFunction`/`GradientReversalLayer` are defined twice; the
  second shadows the first (identical bodies). (2) The head outputs unbounded
  regression scores; `softmax(pred)` downstream yields **uncalibrated**
  "probabilities".

## Evaluate.py
- Added module docstring.
- **Added `bootstrap_auc_ci(...)`** — NEW helper returning AUC point estimate +
  95% bootstrap CI, which the manuscript currently lacks.
- Documented that `plot_xai_on_image` is defined twice (second shadows first)
  and that all metrics are only as trustworthy as the loader they receive (use a
  patient-grouped held-out cohort).

## preprocessing.py
- Added module docstring and `import os`.
- **Removed hardcoded absolute paths** by introducing `CONCH_MODEL_PATH` and
  `DIAGNOSTIC_MODEL_PATH` module constants read from environment variables
  (`ISTAMP_CONCH_MODEL_PATH`, `ISTAMP_DIAGNOSTIC_MODEL_PATH`); defaults reproduce
  the original values.
- **Fixed a real bug in `score_all_tiles`**: original passed an unsupported
  `transform=` kwarg to `TileInferenceDataset` and used an undefined
  `DataLoader`; the copy constructs the dataset with supported kwargs and imports
  `DataLoader` locally. (This function was dead/broken; the working path is
  `add_diagnostic_score_to_qc_df`.)
- Documented the background-seed concern: `generate_random_points` samples over
  the whole image with no tissue mask.

## hyperparameter.py
- Added module docstring.
- Extended `set_seed` with commented cuDNN-determinism toggles for bit-repro GPU
  runs.
- Documented that the same (potentially leaky) loaders are reused across all
  Optuna trials and the final fit.
