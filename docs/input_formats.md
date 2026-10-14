# Input formats
| Format | Support | Notes |
|---|---|---|
| PyTorch-Geometric `.pt` graphs | **Primary** | dict with `pyg_dataset_by_hop`, `patient_id`, `image_id`; node features are 512-d CONCH embeddings. Full inference works on these. |
| TIFF / multi-TIFF | Viewer + metadata + QC | inference requires CONCH embedding (see below). |
| DICOM / DICOM series | Viewer + metadata (pydicom) | SRH originals are DICOM; private/vendor (NEO) tags surfaced. |
| JPEG | Viewer | converted SRH images used for overlays. |

**Raw new images** must be embedded into 512-d node features by the CONCH (ViT-B/16) foundation
model before graph inference. CONCH weights are **not redistributed**; a `CONCHEncoder` stub
documents the required interface. Until embeddings are available, raw images get viewer/metadata/QC
only.
