# iSTAMP: Intraoperative Spatially-informed Tumor Architecture Mapping & Profiling

> **Real-time, label-free molecular classification for glioblastoma** using stimulated Raman microscopy (SRH/SRS) and graph-based deep learning to predict DNA-methylation subtypes intraoperatively.

![status-badge](https://img.shields.io/badge/status-research-prototype)
![python-badge](https://img.shields.io/badge/python-3.10%2B-blue)
![license-badge](https://img.shields.io/badge/license-MIT-green)

---

## Abstract (from the manuscript)

Glioblastoma is a highly malignant brain tumor in which maximal safe resection is associated with improved survival, yet the oncological benefit of resection varies by molecular subtype. Recent work has shown that DNA methylation–defined subtypes, particularly receptor tyrosine kinase (RTK) I and II, benefit from complete CE (contrast-enriched) resection compared to mesenchymal tumors, highlighting the need for pre- or intraoperative tools that guide resection based on tumor biology. Here, we present **iSTAMP** (*intraoperative Spatially-informed Tumor Architecture Mapping and Profiling*)—a real-time, label-free molecular classification framework using stimulated Raman scattering microscopy and graph-based deep learning to predict glioblastoma epigenetic subtypes intraoperatively (within 5–7 minutes). Across 1,295 intraoperative tissue samples from 237 patients profiled with EPIC methylation arrays, our graph attention network achieved high predictive performance for all major subtypes (AUC 0.88–0.99), with spatially stable predictions across tumor regions. RTK subtypes, but not mesenchymal tumors, showed significant survival benefit from GTR (HR = 0.42, P = 6.1 × 10⁻⁶). Explainable AI revealed subtype-specific histopathological features (e.g., necrosis/macrophage infiltration in mesenchymal; glio-fibrillary matrix/axon-rich regions in RTK). Spatial transcriptomics validated cellular correlates of subtype-specific SRH features. These findings support integrating Raman-based molecular diagnostics into intraoperative workflows to guide biologically informed surgical strategies in glioblastoma.

---

## What’s in this repo

- **/istamp/**
  - `models/` — Graph Attention Network (GAT) and utilities
  - `inference/` — slide/patch → graph → subtype pipeline (5–7 min)
  - `explain/` — XAI methods (attention maps, IG, Grad-CAM for SRH)
  - `prep/` — SRH patch extraction, QC, affine alignment helpers
  - `metrics/` — AUC/PR, calibration, spatial stability
- **/configs/** — YAML configs for inference/training
- **/scripts/** — CLI entry points (`istamp-infer`, `istamp-train`, `istamp-explain`)
- **/notebooks/** — example walkthroughs (toy data)
- **/assets/** — figures, schematic diagrams (non-patient)
- **/tests/** — minimal tests for CI
- `environment.yml` — conda environment
- `LICENSE`, `README.md`

> **Note:** Clinical data and patient images are **not** included. See **Data Access** below.

---

## Key features

- **Real-time inference** (target 5–7 minutes) from SRH/SRS snapshot to subtype.
- **Graph Attention Network** integrating patch-level features + spatial context.
- **Strong performance** across major GBM methylation subtypes (AUC 0.88–0.99).
- **Explainable AI**: attention overlays, integrated gradients, Grad-CAM.
- **Spatial robustness**: stability across regions and multi-sample sites.

---

## Installation

```bash
# Clone
git clone https://github.com/<your-org>/iSTAMP.git
cd iSTAMP

# Create environment
conda env create -f environment.yml
conda activate istamp

# Or using pip (requires CUDA/cuDNN preinstalled for GPU)
pip install -e .


---

### LICENSE (MIT)

Create a file named `LICENSE` with the following contents:

```text
MIT License

Copyright (c) 2025 Dieter Henrik Heiland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
