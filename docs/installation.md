# Installation
## Requirements
Python 3.10+, and the packages in `requirements.txt`. A GPU is recommended for training; CPU is
sufficient for inference on precomputed graphs.
## Steps
```bash
git clone <repo-url> && cd istamp
conda env create -f environment.yml && conda activate istamp   # or pip install -r requirements.txt
pip install -e .                                                 # exposes `ramanmolecularclassifier`
```
`torch_geometric` may require matching wheels for your torch/CUDA — see the PyG install guide.
On Apple Silicon, GATv2 currently runs on CPU (MPS lacks `scatter_reduce`).
