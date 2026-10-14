# Troubleshooting
- **`scatter_reduce` not implemented on MPS** → run on CPU (`device="cpu"`) or set
  `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **torch_geometric import errors** → install PyG wheels matching your torch/CUDA.
- **Qt fails to start headless** → set `QT_QPA_PLATFORM=offscreen`.
- **Raw image inference unavailable** → CONCH embeddings required; use precomputed `.pt` graphs.
- **Model keys missing on load** → ensure `bias_classes` matches the checkpoint (Center=4, sample_type=2, image_id=1274, patient_id=233) and `load_state_dict(strict=False)`.
