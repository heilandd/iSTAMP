"""Reconstruct and load the trained ``RamanGATv2MultiBias`` model.

The model class lives in the original research package
(``Raman_Project/Package/models.py``).  To keep this application
self-contained and cross-platform we *vendor* a clean copy of the architecture
here (:func:`_build_model`) so the app does not depend on the original source
tree being present.  The vendored definition is byte-for-byte compatible with
the trained checkpoint: ``load_state_dict(strict=False)`` reports **0 missing
keys**.

The default checkpoint path points at the revision asset, but any path may be
supplied (e.g. a bundled copy inside a PyInstaller app).  Loading is cached so
the GUI and batch paths share a single in-memory model.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

# --------------------------------------------------------------------------- #
# Default asset locations (overridable by the caller / env var).
# --------------------------------------------------------------------------- #
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "RMC_CHECKPOINT",
        "/Users/henrikheiland/Desktop/Project_Raman/Revision/best_model_full.pt",
    )
)

#: Hyper-parameters of the trained checkpoint (verified against
#: ``best_model_full.pt['params']``).
DEFAULT_MODEL_CONFIG = dict(
    in_channels=512,
    hidden_channels=256,           # hidden_multiplier (32) * num_heads (8)
    out_channels=5,
    edge_dim=1,
    num_heads=8,
    num_layers=4,
    dropout=0.43582274185083414,
    use_batch_norm=False,
    bias_classes={"Center": 4, "sample_type": 2, "image_id": 1274, "patient_id": 233},
    bias_hidden_dim=128,
    grl_alpha=0.126491638357797,
)


# --------------------------------------------------------------------------- #
# Vendored architecture (identical to RamanGATv2MultiBias).
# --------------------------------------------------------------------------- #
class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):  # noqa: D401
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class _GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.alpha)


class _BiasAdversarialHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 128, grl_alpha: float = 1.0):
        super().__init__()
        self.grl = _GradientReversalLayer(alpha=grl_alpha)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(self.grl(x))


class RamanGATv2MultiBias(nn.Module):
    """GATv2 backbone with multiple gradient-reversal bias-adversarial heads.

    Forward expects a :class:`torch_geometric.data.Data` (or batched ``Batch``)
    object with ``.x`` ``[N, in_channels]``, ``.edge_index``, ``.edge_attr``
    ``[E, edge_dim]`` and ``.batch``.  Returns a dict whose ``"pred"`` entry is
    ``[num_graphs, out_channels]`` raw class scores.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        out_channels: int = 5,
        edge_dim: int = 1,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        bias_classes: Optional[dict] = None,
        bias_hidden_dim: int = 128,
        grl_alpha: float = 1.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.bias_classes = bias_classes or {}

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    concat=True,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(
                nn.BatchNorm1d(hidden_channels) if use_batch_norm else nn.Identity()
            )

        graph_dim = hidden_channels * 2
        self.pred_head = nn.Sequential(
            nn.Linear(graph_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        self.bias_heads = nn.ModuleDict()
        for bias_name, n_classes in self.bias_classes.items():
            self.bias_heads[bias_name] = _BiasAdversarialHead(
                in_dim=graph_dim,
                n_classes=n_classes,
                hidden_dim=bias_hidden_dim,
                grl_alpha=grl_alpha,
            )

    def forward(self, data, return_attention: bool = False):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = (
            data.edge_attr.float() if getattr(data, "edge_attr", None) is not None else None
        )
        batch = data.batch

        x = self.input_proj(x)

        attention_outputs = []
        for conv, norm in zip(self.convs, self.norms):
            if return_attention:
                x_out, att = conv(
                    x, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
                attention_outputs.append(att)
            else:
                x_out = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x_out)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x
        graph_mean = global_mean_pool(node_embeddings, batch)
        graph_max = global_max_pool(node_embeddings, batch)
        graph_embedding = torch.cat([graph_mean, graph_max], dim=1)

        pred = self.pred_head(graph_embedding)

        out = {
            "pred": pred,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
        }
        if return_attention:
            out["attention_weights"] = attention_outputs
        if self.bias_heads:
            out["bias_logits"] = {
                name: head(graph_embedding) for name, head in self.bias_heads.items()
            }
        return out


# --------------------------------------------------------------------------- #
# Public loading API.
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    """A loaded model bundled with provenance metadata."""

    model: RamanGATv2MultiBias
    checkpoint_path: Path
    config: dict
    device: str = "cpu"
    missing_keys: list = field(default_factory=list)
    unexpected_keys: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def version_string(self) -> str:
        return (
            f"RamanGATv2MultiBias | {self.config['num_layers']} layers x "
            f"{self.config['num_heads']} heads | hidden={self.config['hidden_channels']} | "
            f"ckpt={self.checkpoint_path.name}"
        )


def _build_model(config: Optional[dict] = None) -> RamanGATv2MultiBias:
    """Instantiate an (untrained) model from a hyper-parameter dict."""
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if config:
        cfg.update(config)
    return RamanGATv2MultiBias(**cfg)


def load_model(
    checkpoint_path: Optional[os.PathLike | str] = None,
    config: Optional[dict] = None,
    device: str = "cpu",
    strict: bool = False,
) -> LoadedModel:
    """Reconstruct the architecture and load the trained weights.

    Parameters
    ----------
    checkpoint_path:
        Path to ``best_model_full.pt`` (a dict with ``model_state_dict``).
        Defaults to :data:`DEFAULT_CHECKPOINT`.
    config:
        Optional override of :data:`DEFAULT_MODEL_CONFIG`.
    device:
        Torch device string (``"cpu"`` recommended for portability).
    strict:
        Passed to ``load_state_dict``.  Kept ``False`` because the trained
        checkpoint legitimately matches with 0 missing/unexpected keys but we
        want robustness across torch_geometric versions.

    Returns
    -------
    LoadedModel
    """
    ckpt_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{ckpt_path}'. "
            "Set RMC_CHECKPOINT or pass checkpoint_path explicitly."
        )

    cfg = dict(DEFAULT_MODEL_CONFIG)
    if config:
        cfg.update(config)

    model = _build_model(cfg)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        extra = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
        # Keep only JSON-friendly provenance to avoid carrying large tensors.
        extra = {
            k: v for k, v in extra.items() if k in ("params", "lambda_bias", "epoch", "best_val_loss")
        }
    else:  # a bare state-dict
        state_dict = ckpt
        extra = {}

    result = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        checkpoint_path=ckpt_path,
        config=cfg,
        device=device,
        missing_keys=list(result.missing_keys),
        unexpected_keys=list(result.unexpected_keys),
        extra=extra,
    )


@functools.lru_cache(maxsize=4)
def get_cached_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> LoadedModel:
    """Return a process-wide cached :class:`LoadedModel`.

    The cache key is the (resolved) checkpoint path string plus device, so the
    GUI and batch paths reuse the same weights instead of reloading 15 MB each
    time.  ``config`` is intentionally not part of the key (the default config
    matches the shipped checkpoint).
    """
    key = str(Path(checkpoint_path)) if checkpoint_path else None
    return load_model(key, device=device)
