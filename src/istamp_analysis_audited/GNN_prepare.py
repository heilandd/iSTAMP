# =============================================================================
# MsClaude optimized copy — original unchanged at
#   /Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/GNN_prepare.py
#
# Purpose: merge per-image PyG files and build train/val loaders.
#
# Light improvements only (NO scientific-logic changes to existing functions):
#   - Added module docstring, type hints, and a deterministic seed helper.
#   - Added `make_patient_grouped_loaders`: a NEW, opt-in alternative to
#     `make_train_val_loaders` that splits by `patient_id` so all subgraphs
#     of a patient stay on one side of the split.
#     >>> RATIONALE (data leakage): the ORIGINAL `make_train_val_loaders`
#         calls torch random_split on the POOLED subgraph dataset, so
#         subgraphs from the same patient/image — and overlapping ego-graphs
#         — can appear in BOTH train and validation. This inflates the
#         reported validation metrics. The original function is preserved
#         verbatim for reproducibility; use the new helper for a leakage-free
#         evaluation.
# =============================================================================
"""GNN data-preparation utilities for the iSTAMP Raman/GNN pipeline."""

from pathlib import Path
from typing import Optional, Sequence
import random
import torch
from torch.utils.data import random_split
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from tqdm import tqdm


def set_global_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch RNGs for reproducible splits and training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def load_pyg_result(in_path, map_location="cpu"):
    in_path = Path(in_path)
    return torch.load(in_path, map_location=map_location, weights_only=False)


def merge_pyg_files_for_hop(
    pt_files,
    hop=3,
    map_location="cpu",
    save_path=None,
):
    """
    Merge multiple saved PyG result files into one dataset for a single hop.

    Parameters
    ----------
    pt_files : list[str | Path]
        Paths to saved .pt files
    hop : int
        Which hop neighborhood to extract
    map_location : str
        Torch map_location
    save_path : str | Path | None
        Optional output path for saving merged dataset

    Returns
    -------
    merged_dataset : list
        List of PyG Data objects
    merged_payload : dict
        Metadata + merged dataset
    """
    merged_dataset = []
    source_metadata = []

    for pt_file in pt_files:
        payload = load_pyg_result(pt_file, map_location=map_location)

        available_hops = payload.get("available_hops", list(payload["pyg_dataset_by_hop"].keys()))
        if hop not in payload["pyg_dataset_by_hop"]:
            raise ValueError(
                f"hop={hop} not found in {pt_file}. Available hops: {available_hops}"
            )

        ds = payload["pyg_dataset_by_hop"][hop]
        merged_dataset.extend(ds)

        source_metadata.append({
            "file": str(pt_file),
            "patient_id": payload.get("patient_id"),
            "sample_id": payload.get("sample_id"),
            "image_id": payload.get("image_id"),
            "n_graphs": len(ds),
        })

    merged_payload = {
        "hop": hop,
        "dataset": merged_dataset,
        "n_graphs": len(merged_dataset),
        "sources": source_metadata,
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged_payload, save_path)
        print(f"Saved merged hop-{hop} dataset to: {save_path}")

    return merged_dataset, merged_payload

def inspect_dataset(dataset):
    print("Number of graphs:", len(dataset))
    g = dataset[0]
    print(g)
    print("x shape:", g.x.shape)
    print("edge_index shape:", g.edge_index.shape)
    print("edge_attr shape:", None if getattr(g, "edge_attr", None) is None else g.edge_attr.shape)
    print("y shape:", None if getattr(g, "y", None) is None else g.y.shape)


def make_train_val_loaders(dataset, batch_size=4, train_fraction=0.8, seed=42):
    n_total = len(dataset)
    n_train = max(1, int(n_total * train_fraction))
    n_val = n_total - n_train

    if n_val == 0:
        n_val = 1
        n_train = n_total - 1

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds =torch.utils.data.random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds, train_loader, val_loader


def make_patient_grouped_loaders(
    dataset,
    batch_size: int = 4,
    train_fraction: float = 0.7,
    seed: int = 42,
    group_attr: str = "patient_id",
):
    """Leakage-free train/val split: partition by `group_attr` (patient_id).

    NEW helper (MsClaude). Unlike `make_train_val_loaders`, this guarantees
    that every subgraph belonging to a given patient lands entirely in either
    the train OR the validation set — never both. Use this to produce a
    held-out validation cohort that supports the manuscript's no-leakage
    claim.

    Returns (train_ds, val_ds, train_loader, val_loader, split_info).
    """
    # Collect the group label for each graph.
    groups = []
    for data in dataset:
        val = getattr(data, group_attr, None)
        if val is None:
            raise ValueError(
                f"Graph is missing grouping attribute '{group_attr}'. "
                "Cannot perform a leakage-free patient-level split."
            )
        groups.append(str(val))

    unique_patients = sorted(set(groups))
    rng = random.Random(seed)
    shuffled = unique_patients[:]
    rng.shuffle(shuffled)

    n_train_patients = max(1, int(round(len(shuffled) * train_fraction)))
    train_patients = set(shuffled[:n_train_patients])
    val_patients = set(shuffled[n_train_patients:])
    if len(val_patients) == 0:  # ensure non-empty val cohort
        val_patients = {shuffled[-1]}
        train_patients = set(shuffled[:-1])

    train_idx = [i for i, g in enumerate(groups) if g in train_patients]
    val_idx = [i for i, g in enumerate(groups) if g in val_patients]

    train_ds = [dataset[i] for i in train_idx]
    val_ds = [dataset[i] for i in val_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    split_info = {
        "n_train_patients": len(train_patients),
        "n_val_patients": len(val_patients),
        "n_train_graphs": len(train_ds),
        "n_val_graphs": len(val_ds),
        "train_patients": sorted(train_patients),
        "val_patients": sorted(val_patients),
        # Sanity check: the two patient sets MUST be disjoint.
        "patient_overlap": sorted(train_patients & val_patients),
    }
    assert len(split_info["patient_overlap"]) == 0, "Patient overlap detected!"

    return train_ds, val_ds, train_loader, val_loader, split_info


def add_domain_labels_from_attr(dataset, attr_name="image_id"):
    """
    Add integer domain_label to each PyG graph from a string attribute like image_id.
    """
    values = []
    for data in dataset:
        val = getattr(data, attr_name, None)
        if val is None:
            raise ValueError(f"Graph is missing attribute '{attr_name}'")
        values.append(str(val))

    unique_vals = sorted(set(values))
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}

    for data in dataset:
        domain_idx = val_to_idx[str(getattr(data, attr_name))]
        data.domain_label = torch.tensor(domain_idx, dtype=torch.long)

    return dataset, val_to_idx

import torch


def add_multi_bias_labels(
    dataset,
    bias_attrs=("Center", "sample_type", "image_id", "patient_id"),
):
    """
    Add integer bias labels to each graph:
    - data.bias_center
    - data.bias_sample_type
    - data.bias_image_id
    - data.bias_patient_id

    Returns
    -------
    dataset
    mappings : dict[attr_name -> dict[str, int]]
    """
    mappings = {}

    for attr in bias_attrs:
        values = []
        for data in dataset:
            val = getattr(data, attr, None)
            if val is None:
                raise ValueError(f"Graph missing bias attribute '{attr}'")
            values.append(str(val))

        uniq = sorted(set(values))
        mapping = {v: i for i, v in enumerate(uniq)}
        mappings[attr] = mapping

        safe_attr = f"bias_{attr.replace('.', '_')}".replace(" ", "_")

        for data in dataset:
            idx = mapping[str(getattr(data, attr))]
            setattr(data, safe_attr, torch.tensor(idx, dtype=torch.long))

    return dataset, mappings


