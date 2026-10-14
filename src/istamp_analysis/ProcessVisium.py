import numpy as np
import cv2
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union


def redefine_in_tissue_by_polygon(
    adata,
    library_id="V3",
    img_key="custom",
    scale_factor_key="tissue_custom_scalef",
    background_threshold=245,
    min_area=5000,
    close_kernel_size=15,
    open_kernel_size=7,
    simplify_tolerance=5.0,
    use_convex_hull=False,
    obs_key="in_tissue_recomputed",
    plot_debug=False,
):
    """
    Recompute in-tissue annotation for Visium spots by thresholding the image,
    extracting a tissue contour/polygon, and testing whether each spot lies
    inside the polygon.

    Parameters
    ----------
    adata : AnnData
        AnnData object with spatial information.
    library_id : str
        Key in adata.uns["spatial"] (e.g. "V3").
    img_key : str
        Image key in adata.uns["spatial"][library_id]["images"].
    scale_factor_key : str
        Scale factor key in adata.uns["spatial"][library_id]["scalefactors"].
    background_threshold : int
        Pixels brighter than this value are treated as background.
        Lower this if tissue is very pale.
    min_area : int
        Minimum contour area to keep.
    close_kernel_size : int
        Kernel size for morphological closing.
    open_kernel_size : int
        Kernel size for morphological opening.
    simplify_tolerance : float
        Polygon simplification tolerance in pixel units.
    use_convex_hull : bool
        If True, replace tissue contour with its convex hull.
    obs_key : str
        Column name to store the recomputed in-tissue flag.
    plot_debug : bool
        If True, plot intermediate results.

    Returns
    -------
    adata : AnnData
        Modified AnnData with new obs column.
    tissue_polygon : shapely Polygon or MultiPolygon
        The extracted tissue polygon.
    mask : np.ndarray
        Binary tissue mask.
    """

    import matplotlib.pyplot as plt

    # -----------------------------
    # 1. Load image
    # -----------------------------
    img = adata.uns["spatial"][library_id]["images"][img_key]

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Image must be 2D grayscale or 3D RGB.")

    # -----------------------------
    # 2. Threshold: tissue = non-white
    # -----------------------------
    # white background -> high grayscale values
    mask = gray < background_threshold
    mask = mask.astype(np.uint8) * 255

    # -----------------------------
    # 3. Morphological cleanup
    # -----------------------------
    close_kernel = np.ones((close_kernel_size, close_kernel_size), np.uint8)
    open_kernel = np.ones((open_kernel_size, open_kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    # -----------------------------
    # 4. Find contours
    # -----------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise ValueError("No tissue contours found. Try increasing background_threshold.")

    # Keep only large contours
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

    if len(large_contours) == 0:
        raise ValueError(
            "Contours were found, but all were smaller than min_area. "
            "Try lowering min_area."
        )

    # -----------------------------
    # 5. Convert contours to polygons
    # -----------------------------
    polygons = []
    for cnt in large_contours:
        cnt = cnt.squeeze()

        if cnt.ndim != 2 or cnt.shape[0] < 3:
            continue

        poly = Polygon(cnt)

        if not poly.is_valid:
            poly = poly.buffer(0)

        if simplify_tolerance is not None and simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        if use_convex_hull:
            poly = poly.convex_hull

        if poly.is_valid and not poly.is_empty and poly.area > 0:
            polygons.append(poly)

    if len(polygons) == 0:
        raise ValueError("No valid polygons could be constructed from contours.")

    tissue_polygon = unary_union(polygons)

    # -----------------------------
    # 6. Map spots to image pixel coordinates
    # -----------------------------
    coords = adata.obsm["spatial"]
    scale_factor = adata.uns["spatial"][library_id]["scalefactors"][scale_factor_key]

    # Visium coordinates are usually in fullres coordinates.
    # Image pixel coordinates are obtained by multiplying by the scale factor.
    pixel_coords = coords * scale_factor

    # -----------------------------
    # 7. Point-in-polygon test
    # -----------------------------
    in_tissue = []
    for x, y in pixel_coords:
        pt = Point(float(x), float(y))
        inside = tissue_polygon.contains(pt) or tissue_polygon.touches(pt)
        in_tissue.append(int(inside))

    adata.obs[obs_key] = np.array(in_tissue, dtype=int)

    # -----------------------------
    # 8. Optional debug plots
    # -----------------------------
    if plot_debug:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img)
        axes[0].set_title("Original image")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Thresholded tissue mask")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].set_title("Polygon + spots")
        axes[2].axis("off")

        # Plot polygon(s)
        def plot_poly(ax, poly, color="red", lw=2):
            if poly.geom_type == "Polygon":
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=lw)
            elif poly.geom_type == "MultiPolygon":
                for p in poly.geoms:
                    x, y = p.exterior.xy
                    ax.plot(x, y, color=color, linewidth=lw)

        plot_poly(axes[2], tissue_polygon)

        inside_mask = np.array(in_tissue, dtype=bool)
        outside_mask = ~inside_mask

        axes[2].scatter(
            pixel_coords[inside_mask, 0],
            pixel_coords[inside_mask, 1],
            s=10,
            c="lime",
            label="inside",
        )
        axes[2].scatter(
            pixel_coords[outside_mask, 0],
            pixel_coords[outside_mask, 1],
            s=10,
            c="blue",
            label="outside",
        )
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    return adata, tissue_polygon, mask


from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import networkx as nx
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# =========================
# Helpers: image + geometry
# =========================

def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """
    Convert image to HWC uint8 RGB.
    Accepts HWC, CHW, grayscale.
    """
    arr = np.asarray(img)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    elif arr.ndim == 3:
        # if CHW -> HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        if arr.shape[-1] == 4:
            arr = arr[..., :3]

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def get_custom_raman_image_from_adata(adata, sample_key: str | None = None) -> np.ndarray:
    """
    Extract Raman image from:
    adata.uns["spatial"][sample_key]["images"]["custom"]
    """
    if "spatial" not in adata.uns:
        raise KeyError("adata.uns['spatial'] not found.")

    spatial_dict = adata.uns["spatial"]

    if sample_key is None:
        if len(spatial_dict) != 1:
            raise ValueError(
                f"Multiple samples found in adata.uns['spatial']: {list(spatial_dict.keys())}. "
                "Please provide sample_key explicitly."
            )
        sample_key = list(spatial_dict.keys())[0]

    if sample_key not in spatial_dict:
        raise KeyError(f"sample_key='{sample_key}' not found in adata.uns['spatial'].")

    if "images" not in spatial_dict[sample_key]:
        raise KeyError(f"adata.uns['spatial'][{sample_key!r}]['images'] not found.")

    if "custom" not in spatial_dict[sample_key]["images"]:
        raise KeyError(f"adata.uns['spatial'][{sample_key!r}]['images']['custom'] not found.")

    img = spatial_dict[sample_key]["images"]["custom"]
    return _to_hwc_uint8(img)


def get_spatial_coordinates_from_adata(
    adata,
    spatial_key: str = "spatial",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """
    Extract spot coordinates from adata.obsm[spatial_key].
    Returns DataFrame with point_id, x, y.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] must have at least 2 columns.")

    df = pd.DataFrame({
        "point_id": adata.obs_names.astype(str),
        x_col: coords[:, 0].astype(float),
        y_col: coords[:, 1].astype(float),
    })
    return df


def extract_tile_from_image(
    image_hwc: np.ndarray,
    center_x: float,
    center_y: float,
    tile_size: int = 448,
    pad_mode: str = "constant",
) -> np.ndarray:
    """
    Extract tile centered at (x, y), padding if needed.
    """
    h, w, c = image_hwc.shape
    half = tile_size // 2

    cx = int(round(center_x))
    cy = int(round(center_y))

    x0 = cx - half
    x1 = cx + half
    y0 = cy - half
    y1 = cy + half

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        image_pad = np.pad(
            image_hwc,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=pad_mode,
        )
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
        tile = image_pad[y0:y1, x0:x1, :]
    else:
        tile = image_hwc[y0:y1, x0:x1, :]

    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
        # final safety pad
        out = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        out[:tile.shape[0], :tile.shape[1], :] = tile
        tile = out

    return tile


# =========================
# CONCH embedding
# =========================

class ConchEmbedder:
    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
    ):
        from conch.open_clip_custom import create_model_from_pretrained

        self.model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16",
            self.model_path,
        )
        self.model.to(self.device)
        self.model.eval()

    def embed_pil(self, images: Sequence[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embs = []

        with torch.inference_mode():
            for i in tqdm(range(0, len(images), batch_size), desc="CONCH embedding", unit="batch"):
                batch_imgs = images[i:i + batch_size]
                batch_tensor = torch.stack(
                    [self.preprocess(img.convert("RGB")) for img in batch_imgs]
                ).to(self.device)

                embs = self.model.encode_image(
                    batch_tensor,
                    proj_contrast=False,
                    normalize=False,
                )
                all_embs.append(embs.detach().cpu().numpy())

        if len(all_embs) == 0:
            return np.empty((0, 512), dtype=np.float32)

        return np.vstack(all_embs).astype(np.float32)


# =========================
# Graph construction
# =========================

def build_knn_spot_graph(
    coords_df: pd.DataFrame,
    n_neighbors: int = 6,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[pd.DataFrame, nx.Graph]:
    """
    Build undirected kNN graph on ST spots.
    """
    coords = coords_df[[x_col, y_col]].to_numpy(dtype=float)
    point_ids = coords_df["point_id"].astype(str).tolist()

    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    dists, inds = nn.kneighbors(coords)

    edges = set()
    rows = []

    for i in range(len(coords)):
        for j, dist in zip(inds[i][1:], dists[i][1:]):  # skip self
            a, b = sorted((i, j))
            if (a, b) in edges:
                continue
            edges.add((a, b))
            rows.append({
                "from_idx": a,
                "to_idx": b,
                "from_id": point_ids[a],
                "to_id": point_ids[b],
                "distance": float(dist),
            })

    edge_df = pd.DataFrame(rows)

    G = nx.Graph()
    for row in coords_df.itertuples(index=False):
        G.add_node(str(row.point_id), x=float(row.x), y=float(row.y))

    for row in edge_df.itertuples(index=False):
        G.add_edge(str(row.from_id), str(row.to_id), distance=float(row.distance))

    return edge_df, G


def prune_long_edges(
    edge_df: pd.DataFrame,
    distance_quantile: float | None = 0.95,
    max_distance: float | None = None,
) -> tuple[pd.DataFrame, float]:
    if max_distance is None:
        if distance_quantile is None:
            raise ValueError("Provide either distance_quantile or max_distance.")
        max_distance = float(edge_df["distance"].quantile(distance_quantile))

    pruned = edge_df.loc[edge_df["distance"] <= max_distance].copy()
    return pruned, max_distance


def build_networkx_graph_from_edge_df(
    coords_df: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> nx.Graph:
    G = nx.Graph()

    for row in coords_df.itertuples(index=False):
        G.add_node(str(row.point_id), x=float(row.x), y=float(row.y))

    for row in edge_df.itertuples(index=False):
        G.add_edge(str(row.from_id), str(row.to_id), distance=float(row.distance))

    return G


# =========================
# PyG graph construction
# =========================

def edge_distance_to_weight(distance: float, sigma: float = 300.0) -> float:
    """
    Optional distance transform for edge_attr.
    """
    return float(np.exp(-(distance ** 2) / (2 * sigma ** 2)))


def build_pyg_graph_for_center_spot(
    center_id: str,
    G: nx.Graph,
    edge_df: pd.DataFrame,
    coords_df: pd.DataFrame,
    emb_df: pd.DataFrame,
    hop: int = 3,
    use_edge_weight: bool = False,
    edge_sigma: float = 300.0,
) -> Data:
    """
    Build one PyG graph centered on one ST spot.
    """
    node_ids = sorted(nx.ego_graph(G, center_id, radius=hop).nodes())
    node_set = set(node_ids)

    sub_edge_df = edge_df.loc[
        edge_df["from_id"].astype(str).isin(node_set) &
        edge_df["to_id"].astype(str).isin(node_set)
    ].copy()

    coords_sub = coords_df.set_index("point_id").loc[node_ids].reset_index()
    emb_sub = emb_df.set_index("point_id").loc[node_ids].reset_index()

    feature_cols = [c for c in emb_sub.columns if c.startswith("CONCH_")]
    X = emb_sub[feature_cols].to_numpy(dtype=np.float32)
    pos = coords_sub[["x", "y"]].to_numpy(dtype=np.float32)

    id_to_idx = {pid: i for i, pid in enumerate(node_ids)}

    edge_index = []
    edge_attr = []

    for row in sub_edge_df.itertuples(index=False):
        a = id_to_idx[str(row.from_id)]
        b = id_to_idx[str(row.to_id)]

        edge_index.append([a, b])
        edge_index.append([b, a])

        if use_edge_weight:
            w = edge_distance_to_weight(float(row.distance), sigma=edge_sigma)
            edge_attr.append([w])
            edge_attr.append([w])
        else:
            edge_attr.append([float(row.distance)])
            edge_attr.append([float(row.distance)])

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(pos, dtype=torch.float),
    )

    data.seed_id = str(center_id)
    data.node_ids = node_ids
    data.hop = int(hop)
    data.num_nodes_in_subgraph = len(node_ids)

    return data


# =========================
# Main prediction function
# =========================

def predict_raman_scores_for_st_spots(
    adata,
    model,
    sample_key: str | None = None,
    conch_model_path: str | Path = "/Users/henrikheiland/CONCH/checkpoints/conch/pytorch_model.bin",
    spatial_key: str = "spatial",
    image_name: str = "custom",
    tile_size: int = 448,
    hop: int = 3,
    n_neighbors: int = 6,
    prune_quantile: float | None = 0.95,
    prune_max_distance: float | None = None,
    batch_size_conch: int = 32,
    batch_size_graph: int = 16,
    use_edge_weight: bool = False,
    edge_sigma: float = 300.0,
    device: str | None = None,
    labels: Sequence[str] = ("GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"),
    add_probabilities: bool = True,
    add_logits: bool = True,
    confidence_mode: str = "softmax",
    return_intermediates: bool = False,
):
    """
    Full preprocessing + prediction pipeline:
    ST spots -> Raman tiles -> CONCH embeddings -> k-hop PyG graphs -> model predictions -> adata.obs.

    Parameters
    ----------
    adata
        AnnData with:
        - adata.obsm["spatial"] coordinates
        - adata.uns["spatial"][sample]["images"]["custom"] Raman image
    model
        Trained GNN model. Expected forward: out = model(data, return_attention=False), out["pred"] shape [B, n_labels]
    sample_key
        Key into adata.uns["spatial"] if multiple samples exist
    conch_model_path
        Local CONCH checkpoint
    hop
        k-hop neighborhood around each ST spot
    n_neighbors
        kNN graph connectivity for ST spots
    prune_quantile / prune_max_distance
        Optional edge pruning
    add_probabilities / add_logits
        Whether to write probs/logits into adata.obs
    confidence_mode
        "softmax" or "margin"
    return_intermediates
        If True, return dict with coords, edges, embeddings, pyg dataset, predictions

    Returns
    -------
    adata
        Modified AnnData with Raman prediction columns in adata.obs
    optional dict
        Intermediates if return_intermediates=True
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. image + spot coordinates
    raman_img = get_custom_raman_image_from_adata(adata, sample_key=sample_key)
    coords_df = get_spatial_coordinates_from_adata(adata, spatial_key=spatial_key)
    coords_df["point_id"] = coords_df["point_id"].astype(str)

    # 2. extract tiles for ALL ST spots
    pil_tiles = []
    for row in tqdm(coords_df.itertuples(index=False), total=len(coords_df), desc="Extract ST spot tiles", unit="spot"):
        tile = extract_tile_from_image(
            raman_img,
            center_x=float(row.x),
            center_y=float(row.y),
            tile_size=tile_size,
        )
        pil_tiles.append(Image.fromarray(tile.astype(np.uint8)))

    # 3. CONCH embeddings for ALL spots
    embedder = ConchEmbedder(model_path=conch_model_path, device=device)
    embs = embedder.embed_pil(pil_tiles, batch_size=batch_size_conch)

    emb_cols = [f"CONCH_{i+1}" for i in range(embs.shape[1])]
    emb_df = pd.DataFrame(embs, columns=emb_cols)
    emb_df.insert(0, "point_id", coords_df["point_id"].values)

    # 4. Build spot graph
    edge_df, G0 = build_knn_spot_graph(coords_df, n_neighbors=n_neighbors)

    if prune_quantile is not None or prune_max_distance is not None:
        edge_df_pruned, prune_threshold = prune_long_edges(
            edge_df,
            distance_quantile=prune_quantile,
            max_distance=prune_max_distance,
        )
    else:
        edge_df_pruned = edge_df.copy()
        prune_threshold = None

    G = build_networkx_graph_from_edge_df(coords_df, edge_df_pruned)

    # 5. Build one PyG graph per ST spot
    pyg_dataset = []
    for center_id in tqdm(coords_df["point_id"].tolist(), desc=f"Build {hop}-hop PyG graphs", unit="graph"):
        data = build_pyg_graph_for_center_spot(
            center_id=str(center_id),
            G=G,
            edge_df=edge_df_pruned,
            coords_df=coords_df,
            emb_df=emb_df,
            hop=hop,
            use_edge_weight=use_edge_weight,
            edge_sigma=edge_sigma,
        )
        pyg_dataset.append(data)

    # 6. Run model prediction
    loader = DataLoader(pyg_dataset, batch_size=batch_size_graph, shuffle=False)

    model.eval()
    model.to(device)

    all_logits = []
    all_seed_ids = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict Raman subtype scores", unit="batch"):
            batch = batch.to(device)
            out = model(batch, return_attention=False)
            logits = out["pred"].detach().cpu().numpy()
            all_logits.append(logits)

            batch_cpu = batch.cpu().to_data_list()
            all_seed_ids.extend([g.seed_id for g in batch_cpu])

    all_logits = np.vstack(all_logits)

    if add_probabilities:
        all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    else:
        all_probs = None

    pred_class_idx = np.argmax(all_logits, axis=1)
    pred_class_label = [labels[i] for i in pred_class_idx]

    if confidence_mode == "softmax":
        confidence = all_probs[np.arange(len(pred_class_idx)), pred_class_idx]
    elif confidence_mode == "margin":
        tmp = np.sort(all_logits, axis=1)[:, ::-1]
        confidence = tmp[:, 0] - tmp[:, 1]
    else:
        raise ValueError("confidence_mode must be 'softmax' or 'margin'")

    pred_df = pd.DataFrame({
        "point_id": [str(x) for x in all_seed_ids],
        "raman_pred_class": pred_class_label,
        "raman_pred_class_idx": pred_class_idx,
        "raman_confidence": confidence,
    })

    if add_logits:
        for j, lab in enumerate(labels):
            pred_df[f"raman_logit_{lab}"] = all_logits[:, j]

    if add_probabilities:
        for j, lab in enumerate(labels):
            pred_df[f"raman_prob_{lab}"] = all_probs[:, j]

    # 7. Write back into adata.obs by spot ID
    obs_df = adata.obs.copy()
    obs_df["point_id"] = adata.obs_names.astype(str)

    obs_df = obs_df.merge(pred_df, on="point_id", how="left")
    obs_df.index = adata.obs_names
    adata.obs = obs_df.drop(columns=["point_id"])

    if return_intermediates:
        return adata, {
            "coords_df": coords_df,
            "edge_df": edge_df,
            "edge_df_pruned": edge_df_pruned,
            "prune_threshold": prune_threshold,
            "emb_df": emb_df,
            "pyg_dataset": pyg_dataset,
            "prediction_df": pred_df,
        }

    return adata








