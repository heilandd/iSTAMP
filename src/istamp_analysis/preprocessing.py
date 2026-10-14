from pathlib import Path
import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel
from PIL import Image
import numpy as np
import spatialdata as sd
from spatialdata.models import Image2DModel
from spatialdata.transformations import Identity
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def load_image_as_spatialdata(
    image_path: str | Path,
    image_name: str = "raman_image",
    crop_margin: int = 0,
) -> sd.SpatialData:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # (y, x, c)

    if crop_margin > 0:
        h, w, _ = arr.shape
        if 2 * crop_margin >= h or 2 * crop_margin >= w:
            raise ValueError(
                f"crop_margin={crop_margin} is too large for image shape {(h, w)}"
            )
        arr = arr[crop_margin:h - crop_margin, crop_margin:w - crop_margin, :]

    image_element = Image2DModel.parse(
        arr,
        dims=("y", "x", "c"),
    )

    sdata = sd.SpatialData(images={image_name: image_element})
    return sdata


def load_image_with_metadata(
    image_path: str | Path,
    image_name: str = "raman_image",
    crop_margin: int = 0,
) -> tuple[sd.SpatialData, dict]:
    sdata = load_image_as_spatialdata(
        image_path=image_path,
        image_name=image_name,
        crop_margin=crop_margin,
    )

    img = sdata.images[image_name]
    _, height, width = img.shape  # expected order: (c, y, x)

    meta = {
        "image_path": str(image_path),
        "image_name": image_name,
        "crop_margin": crop_margin,
        "height": int(height),
        "width": int(width),
    }
    return sdata, meta


def load_image_as_multiscale_sdata(
    image_path,
    image_name="raw_image",
    crop_margin=0,
    scale_factors=(2, 2, 2),
    to_grayscale=False,
):
    image_path = Path(image_path)

    img = Image.open(image_path)

    if to_grayscale:
        img = img.convert("L")   # 1 channel
        arr = np.array(img)
        arr = arr[..., None]     # (y, x) -> (y, x, 1)
        c_coords = ["intensity"]
    else:
        img = img.convert("RGB") # 3 channels
        arr = np.array(img)      # (y, x, 3)
        c_coords = ["r", "g", "b"]

    if crop_margin > 0:
        h, w = arr.shape[:2]
        if 2 * crop_margin >= h or 2 * crop_margin >= w:
            raise ValueError(
                f"crop_margin={crop_margin} too large for image shape {(h, w)}"
            )
        arr = arr[crop_margin:h - crop_margin, crop_margin:w - crop_margin, :]

    image_element = Image2DModel.parse(
        arr,
        dims=("y", "x", "c"),
        c_coords=c_coords,
        transformations={"global": Identity()},
        scale_factors=scale_factors,
    )

    sdata = sd.SpatialData(images={image_name: image_element})
    return sdata

import numpy as np
import pandas as pd
from spatialdata.models import PointsModel


def get_image_dataarray(image_element):
    """
    Return a single-scale DataArray from either:
    - a single-scale SpatialData image (DataArray), or
    - a multiscale SpatialData image (DataTree -> scale0 -> image)
    """
    # multiscale image
    if hasattr(image_element, "keys") and "scale0" in image_element.keys():
        return image_element["scale0"]["image"]

    # single-scale image
    return image_element


def generate_random_points(
    sdata,
    image_name: str,
    points_name: str = "random_points",
    n_points: int = 500,
    patch_size: int = 448,
    min_distance: float | None = None,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    img = get_image_dataarray(sdata.images[image_name])

    # now img is a DataArray with dims typically ('c', 'y', 'x')
    _, height, width = img.shape

    margin = patch_size // 2
    x_min, x_max = margin, width - margin
    y_min, y_max = margin, height - margin

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Patch size {patch_size} is too large for image size {(height, width)}."
        )

    points = []
    attempts = 0
    max_attempts = n_points * 100

    while len(points) < n_points and attempts < max_attempts:
        x = rng.integers(x_min, x_max)
        y = rng.integers(y_min, y_max)

        if min_distance is not None and len(points) > 0:
            dists = [np.hypot(x - px, y - py) for px, py in points]
            if min(dists) < min_distance:
                attempts += 1
                continue

        points.append((x, y))
        attempts += 1

    if len(points) == 0:
        raise RuntimeError("No valid points could be generated.")

    points = np.asarray(points)

    df = pd.DataFrame(
        {
            "x": points[:, 0],
            "y": points[:, 1],
            "point_id": [f"p_{i:05d}" for i in range(len(points))],
        }
    )

    # PointsModel.parse accepts a dataframe directly; if x/y are present,
    # coordinates can be inferred, but being explicit is fine too.
    points_element = PointsModel.parse(df, coordinates={"x": "x", "y": "y"})
    sdata.points[points_name] = points_element

    return sdata


import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from spatialdata.models import ShapesModel


def points_to_tiles(
    sdata,
    points_name: str = "random_points",
    shapes_name: str = "tiles_448",
    tile_size: int = 448,
):
    """
    Convert point centers in sdata.points[points_name] into square tiles
    of size tile_size x tile_size, centered on each point.
    """
    pts = sdata.points[points_name]

    # Dask dataframe -> pandas if needed
    if hasattr(pts, "compute"):
        pts = pts.compute()

    half = tile_size / 2.0

    geoms = [
        box(x - half, y - half, x + half, y + half)
        for x, y in zip(pts["x"], pts["y"])
    ]

    gdf = gpd.GeoDataFrame(
        {
            "tile_id": [f"tile_{i:05d}" for i in range(len(pts))],
            "point_id": pts["point_id"].tolist() if "point_id" in pts.columns else [f"p_{i:05d}" for i in range(len(pts))],
            "center_x": pts["x"].to_numpy(),
            "center_y": pts["y"].to_numpy(),
            "tile_size": tile_size,
        },
        geometry=geoms,
    )

    sdata.shapes[shapes_name] = ShapesModel.parse(gdf)
    return sdata

def get_image_scale0(image_element):
    if hasattr(image_element, "keys") and "scale0" in image_element.keys():
        return image_element["scale0"]["image"]
    return image_element

from PIL import Image


def extract_tile_array(
    sdata,
    image_name: str,
    center_x: int,
    center_y: int,
    tile_size: int = 448,
):
    """
    Extract a tile as a numpy array of shape (tile_size, tile_size, c)
    from the highest-resolution image.
    """
    img = get_image_scale0(sdata.images[image_name])
    img_np = img.transpose("y", "x", "c").values

    half = tile_size // 2
    x0 = int(center_x - half)
    x1 = int(center_x + half)
    y0 = int(center_y - half)
    y1 = int(center_y + half)

    tile = img_np[y0:y1, x0:x1, :]
    return tile

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel, laplace
from skimage.measure import shannon_entropy


def compute_tile_qc_features(tile: np.ndarray) -> dict:
    """
    Compute simple QC / diagnostic proxy features for one tile.
    Expects tile shape (H, W, C), usually RGB.
    """
    tile = tile.astype(np.float32)

    out = {}

    # Per-channel means and stds
    if tile.ndim == 3 and tile.shape[2] >= 3:
        for i, ch in enumerate(["r", "g", "b"]):
            out[f"{ch}_mean"] = float(tile[..., i].mean())
            out[f"{ch}_std"] = float(tile[..., i].std())

    # Global grayscale features
    gray = rgb2gray(tile / 255.0) if tile.ndim == 3 else tile / 255.0

    out["gray_mean"] = float(gray.mean())
    out["gray_std"] = float(gray.std())
    out["gray_entropy"] = float(shannon_entropy((gray * 255).astype(np.uint8)))

    # Edge content
    sob = sobel(gray)
    out["sobel_mean"] = float(sob.mean())
    out["sobel_std"] = float(sob.std())

    # Blur / focus proxy
    lap = laplace(gray)
    out["laplace_var"] = float(lap.var())

    # White / dark fractions
    out["white_fraction"] = float((gray > 0.9).mean())
    out["dark_fraction"] = float((gray < 0.1).mean())

    # Colorfulness proxy
    if tile.ndim == 3 and tile.shape[2] >= 3:
        rg = tile[..., 0] - tile[..., 1]
        yb = 0.5 * (tile[..., 0] + tile[..., 1]) - tile[..., 2]
        out["colorfulness"] = float(
            np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        )

    return out

from tqdm import tqdm


def compute_qc_for_all_tiles(
    sdata,
    image_name: str,
    points_name: str = "random_points",
    tile_size: int = 448,
):
    pts = sdata.points[points_name]

    if hasattr(pts, "compute"):
        pts = pts.compute()

    rows = []

    for row in tqdm(pts.itertuples(), total=len(pts), desc="QC tiles", unit="tile"):
        cx = int(row.x)
        cy = int(row.y)

        tile = extract_tile_array(
            sdata,
            image_name=image_name,
            center_x=cx,
            center_y=cy,
            tile_size=tile_size,
        )

        feats = compute_tile_qc_features(tile)
        feats["x"] = cx
        feats["y"] = cy
        feats["point_id"] = getattr(row, "point_id", None)

        rows.append(feats)

    return pd.DataFrame(rows)


## Get Dignostic Score

import torch
import torch.nn as nn
from torchvision import models
def build_diagnostic_model(architecture: str = "resnet18", num_classes: int = 2):
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")
def load_diagnostic_model(model_path: str | Path, device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)

    architecture = checkpoint["architecture"]
    model = build_diagnostic_model(architecture=architecture, num_classes=2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
class TileInferenceDataset(Dataset):
    def __init__(self, sdata, qc_df, image_name="raw_image", tile_size=448, input_size=224):
        self.sdata = sdata
        self.qc_df = qc_df.reset_index(drop=True)
        self.image_name = image_name
        self.tile_size = tile_size
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.qc_df)

    def __getitem__(self, idx):
        row = self.qc_df.iloc[idx]

        tile = extract_tile_array(
            self.sdata,
            image_name=self.image_name,
            center_x=int(row["x"]),
            center_y=int(row["y"]),
            tile_size=self.tile_size,
        )

        tile = Image.fromarray(tile.astype(np.uint8))
        tile = self.transform(tile)

        return tile, row["point_id"]
def score_all_tiles(model, sdata, qc_df, image_name="raw_image", tile_size=448, batch_size=32, device="cpu"):
    infer_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = TileInferenceDataset(
        sdata=sdata,
        qc_df=qc_df,
        image_name=image_name,
        tile_size=tile_size,
        transform=infer_tf,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_ids = []
    all_scores = []
    all_preds = []

    with torch.no_grad():
        for x, point_ids in tqdm(loader, desc="Scoring all tiles", unit="batch"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_ids.extend(point_ids)
            all_scores.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    score_df = pd.DataFrame({
        "point_id": all_ids,
        "diagnostic_score": all_scores,
        "diagnostic_pred": all_preds,
    })

    return score_df      
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
def add_diagnostic_score_to_qc_df(
    sdata,
    qc_df: pd.DataFrame,
    model_path: str | Path,
    image_name: str = "raw_image",
    tile_size: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
):
    """
    Load a saved diagnostic model and add diagnostic scores to qc_df.

    Returns
    -------
    pd.DataFrame
        qc_df with added columns:
        - diagnostic_score
        - diagnostic_pred
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_diagnostic_model(model_path, device=device)

    if tile_size is None:
        tile_size = checkpoint.get("tile_size", 448)

    input_size = checkpoint.get("input_size", 224)
    image_name = image_name or checkpoint.get("image_name", "raw_image")

    ds = TileInferenceDataset(
        sdata=sdata,
        qc_df=qc_df,
        image_name=image_name,
        tile_size=tile_size,
        input_size=input_size,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_ids = []
    all_scores = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for x, point_ids in tqdm(loader, desc="Computing diagnostic score", unit="batch"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_ids.extend(point_ids)
            all_scores.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    score_df = pd.DataFrame({
        "point_id": all_ids,
        "diagnostic_score": all_scores,
        "diagnostic_pred": all_preds,
    })

    out = qc_df.merge(score_df, on="point_id", how="left")
    return out


from spatialdata.models import PointsModel, ShapesModel


def subset_sdata_by_diagnostic_score(
    sdata,
    qc_df,
    threshold: float = 0.8,
    prams: str = "diagnostic_score",
    points_name: str = "random_points",
    shapes_name: str = "tiles_448",
    new_points_name: str = "diagnostic_points",
    new_shapes_name: str = "diagnostic_tiles",
):
    keep_df = qc_df.loc[qc_df[prams] > threshold].copy()
    keep_ids = set(keep_df["point_id"].astype(str))

    pts = sdata.points[points_name]
    if hasattr(pts, "compute"):
        pts = pts.compute()
    pts = pts.loc[pts["point_id"].astype(str).isin(keep_ids)].copy()

    shp = sdata.shapes[shapes_name]
    if hasattr(shp, "compute"):
        shp = shp.compute()
    shp = shp.loc[shp["point_id"].astype(str).isin(keep_ids)].copy()

    sdata.points[new_points_name] = PointsModel.parse(
        pts,
        coordinates={"x": "x", "y": "y"},
    )
    sdata.shapes[new_shapes_name] = ShapesModel.parse(shp)

    return sdata, keep_df



## Build Graph:

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def build_delaunay_graph_from_points(
    points_df: pd.DataFrame,
    point_id_col: str = "point_id",
    x_col: str = "x",
    y_col: str = "y",
):
    coords = points_df[[x_col, y_col]].to_numpy(dtype=float)
    point_ids = points_df[point_id_col].astype(str).tolist()

    tri = Delaunay(coords)
    simplices = tri.simplices

    edges = set()
    for simplex in simplices:
        for i, j in [(0, 1), (1, 2), (0, 2)]:
            a = simplex[i]
            b = simplex[j]
            edges.add(tuple(sorted((a, b))))

    rows = []
    for a, b in edges:
        xa, ya = coords[a]
        xb, yb = coords[b]
        rows.append({
            "from_idx": a,
            "to_idx": b,
            "from_id": point_ids[a],
            "to_id": point_ids[b],
            "distance": float(np.hypot(xa - xb, ya - yb)),
        })

    return pd.DataFrame(rows)


def build_knn_graph_from_points(
    points_df: pd.DataFrame,
    k: int = 6,
    point_id_col: str = "point_id",
    x_col: str = "x",
    y_col: str = "y",
):
    coords = points_df[[x_col, y_col]].to_numpy(dtype=float)
    point_ids = points_df[point_id_col].astype(str).tolist()

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean")
    nn.fit(coords)
    dists, inds = nn.kneighbors(coords)

    edges = set()
    rows = []

    for i in range(len(coords)):
        for j, dist in zip(inds[i][1:], dists[i][1:]):  # skip self
            edge = tuple(sorted((i, j)))
            if edge in edges:
                continue
            edges.add(edge)

            rows.append({
                "from_idx": edge[0],
                "to_idx": edge[1],
                "from_id": point_ids[edge[0]],
                "to_id": point_ids[edge[1]],
                "distance": float(dist),
            })

    return pd.DataFrame(rows)

def prune_long_edges(
    edge_df: pd.DataFrame,
    max_distance: float | None = None,
    distance_quantile: float | None = 0.90,
):
    if max_distance is None:
        if distance_quantile is None:
            raise ValueError("Either max_distance or distance_quantile must be provided.")
        max_distance = float(edge_df["distance"].quantile(distance_quantile))

    pruned = edge_df.loc[edge_df["distance"] <= max_distance].copy()
    return pruned, max_distance

import networkx as nx
def build_networkx_graph(points_df: pd.DataFrame, edge_df: pd.DataFrame):
    G = nx.Graph()

    for row in points_df.itertuples(index=False):
        G.add_node(
            str(row.point_id),
            x=float(row.x),
            y=float(row.y),
        )

    for row in edge_df.itertuples(index=False):
        G.add_edge(
            str(row.from_id),
            str(row.to_id),
            distance=float(row.distance),
        )

    return G

from tqdm import tqdm


def generate_subgraphs_from_seeds(
    G,
    seed_ids: list[str] | None = None,
    hop: int = 3,
    max_overlap_fraction: float = 0.2,
    min_nodes: int = 5,
):
    if seed_ids is None:
        seed_ids = list(G.nodes())

    covered_nodes = set()
    subgraphs = []

    for seed in tqdm(seed_ids, desc=f"Generating {hop}-hop subgraphs", unit="seed"):
        ego = nx.ego_graph(G, seed, radius=hop)
        node_ids = sorted(ego.nodes())

        if len(node_ids) < min_nodes:
            continue

        overlap_fraction = len(set(node_ids) & covered_nodes) / len(node_ids)
        if overlap_fraction > max_overlap_fraction:
            continue

        subgraphs.append({
            "seed_id": str(seed),
            "node_ids": node_ids,
            "hop": hop,
        })

        covered_nodes.update(node_ids)

    return subgraphs

import torch
from torch_geometric.data import Data

def build_pyg_data_from_subgraph(
    adata,
    subgraph: dict,
    edge_df: pd.DataFrame,
    feature_key: str = "X_conch",
    label_cols: list[str] | None = None,
    extra_graph_obs_cols: list[str] | None = None,
):
    """
    Build one PyG Data object from one subgraph.

    Parameters
    ----------
    adata
        AnnData with obs_names = point_id
    subgraph
        dict with seed_id, node_ids, hop
    edge_df
        global edge table
    feature_key
        "X_conch" from adata.obsm, or "X" to use adata.X
    label_cols
        graph-level target columns from adata.obs
    extra_graph_obs_cols
        extra graph-level metadata copied into the PyG object
    """
    node_ids = [str(n) for n in subgraph["node_ids"]]
    node_set = set(node_ids)

    # subset AnnData in exact node order
    adata_sub = adata[node_ids].copy()

    # node features
    if feature_key == "X":
        X = adata_sub.X
        if not isinstance(X, np.ndarray):
            X = X.toarray()
    else:
        X = adata_sub.obsm[feature_key]

    X = np.asarray(X, dtype=np.float32)

    # spatial coords
    coords = np.asarray(adata_sub.obsm["spatial"], dtype=np.float32)

    # subgraph edges
    sub_edge_df = edge_df.loc[
        edge_df["from_id"].astype(str).isin(node_set) &
        edge_df["to_id"].astype(str).isin(node_set)
    ].copy()

    # map point_id -> local node index
    id_to_idx = {pid: i for i, pid in enumerate(node_ids)}

    edge_index = []
    edge_attr = []

    for row in sub_edge_df.itertuples(index=False):
        a = id_to_idx[str(row.from_id)]
        b = id_to_idx[str(row.to_id)]

        # undirected graph -> both directions
        edge_index.append([a, b])
        edge_index.append([b, a])

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
        pos=torch.tensor(coords, dtype=torch.float),
    )

    # graph-level labels
    if label_cols is not None:
        y_vals = adata_sub.obs.iloc[0][label_cols].to_numpy(dtype=np.float32)
        data.y = torch.tensor(y_vals, dtype=torch.float)

    # graph-level metadata
    data.seed_id = str(subgraph["seed_id"])
    data.node_ids = node_ids
    data.hop = int(subgraph["hop"])
    data.num_nodes_in_subgraph = len(node_ids)

    if extra_graph_obs_cols is not None:
        for col in extra_graph_obs_cols:
            val = adata_sub.obs.iloc[0][col]
            setattr(data, col, val)

    return data


def build_pyg_dataset_from_spatial_adata(
    sdata,
    adata,
    points_name: str = "diagnostic_points",
    graph_method: str = "delaunay",   # or "knn"
    knn_k: int = 6,
    prune_quantile: float | None = 0.90,
    prune_max_distance: float | None = None,
    hop_values: list[int] = [3],
    max_overlap_fraction: float = 0.2,
    min_nodes: int = 5,
    feature_key: str = "X_conch",
    label_cols: list[str] | None = None,
    extra_graph_obs_cols: list[str] | None = None,
):
    """
    Build a full PyG dataset from sdata + adata.

    Returns
    -------
    result : dict
        {
            "points_df": ...,
            "edge_df": ...,
            "edge_df_pruned": ...,
            "graph": ...,
            "subgraphs_by_hop": ...,
            "pyg_dataset_by_hop": ...
        }
    """
    pts = sdata.points[points_name]
    if hasattr(pts, "compute"):
        pts = pts.compute()
    pts = pts.copy()
    pts["point_id"] = pts["point_id"].astype(str)

    # graph construction
    if graph_method == "delaunay":
        edge_df = build_delaunay_graph_from_points(pts)
    elif graph_method == "knn":
        edge_df = build_knn_graph_from_points(pts, k=knn_k)
    else:
        raise ValueError(f"Unsupported graph_method: {graph_method}")

    # pruning
    if prune_quantile is not None or prune_max_distance is not None:
        edge_df_pruned, threshold = prune_long_edges(
            edge_df,
            max_distance=prune_max_distance,
            distance_quantile=prune_quantile,
        )
    else:
        edge_df_pruned = edge_df.copy()
        threshold = None

    G = build_networkx_graph(pts, edge_df_pruned)

    subgraphs_by_hop = {}
    pyg_dataset_by_hop = {}

    seed_ids = list(G.nodes())

    for hop in hop_values:
        subgraphs = generate_subgraphs_from_seeds(
            G,
            seed_ids=seed_ids,
            hop=hop,
            max_overlap_fraction=max_overlap_fraction,
            min_nodes=min_nodes,
        )

        pyg_dataset = []
        for sg in tqdm(subgraphs, desc=f"Building PyG dataset for hop={hop}", unit="subgraph"):
            data = build_pyg_data_from_subgraph(
                adata=adata,
                subgraph=sg,
                edge_df=edge_df_pruned,
                feature_key=feature_key,
                label_cols=label_cols,
                extra_graph_obs_cols=extra_graph_obs_cols,
            )
            pyg_dataset.append(data)

        subgraphs_by_hop[hop] = subgraphs
        pyg_dataset_by_hop[hop] = pyg_dataset

    return {
        "points_df": pts,
        "edge_df": edge_df,
        "edge_df_pruned": edge_df_pruned,
        "prune_threshold": threshold,
        "graph": G,
        "subgraphs_by_hop": subgraphs_by_hop,
        "pyg_dataset_by_hop": pyg_dataset_by_hop,
    }




## Get CONCH Embedding
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


class ConchEmbedder:
    def __init__(
        self,
        model_path: str | Path = "/Users/henrikheiland/CONCH/checkpoints/conch/pytorch_model.bin",
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

def extract_tiles_for_points(
    sdata,
    points_df: pd.DataFrame,
    image_name: str = "raw_image",
    tile_size: int = 448,
):
    tiles = []
    meta_rows = []

    for row in tqdm(points_df.itertuples(index=False), total=len(points_df), desc="Extracting tiles", unit="tile"):
        tile = extract_tile_array(
            sdata=sdata,
            image_name=image_name,
            center_x=int(row.x),
            center_y=int(row.y),
            tile_size=tile_size,
        )

        tiles.append(Image.fromarray(tile.astype(np.uint8)))
        meta_rows.append(
            {
                "point_id": str(row.point_id),
                "x": float(row.x),
                "y": float(row.y),
            }
        )

    meta_df = pd.DataFrame(meta_rows)
    return tiles, meta_df

def compute_conch_embeddings_for_points(
    sdata,
    points_df: pd.DataFrame,
    image_name: str = "raw_image",
    tile_size: int = 448,
    conch_model_path: str | Path = "/Users/henrikheiland/CONCH/checkpoints/conch/pytorch_model.bin",
    batch_size: int = 32,
    device: str | None = None,
):
    """
    Compute one CONCH embedding per point/tile.

    Returns
    -------
    emb_df : pd.DataFrame
        Columns:
        - point_id
        - x
        - y
        - CONCH_1 ... CONCH_n
    """
    tiles, meta_df = extract_tiles_for_points(
        sdata=sdata,
        points_df=points_df,
        image_name=image_name,
        tile_size=tile_size,
    )

    embedder = ConchEmbedder(
        model_path=conch_model_path,
        device=device,
    )

    embs = embedder.embed_pil(tiles, batch_size=batch_size)

    emb_cols = [f"CONCH_{i+1}" for i in range(embs.shape[1])]
    emb_df = pd.DataFrame(embs, columns=emb_cols)
    emb_df = pd.concat([meta_df.reset_index(drop=True), emb_df], axis=1)

    return emb_df

import numpy as np
import pandas as pd
import anndata as ad


def build_adata_with_conch(
    qc_df_diag: pd.DataFrame,
    emb_df: pd.DataFrame,
):
    qc_df_diag = qc_df_diag.copy()
    emb_df = emb_df.copy()

    qc_df_diag["point_id"] = qc_df_diag["point_id"].astype(str)
    emb_df["point_id"] = emb_df["point_id"].astype(str)

    merged = qc_df_diag.merge(
        emb_df,
        on=["point_id", "x", "y"],
        how="inner",
    ).copy()

    merged = merged.set_index("point_id")

    conch_cols = [c for c in merged.columns if c.startswith("CONCH_")]
    X_conch = merged[conch_cols].to_numpy(dtype=np.float32)

    obs = merged.drop(columns=conch_cols).copy()

    adata = ad.AnnData(
        X=X_conch,
        obs=obs,
    )

    adata.obsm["X_conch"] = X_conch
    adata.obsm["spatial"] = merged[["x", "y"]].to_numpy(dtype=np.float32)

    return adata


from spatialdata.models import TableModel


def attach_adata_to_sdata(
    sdata,
    adata,
    table_name: str = "tile_table",
    region_name: str = "diagnostic_points",
    region_key: str = "region",
    instance_key: str = "point_id",
):
    adata = adata.copy()

    # make sure instance key exists as a real obs column
    adata.obs[instance_key] = adata.obs_names.astype(str)

    # region annotation required for SpatialData tables
    adata.obs[region_key] = region_name

    table = TableModel.parse(
        adata,
        region=region_name,
        region_key=region_key,
        instance_key=instance_key,
    )

    sdata.tables[table_name] = table
    return sdata

import scanpy as sc

def run_umap_on_conch(
    adata,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    resolution: float = 0.5,
    use_rep: str = "X_conch",
):
    # 1. Build neighborhood graph
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
    )

    # 2. UMAP
    sc.tl.umap(adata, min_dist=min_dist)

    # 3. Clustering (Leiden)
    sc.tl.leiden(adata, resolution=resolution)

    return adata


## Store data

from pathlib import Path
import torch


def save_pyg_result(
    result: dict,
    out_path: str | Path,
    patient_id: str | None = None,
    sample_id: str | None = None,
    image_id: str | None = None,
    image_path: str | None = None,
    graph_method: str | None = None,
):
    """
    Save only the PyG-relevant result content.

    Parameters
    ----------
    result
        Output of build_pyg_dataset_from_spatial_adata()
    out_path
        File path, e.g. ".../sample_01_pyg.pt"
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # annotate graphs with origin metadata before saving
    pyg_dataset_by_hop = result["pyg_dataset_by_hop"]

    for hop, dataset in pyg_dataset_by_hop.items():
        for data in dataset:
            data.patient_id = patient_id
            data.sample_id = sample_id
            data.image_id = image_id
            data.image_path = image_path
            data.graph_method = graph_method
            data.hop = hop

    payload = {
        "pyg_dataset_by_hop": pyg_dataset_by_hop,
        "prune_threshold": result.get("prune_threshold", None),
        "graph_method": graph_method,
        "patient_id": patient_id,
        "sample_id": sample_id,
        "image_id": image_id,
        "image_path": image_path,
        "available_hops": sorted(list(pyg_dataset_by_hop.keys())),
    }

    torch.save(payload, out_path)
    print(f"Saved PyG result to: {out_path}")



from pathlib import Path
import torch


def load_pyg_result(
    in_path: str | Path,
    map_location: str = "cpu",
):
    in_path = Path(in_path)
    payload = torch.load(
        in_path,
        map_location=map_location,
        weights_only=False,   # important for PyG Data objects
    )
    return payload


#### Full Pipeline

def run_preprocessing(image_path, meta, patient, return_folder, min_tiles=50):
  
  data = return_folder
  
  print('##### Step 1: Preprocess Image ##### ')

  sdata = load_image_as_multiscale_sdata(
      image_path,
      image_name="raw_image",
      crop_margin=200,
      scale_factors=(2, 2, 2),
      to_grayscale=False,   # set True if you want 1 channel
  )
  
  sdata = generate_random_points(
      sdata,
      image_name="raw_image",  
      n_points=500,
      patch_size=448,
      min_distance=200)
  sdata = points_to_tiles(
      sdata,
      points_name="random_points",
      shapes_name="tiles_448",
      tile_size=448,
  )
  
  print('##### Step 2: QC ##### ')
  qc_df = compute_qc_for_all_tiles(
      sdata,
      image_name="raw_image",
      points_name="random_points",
      tile_size=448,
  )
  
  print('##### Step 3: Predict Diagnostic value ##### ')
  qc_df = add_diagnostic_score_to_qc_df(
      sdata=sdata,
      qc_df=qc_df,
      model_path="/Users/henrikheiland/Desktop/Project_Raman/Raman_Project/Package/models/diagnostic_model.pth",
      image_name="raw_image",
      batch_size=32,
  )
  
  sdata, qc_df_diag = subset_sdata_by_diagnostic_score(
      sdata,
      qc_df=qc_df,
      threshold=0.8,
      prams = "diagnostic_score",
      points_name="random_points",
      shapes_name="tiles_448",
      new_points_name="diagnostic_points",
      new_shapes_name="diagnostic_tiles",
  )
  
  n_tiles = len(qc_df_diag)
  if n_tiles < min_tiles:
      print(f"❌ Skipping patient {patient}: only {n_tiles} diagnostic tiles (<{min_tiles})")
      return None
  
  
  print('##### Step 4: Feature Extraction ##### ')
  
  conch_model_path = "/Users/henrikheiland/CONCH/checkpoints/conch/pytorch_model.bin"

  pts_diag = sdata.points["diagnostic_points"]
  if hasattr(pts_diag, "compute"):
      pts_diag = pts_diag.compute()
  
  emb_df = compute_conch_embeddings_for_points(
      sdata=sdata,
      points_df=pts_diag,
      image_name="raw_image",
      tile_size=448,
      conch_model_path=conch_model_path,
      batch_size=32,
  )
  
  print(emb_df.shape)
  print(emb_df.iloc[:5, :8])
  
  adata = build_adata_with_conch(
      qc_df_diag=qc_df_diag,
      emb_df=emb_df,
  )
  adata.X = adata.obsm["X_conch"].copy()
  
  sdata = attach_adata_to_sdata(
      sdata=sdata,
      adata=adata,
      table_name="diagnostic_conch_table",
      region_name="diagnostic_points",
  )
  
  print('##### Step 5: Add meta information ##### ')
  vars_to_add = ["Sample.Number.internal", "image_id", "Primary", "GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC", "MGMTStatus"]
  for var in vars_to_add:
    sdata.tables["diagnostic_conch_table"].obs[var] = meta[var].values[patient]


  print('##### Step 6: Build PYG ##### ')
  label_cols = ["GBM_MES_TYP", "GBM_MES_ATYP", "GBM_RTK1", "GBM_RTK2", "GBM_PNC"]
  
  result = build_pyg_dataset_from_spatial_adata(
      sdata=sdata,
      adata=sdata.tables["diagnostic_conch_table"],
      points_name="diagnostic_points",
      graph_method="delaunay",
      prune_quantile=0.90,
      hop_values=[1, 2, 3],
      max_overlap_fraction=0.2,
      min_nodes=5,
      feature_key="X_conch",
      label_cols=label_cols,
      extra_graph_obs_cols=["Sample.Number.internal", "image_id"],
  )
  print('##### Step 7: Store Data ##### ')
  
  patient_id = sdata.tables["diagnostic_conch_table"].obs["Sample.Number.internal"].values[0]
  image_id = sdata.tables["diagnostic_conch_table"].obs["image_id"].values[0]
  out_path=f"{data}/{image_id}_pyg.pt"
  
  sdata.write(f'{data}/{image_id}.zarr')
  
  save_pyg_result(
      result=result,
      out_path=out_path,
      patient_id=patient_id,
      sample_id=image_id,
      image_id=image_id,
      image_path=image_path,
      graph_method="delaunay",
  )





