"""End-to-end analysis pipeline (UI-free, head-less friendly).

Ties together graph loading, inference, QC, heatmap rendering and report
generation for one image.  Both the GUI worker thread and the example-output
generator call :func:`analyse_graph_file`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .. import __version__
from .graph_io import GraphBundle, load_graph_bundle
from .heatmap import build_heatmap, save_heatmap
from .image_io import LoadedImage, load_image
from .inference import ImagePrediction, predict_image
from .model_loader import LoadedModel, get_cached_model
from .qc import QCReport, compute_qc
from ..reporting.report_engine import (
    ReportInputs,
    ReportOutputs,
    generate_reports,
)


@dataclass
class AnalysisResult:
    bundle: GraphBundle
    prediction: ImagePrediction
    qc: QCReport
    hop: int
    heatmap_paths: List[Path]
    reports: Optional[ReportOutputs]
    runtime_seconds: float
    background_image: Optional[LoadedImage] = None


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)


def analyse_graph_file(
    graph_path: str | Path,
    loaded_model: Optional[LoadedModel] = None,
    checkpoint_path: Optional[str] = None,
    hop: Optional[int] = None,
    out_dir: Optional[str | Path] = None,
    make_reports: bool = True,
    load_background: bool = True,
    image_downsample: int = 8,
    rbf_heatmap: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> AnalysisResult:
    """Run the full analysis on one precomputed graph file.

    Parameters
    ----------
    graph_path:
        Path to an ``Img_*_pyg.pt`` graph bundle.
    loaded_model / checkpoint_path:
        Either pass an already-loaded model, or a checkpoint path (cached load).
    hop:
        Hop level to use; defaults to the bundle's largest hop.
    out_dir:
        Where reports + heatmaps are written.  Defaults to ``<graph_dir>/analysis``.
    load_background:
        Try to load the SRH image referenced by the graph for heatmap overlay.
    image_downsample:
        Down-sampling stride for the (large) background SRH image.
    """
    t0 = time.time()
    graph_path = Path(graph_path)
    _log(log, f"Loading graph bundle: {graph_path.name}")
    bundle = load_graph_bundle(graph_path)

    if loaded_model is None:
        _log(log, "Loading model (cached) ...")
        loaded_model = get_cached_model(checkpoint_path)
    model = loaded_model.model

    use_hop = hop if hop is not None else bundle.default_hop()
    subs = bundle.subgraphs(use_hop)
    _log(log, f"Running inference on {len(subs)} subgraph(s) at hop {use_hop} ...")

    prediction = predict_image(
        model, subs, device=loaded_model.device,
        true_target=bundle.true_target(use_hop), hop=use_hop,
    )
    _log(log, f"Predicted: {prediction.predicted_label} "
              f"(p={prediction.predicted_probability:.3f})")

    qc = compute_qc(bundle, prediction, hop=use_hop)
    _log(log, f"QC: {'PASS' if qc.diagnostic else 'REVIEW'} - {qc.notes}")

    out_dir = Path(out_dir) if out_dir else graph_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_id = bundle.image_id or graph_path.stem

    # Background image (optional).
    background = None
    bg_array = None
    ds = 1
    if load_background and bundle.image_path:
        try:
            background = load_image(bundle.image_path, downsample=image_downsample)
            bg_array = background.display_array()
            ds = image_downsample
            _log(log, f"Loaded background SRH image ({background.fmt}).")
        except Exception as exc:  # noqa: BLE001
            _log(log, f"Background image unavailable: {exc}")

    # Heatmaps: predicted-class probability + confidence.
    heatmap_paths: List[Path] = []
    for scalar, suffix in (("predicted_class_prob", "predclass"), ("confidence", "confidence")):
        hm = build_heatmap(
            prediction, scalar=scalar,
            background_image=bg_array, image_downsample=ds,
            rbf=rbf_heatmap,
        )
        png = save_heatmap(hm, out_dir / f"{image_id}_heatmap_{suffix}.png")
        heatmap_paths.append(png)
    # One PDF heatmap as well (predicted class).
    hm_pdf = build_heatmap(
        prediction, scalar="predicted_class_prob",
        background_image=bg_array, image_downsample=ds, rbf=rbf_heatmap,
    )
    heatmap_paths.append(save_heatmap(hm_pdf, out_dir / f"{image_id}_heatmap_predclass.pdf"))
    _log(log, f"Wrote {len(heatmap_paths)} heatmap file(s).")

    runtime = time.time() - t0

    reports = None
    if make_reports:
        graph_config = {
            "graph_method": bundle.graph_method,
            "hop_used": use_hop,
            "available_hops": bundle.available_hops,
            "prune_threshold": bundle.prune_threshold,
            "n_subgraphs": len(subs),
            "node_feature_dim": 512,
        }
        preprocessing = {
            "node_features": "CONCH 512-d SRH tile embeddings (precomputed)",
            "edge_attr": "Delaunay edge distance (1-d)",
            "aggregation": "softmax per subgraph -> mean over subgraphs",
            "device": loaded_model.device,
            "image_downsample": ds,
        }
        inp = ReportInputs(
            image_id=image_id,
            prediction=prediction,
            qc=qc,
            metadata=bundle.metadata(),
            model_version=loaded_model.version_string,
            graph_config=graph_config,
            preprocessing=preprocessing,
            runtime_seconds=runtime,
            heatmap_paths=heatmap_paths,
            out_dir=out_dir,
        )
        _log(log, "Generating reports (PDF / HTML / xlsx / CSV) ...")
        reports = generate_reports(inp)
        _log(log, "Reports written.")

    return AnalysisResult(
        bundle=bundle,
        prediction=prediction,
        qc=qc,
        hop=use_hop,
        heatmap_paths=heatmap_paths,
        reports=reports,
        runtime_seconds=time.time() - t0,
        background_image=background,
    )
