"""Tests for inference, QC, heatmap, reports and the end-to-end pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ramanmolecularclassifier.core import graph_io, heatmap, inference, qc
from ramanmolecularclassifier.core.conch import CONCHEncoder, CONCHUnavailableError
from ramanmolecularclassifier.core.model_loader import load_model
from ramanmolecularclassifier.core.pipeline import analyse_graph_file


@pytest.fixture(scope="module")
def loaded(checkpoint, graph_file):
    lm = load_model(checkpoint)
    bundle = graph_io.load_graph_bundle(graph_file)
    return lm, bundle


def test_predict_image_shapes(loaded):
    lm, bundle = loaded
    hop = bundle.default_hop()
    subs = bundle.subgraphs(hop)
    pred = inference.predict_image(lm.model, subs, true_target=bundle.true_target(hop), hop=hop)
    assert pred.mean_probabilities.shape == (5,)
    assert np.isclose(pred.mean_probabilities.sum(), 1.0, atol=1e-5)
    assert 0.0 <= pred.predicted_probability <= 1.0
    assert pred.predicted_label in pred.labels
    assert len(pred.subgraph_predictions) == len(subs)
    assert pred.per_subgraph_positions().shape == (len(subs), 2)


def test_prediction_is_deterministic(loaded):
    lm, bundle = loaded
    subs = bundle.subgraphs(bundle.default_hop())
    p1 = inference.predict_image(lm.model, subs)
    p2 = inference.predict_image(lm.model, subs)
    assert np.allclose(p1.mean_probabilities, p2.mean_probabilities, atol=1e-6)


def test_qc_metrics(loaded):
    lm, bundle = loaded
    hop = bundle.default_hop()
    pred = inference.predict_image(lm.model, bundle.subgraphs(hop), hop=hop)
    report = qc.compute_qc(bundle, pred, hop=hop)
    d = report.as_dict()
    assert d["n_subgraphs"] == pred.n_subgraphs
    assert 0.0 <= d["fraction_low_confidence"] <= 1.0
    assert d["n_nodes_total"] > 0
    assert isinstance(d["diagnostic"], bool)


def test_heatmap_builds(loaded):
    lm, bundle = loaded
    hop = bundle.default_hop()
    pred = inference.predict_image(lm.model, bundle.subgraphs(hop), hop=hop)
    res = heatmap.build_heatmap(pred, scalar="predicted_class_prob")
    assert res.grid.ndim == 2
    assert res.n_points == pred.n_subgraphs


def test_conch_stub_raises():
    enc = CONCHEncoder()
    assert enc.is_available() is False
    with pytest.raises(CONCHUnavailableError):
        enc.encode_tiles(np.zeros((1, 8, 8, 3), dtype=np.uint8))


def test_pipeline_end_to_end(checkpoint, graph_file, tmp_path):
    result = analyse_graph_file(
        graph_file, checkpoint_path=str(checkpoint),
        out_dir=tmp_path, load_background=False,
    )
    assert result.reports is not None
    assert result.reports.pdf.exists()
    assert result.reports.html.exists()
    assert result.reports.xlsx.exists()
    assert result.reports.csv.exists()
    assert all(Path(p).exists() for p in result.heatmap_paths)
    assert result.prediction.predicted_label in result.prediction.labels
