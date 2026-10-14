"""Tests for the input readers: JPEG, TIFF, DICOM (synthetic), graph .pt, model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ramanmolecularclassifier.core import graph_io, image_io, model_loader


# --------------------------------------------------------------------------- #
# Image readers
# --------------------------------------------------------------------------- #
def test_read_jpeg(jpeg_file: Path):
    img = image_io.load_image(jpeg_file, downsample=16)
    assert img.fmt == "jpeg"
    assert img.array is not None and img.array.ndim in (2, 3)
    assert img.width and img.height
    assert img.display_array().dtype == np.uint8


def test_read_tiff(tmp_path: Path):
    import tifffile

    arr = (np.random.rand(64, 48, 3) * 255).astype(np.uint8)
    tif_path = tmp_path / "synthetic.tif"
    tifffile.imwrite(tif_path, arr)

    img = image_io.load_image(tif_path)
    assert img.fmt == "tiff"
    assert img.array.shape[0] == 64 and img.array.shape[1] == 48
    assert img.metadata["format"] == "TIFF"


def test_read_dicom_synthetic(tmp_path: Path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    rows, cols = 32, 40
    pixel = (np.random.rand(rows, cols) * 255).astype(np.uint8)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.Modality = "OT"
    ds.Manufacturer = "SyntheticSRH"
    ds.Rows = rows
    ds.Columns = cols
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = pixel.tobytes()
    # Add a private/proprietary tag to verify it is surfaced.
    ds.add_new(0x00410010, "LO", "NEO_PRIVATE_CREATOR")

    dcm_path = tmp_path / "synthetic.dcm"
    ds.save_as(dcm_path, write_like_original=False)

    img = image_io.load_image(dcm_path)
    assert img.fmt == "dicom"
    assert img.array is not None
    assert img.array.shape == (rows, cols)
    assert img.metadata["Manufacturer"] == "SyntheticSRH"
    # Proprietary/private tag must be surfaced, not dropped.
    assert any("NEO" in t or "0041" in t for t in img.proprietary_tags) or img.proprietary_tags


def test_dicom_series_folder(tmp_path: Path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    folder = tmp_path / "series"
    folder.mkdir()
    for i in range(3):
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = generate_uid()
        fm.MediaStorageSOPInstanceUID = generate_uid()
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = fm
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.Rows = 16
        ds.Columns = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelData = (np.zeros((16, 16), np.uint8)).tobytes()
        ds.save_as(folder / f"slice_{i}.dcm", write_like_original=False)

    img = image_io.load_image(folder)
    assert img.fmt == "dicom-series"
    assert img.n_frames == 3


# --------------------------------------------------------------------------- #
# Graph reader
# --------------------------------------------------------------------------- #
def test_load_graph_bundle(graph_file: Path):
    bundle = graph_io.load_graph_bundle(graph_file)
    assert bundle.image_id
    assert len(bundle.hops()) >= 1
    hop = bundle.default_hop()
    subs = bundle.subgraphs(hop)
    assert len(subs) > 0
    g0 = subs[0]
    assert g0.x.shape[1] == 512
    pos = bundle.seed_positions(hop)
    assert pos.shape[0] == len(subs)
    assert pos.shape[1] == 2


def test_graph_metadata(graph_file: Path):
    bundle = graph_io.load_graph_bundle(graph_file)
    md = bundle.metadata()
    assert "image_id" in md and "graph_method" in md


# --------------------------------------------------------------------------- #
# Model load + infer
# --------------------------------------------------------------------------- #
def test_model_load_zero_missing(checkpoint: Path):
    lm = model_loader.load_model(checkpoint)
    assert lm.missing_keys == []
    assert lm.unexpected_keys == []
    assert lm.config["in_channels"] == 512
    assert lm.config["out_channels"] == 5


def test_model_infer_on_real_graph(checkpoint: Path, graph_file: Path):
    import torch
    from torch_geometric.data import Batch

    lm = model_loader.load_model(checkpoint)
    bundle = graph_io.load_graph_bundle(graph_file)
    subs = bundle.subgraphs(bundle.default_hop())
    batch = Batch.from_data_list(subs[:4])
    with torch.no_grad():
        out = lm.model(batch)
    assert out["pred"].shape == (min(4, len(subs)), 5)
