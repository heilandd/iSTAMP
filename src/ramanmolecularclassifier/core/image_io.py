"""Read SRH images and their metadata.

Supports:

* **JPEG / PNG** via Pillow (the converted SRH images present on disk).
* **TIFF / multi-page TIFF** via tifffile (falls back to Pillow).
* **DICOM** single files *and* folders / series via pydicom.

For each format we extract a small, display-friendly metadata dict and an
``numpy`` image array (down-sampled on request so the GUI can show huge
5400x5900 SRH frames without exhausting memory).  Proprietary / vendor-specific
DICOM tags (e.g. SRH "NEO" private tags) are surfaced rather than silently
dropped, so the user can see that they exist even when they cannot be decoded.

This module never embeds images into CONCH features - see
:mod:`ramanmolecularclassifier.core.conch` for why raw-image inference is
unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

JPEG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
TIFF_EXTS = {".tif", ".tiff"}
DICOM_EXTS = {".dcm", ".dicom", ".ima"}


@dataclass
class LoadedImage:
    """An image array plus extracted metadata and provenance."""

    array: Optional[np.ndarray]      # HxW or HxWxC uint8/uint16, may be down-sampled
    path: Path
    fmt: str                         # 'jpeg' | 'tiff' | 'dicom' | 'dicom-series'
    width: Optional[int]
    height: Optional[int]
    n_frames: int = 1
    downsample: int = 1
    metadata: Dict[str, object] = field(default_factory=dict)
    proprietary_tags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def display_array(self) -> Optional[np.ndarray]:
        """Return an array safe to hand to a Qt/matplotlib display (uint8)."""
        if self.array is None:
            return None
        arr = self.array
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float64)
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            arr = (arr - mn) / (mx - mn + 1e-9) * 255.0
            arr = arr.astype(np.uint8)
        return arr


def detect_format(path: Path) -> str:
    if path.is_dir():
        return "dicom-series"
    ext = path.suffix.lower()
    if ext in JPEG_EXTS:
        return "jpeg"
    if ext in TIFF_EXTS:
        return "tiff"
    if ext in DICOM_EXTS:
        return "dicom"
    # Unknown extension: sniff DICOM magic bytes.
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            if fh.read(4) == b"DICM":
                return "dicom"
    except OSError:
        pass
    return "jpeg"  # best-effort fall-through; Pillow will raise if unreadable


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def load_jpeg(path: Path, downsample: int = 1) -> LoadedImage:
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        w, h = im.size
        arr = np.asarray(im)
    arr = _downsample(arr, downsample)
    return LoadedImage(
        array=arr,
        path=path,
        fmt="jpeg",
        width=w,
        height=h,
        downsample=downsample,
        metadata={
            "format": im.format if hasattr(im, "format") else "JPEG",
            "mode": im.mode,
            "size_px": f"{w} x {h}",
        },
    )


def load_tiff(path: Path, downsample: int = 1) -> LoadedImage:
    try:
        import tifffile

        with tifffile.TiffFile(path) as tf:
            n_frames = len(tf.pages)
            page = tf.pages[0]
            arr = page.asarray()
            meta = {
                "format": "TIFF",
                "n_pages": n_frames,
                "dtype": str(arr.dtype),
                "shape": str(arr.shape),
            }
            # Surface a few standard TIFF tags if present.
            for tag in ("ImageDescription", "Software", "Make", "Model"):
                if tag in page.tags:
                    meta[tag] = str(page.tags[tag].value)[:200]
        h, w = arr.shape[:2]
        arr = _downsample(arr, downsample)
        return LoadedImage(
            array=arr,
            path=path,
            fmt="tiff",
            width=w,
            height=h,
            n_frames=n_frames,
            downsample=downsample,
            metadata=meta,
        )
    except Exception as exc:  # noqa: BLE001 - fall back to Pillow
        img = load_jpeg(path, downsample=downsample)
        img.fmt = "tiff"
        img.warnings.append(f"tifffile failed ({exc}); read first page via Pillow.")
        return img


def _dicom_metadata(ds) -> tuple[Dict[str, object], List[str]]:
    """Extract common DICOM tags + list private / proprietary tags."""
    common = {}
    for kw in (
        "Modality",
        "Manufacturer",
        "ManufacturerModelName",
        "SeriesDescription",
        "StudyDescription",
        "Rows",
        "Columns",
        "PhotometricInterpretation",
        "BitsAllocated",
        "PatientID",
        "PixelSpacing",
    ):
        if kw in ds:
            try:
                common[kw] = str(getattr(ds, kw))
            except Exception:  # noqa: BLE001
                common[kw] = "<unreadable>"

    private = []
    for elem in ds:
        try:
            if elem.tag.is_private:
                name = elem.keyword or "<private>"
                private.append(f"{elem.tag} {name}")
        except Exception:  # noqa: BLE001
            continue
    return common, private


def load_dicom(path: Path, downsample: int = 1) -> LoadedImage:
    import pydicom

    ds = pydicom.dcmread(str(path), force=True)
    common, private = _dicom_metadata(ds)

    arr = None
    warnings: List[str] = []
    width = int(common.get("Columns", 0)) or None
    height = int(common.get("Rows", 0)) or None
    try:
        arr = ds.pixel_array
        height, width = arr.shape[:2]
        arr = _downsample(arr, downsample)
    except Exception as exc:  # noqa: BLE001
        warnings.append(
            "Pixel data could not be decoded "
            f"({exc}). Likely a proprietary/compressed SRH transfer syntax; "
            "metadata is still available."
        )

    meta = {"format": "DICOM", **common}
    return LoadedImage(
        array=arr,
        path=path,
        fmt="dicom",
        width=width,
        height=height,
        downsample=downsample,
        metadata=meta,
        proprietary_tags=private,
        warnings=warnings,
    )


def load_dicom_series(folder: Path, downsample: int = 1) -> LoadedImage:
    import pydicom

    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and (p.suffix.lower() in DICOM_EXTS or _looks_like_dicom(p))
    )
    if not files:
        raise FileNotFoundError(f"No DICOM files found in folder '{folder}'.")

    first = load_dicom(files[0], downsample=downsample)
    first.fmt = "dicom-series"
    first.n_frames = len(files)
    first.metadata["series_files"] = len(files)
    first.path = folder
    return first


def _looks_like_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def load_image(path: os.PathLike | str, downsample: int = 1) -> LoadedImage:  # type: ignore[name-defined]
    """Dispatch to the correct reader based on path / content.

    Parameters
    ----------
    path:
        File or (for DICOM series) folder path.
    downsample:
        Integer stride applied to rows & columns to limit memory for the GUI.
        ``1`` returns full resolution.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image path not found: {p}")
    fmt = detect_format(p)
    if fmt == "dicom-series":
        return load_dicom_series(p, downsample=downsample)
    if fmt == "dicom":
        return load_dicom(p, downsample=downsample)
    if fmt == "tiff":
        return load_tiff(p, downsample=downsample)
    return load_jpeg(p, downsample=downsample)


import os  # noqa: E402
