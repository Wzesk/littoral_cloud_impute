"""Cloud removal utilities using VPint2 and SEnSeIv2.

This module provides a simple function to perform cloud removal on windowed
Sentinel-2 data using:
- SEnSeIv2 for cloud/shadow segmentation (to build a mask)
- VPint2 for cloud pixel imputation using a companion feature image

Notes:
- Heavy model assets are cached so repeated calls are fast.
- File loading supports Sentinel SAFE .zip and GeoTIFF .tif/.tiff.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Prefer environment overrides for local package paths, else fall back to known paths
_VPINT_PATH = os.environ.get("VPINT_PATH", "/home/walter_littor_al/VPint")
_SENSEIV2_PATH = os.environ.get("SENSEIV2_PATH", "/home/walter_littor_al/SEnSeIv2")
if _VPINT_PATH and _VPINT_PATH not in sys.path:
    sys.path.append(_VPINT_PATH)
if _SENSEIV2_PATH and _SENSEIV2_PATH not in sys.path:
    sys.path.append(_SENSEIV2_PATH)

# VPint2 imports
from VPint.VPint2 import VPint2_interpolator  # type: ignore
from VPint.utils.EO_utils import (  # type: ignore
    load_product_windowed,
    load_tiff_windowed,
)

# SEnSeIv2 imports
from senseiv2.inference import CloudMask  # type: ignore
from senseiv2.utils import get_model_files  # type: ignore
from senseiv2.constants import SENTINEL2_BANDS, SENTINEL2_DESCRIPTORS  # type: ignore
from skimage.transform import resize  # type: ignore
import re
import csv

def remove_clouds(
    target_path: str,
    features_path: str,
    *,
    y_size: int = 256,
    x_size: int = 256,
    y_offset: Optional[int] = None,
    x_offset: Optional[int] = None,
    threshold: float = 0.2,
    include_shadow: bool = False,
    device: str = "auto",
    return_mask: bool = False,
):
    """Impute cloudy pixels using VPint2 guided by a SEnSeIv2 cloud mask.

    Steps:
    1) Load co-windowed target and features (SAFE .zip or GeoTIFF .tif/.tiff)
    2) Segment clouds/shadows on the target with SEnSeIv2 to build a binary mask
    3) Run VPint2 interpolation with the mask to fill cloudy pixels

    Args:
        target_path: Path to target image (.SAFE.zip or .tif/.tiff)
        features_path: Path to features image (.SAFE.zip or .tif/.tiff)
        y_size, x_size: Window size in pixels
        y_offset, x_offset: Optional window top-left offsets in pixels. If omitted,
            the window is centered within the target image by default.
        threshold: VPint2 similarity threshold
        include_shadow: If True, treat model's shadow class as cloudy
        device: "auto" to pick CUDA when available, else "cpu"; or explicit "cuda"/"cpu"
        return_mask: If True, also return the binary cloud mask used

    Returns:
        pred: ndarray[H, W, C] of imputed target window
        mask (optional): ndarray[H, W] uint8 binary mask used for imputation
    """
    # Compute centered offsets if not provided
    if x_offset is None or y_offset is None:
        size_y, size_x = _get_image_size(target_path)
        if y_size > size_y or x_size > size_x:
            raise ValueError(
                f"Requested window {x_size}x{y_size} exceeds target size {size_x}x{size_y}."
            )
        if x_offset is None:
            x_offset = max(0, (size_x - x_size) // 2)
        if y_offset is None:
            y_offset = max(0, (size_y - y_size) // 2)

    print(f"Extracting {y_size}x{x_size} window at ({x_offset}, {y_offset})")

    target = _load_window(target_path, y_size, x_size, y_offset, x_offset)
    features = _load_window(features_path, y_size, x_size, y_offset, x_offset)

    cld_mask = build_cloud_mask(target, include_shadow=include_shadow, device=device)
    if cld_mask.shape != target.shape[:2]:
        # Ensure nearest-neighbor resize for masks
        cld_mask = resize(
            cld_mask.astype("float32"), target.shape[:2], order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

    vp = VPint2_interpolator(target, features, mask=cld_mask, bands_first=False, threshold=threshold)
    pred = vp.run()
    return (pred, cld_mask) if return_mask else pred



def preprocess_s2_window(
    target_arr: np.ndarray,
    band_names: Optional[Sequence[str]] = None,
    min_size: int = 1068,
) -> Tuple[np.ndarray, List[dict]]:
    """Convert a VPint-style S2 window (H, W, C uint16) to SEnSeIv2 input.

    Produces channel-first float array and corresponding band descriptors.
    Resizes to min_size if the window is smaller than the model's preferred size.
    """
    vpint_index = {
        "B01": 0,
        "B02": 1,
        "B03": 2,
        "B04": 3,
        "B05": 4,
        "B06": 5,
        "B07": 6,
        "B08": 7,
        "B8A": 8,
        "B09": 9,
        "B11": 10,
        "B12": 11,
    }
    if band_names is None:
        band_names = list(vpint_index.keys())
    band_names = [b for b in band_names if b in vpint_index]
    band_idxs = [vpint_index[b] for b in band_names]

    h, w = target_arr.shape[:2]
    sub = target_arr[..., band_idxs].astype("float32")
    sub = (sub / 10000.0).clip(0.0, 1.0)
    if h < min_size or w < min_size:
        im_resized = np.zeros((len(band_idxs), min_size, min_size), dtype="float32")
        for i in range(len(band_idxs)):
            im_resized[i, ...] = resize(
                sub[..., i], (min_size, min_size), order=1, preserve_range=True, anti_aliasing=False
            )
        im = im_resized
    else:
        im = sub.transpose(2, 0, 1)

    s2_desc_by_name = {b["name"]: d for b, d in zip(SENTINEL2_BANDS, SENTINEL2_DESCRIPTORS)}
    descriptors = [s2_desc_by_name[b] for b in band_names]
    return im, descriptors

def _select_device(requested: str) -> str:
    """Resolve device string. "auto" -> cuda if available else cpu."""
    if requested.lower() == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return requested


@lru_cache(maxsize=1)
def senseiv2_setup(device: str = "auto") -> Tuple[CloudMask, List[str]]:
    """Load and cache the SEnSeIv2 model and default S2 band names.

    Returns a tuple of (model, band_names) where band_names are a subset of S2 bands
    expected by the model and present in VPint arrays.
    """
    DEVICE = _select_device(device)
    model_name = "SEnSeIv2-SegFormerB2-alldata-ambiguous"
    config, weights = get_model_files(model_name)
    model = CloudMask(config, weights, verbose=False, categorise=True, device=DEVICE)
    band_names = [
        b["name"]
        for b in SENTINEL2_BANDS
        if b["name"] in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    ]
    return model, band_names


def build_cloud_mask(
    target: np.ndarray,
    *,
    include_shadow: bool = False,
    device: str = "auto",
    stride: int = 357,
) -> np.ndarray:
    """Generate a binary cloud mask (H, W) from a target window using SEnSeIv2.

    include_shadow: include the shadow class in the mask if True.
    stride: SEnSeIv2 inference stride.
    """
    model, band_names = senseiv2_setup(device)
    im, descriptors = preprocess_s2_window(target, band_names)

    mask4 = model(im, descriptors=descriptors, stride=stride)
    # Expected classes: 0-clear, 1-cloud, 2-thick cloud, 3-shadow
    cloud_like = (mask4 == 1) | (mask4 == 2)
    if include_shadow:
        cloud_like = cloud_like | (mask4 == 3)
    return cloud_like.astype(np.uint8)


def _load_window(path: str, y_size: int, x_size: int, y_offset: int, x_offset: int) -> np.ndarray:
    """Helper to load a window from a SAFE zip or GeoTIFF."""
    lower = path.lower()
    if lower.endswith(".zip"):
        return load_product_windowed(path, y_size, x_size, y_offset, x_offset)
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        return load_tiff_windowed(path, y_size, x_size, y_offset, x_offset)
    raise ValueError(f"Unsupported file format: {path}")


def _get_image_size(path: str) -> Tuple[int, int]:
    """Return (height, width) for a SAFE zip or GeoTIFF using rasterio.

    For SAFE zips, opens the first subdataset to infer base resolution size.
    """
    import rasterio  # local import to avoid hard dep at import time

    lower = path.lower()
    if lower.endswith(".zip"):
        with rasterio.open(path) as raw:
            subdatasets = raw.subdatasets
        if not subdatasets:
            raise ValueError("SAFE product has no subdatasets: " + path)
        with rasterio.open(subdatasets[1]) as ds:
            return ds.height, ds.width
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        with rasterio.open(path) as ds:
            return ds.height, ds.width
    raise ValueError(f"Unsupported file format: {path}")


def _extract_image_date(tags: dict, filename: str) -> str:
    """Extract an image date string from raster tags or filename.

    Tries common tag keys first, then falls back to parsing YYYYMMDD or
    YYYYMMDDTHHMMSS from filename. Returns an ISO-like string if possible.
    """
    # Try common tag keys
    cand_keys = [
        "TIFFTAG_DATETIME",
        "DATETIME",
        "DateTime",
        "ACQUISITION_DATE",
        "ACQUISITIONDATETIME",
        "SENSING_TIME",
    ]
    for k in cand_keys:
        v = tags.get(k)
        if v:
            # Format variants:
            #  - TIFFTAG_DATETIME: 'YYYY:MM:DD HH:MM:SS'
            #  - Others: ISO-ish or freeform
            m = re.match(r"(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", v)
            if m:
                y, mo, d, h, mi, s = m.groups()
                return f"{y}-{mo}-{d}T{h}:{mi}:{s}"
            m = re.match(r"(\d{4})[-:]?(\d{2})[-:]?(\d{2})[ T]?([0-9:]{0,8})", v)
            if m:
                y, mo, d, t = m.groups()
                if t:
                    t = t.replace(":", "")
                    t = (t + "000000")[:6]
                    return f"{y}-{mo}-{d}T{t[0:2]}:{t[2:4]}:{t[4:6]}"
                return f"{y}-{mo}-{d}"
            return str(v)

    # Fallback to filename patterns: YYYYMMDDTHHMMSS or YYYYMMDD
    m = re.search(r"(\d{8})T(\d{6})", filename)
    if m:
        ymd, hms = m.groups()
        return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}T{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"
    m = re.search(r"(\d{8})", filename)
    if m:
        ymd = m.group(1)
        return f"{ymd[0:4]}-{ymd[4:6]}-{ymd[6:8]}"
    return ""


__all__ = [
    "remove_clouds",
    "build_cloud_mask",
    "preprocess_s2_window",
    "senseiv2_setup",
    "batch_remove_clouds_folder",
]


def batch_remove_clouds_folder(
    folder: str,
    *,
    include_shadow: bool = False,
    device: str = "auto",
    overwrite: bool = False,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    executor: str = "thread",
) -> List[str]:
    """Process all .tif/.tiff in a folder and save cloudless predictions.

    Steps:
      1) Scan folder for TIFFs (non-recursive)
      2) Compute cloud mask for each (full-image) and choose the image with lowest
         cloud fraction as the 'features' reference
      3) Create '<folder>/cloudless' if missing
      4) Run remove_clouds for each image (including the feature image) with
         full-image window and save '<name>_pred.tif' to 'cloudless'

    Returns:
      List of saved file paths under the 'cloudless' folder.
    """
    # Collect TIFF files
    names = [n for n in os.listdir(folder) if n.lower().endswith((".tif", ".tiff"))]
    paths = [os.path.join(folder, n) for n in names]
    if not paths:
        raise FileNotFoundError(f"No TIFF files found in folder: {folder}")

    # Determine worker count based on device and requested parallelism
    resolved_device = _select_device(device)
    if not parallel:
        workers = 1
    else:
        if max_workers is not None:
            workers = max(1, max_workers)
        else:
            try:
                cpu_n = os.cpu_count() or 4
            except Exception:
                cpu_n = 4
            # On CUDA default to 1 to avoid GPU OOM/contention unless overridden
            workers = 1 if resolved_device == "cuda" else min(cpu_n, len(paths))

    # Stage 1: Compute masks and cloud fractions (optionally parallel)
    cloud_fractions: List[float] = [1.0] * len(paths)
    sizes: List[Tuple[int, int]] = [(0, 0)] * len(paths)

    def _compute_cloud_fraction(idx_path: Tuple[int, str]) -> Tuple[int, Tuple[int, int], float]:
        idx, p = idx_path
        try:
            size_y, size_x = _get_image_size(p)
            img = load_tiff_windowed(p, size_y, size_x, 0, 0)
            mask = build_cloud_mask(img, include_shadow=include_shadow, device=device)
            frac = float(mask.mean()) if mask.size else 1.0
            return idx, (size_y, size_x), frac
        except Exception as e:
            print(f"[WARN] Failed to compute mask for {p}: {e}")
            return idx, (0, 0), 1.0

    if workers == 1:
        for idx, p in enumerate(paths):
            i, size, frac = _compute_cloud_fraction((idx, p))
            sizes[i] = size
            cloud_fractions[i] = frac
    else:
        import concurrent.futures as cf

        Exec = cf.ThreadPoolExecutor if executor == "thread" else cf.ProcessPoolExecutor
        with Exec(max_workers=workers) as pool:
            for i, size, frac in pool.map(_compute_cloud_fraction, list(enumerate(paths))):
                sizes[i] = size
                cloud_fractions[i] = frac

    # Choose features image (lowest cloud fraction)
    best_idx = int(np.argmin(cloud_fractions))
    features_path = paths[best_idx]
    feat_h, feat_w = sizes[best_idx]
    if feat_h <= 0 or feat_w <= 0:
        raise RuntimeError("Failed to determine size of the selected features image.")
    print(f"Selected features image: {os.path.basename(features_path)} (cloud={cloud_fractions[best_idx]:.3f})")

    # Prepare output folder
    out_dir = os.path.join(folder, "cloudless")
    os.makedirs(out_dir, exist_ok=True)

    saved: List[str] = []
    report_rows: List[List[str]] = []

    def _process_and_save(p: str) -> Optional[Tuple[str, List[str]]]:
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}_pred.tif")
        if (not overwrite) and os.path.exists(out_path):
            print(f"[SKIP] Exists: {out_path}")
            # Build quick metadata row from existing file
            try:
                import rasterio  # type: ignore
                with rasterio.open(p) as src:
                    height, width = src.height, src.width
                    crs = str(src.crs) if src.crs else ""
                    transform = str(src.transform)
                    tags = src.tags()
                    band_names = [d for d in (src.descriptions or []) if d]
                image_date = _extract_image_date(tags, os.path.basename(p))
                cloudless_size = f"{height}x{width}"
                original_size = cloudless_size
                # We don’t recompute mask here; set unknown
                cloud_pct = ""
                row = [
                    os.path.basename(p),
                    image_date,
                    cloud_pct,
                    original_size,
                    cloudless_size,
                    crs,
                    transform,
                    ";".join(band_names),
                ]
            except Exception:
                row = [os.path.basename(p), "", "", "", "", "", "", ""]
            return out_path, row
        try:
            # Ensure matching size; if different, skip to avoid VPint shape mismatch
            h, w = _get_image_size(p)
            if (h, w) != (feat_h, feat_w):
                print(f"[WARN] Skipping {p}: size {w}x{h} != features {feat_w}x{feat_h}")
                return None

            pred = remove_clouds(
                p,
                features_path,
                y_size=h,
                x_size=w,
                y_offset=0,
                x_offset=0,
                include_shadow=include_shadow,
                device=device,
            )

            # Save prediction as GeoTIFF, copying georeferencing when possible
            try:
                import rasterio  # type: ignore

                height, width, bands = pred.shape
                with rasterio.open(p) as src:
                    meta = src.meta.copy()
                meta.update(driver="GTiff", height=height, width=width, count=bands, dtype=pred.dtype)
                with rasterio.open(out_path, "w", **meta) as dst:
                    for i in range(bands):
                        dst.write(pred[:, :, i], i + 1)
            except Exception:
                # Fallback: write a plain raster if metadata copy fails
                import rasterio  # type: ignore

                height, width, bands = pred.shape
                with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=bands,
                    dtype=pred.dtype,
                ) as dst:
                    for i in range(bands):
                        dst.write(pred[:, :, i], i + 1)

            print(f"[OK] Saved: {out_path}")

            # Collect metadata for report
            try:
                import rasterio  # type: ignore
                with rasterio.open(p) as src:
                    height, width = src.height, src.width
                    crs = str(src.crs) if src.crs else ""
                    transform = str(src.transform)
                    tags = src.tags()
                    band_names = [d for d in (src.descriptions or []) if d]
                image_date = _extract_image_date(tags, os.path.basename(p))
                original_size = f"{height}x{width}"
                cloudless_size = original_size  # outputs match input size

                # Recompute cloud fraction here to record exact value used for selection
                img = load_tiff_windowed(p, height, width, 0, 0)
                mask = build_cloud_mask(img, include_shadow=include_shadow, device=device)
                cloud_pct = f"{(float(mask.mean()) * 100.0):.2f}"
                row = [
                    os.path.basename(p),
                    image_date,
                    cloud_pct,
                    original_size,
                    cloudless_size,
                    crs,
                    transform,
                    ";".join(band_names),
                ]
            except Exception:
                row = [os.path.basename(p), "", "", "", "", "", "", ""]
            return out_path, row
        except Exception as e:
            print(f"[ERROR] Failed processing {p}: {e}")
            return None

    if workers == 1:
        for p in paths:
            out = _process_and_save(p)
            if out:
                out_path, row = out
                saved.append(out_path)
                report_rows.append(row)
    else:
        import concurrent.futures as cf

        Exec = cf.ThreadPoolExecutor if executor == "thread" else cf.ProcessPoolExecutor
        with Exec(max_workers=workers) as pool:
            for out in pool.map(_process_and_save, paths):
                if out:
                    out_path, row = out
                    saved.append(out_path)
                    report_rows.append(row)

    # Write CSV report
    report_path = os.path.join(out_dir, "cloudless_report.csv")
    header = [
        "image_name",
        "image_date",
        "cloud_coverage %",
        "original_image_size",
        "cloudless_image_size",
        "image_CRS",
        "image_Transform",
        "band_names",
    ]
    try:
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(report_rows)
        print(f"[OK] Report saved: {report_path}")
    except Exception as e:
        print(f"[WARN] Failed to write report {report_path}: {e}")

    return saved