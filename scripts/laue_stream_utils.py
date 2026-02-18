#!/usr/bin/env python3
"""
laue_stream_utils.py — Shared utilities for the LaueMatching streaming workflow.

Provides:
  - Configuration parsing (classic text format used by LaueMatching)
  - H5 image loading
  - Image preprocessing pipeline (background, threshold, components, blur)
  - Output file parsing (solutions.txt, spots.txt with ImageNr column)
  - Orientation filtering by unique-spot count
  - TCP / networking helpers

These are extracted from RunImage.py so that
laue_image_server.py, laue_postprocess.py, and laue_orchestrator.py
can all share the same code without depending on the full RunImage.py.

Author: Hemant Sharma (hsharma@anl.gov)
"""

import os
import sys
import time
import json
import socket
import struct
import logging
import tempfile
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import cv2
import scipy.ndimage as ndimg

# Optional heavy imports — gracefully degrade
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import diplib as dip
    HAS_DIPLIB = True
except ImportError:
    HAS_DIPLIB = False

try:
    from skimage import filters, restoration, exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("LaueStream")

def setup_logger(
    name: str = "LaueStream",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
) -> logging.Logger:
    """Configure and return a logger."""
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s",
                            datefmt="%H:%M:%S")
    if console_output and not _logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        _logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
    return _logger

# ---------------------------------------------------------------------------
# Configuration — lightweight dict-based config for streaming
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    # Detector
    "nr_px_x":          2048,
    "nr_px_y":          2048,
    "px_x":             0.2,
    "px_y":             0.2,
    "distance":         0.513,
    # Energy range
    "elo":              5.0,
    "ehi":              30.0,
    # Crystallography
    "space_group":      225,
    "symmetry":         "F",
    "lattice_parameter":"0.3615 0.3615 0.3615 90 90 90",
    "r_array":          "-1.2 -1.2 -1.2",
    "p_array":          "0 0 0.513",
    # Thresholding
    "threshold_method": "adaptive",
    "threshold_value":  0.0,
    "threshold_percentile": 90.0,
    # Image processing
    "min_area":         10,
    "filter_radius":    101,
    "median_passes":    1,
    "watershed_enabled":True,
    "gaussian_factor":  0.25,
    "enhance_contrast": False,
    "denoise_image":    False,
    "denoise_strength": 1.0,
    "edge_enhancement": False,
    # Matching
    "min_intensity":    0.0,
    "min_good_spots":   2,
    "min_nr_spots":     5,
    "max_angle":        2.0,
    "max_laue_spots":   400,
    "orientation_spacing": 0.4,
    # Files
    "background_file":  "",
    "orientation_file": "orientations.bin",
    "hkl_file":         "hkls.bin",
    "result_dir":       "results",
    # H5 data location
    "h5_location":      "/entry/data/data",
}


def parse_config(config_file: str) -> Dict[str, Any]:
    """
    Parse a classic LaueMatching text config file into a flat dictionary.

    Returns a dict with the same keys as DEFAULT_CONFIG, overridden by
    values found in the file.
    """
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(config_file):
        logger.warning(f"Config file '{config_file}' not found — using defaults.")
        return cfg

    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()
            parts = line.split()
            if len(parts) < 2:
                continue
            key, rest = parts[0], parts[1:]

            try:
                if key == "SpaceGroup":
                    cfg["space_group"] = int(rest[0])
                elif key == "Symmetry":
                    cfg["symmetry"] = rest[0]
                elif key == "LatticeParameter" and len(rest) >= 6:
                    cfg["lattice_parameter"] = " ".join(rest[:6])
                elif key == "R_Array" and len(rest) >= 3:
                    cfg["r_array"] = " ".join(rest[:3])
                elif key == "P_Array" and len(rest) >= 3:
                    cfg["p_array"] = " ".join(rest[:3])
                    cfg["distance"] = float(rest[2])
                elif key == "NrPxX":
                    cfg["nr_px_x"] = int(rest[0])
                elif key == "NrPxY":
                    cfg["nr_px_y"] = int(rest[0])
                elif key == "PxX":
                    cfg["px_x"] = float(rest[0])
                elif key == "PxY":
                    cfg["px_y"] = float(rest[0])
                elif key == "Elo":
                    cfg["elo"] = float(rest[0])
                elif key == "Ehi":
                    cfg["ehi"] = float(rest[0])
                elif key == "MinIntensity":
                    cfg["min_intensity"] = float(rest[0])
                elif key == "MinGoodSpots":
                    cfg["min_good_spots"] = int(rest[0])
                elif key == "MinNrSpots":
                    cfg["min_nr_spots"] = int(rest[0])
                elif key == "MaxAngle":
                    cfg["max_angle"] = float(rest[0])
                elif key == "MaxNrLaueSpots":
                    cfg["max_laue_spots"] = int(rest[0])
                elif key == "OrientationSpacing":
                    cfg["orientation_spacing"] = float(rest[0])
                elif key == "ThresholdMethod":
                    cfg["threshold_method"] = rest[0].lower()
                elif key == "Threshold":
                    cfg["threshold_value"] = float(rest[0])
                elif key == "ThresholdPercentile":
                    cfg["threshold_percentile"] = float(rest[0])
                elif key == "MinArea":
                    cfg["min_area"] = int(rest[0])
                elif key == "FilterRadius":
                    cfg["filter_radius"] = int(rest[0])
                elif key == "NMeadianPasses":
                    cfg["median_passes"] = int(rest[0])
                elif key == "WatershedImage":
                    cfg["watershed_enabled"] = bool(int(rest[0]))
                elif key == "GaussianFactor":
                    cfg["gaussian_factor"] = float(rest[0])
                elif key == "EnhanceContrast":
                    cfg["enhance_contrast"] = bool(int(rest[0]))
                elif key == "DenoiseImage":
                    cfg["denoise_image"] = bool(int(rest[0]))
                elif key == "DenoiseStrength":
                    cfg["denoise_strength"] = float(rest[0])
                elif key == "EdgeEnhancement":
                    cfg["edge_enhancement"] = bool(int(rest[0]))
                elif key == "BackgroundFile":
                    cfg["background_file"] = rest[0]
                elif key == "OrientationFile":
                    cfg["orientation_file"] = rest[0]
                elif key == "HKLFile":
                    cfg["hkl_file"] = rest[0]
                elif key == "ResultDir":
                    cfg["result_dir"] = rest[0]
                elif key == "H5Location":
                    cfg["h5_location"] = rest[0]
            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping malformed config line '{line}': {e}")

    return cfg


# ---------------------------------------------------------------------------
# H5 image loading
# ---------------------------------------------------------------------------

def load_h5_image(
    path: str,
    h5_location: str = "/entry/data/data",
    frame_index: Optional[int] = None,
) -> np.ndarray:
    """
    Load a 2-D image from an HDF5 file.

    Args:
        path:        Path to the .h5 file.
        h5_location: Internal dataset path.
        frame_index: If the dataset is 3-D, which frame to read (None → first).

    Returns:
        2-D numpy array (Y, X) of float64.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for H5 file loading")

    with h5py.File(path, "r") as hf:
        # Try specified path first, then common alternatives
        dset_path = h5_location
        if dset_path not in hf:
            for alt in ["/entry/data/data", "/entry1/data/data",
                        "/entry/data/raw_data", "/data"]:
                if alt in hf:
                    dset_path = alt
                    break
            else:
                raise KeyError(
                    f"Could not find image data at '{h5_location}' "
                    f"or standard paths in {path}"
                )

        dset = hf[dset_path]
        if dset.ndim == 3:
            idx = frame_index if frame_index is not None else 0
            data = np.array(dset[idx], dtype=np.float64)
        elif dset.ndim == 2:
            data = np.array(dset[()], dtype=np.float64)
        else:
            raise ValueError(
                f"Unexpected dataset dimensions ({dset.ndim}) in {path}"
            )

    return data


def count_h5_frames(path: str, h5_location: str = "/entry/data/data") -> int:
    """Return the number of frames in an H5 dataset (1 for 2-D data)."""
    if not HAS_H5PY:
        raise ImportError("h5py is required")
    with h5py.File(path, "r") as hf:
        dset_path = h5_location
        if dset_path not in hf:
            for alt in ["/entry/data/data", "/entry1/data/data",
                        "/entry/data/raw_data", "/data"]:
                if alt in hf:
                    dset_path = alt
                    break
            else:
                return 0
        dset = hf[dset_path]
        return dset.shape[0] if dset.ndim == 3 else 1


# ---------------------------------------------------------------------------
# Image pre-processing pipeline  (mirrors RunImage.py steps 1–6)
# ---------------------------------------------------------------------------

def compute_background(
    image: np.ndarray,
    filter_radius: int = 101,
    median_passes: int = 1,
) -> np.ndarray:
    """
    Compute background via diplib median filter.

    Falls back to scipy median if diplib is unavailable.
    """
    if filter_radius <= 0 or median_passes <= 0:
        return np.zeros_like(image, dtype=np.float64)

    if HAS_DIPLIB:
        bg = dip.Image(image)
        for _ in range(median_passes):
            bg = dip.MedianFilter(bg, filter_radius)
        return np.array(bg).astype(np.float64)
    else:
        # Fallback: scipy median filter (slower for large radii)
        bg = image.astype(np.float64)
        for _ in range(median_passes):
            bg = ndimg.median_filter(bg, size=filter_radius)
        return bg


def load_background(
    background_file: str,
    nr_px_x: int,
    nr_px_y: int,
) -> np.ndarray:
    """Load a pre-computed background from a raw binary file."""
    expected = nr_px_x * nr_px_y * np.dtype(np.float64).itemsize
    if background_file and os.path.exists(background_file):
        actual = os.path.getsize(background_file)
        if actual == expected:
            return np.fromfile(
                background_file, dtype=np.float64
            ).reshape((nr_px_y, nr_px_x))
        else:
            logger.warning(
                f"Background file size mismatch (expected {expected}, "
                f"got {actual}). Ignoring."
            )
    return np.zeros((nr_px_y, nr_px_x), dtype=np.float64)


def enhance_image(
    image: np.ndarray,
    *,
    denoise: bool = False,
    denoise_strength: float = 1.0,
    contrast: bool = False,
    edge: bool = False,
) -> np.ndarray:
    """
    Apply optional image enhancements (denoise, CLAHE, edge sharpening).

    All operations are skipped if skimage is unavailable.
    """
    if not HAS_SKIMAGE:
        return image

    enhanced = image.copy().astype(np.float32)

    # 1. Denoising
    if denoise:
        img_min, img_max = enhanced.min(), enhanced.max()
        if img_max > img_min:
            norm = (enhanced - img_min) / (img_max - img_min)
            denoised = restoration.denoise_nl_means(
                norm, h=denoise_strength, fast_mode=True,
                patch_size=5, patch_distance=7, channel_axis=None,
            )
            enhanced = (denoised * (img_max - img_min) + img_min).astype(np.float32)

    # 2. Contrast (CLAHE)
    if contrast:
        img_min, img_max = enhanced.min(), enhanced.max()
        if img_max > img_min:
            norm = (enhanced - img_min) / (img_max - img_min)
            u16 = (norm * 65535).astype(np.uint16)
            eq = exposure.equalize_adapthist(u16, clip_limit=0.03).astype(np.float32)
            enhanced = eq / 65535.0 * (img_max - img_min) + img_min

    # 3. Edge sharpening (unsharp mask)
    if edge:
        blurred = filters.gaussian(enhanced, sigma=1.0)
        enhanced = enhanced + 0.5 * (enhanced - blurred)
        enhanced = np.maximum(enhanced, 0)

    return enhanced


def apply_threshold(
    image: np.ndarray,
    method: str = "adaptive",
    fixed_value: float = 0.0,
    percentile: float = 90.0,
) -> Tuple[np.ndarray, float]:
    """
    Threshold an image, returning (thresholded_image, threshold_used).

    Methods: adaptive, percentile, otsu, fixed.
    """
    if method == "percentile":
        thresh = np.percentile(image.ravel(), percentile)
    elif method == "otsu" and HAS_SKIMAGE:
        thresh = filters.threshold_otsu(image)
    elif method == "fixed":
        thresh = fixed_value
    else:
        # adaptive (default)
        std = np.std(image)
        thresh = max(60.0 * (1.0 + std // 60.0), 1.0)

    out = image.copy()
    out[image <= thresh] = 0
    return out, float(thresh)


def find_connected_components(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Find connected components in a thresholded image (OpenCV).

    Returns (labels, bboxes, areas, nlabels).
    bboxes and areas correspond to labels 1..nlabels-1.
    """
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
    nlabels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, 8, cv2.CV_32S
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    bboxes = stats[1:, : cv2.CC_STAT_HEIGHT + 1]
    return labels, bboxes, areas, nlabels


def filter_small_components(
    image: np.ndarray,
    labels: np.ndarray,
    bboxes: np.ndarray,
    areas: np.ndarray,
    nlabels: int,
    min_area: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Remove components smaller than *min_area* and compute centers of mass.

    Returns (filtered_image, filtered_labels, centers).
    Centers format: [[label, (cx, cy), area], ...]
    """
    filt_img = image.copy()
    filt_lbl = labels.copy()
    centers: List = []

    for lbl in range(1, nlabels):
        idx = lbl - 1
        if areas[idx] >= min_area:
            x, y, w, h = bboxes[idx]
            mask = labels[y : y + h, x : x + w] == lbl
            region = image[y : y + h, x : x + w] * mask
            try:
                com = ndimg.center_of_mass(region)
                centers.append([lbl, (com[1] + x, com[0] + y), areas[idx]])
            except Exception:
                filt_img[labels == lbl] = 0
                filt_lbl[labels == lbl] = 0
        else:
            filt_img[labels == lbl] = 0
            filt_lbl[labels == lbl] = 0

    return filt_img, filt_lbl, centers


def calculate_gaussian_sigma(
    centers: List,
    pixel_size: float = 0.2,
    distance: float = 0.513,
    orient_spacing: float = 0.4,
) -> float:
    """
    Calculate Gaussian blur sigma from spot spacing.

    Returns sigma in pixels (float, >= 1.0).
    """
    if not centers or len(centers) < 2:
        return 3.0

    coords = [c[1] for c in centers]
    min_dist_sq = float("inf")
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d2 = (coords[i][0] - coords[j][0]) ** 2 + (
                coords[i][1] - coords[j][1]
            ) ** 2
            min_dist_sq = min(min_dist_sq, d2)
    min_px_dist = np.sqrt(min_dist_sq)

    if pixel_size > 0 and distance > 0:
        delta = (distance * np.tan(np.radians(orient_spacing))) / pixel_size
    else:
        delta = min_px_dist

    sigma = 0.25 * min(min_px_dist, delta) if delta > 0 else 0.25 * min_px_dist
    return max(sigma, 1.0)


def preprocess_image(
    raw_image: np.ndarray,
    cfg: Dict[str, Any],
    background: Optional[np.ndarray] = None,
    override_thresh: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Full preprocessing pipeline (RunImage.py steps 1-6).

    1. Background subtraction
    2. Optional enhancement (denoise, contrast, edge)
    3. Thresholding
    4. Connected components
    5. Filter small components
    6. Gaussian blur

    Args:
        raw_image:       Raw 2-D image (Y, X) float64.
        cfg:             Config dict (from parse_config).
        background:      Optional pre-loaded background array.
        override_thresh: If > 0, override threshold method with this value.

    Returns:
        (blurred_image, filtered_thresholded_image, filtered_labels, centers)

        blurred_image is what should be sent to the GPU daemon (double[]).
        filtered_thresholded_image and filtered_labels are kept for
        post-processing / visualization.
    """
    nr_px_x = cfg["nr_px_x"]
    nr_px_y = cfg["nr_px_y"]

    # --- Step 1: Background subtraction ---
    if background is None:
        bg_file = cfg.get("background_file", "")
        background = load_background(bg_file, nr_px_x, nr_px_y)
        # Compute from image if still empty
        if np.count_nonzero(background) == 0:
            background = compute_background(
                raw_image,
                filter_radius=cfg["filter_radius"],
                median_passes=cfg["median_passes"],
            )

    bg_sub = np.maximum(raw_image - background, 0.0)

    # --- Step 2: Enhancement ---
    enhanced = enhance_image(
        bg_sub,
        denoise=cfg.get("denoise_image", False),
        denoise_strength=cfg.get("denoise_strength", 1.0),
        contrast=cfg.get("enhance_contrast", False),
        edge=cfg.get("edge_enhancement", False),
    )

    # --- Step 3: Thresholding ---
    if override_thresh > 0:
        thresholded, _thresh = apply_threshold(
            enhanced, method="fixed", fixed_value=override_thresh,
        )
    else:
        thresholded, _thresh = apply_threshold(
            enhanced,
            method=cfg["threshold_method"],
            fixed_value=cfg["threshold_value"],
            percentile=cfg["threshold_percentile"],
        )

    thresholded_u16 = thresholded.astype(np.uint16)

    # --- Step 4: Connected components ---
    labels, bboxes, areas, nlabels = find_connected_components(thresholded_u16)
    if nlabels <= 1:
        # No components — return zero image
        blurred = np.zeros_like(raw_image, dtype=np.float64)
        return blurred, thresholded_u16, labels, []

    # --- Step 5: Filter small components ---
    filt_img, filt_labels, centers = filter_small_components(
        thresholded_u16, labels, bboxes, areas, nlabels,
        min_area=cfg["min_area"],
    )
    if not centers:
        blurred = np.zeros_like(raw_image, dtype=np.float64)
        return blurred, filt_img, filt_labels, centers

    # --- Step 6: Gaussian blur ---
    sigma = calculate_gaussian_sigma(
        centers,
        pixel_size=cfg["px_x"],
        distance=cfg["distance"],
        orient_spacing=cfg["orientation_spacing"],
    )
    blurred = ndimg.gaussian_filter(filt_img.astype(np.float64), sigma)

    return blurred, filt_img, filt_labels, centers


# ---------------------------------------------------------------------------
# Output file parsing  (solutions.txt / spots.txt)
# ---------------------------------------------------------------------------

def read_solutions(path: str) -> Tuple[np.ndarray, str]:
    """
    Read solutions.txt.

    Returns (orientations_array, header_line).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Solutions file not found: {path}")

    with open(path, "r") as f:
        header = f.readline().strip()

    data = np.genfromtxt(path, skip_header=1)
    if data.size == 0:
        data = np.empty((0, 31))
    elif data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data, header


def read_spots(path: str) -> Tuple[np.ndarray, str]:
    """
    Read spots.txt (may contain ImageNr column).

    Returns (spots_array, header_line).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spots file not found: {path}")

    with open(path, "r") as f:
        header = f.readline().strip()

    data = np.genfromtxt(path, skip_header=1)
    if data.size == 0:
        ncols = len(header.split()) if header else 8
        data = np.empty((0, ncols))
    elif data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    return data, header


def split_spots_by_image(spots: np.ndarray, image_col: int = 0) -> Dict[int, np.ndarray]:
    """
    Split a spots array by the ImageNr column.

    Args:
        spots:     2-D spots array (with ImageNr as a column).
        image_col: Column index that holds ImageNr.

    Returns:
        Dict mapping image_num → sub-array of spots for that image.
    """
    if spots.size == 0:
        return {}
    unique_ids = np.unique(spots[:, image_col].astype(int))
    return {
        int(img_id): spots[spots[:, image_col].astype(int) == img_id]
        for img_id in unique_ids
    }


def split_solutions_by_image(
    solutions: np.ndarray,
    spots: np.ndarray,
    image_col: int = 0,
    grain_col_spots: int = 1,
    grain_col_solutions: int = 0,
) -> Dict[int, np.ndarray]:
    """
    Split solutions by image number using the spots→grain linkage.

    Each solution's GrainNr appears in the spots file; the spots file
    has an ImageNr column.  We determine which images a grain belongs
    to and group solutions accordingly.

    Args:
        solutions:          Solutions array (GrainNr in column grain_col_solutions).
        spots:              Spots array (ImageNr in column image_col, GrainNr in grain_col_spots).
        image_col:          Column index for ImageNr in spots.
        grain_col_spots:    Column index for GrainNr in spots.
        grain_col_solutions:Column index for GrainNr in solutions.

    Returns:
        Dict mapping image_num → sub-array of solutions for that image.
    """
    if solutions.size == 0 or spots.size == 0:
        return {}

    # Build grain→image mapping from spots
    grain_to_images: Dict[int, set] = {}
    for row in spots:
        gn = int(row[grain_col_spots])
        img = int(row[image_col])
        grain_to_images.setdefault(gn, set()).add(img)

    result: Dict[int, List] = {}
    for sol in solutions:
        gn = int(sol[grain_col_solutions])
        for img in grain_to_images.get(gn, set()):
            result.setdefault(img, []).append(sol)

    return {
        img: np.array(rows)
        for img, rows in result.items()
    }


# ---------------------------------------------------------------------------
# Orientation filtering (unique-spot counting)
# ---------------------------------------------------------------------------

def calculate_unique_spots(
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    grain_col: int = 0,
    quality_col: int = 4,
    spot_grain_col: int = 0,
    spot_x_col: int = 5,
    spot_y_col: int = 6,
) -> Dict[int, Dict]:
    """
    Calculate unique spots per orientation, prioritized by quality score.

    Orientations are sorted by quality (descending) using a stable sort so
    that ties preserve the original input order.  Higher-quality orientations
    claim spots first; once a label or pixel position is assigned, lower-
    quality orientations cannot reuse it.

    Returns:
        {grain_nr: {"count": int, "unique_label_count": int,
                     "unique_labels": set, "positions": set}}
    """
    results: Dict[int, Dict] = {}
    if orientations.size == 0:
        return results

    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, 0)

    # Sort by quality descending with *stable* tie-breaking (matches the
    # original Python list .sort() used in RunImage._calculate_unique_spots).
    sorted_idx = np.argsort(orientations[:, quality_col], kind='stable')[::-1]

    assigned_labels: set = set()
    assigned_positions: set = set()

    for idx in sorted_idx:
        orient = orientations[idx]
        gn = int(orient[grain_col])
        results[gn] = {
            "count": 0,
            "unique_label_count": 0,
            "unique_labels": set(),
            "positions": set(),
        }

        orient_spots = spots[spots[:, spot_grain_col].astype(int) == gn]
        if orient_spots.size == 0:
            continue

        for spot in orient_spots:
            try:
                x, y = int(spot[spot_x_col]), int(spot[spot_y_col])
            except (IndexError, ValueError):
                continue
            pos = (x, y)
            if pos in assigned_positions:
                continue

            if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                lbl = labels[y, x]
                if lbl > 0 and lbl in assigned_labels:
                    continue
                results[gn]["positions"].add(pos)
                assigned_positions.add(pos)
                if lbl > 0:
                    results[gn]["unique_labels"].add(lbl)
                    assigned_labels.add(lbl)
            else:
                # Out-of-bounds: still claim the position so lower-quality
                # orientations cannot reuse it (matches original behaviour).
                results[gn]["positions"].add(pos)
                assigned_positions.add(pos)

        results[gn]["count"] = len(results[gn]["positions"])
        results[gn]["unique_label_count"] = len(results[gn]["unique_labels"])

    return results


def filter_orientations(
    orientations: np.ndarray,
    unique_spot_info: Dict[int, Dict],
    min_unique: int = 2,
    grain_col: int = 0,
    quality_col: int = 4,
) -> np.ndarray:
    """
    Filter orientations by minimum unique spot (label) count.

    Returns a new array with only the qualifying rows, sorted by quality.
    """
    if orientations.size == 0:
        return orientations.copy()

    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, 0)

    # Sort by quality descending first
    sorted_orient = orientations[np.argsort(orientations[:, quality_col])[::-1]]

    keep = []
    for row in sorted_orient:
        gn = int(row[grain_col])
        info = unique_spot_info.get(gn, {})
        if info.get("unique_label_count", 0) >= min_unique:
            keep.append(row)

    if keep:
        return np.array(keep)
    return np.empty((0, orientations.shape[1]))


# ---------------------------------------------------------------------------
# Results I/O — sort, H5 output, text file packing
# (extracted from RunImage.py for reuse by laue_postprocess.py)
# ---------------------------------------------------------------------------

def sort_orientations_by_quality(
    orientations: np.ndarray,
    quality_col: int = 4,
) -> np.ndarray:
    """
    Sort orientations by quality score (descending).

    Args:
        orientations: 2-D orientation array.
        quality_col:  Column index containing the quality score.

    Returns:
        Sorted copy of the orientation array.
    """
    if orientations.size == 0 or orientations.ndim == 1:
        return orientations.copy() if orientations.size else orientations

    try:
        sorted_idx = np.argsort(orientations[:, quality_col])[::-1]
        return orientations[sorted_idx].copy()
    except IndexError:
        logger.warning(
            f"Could not sort by quality column {quality_col}; "
            "returning orientations as-is."
        )
        return orientations.copy()


def store_txt_files_in_h5(
    output_path: str,
    h5_file,
) -> None:
    """
    Store the contents of generated text files in an open H5 file handle.

    Each text file is stored as a bytes dataset, and the first line (header)
    is attached as an ``'header'`` attribute.

    Args:
        output_path: Base path for output files (e.g. ``results/image_001``).
        h5_file:     Open ``h5py.File`` handle (mode ``'a'`` or ``'w'``).
    """
    txt_files_map = {
        f"{output_path}.bin.solutions.txt":           "/entry/results/solutions_text",
        f"{output_path}.bin.solutions_filtered.txt":  "/entry/results/solutions_filtered_text",
        f"{output_path}.bin.spots.txt":               "/entry/results/spots_text",
        f"{output_path}.bin.LaueMatching_stdout.txt": "/entry/logs/stdout",
        f"{output_path}.bin.LaueMatching_stderr.txt": "/entry/logs/stderr",
        f"{output_path}.simulation_stdout.txt":       "/entry/logs/simulation_stdout",
        f"{output_path}.bin.unique_spot_counts.txt":  "/entry/results/unique_spot_counts_text",
    }

    # Ensure parent groups exist
    for dataset_path in txt_files_map.values():
        group_path = os.path.dirname(dataset_path)
        if group_path != "/":
            h5_file.require_group(group_path)

    for txt_file_path, dataset_path in txt_files_map.items():
        try:
            if os.path.exists(txt_file_path):
                with open(txt_file_path, "r") as f:
                    lines = f.readlines()
                    header = lines[0].strip() if lines else ""
                    content = "".join(lines)

                if dataset_path in h5_file:
                    del h5_file[dataset_path]
                dataset = h5_file.create_dataset(dataset_path, data=np.bytes_(content))
                if header:
                    dataset.attrs["header"] = header

                logger.debug(f"Stored '{txt_file_path}' in H5 dataset '{dataset_path}'")
        except Exception as e:
            logger.warning(f"Error storing text file '{txt_file_path}' in H5: {e}")


def store_binary_headers_in_h5(
    output_path: str,
    h5_file,
) -> None:
    """
    Store column headers from text files as attributes on binary H5 datasets.

    Args:
        output_path: Base path for output files (e.g. ``results/image_001``).
        h5_file:     Open ``h5py.File`` handle.
    """
    binary_datasets_map = {
        "/entry/results/orientations":          f"{output_path}.bin.solutions.txt",
        "/entry/results/filtered_orientations": f"{output_path}.bin.solutions.txt",
        "/entry/results/spots":                 f"{output_path}.bin.spots.txt",
        "/entry/results/filtered_spots":        f"{output_path}.bin.spots.txt",
    }

    for dataset_path, header_file_path in binary_datasets_map.items():
        try:
            if dataset_path in h5_file and os.path.exists(header_file_path):
                with open(header_file_path, "r") as f:
                    header = f.readline().strip()
                if header:
                    ds = h5_file[dataset_path]
                    ds.attrs["header"] = header
                    columns = [c.strip() for c in header.split() if c.strip()]
                    ds.attrs["columns"] = columns
                    logger.debug(f"Added header and {len(columns)} columns to {dataset_path}")
        except Exception as e:
            logger.warning(f"Error adding header to {dataset_path}: {e}")

    # Unique spots dataset
    usp = "/entry/results/unique_spots_per_orientation"
    if usp in h5_file:
        try:
            ds = h5_file[usp]
            ds.attrs["header"] = "Grain_Nr Unique_Spots"
            ds.attrs["columns"] = ["Grain_Nr", "Unique_Spots"]
        except Exception as e:
            logger.warning(f"Error adding header to {usp}: {e}")

    # Simulated spots dataset
    ssp = "/entry/simulation/simulated_spots"
    if ssp in h5_file:
        try:
            ds = h5_file[ssp]
            if "header" not in ds.attrs:
                ds.attrs["header"] = "X Y GrainID Matched H K L Energy"
                ds.attrs["columns"] = ["X", "Y", "GrainID", "Matched", "H", "K", "L", "Energy"]
        except Exception as e:
            logger.warning(f"Error adding header to {ssp}: {e}")


def create_h5_output(
    output_path: str,
    orientations_unfiltered: np.ndarray,
    filtered_orientations: np.ndarray,
    spots_unfiltered: np.ndarray,
    filtered_spots: np.ndarray,
    orientation_unique_spots: Dict[int, Dict],
) -> None:
    """
    Create / update an HDF5 file with orientation and spot data.

    Saves both unfiltered and filtered datasets, plus unique spot counts.
    Headers are attached as attributes via :func:`store_binary_headers_in_h5`.

    Args:
        output_path:               Base path (e.g. ``results/image_001``).
        orientations_unfiltered:   Sorted orientation array.
        filtered_orientations:     Filtered orientation array.
        spots_unfiltered:          Original spot array.
        filtered_spots:            Filtered spot array.
        orientation_unique_spots:  ``{grain_nr: {"unique_label_count": int, …}}``.
    """
    if not HAS_H5PY:
        logger.error("h5py is required for H5 output but is not installed.")
        return

    output_h5 = f"{output_path}.output.h5"

    # Build unique-count array [Grain_Nr, Unique_Label_Count]
    unique_counts_list = []
    if orientation_unique_spots:
        for grain_nr, data in orientation_unique_spots.items():
            unique_counts_list.append([grain_nr, data.get("unique_label_count", 0)])
    unique_counts_array = (
        np.array(unique_counts_list, dtype=np.int32)
        if unique_counts_list
        else np.empty((0, 2), dtype=np.int32)
    )

    try:
        with h5py.File(output_h5, "a") as hf:
            hf.require_group("/entry/results")

            datasets = {
                "orientations":                  orientations_unfiltered,
                "filtered_orientations":         filtered_orientations,
                "spots":                         spots_unfiltered,
                "filtered_spots":                filtered_spots,
                "unique_spots_per_orientation":   unique_counts_array,
            }

            for name, arr in datasets.items():
                ds_path = f"/entry/results/{name}"
                if ds_path in hf:
                    del hf[ds_path]
                hf.create_dataset(ds_path, data=arr)

            logger.info(f"Saved orientation/spot data in {output_h5}")
            store_binary_headers_in_h5(output_path, hf)

    except Exception as e:
        logger.error(f"Error creating H5 output '{output_h5}': {e}")


# ---------------------------------------------------------------------------
# TCP / networking helpers
# ---------------------------------------------------------------------------

LAUE_STREAM_PORT = 60517


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check whether a TCP port is accepting connections."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except socket.error:
        return False
    finally:
        s.close()


def wait_for_port(
    host: str = "127.0.0.1",
    port: int = LAUE_STREAM_PORT,
    timeout: float = 180.0,
    poll_interval: float = 2.0,
) -> bool:
    """Block until *port* opens or *timeout* seconds elapse."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_port_open(host, port):
            logger.info(f"Port {port} ready ({time.time()-t0:.1f}s)")
            return True
        time.sleep(poll_interval)
    logger.error(f"Timed out waiting for port {port}")
    return False


def send_image(
    sock: socket.socket,
    image_num: int,
    image_data: np.ndarray,
) -> None:
    """
    Send one image to the LaueMatchingGPUStream daemon.

    Wire format: uint16_t image_num  +  double[NrPxX*NrPxY] pixels
    """
    header = struct.pack("<H", image_num)  # little-endian uint16
    pixels = image_data.astype(np.float64).tobytes()
    sock.sendall(header + pixels)


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    """Receive exactly *nbytes* from a socket."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed before full read")
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------------------
# Frame mapping I/O
# ---------------------------------------------------------------------------

def save_frame_mapping(mapping: Dict, path: str) -> None:
    """Atomically save frame mapping to JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(mapping, f, indent=1)
    os.replace(tmp, path)


def load_frame_mapping(path: str) -> Dict:
    """Load frame mapping from JSON, returning {} if missing."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)
