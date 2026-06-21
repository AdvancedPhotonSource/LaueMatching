"""Preprocessor stage — background, threshold, segmentation, blur.

REFACTOR_PLAN §3 / §6.5.  The image pre-processing pipeline (RunImage steps
1-6), lifted from laue_stream_utils so laue_index owns it; lsu re-exports for
back-compat.  Thresholding is delegated to laue_index.thresholds (§6.3).

Optional deps (diplib / scikit-image) degrade gracefully, matching legacy.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import scipy.ndimage as ndimg

from .thresholds import apply_threshold

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

logger = logging.getLogger("LaueStream")

__all__ = [
    "compute_background", "load_background", "enhance_image",
    "find_connected_components", "filter_small_components",
    "calculate_gaussian_sigma", "preprocess_image", "Preprocessor",
]


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

    coords = np.array([c[1] for c in centers])
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    # query k=2: first neighbor is the point itself (distance 0), second is nearest
    dists, _ = tree.query(coords, k=2)
    min_px_dist = float(dists[:, 1].min())

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
    return_intermediates: bool = False,
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
        return_intermediates: If True, return a dict with all intermediate
            arrays instead of the standard 4-tuple.

    Returns:
        If return_intermediates is False (default):
            (blurred_image, filtered_thresholded_image, filtered_labels, centers)
        If return_intermediates is True:
            dict with keys: background, thresholded, labels_unfiltered,
            filt_img, filt_labels, blurred, centers
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
        if return_intermediates:
            return {
                "background": background, "thresholded": thresholded_u16,
                "labels_unfiltered": labels, "filt_img": thresholded_u16,
                "filt_labels": labels, "blurred": blurred, "centers": [],
            }
        return blurred, thresholded_u16, labels, []

    # --- Step 5: Filter small components ---
    filt_img, filt_labels, centers = filter_small_components(
        thresholded_u16, labels, bboxes, areas, nlabels,
        min_area=cfg["min_area"],
    )
    if not centers:
        blurred = np.zeros_like(raw_image, dtype=np.float64)
        if return_intermediates:
            return {
                "background": background, "thresholded": thresholded_u16,
                "labels_unfiltered": labels, "filt_img": filt_img,
                "filt_labels": filt_labels, "blurred": blurred, "centers": centers,
            }
        return blurred, filt_img, filt_labels, centers

    # --- Step 6: Gaussian blur ---
    sigma = calculate_gaussian_sigma(
        centers,
        pixel_size=cfg["px_x"],
        distance=cfg["distance"],
        orient_spacing=cfg["orientation_spacing"],
    )
    blurred = ndimg.gaussian_filter(filt_img.astype(np.float64), sigma)

    if return_intermediates:
        return {
            "background": background, "thresholded": thresholded_u16,
            "labels_unfiltered": labels, "filt_img": filt_img,
            "filt_labels": filt_labels, "blurred": blurred, "centers": centers,
        }
    return blurred, filt_img, filt_labels, centers



class Preprocessor:
    """Config-driven wrapper over :func:`preprocess_image` (REFACTOR_PLAN §6.5)."""

    def __init__(self, cfg: Dict[str, Any], background: Optional[np.ndarray] = None):
        self.cfg = cfg
        self.background = background

    def __call__(self, raw_image: np.ndarray, *, override_thresh: float = 0.0,
                 return_intermediates: bool = False):
        return preprocess_image(
            raw_image, self.cfg, background=self.background,
            override_thresh=override_thresh,
            return_intermediates=return_intermediates)
