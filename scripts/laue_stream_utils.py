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

# REFACTOR_PLAN §6.2/§6.3: the filtering/geometry/threshold implementations live
# in the laue_index package (single source of truth); ensure it is importable
# (repo root one level above scripts/) before the re-exports below.
_INSTALL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _INSTALL_PATH not in sys.path:
    sys.path.insert(0, _INSTALL_PATH)

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
# Image pre-processing pipeline
#
# REFACTOR_PLAN §6.5: moved to laue_index.preprocess (single source of truth);
# re-exported here so RunImage / laue_postprocess callers keep working.
# ---------------------------------------------------------------------------
from laue_index.preprocess import (  # noqa: E402
    compute_background,
    load_background,
    enhance_image,
    find_connected_components,
    filter_small_components,
    calculate_gaussian_sigma,
    preprocess_image,
)
# REFACTOR_PLAN §6.3: thresholding lives in laue_index.thresholds; re-exported
# (byte-for-byte equivalent) so RunImage/preprocess callers are unchanged.  This
# line previously sat inline among the preprocessing funcs that moved out in §6.5.
from laue_index.thresholds import apply_threshold  # noqa: E402,F401


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

    try:
        data = np.loadtxt(path, skiprows=1)
    except ValueError:
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

    try:
        data = np.loadtxt(path, skiprows=1)
    except ValueError:
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
# Orientation filtering + CSL geometry
#
# REFACTOR_PLAN §6.2: the implementations now live in the laue_index package
# (single source of truth).  This module re-exports them under their historical
# names so existing callers (laue_postprocess, laue_image_server, tests) keep
# working unchanged, while the duplicate inline copy in RunImage is removed.
# ---------------------------------------------------------------------------
_INSTALL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _INSTALL_PATH not in sys.path:
    sys.path.insert(0, _INSTALL_PATH)

from laue_index.geometry import (  # noqa: E402
    cubic_proper_ops as _cubic_proper_ops,
    CUBIC_OPS as _CUBIC_OPS,
    CSL_TABLE as _CSL_TABLE,
    disorientation_deg_axis as _disorientation_deg_axis,
    is_csl_related,
)
from laue_index.filtering import (  # noqa: E402
    calculate_unique_spots,
    filter_orientations,
    filter_orientations_robust,
)


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

    Wire format: uint16_t image_num  +  float[NrPxX*NrPxY] pixels
    """
    header = struct.pack("<H", image_num)  # little-endian uint16
    pixels = image_data.astype(np.float32).tobytes()
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
