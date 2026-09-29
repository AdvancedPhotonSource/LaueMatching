#!/usr/bin/env python3
"""
laue_stream_utils.py — Shared utilities for the LaueMatching streaming workflow.

Provides:
  - Configuration parsing (classic text format used by LaueMatching)
  - H5 image loading
  - Image preprocessing pipeline (background, threshold, components, blur)
  - Output file parsing (solutions.txt, spots.txt with ImageNr column)
  - Orientation filtering (re-exported from laue_index.filtering; the "unique"
    spot count there is WINNER-TAKE-ALL across one frame's orientations, not a
    count of distinct observed peaks)
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

# Shared parsing rules (SpaceGroup, Symmetry) so both config readers agree.
from laue_index import config_schema as _schema  # noqa: E402

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
    # The KEYS this parser returns. Every value whose key is a config_schema
    # key is replaced by SCHEMA's default just below, so the literals here are
    # placeholders, not defaults. Keys in config_schema.REQUIRED_KEYS have no
    # default at all: parse_config refuses a file without them.
    # Detector
    "nr_px_x":          None,   # REQUIRED
    "nr_px_y":          None,   # REQUIRED
    "px_x":             None,   # REQUIRED (metres)
    "px_y":             None,   # REQUIRED (metres)
    "distance":         None,   # P_Array[2]
    # Energy range
    "elo":              None,
    "ehi":              None,
    # Crystallography
    "space_group":      None,
    "symmetry":         None,
    "lattice_parameter":None,   # REQUIRED
    "r_array":          None,   # REQUIRED
    "p_array":          None,   # REQUIRED
    # Thresholding
    "threshold_method": None,
    "threshold_value":  None,
    "threshold_percentile": None,
    # Image processing
    "min_area":         None,
    # Detector positions whose spots must not count as evidence. Empty = off.
    # NOTE: adding a parameter needs THREE edits kept in step -- this dict, the
    # parser chain below, AND laue_config.ImageProcessingConfig (plus a
    # config_schema row, which supplies the default). A key present in only
    # some of them is parsed and then silently dropped, so the run proceeds
    # with the setting quietly not applied.
    "exclude_spots_file": None,
    "exclude_spots_dir": None,
    "filter_radius":    None,
    "median_passes":    None,   # REQUIRED (NMeadianPasses)
    "watershed_enabled":None,
    "gaussian_factor":  None,
    "enhance_contrast": None,
    "denoise_image":    None,
    "denoise_strength": None,
    "edge_enhancement": None,
    # Preprocessing pool cap; 0 = let laue_index.workers decide. Read by
    # laue_image_server (the streaming path). LAUE_PREPROCESS_WORKERS overrides.
    "preprocess_workers": None,
    # Matching
    "min_intensity":    None,   # REQUIRED
    # Exclusive-spot floor for the orientation filter (winner-take-all label
    # count). laue_postprocess uses it unless --min-unique is given. REQUIRED:
    # it used to default to 2 here and 5 in RunImage.
    "min_good_spots":   None,
    # 1 = twin/CSL-aware filter, 0 = legacy. REQUIRED: an absent key used to
    # mean legacy here and robust in RunImage.
    "robust_filter":    None,
    "min_nr_spots":     None,
    "max_angle":        None,
    "max_laue_spots":   None,   # REQUIRED
    "orientation_spacing": None,
    "gauss_sigma_max":  None,   # 0 = no cap on the auto matching-blur sigma
    # Files
    "background_file":  None,   # REQUIRED
    "orientation_file": None,
    "hkl_file":         None,
    "result_dir":       None,
    # H5 data location (not a schema key)
    "h5_location":      "/entry/data/data",
}


# One source of defaults (0.8, decision D5): every DEFAULT_CONFIG entry that is
# a schema key takes config_schema.SCHEMA's default, so this parser and
# laue_config's cannot drift apart again. Keys in config_schema.REQUIRED_KEYS
# (their defaults used to disagree between the parsers: MaxNrLaueSpots 400 here
# vs 7 in RunImage vs 500 in the C, MinIntensity 0 vs 50 vs 1000, ...) must be
# present: parse_config raises when any is missing.
_STREAM_NAME = {"maxAngle": "max_angle"}      # schema field -> name used here
for _p in _schema.SCHEMA:
    _k = _STREAM_NAME.get(_p.field, _p.field)
    if _k in DEFAULT_CONFIG:
        DEFAULT_CONFIG[_k] = _p.default
DEFAULT_CONFIG["distance"] = float(DEFAULT_CONFIG["p_array"].split()[2])
del _p, _k

# Keys whose default is another experiment's (the Ni lattice, a 0.513 m
# detector), plus the REQUIRED ones: a malformed line for one of these raises
# instead of being skipped with a warning, so a template's literal __SET_ME__
# cannot run as Ni.
_FATAL_KEYS = _schema.FATAL_KEYS


def _floats(key: str, rest: List[str], n: int) -> List[float]:
    """First *n* values as floats. Same token-count rule as the C and as
    config_schema: too few is an error, extra tokens warn and are ignored."""
    if len(rest) < n:
        raise ValueError(f"{key} needs {n} values, got {len(rest)}")
    if len(rest) > n:
        logger.warning(f"{key}: {len(rest)} values given, using the first {n} "
                       f"(as the C does); the rest are ignored.")
    return [float(v) for v in rest[:n]]


def parse_config(config_file: str) -> Dict[str, Any]:
    """
    Parse a classic LaueMatching text config file into a flat dictionary.

    Returns a dict with the same keys as DEFAULT_CONFIG, overridden by
    values found in the file. A key in config_schema.REQUIRED_KEYS that is
    absent, or a malformed value for one of config_schema.FATAL_KEYS
    (SpaceGroup, Symmetry, Elo, Ehi and the required keys), raises
    ValueError naming the key(s); other malformed lines are skipped with a
    warning.
    """
    cfg = dict(DEFAULT_CONFIG)
    if not os.path.exists(config_file):
        # Used to return the built-in defaults; there are none for the
        # required keys any more.
        raise ValueError(f"Config file '{config_file}' not found.")

    present = set()
    with open(config_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()
            parts = line.split()
            if not parts:
                continue
            key, rest = parts[0], parts[1:]
            present.add(key)
            if not rest:
                if key in _FATAL_KEYS:
                    raise ValueError(f"{config_file}: {key} has no value. {key} has "
                                     f"no safe default -- set it for this experiment.")
                continue

            try:
                if key == "SpaceGroup":
                    cfg["space_group"] = _schema.parse_space_group(rest[0])
                elif key == "Symmetry":
                    if len(rest[0]) != 1 or rest[0] not in _schema.SYMMETRY_LETTERS:
                        raise ValueError(_schema.SYMMETRY_RULE)
                    cfg["symmetry"] = rest[0]
                elif key == "LatticeParameter":
                    _floats(key, rest, 6)
                    cfg["lattice_parameter"] = " ".join(rest[:6])
                elif key == "R_Array":
                    _floats(key, rest, 3)
                    cfg["r_array"] = " ".join(rest[:3])
                elif key == "P_Array":
                    cfg["distance"] = _floats(key, rest, 3)[2]
                    cfg["p_array"] = " ".join(rest[:3])
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
                elif key == "RobustFilter":
                    cfg["robust_filter"] = bool(int(rest[0]))
                elif key == "PreprocessWorkers":
                    cfg["preprocess_workers"] = int(rest[0])
                elif key == "MinNrSpots":
                    cfg["min_nr_spots"] = int(rest[0])
                elif key == "MaxAngle":
                    cfg["max_angle"] = float(rest[0])
                elif key == "MaxNrLaueSpots":
                    cfg["max_laue_spots"] = int(rest[0])
                elif key == "OrientationSpacing":
                    cfg["orientation_spacing"] = float(rest[0])
                elif key == "GaussSigmaMax":
                    cfg["gauss_sigma_max"] = float(rest[0])
                elif key == "ThresholdMethod":
                    cfg["threshold_method"] = rest[0].lower()
                elif key == "Threshold":
                    cfg["threshold_value"] = float(rest[0])
                elif key == "ThresholdPercentile":
                    cfg["threshold_percentile"] = float(rest[0])
                elif key == "MinArea":
                    cfg["min_area"] = int(rest[0])
                elif key == "ExcludeSpotsFile":
                    cfg["exclude_spots_file"] = rest[0]
                elif key == "ExcludeSpotsDir":
                    cfg["exclude_spots_dir"] = rest[0]
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
                if key in _FATAL_KEYS:
                    raise ValueError(
                        f"{config_file}: {key} {' '.join(rest)!r} is not valid "
                        f"({e}). {key} has no safe default -- set it for this "
                        f"experiment.") from e
                logger.warning(f"Skipping malformed config line '{line}': {e}")

    missing = _schema.missing_required(present)
    if missing:
        raise ValueError(_schema.missing_required_message(missing, config_file))
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
    save_background,
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
# What the image server actually preprocessed with
#
# Post-processing re-runs the preprocessing to embed /entry/data in each output
# h5. It used to do so with a per-frame median background (the server computes
# ONE background, from the first frame, or loads BackgroundFile) and with
# ExcludeSpotsFile/Dir resolved against its own cwd (the server runs in the
# output dir), so the embedded images were not the ones that were indexed. The
# server now saves its background and records absolute paths in this file, next
# to frame_mapping.json; post-processing reads it back.
# ---------------------------------------------------------------------------
PREPROCESS_RECORD = "stream_preprocess.json"
STREAM_BACKGROUND = "stream_background.bin"


def preprocess_record_path(mapping_file: str) -> str:
    """The record written beside *mapping_file*."""
    return os.path.join(os.path.dirname(os.path.abspath(mapping_file)),
                        PREPROCESS_RECORD)


def write_preprocess_record(path: str, background_file: str,
                            cfg: Dict[str, Any]) -> None:
    rec = {
        "background_file": os.path.abspath(background_file),
        "exclude_spots_file": (os.path.abspath(cfg["exclude_spots_file"])
                               if cfg.get("exclude_spots_file") else ""),
        "exclude_spots_dir": (os.path.abspath(cfg["exclude_spots_dir"])
                              if cfg.get("exclude_spots_dir") else ""),
        "nr_px_x": int(cfg["nr_px_x"]),
        "nr_px_y": int(cfg["nr_px_y"]),
    }
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


def read_preprocess_record(path: str) -> Optional[Dict[str, Any]]:
    """The record, or None if there is none (a run from before 0.8)."""
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def cfg_for_frame(cfg: Dict[str, Any], h5_path: str) -> Dict[str, Any]:
    """*cfg* with this frame's ExcludeSpotsDir mask attached, if there is one.

    A copy, because the mask is frame-specific and *cfg* is shared: caching it
    on the shared dict would leak one frame's exclusions onto every later frame.
    """
    _dir = cfg.get("exclude_spots_dir", "")
    if not _dir:
        return cfg
    from laue_index.preprocess import exclusion_file_for_frame, load_exclusion_mask
    _f = exclusion_file_for_frame(_dir, h5_path)
    cfg = dict(cfg)
    cfg["_exclude_mask_frame"] = (
        load_exclusion_mask(_f, cfg["nr_px_y"], cfg["nr_px_x"]) if _f else None)
    return cfg


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
        # Tolerate torn/truncated rows (e.g. daemon terminated mid-write on a
        # dense frame): keep only rows with the modal column count.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = np.genfromtxt(path, skip_header=1, invalid_raise=False)
        if data.ndim == 2:
            good = ~np.isnan(data).any(axis=1)
            if (~good).sum():
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"read_solutions: dropped {(~good).sum()} malformed row(s) "
                    f"from {path}"
                )
            data = data[good]
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


# ---------------------------------------------------------------------------
# Results I/O — HDF5 output
#
# REFACTOR_PLAN §6.5: moved to laue_index.output (single source of truth);
# re-exported here so RunImage / laue_postprocess callers keep working.
# ---------------------------------------------------------------------------
from laue_index.output import (  # noqa: E402
    store_txt_files_in_h5,
    store_binary_headers_in_h5,
    create_h5_output,
)


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
