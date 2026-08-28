"""OutputWriter stage — HDF5 result writing.

REFACTOR_PLAN §3 / §6.5.  The .output.h5 writer (orientations/spots/unique-spots
datasets + header/column attributes + embedded text-file copies), lifted from
laue_stream_utils so laue_index owns it; lsu re-exports for back-compat.
"""
from __future__ import annotations

import logging
import os

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

logger = logging.getLogger("LaueStream")

__all__ = ["store_txt_files_in_h5", "store_binary_headers_in_h5", "create_h5_output"]


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
