#!/usr/bin/env python
"""
laue_postprocess.py — Post-processing for LaueMatchingGPUStream results

Reads the daemon's appended output files (solutions.txt, spots.txt),
splits them by ImageNr, applies unique-spot filtering, and generates
per-image HDF5 output + an interactive HTML visualization with an image
selector dropdown.

Usage:
    python laue_postprocess.py \
        --solutions solutions.txt \
        --spots spots.txt \
        --mapping frame_mapping.json \
        --config params.txt \
        --output-dir results/ \
        [--image-nr 0]           # 0 = all images, N = specific image
        [--min-unique 2]         # Minimum unique spots to keep orientation
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import laue_stream_utils as lsu
import laue_visualization as lv

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("laue_postprocess")


def _setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_single_image(
    image_nr: int,
    orientations: np.ndarray,
    spots: np.ndarray,
    cfg: Dict[str, Any],
    output_dir: str,
    min_unique: int = 2,
    mapping_info: Optional[Dict] = None,
    generate_per_image_viz: bool = False,
    labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Process results for a single image number.

    1. Use real image segmentation labels (if provided), else build dummy labels.
    2. Calculate unique spots per orientation.
    3. Filter orientations by minimum unique spots.
    4. Sort filtered orientations by quality (descending).
    5. Save per-image H5 output.
    6. Optionally generate per-image interactive visualization.
    7. Return summary dict for visualization.

    Args:
        generate_per_image_viz: If True, produce a per-image interactive
            Plotly HTML via ``lv.create_interactive_visualization``.

    Returns:
        Dict with keys: image_nr, n_orientations, n_filtered, n_spots,
                        filtered_orientations, filtered_spots, file, frame
    """
    nr_px_x = cfg["nr_px_x"]
    nr_px_y = cfg["nr_px_y"]

    result = {
        "image_nr": image_nr,
        "n_orientations": len(orientations),
        "n_filtered": 0,
        "n_spots": len(spots),
        "filtered_orientations": np.empty((0,)),
        "filtered_spots": np.empty((0,)),
    }
    if mapping_info:
        result["file"] = mapping_info.get("file", "")
        result["frame"] = mapping_info.get("frame", -1)

    if orientations.size == 0 or spots.size == 0:
        logger.warning(f"Image {image_nr}: no orientations or spots to process")
        return result

    # Build a simple label image from spot positions for unique-spot calc.
    # Each unique (x,y) location gets a distinct label.
    # Use a compact array covering just the bounding box of spot positions
    # to avoid allocating a full 2048×2048 image (16MB) per frame.
    # Determine column indices for spots and orientations.
    # Stream format: ImageNr=col0, GrainNr=col1, ..., X=col6, Y=col7 (12 cols)
    # RunImage format: GrainNr=col0, ..., X=col5, Y=col6 (11 cols)
    n_cols = spots.shape[1] if spots.ndim == 2 else 0
    n_sol_cols = orientations.shape[1] if orientations.ndim == 2 else 0

    # Stream spots have ImageNr prepended (12 cols vs 11)
    if n_cols > 10:
        spot_x_col = 6
        spot_y_col = 7
        spot_grain_col = 1
    else:
        spot_x_col = 5
        spot_y_col = 6
        spot_grain_col = 0

    # Stream solutions have ImageNr prepended (>30 cols typical)
    # RunImage solutions have GrainNr at col 0; stream has ImageNr at col 0,
    # GrainNr at col 1.  Use same heuristic as spots.
    if n_sol_cols > 30:
        orient_grain_col = 1
        orient_quality_col = 5  # NMatches*sqrt(Intensity)
    else:
        orient_grain_col = 0
        orient_quality_col = 4

    # Use real image segmentation labels if provided, otherwise build
    # simple per-pixel labels as a fallback.
    if labels is not None:
        logger.debug(f"Image {image_nr}: using real segmentation labels")
    else:
        # Fallback: each unique (x,y) position gets its own label.
        labels = np.zeros((nr_px_y, nr_px_x), dtype=np.int32)
        label_counter = 1
        spot_xs = spots[:, spot_x_col].astype(int)
        spot_ys = spots[:, spot_y_col].astype(int)
        valid = (spot_xs >= 0) & (spot_xs < nr_px_x) & (spot_ys >= 0) & (spot_ys < nr_px_y)
        for x, y in zip(spot_xs[valid], spot_ys[valid]):
            if labels[y, x] == 0:
                labels[y, x] = label_counter
                label_counter += 1

    # Calculate unique spots per orientation
    unique_info = lsu.calculate_unique_spots(
        orientations, spots, labels,
        grain_col=orient_grain_col, spot_grain_col=spot_grain_col,
        spot_x_col=spot_x_col, spot_y_col=spot_y_col,
        quality_col=orient_quality_col,
    )

    # Filter orientations
    filtered_orient = lsu.filter_orientations(
        orientations, unique_info, min_unique=min_unique,
        grain_col=orient_grain_col, quality_col=orient_quality_col,
    )

    # Filter spots to only those belonging to filtered orientations
    if filtered_orient.size > 0:
        kept_grains = set(filtered_orient[:, orient_grain_col].astype(int))
        grain_ids = spots[:, spot_grain_col].astype(int)
        mask = np.isin(grain_ids, list(kept_grains))
        filtered_spots = spots[mask] if mask.any() else np.empty((0, n_cols))
    else:
        filtered_spots = np.empty((0, n_cols))

    # Sort filtered orientations by quality (descending)
    if filtered_orient.size > 0:
        filtered_orient = lsu.sort_orientations_by_quality(
            filtered_orient, quality_col=orient_quality_col,
        )

    result["n_filtered"] = len(filtered_orient)
    result["filtered_orientations"] = filtered_orient
    result["filtered_spots"] = filtered_spots
    result["unique_info"] = unique_info

    logger.info(
        f"Image {image_nr}: {len(orientations)} orient → "
        f"{len(filtered_orient)} kept (≥{min_unique} unique spots), "
        f"{len(filtered_spots)} spots"
    )

    # --- Save per-image H5 ---
    if HAS_H5PY:
        _save_image_h5(
            image_nr, output_dir,
            orientations, filtered_orient,
            spots, filtered_spots,
            unique_info, mapping_info,
        )

    # --- Optional per-image interactive visualization ---
    if generate_per_image_viz and filtered_orient.size > 0:
        try:
            from laue_config import VisualizationConfig
            vis_cfg = VisualizationConfig()
            out_base = os.path.join(output_dir, f"image_{image_nr:05d}")
            # Build a simple filtered image (label mask) for background
            bg_image = labels.astype(float)
            lv.create_interactive_visualization(
                out_base, filtered_orient, filtered_spots, labels,
                bg_image, vis_cfg, labels.shape, unique_info,
            )
            logger.info(f"  Per-image viz saved: {out_base}.interactive.html")
        except Exception as e:
            logger.warning(f"  Per-image viz failed for image {image_nr}: {e}")

    return result


def _save_image_h5(
    image_nr: int,
    output_dir: str,
    orientations: np.ndarray,
    filtered_orientations: np.ndarray,
    spots: np.ndarray,
    filtered_spots: np.ndarray,
    unique_info: Dict[int, Dict],
    mapping_info: Optional[Dict] = None,
) -> None:
    """Save per-image results to HDF5 file."""
    h5_path = os.path.join(output_dir, f"image_{image_nr:05d}.output.h5")

    # Build unique counts array [GrainNr, UniqueLabels]
    unique_counts = []
    for gn, info in unique_info.items():
        unique_counts.append([gn, info.get("unique_label_count", 0)])
    uc_arr = np.array(unique_counts, dtype=np.int32) if unique_counts else np.empty((0, 2), dtype=np.int32)

    try:
        with h5py.File(h5_path, "w") as hf:
            grp = hf.require_group("/entry/results")
            grp.create_dataset("orientations", data=orientations)
            grp.create_dataset("filtered_orientations", data=filtered_orientations)
            grp.create_dataset("spots", data=spots)
            grp.create_dataset("filtered_spots", data=filtered_spots)
            grp.create_dataset("unique_spots_per_orientation", data=uc_arr)

            # Metadata
            grp.attrs["image_nr"] = image_nr
            if mapping_info:
                grp.attrs["source_file"] = mapping_info.get("file", "")
                grp.attrs["source_frame"] = mapping_info.get("frame", -1)

        logger.info(f"  Saved {h5_path}")

    except Exception as e:
        logger.error(f"  Error saving H5 for image {image_nr}: {e}")


# ---------------------------------------------------------------------------
# Combined visualization
# ---------------------------------------------------------------------------

def generate_visualization(
    all_results: List[Dict[str, Any]],
    output_dir: str,
    cfg: Dict[str, Any],
    frame_mapping: Dict,
) -> None:
    """
    Generate an interactive HTML visualization with an image-selector
    dropdown.  Uses a single pair of traces with data-swap buttons
    to keep the HTML file small regardless of the number of images.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly not available — skipping visualization.")
        return

    if not all_results:
        logger.warning("No results to visualize.")
        return

    nr_px_x = cfg["nr_px_x"]
    nr_px_y = cfg["nr_px_y"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Orientations (Euler φ₁ vs Φ)", "Spot Positions"),
        horizontal_spacing=0.12,
    )

    # Detect column indices from the first non-empty result
    orient_cols = (1, 2, 4, 0)  # phi1, Phi, quality, grain (RunImage default)
    spot_sx, spot_sy = 5, 6
    for res in all_results:
        o = res.get("filtered_orientations", np.empty((0,)))
        s = res.get("filtered_spots", np.empty((0,)))
        if o.ndim == 2 and o.shape[1] > 30:
            orient_cols = (2, 3, 5, 1)  # stream format: shifted by 1
        if s.ndim == 2 and s.shape[1] > 10:
            spot_sx, spot_sy = 6, 7
        break

    phi1_col, phi_col, q_col, g_col = orient_cols

    # Sort results by image number
    sorted_results = sorted(all_results, key=lambda r: r["image_nr"])

    # Start with the first image's data
    first = sorted_results[0] if sorted_results else {}
    first_orient = first.get("filtered_orientations", np.empty((0,)))
    first_spots = first.get("filtered_spots", np.empty((0,)))

    # Orientation scatter (single trace, data swapped by buttons)
    if first_orient.ndim == 2 and first_orient.size > 0:
        ox, oy, oc = first_orient[:, phi1_col], first_orient[:, phi_col], first_orient[:, q_col]
        otext = [
            f"Grain {int(o[g_col])}<br>φ₁={o[phi1_col]:.1f} Φ={o[phi_col]:.1f}<br>Q={o[q_col]:.2f}"
            for o in first_orient
        ]
    else:
        ox, oy, oc, otext = [], [], [], []

    fig.add_trace(
        go.Scatter(
            x=ox, y=oy, mode="markers",
            marker=dict(size=8, color=oc, colorscale="Viridis",
                        showscale=True, colorbar=dict(title="Quality", x=0.45)),
            text=otext, hoverinfo="text", name="Orientations",
        ),
        row=1, col=1,
    )

    # Spot scatter (single trace)
    if first_spots.ndim == 2 and first_spots.size > 0:
        spx, spy = first_spots[:, spot_sx], first_spots[:, spot_sy]
    else:
        spx, spy = [], []

    fig.add_trace(
        go.Scatter(
            x=spx, y=spy, mode="markers",
            marker=dict(size=4, color="red", opacity=0.6),
            name="Spots",
        ),
        row=1, col=2,
    )

    # Build dropdown buttons that swap data via restyle
    buttons = []
    for res in sorted_results:
        img_nr = res["image_nr"]
        orient = res.get("filtered_orientations", np.empty((0,)))
        spots = res.get("filtered_spots", np.empty((0,)))

        # Orientation data
        if orient.ndim == 2 and orient.size > 0:
            bx = orient[:, phi1_col].tolist()
            by = orient[:, phi_col].tolist()
            bc = orient[:, q_col].tolist()
            bt = [
                f"Grain {int(o[g_col])}<br>φ₁={o[phi1_col]:.1f} Φ={o[phi_col]:.1f}<br>Q={o[q_col]:.2f}"
                for o in orient
            ]
        else:
            bx, by, bc, bt = [[]], [[]], [[]], [[]]

        # Spot data
        if spots.ndim == 2 and spots.size > 0:
            sx_data = spots[:, spot_sx].tolist()
            sy_data = spots[:, spot_sy].tolist()
        else:
            sx_data, sy_data = [[]], [[]]

        # Build label
        map_entry = frame_mapping.get(str(img_nr), {})
        src_file = map_entry.get("file", "")
        src_frame = map_entry.get("frame", "")
        label = f"Image {img_nr}"
        if src_file:
            label += f" ({src_file}"
            if src_frame != "":
                label += f" f{src_frame}"
            label += ")"

        n_orient = res.get("n_filtered", 0)
        n_spots = len(res.get("filtered_spots", []))

        buttons.append(dict(
            method="restyle",
            args=[
                {"x": [bx, sx_data], "y": [by, sy_data],
                 "text": [bt, [None]], "marker.color": [bc, ["red"]]},
                [0, 1],  # trace indices to update
            ],
            label=f"{label} — {n_orient} orient, {n_spots} spots",
        ))

    fig.update_layout(
        title="LaueMatching Stream Results",
        updatemenus=[dict(
            type="dropdown",
            direction="down",
            active=0,
            x=0.5,
            xanchor="center",
            y=1.18,
            yanchor="top",
            buttons=buttons,
            pad=dict(t=10),
        )],
        height=600,
        width=1200,
        template="plotly_dark",
        showlegend=False,
    )
    fig.update_xaxes(title_text="φ₁ (deg)", row=1, col=1)
    fig.update_yaxes(title_text="Φ (deg)", row=1, col=1)
    fig.update_xaxes(title_text="X (px)", range=[0, nr_px_x], row=1, col=2)
    fig.update_yaxes(title_text="Y (px)", range=[0, nr_px_y], autorange="reversed", row=1, col=2)

    html_path = os.path.join(output_dir, "stream_results.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    logger.info(f"Visualization saved to {html_path}")


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _write_summary(all_results: List[Dict], output_dir: str, frame_mapping: Dict) -> None:
    """Write a plain-text summary table to results/summary.txt."""
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"{'ImgNr':>6} {'File':<40} {'Frame':>5} "
                f"{'Orient':>7} {'Filt':>5} {'Spots':>6}\n")
        f.write("-" * 80 + "\n")
        for r in sorted(all_results, key=lambda x: x["image_nr"]):
            f.write(
                f"{r['image_nr']:6d} "
                f"{r.get('file', ''):40s} "
                f"{r.get('frame', -1):5d} "
                f"{r['n_orientations']:7d} "
                f"{r['n_filtered']:5d} "
                f"{r['n_spots']:6d}\n"
            )
    logger.info(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def postprocess(
    solutions_path: str,
    spots_path: str,
    config_file: str,
    output_dir: str,
    mapping_file: str = "frame_mapping.json",
    labels_file: str = "",
    image_nr: int = 0,
    min_unique: int = 2,
    nprocs: int = 1,
) -> None:
    """
    Main post-processing entry point.

    Args:
        solutions_path: Path to solutions.txt from daemon.
        spots_path:     Path to spots.txt from daemon.
        config_file:    Path to params.txt.
        output_dir:     Directory for output H5 and visualization files.
        mapping_file:   Path to frame_mapping.json.
        labels_file:    Path to labels.h5 (image segmentation labels).
        image_nr:       0 = process all images, N = specific image only.
        min_unique:     Minimum unique spots to keep an orientation.
        nprocs:         Number of parallel processes (default: 1 = serial).
    """
    cfg = lsu.parse_config(config_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load frame mapping
    frame_mapping = lsu.load_frame_mapping(mapping_file)

    # Read solutions and spots
    logger.info(f"Reading solutions from {solutions_path}")
    solutions, _sol_hdr = lsu.read_solutions(solutions_path)
    logger.info(f"  {len(solutions)} solutions loaded")

    logger.info(f"Reading spots from {spots_path}")
    spots, _spot_hdr = lsu.read_spots(spots_path)
    logger.info(f"  {len(spots)} spots loaded")

    if solutions.size == 0 or spots.size == 0:
        logger.error("No solutions or spots — nothing to post-process.")
        return

    # Split by ImageNr — solutions already have ImageNr in col 0 in stream
    # format, so split directly with vectorized groupby.
    spots_by_image = lsu.split_spots_by_image(spots, image_col=0)

    # Solutions: split directly by ImageNr (col 0) — much faster than the
    # indirect grain→spots→image path used by split_solutions_by_image.
    sol_img_ids = solutions[:, 0].astype(int)
    unique_sol_imgs = np.unique(sol_img_ids)
    solutions_by_image = {
        int(img): solutions[sol_img_ids == img]
        for img in unique_sol_imgs
    }

    if image_nr > 0:
        # Process only a specific image
        target_images = [image_nr]
    else:
        # Process all available
        target_images = sorted(
            set(spots_by_image.keys()) | set(solutions_by_image.keys())
        )

    logger.info(f"Processing {len(target_images)} image(s): {target_images[:10]}{'...' if len(target_images) > 10 else ''}")

    # Load image segmentation labels if available
    labels_h5f = None
    if labels_file and os.path.isfile(labels_file) and HAS_H5PY:
        labels_h5f = h5py.File(labels_file, "r")
        logger.info(f"Using image segmentation labels from {labels_file}")
    elif labels_file and not os.path.isfile(labels_file):
        logger.warning(f"Labels file not found: {labels_file} — using fallback per-pixel labels")

    def _load_labels(img_num: int) -> Optional[np.ndarray]:
        """Load labels for a given image number from labels.h5."""
        if labels_h5f is None:
            return None
        ds_name = f"labels/{img_num}"
        if ds_name in labels_h5f:
            return np.array(labels_h5f[ds_name])
        return None

    # Process each image (parallel or serial)
    all_results: List[Dict[str, Any]] = []
    n_images = len(target_images)

    if nprocs > 1 and n_images > 1:
        effective_procs = min(nprocs, n_images)
        logger.info(f"Using {effective_procs} parallel workers")

        # Build argument tuples for each image
        futures = {}
        with ProcessPoolExecutor(max_workers=effective_procs) as pool:
            for img in target_images:
                img_solutions = solutions_by_image.get(
                    img, np.empty((0, solutions.shape[1]))
                )
                img_spots = spots_by_image.get(
                    img, np.empty((0, spots.shape[1]))
                )
                map_info = frame_mapping.get(str(img), None)

                img_labels = _load_labels(img)

                fut = pool.submit(
                    process_single_image,
                    image_nr=img,
                    orientations=img_solutions,
                    spots=img_spots,
                    cfg=cfg,
                    output_dir=output_dir,
                    min_unique=min_unique,
                    mapping_info=map_info,
                    labels=img_labels,
                )
                futures[fut] = img

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    all_results.append(res)
                except Exception as e:
                    img = futures[fut]
                    logger.error(f"Image {img} failed: {e}")
    else:
        for img in target_images:
            img_solutions = solutions_by_image.get(
                img, np.empty((0, solutions.shape[1]))
            )
            img_spots = spots_by_image.get(
                img, np.empty((0, spots.shape[1]))
            )
            map_info = frame_mapping.get(str(img), None)
            img_labels = _load_labels(img)

            res = process_single_image(
                image_nr=img,
                orientations=img_solutions,
                spots=img_spots,
                cfg=cfg,
                output_dir=output_dir,
                min_unique=min_unique,
                mapping_info=map_info,
                labels=img_labels,
            )
            all_results.append(res)

    # Close labels file
    if labels_h5f is not None:
        labels_h5f.close()

    # Summary
    _write_summary(all_results, output_dir, frame_mapping)

    # Visualization
    generate_visualization(all_results, output_dir, cfg, frame_mapping)

    logger.info(f"Post-processing complete. Output in {output_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process LaueMatchingGPUStream results"
    )
    parser.add_argument(
        "--solutions", required=True,
        help="Path to solutions.txt"
    )
    parser.add_argument(
        "--spots", required=True,
        help="Path to spots.txt"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to params.txt configuration file"
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory (default: results)"
    )
    parser.add_argument(
        "--mapping", default="frame_mapping.json",
        help="Path to frame_mapping.json (default: frame_mapping.json)"
    )
    parser.add_argument(
        "--labels", default="",
        help="Path to labels.h5 with image segmentation labels (default: none)"
    )
    parser.add_argument(
        "--image-nr", type=int, default=0,
        help="Process specific image number (0 = all, default: 0)"
    )
    parser.add_argument(
        "--min-unique", type=int, default=2,
        help="Minimum unique spots to keep orientation (default: 2)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )
    parser.add_argument(
        "--nprocs", type=int, default=1,
        help="Number of parallel processes for per-image processing (default: 1)"
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    # Validate inputs
    for path, label in [
        (args.solutions, "Solutions file"),
        (args.spots, "Spots file"),
        (args.config, "Config file"),
    ]:
        if not os.path.isfile(path):
            logger.error(f"{label} not found: {path}")
            sys.exit(1)

    postprocess(
        solutions_path=args.solutions,
        spots_path=args.spots,
        config_file=args.config,
        output_dir=args.output_dir,
        mapping_file=args.mapping,
        labels_file=args.labels,
        image_nr=args.image_nr,
        min_unique=args.min_unique,
        nprocs=args.nprocs,
    )


if __name__ == "__main__":
    main()
