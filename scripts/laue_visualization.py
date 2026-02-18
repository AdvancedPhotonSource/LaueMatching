#!/usr/bin/env python3
"""
laue_visualization.py — Standalone visualization functions for Laue diffraction.

Extracted from RunImage.py's EnhancedImageProcessor so they can be reused by
laue_postprocess.py and other entry-points without importing the full
EnhancedImageProcessor class.

All functions are pure (no ``self``); they receive data and configuration as
explicit arguments.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage as ndimg

# Optional heavy imports --------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger("LaueStream")


# ── helpers ───────────────────────────────────────────────────────────────

def _figure_size(image_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Return (width, height) in inches given an image (Y, X) shape."""
    ny, nx = image_shape
    base = 12.0
    aspect = nx / max(ny, 1)
    return (base * aspect, base)


# ── 1. visualize_results (dispatcher) ────────────────────────────────────

def visualize_results(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    filtered_image: np.ndarray,
    vis_config,
    image_shape: Tuple[int, int],
    orientation_unique_spots: Optional[Dict[int, Dict]] = None,
    config_obj=None,
    result_dir: str = "results",
) -> Dict[str, Any]:
    """
    Generate visualisations based on *vis_config*.

    Args:
        output_path:             Base file path (e.g. ``results/image_001``).
        orientations:            Filtered orientation array.
        spots:                   Filtered spot array.
        labels:                  Final segmentation labels.
        filtered_image:          Filtered thresholded image.
        vis_config:              ``VisualizationConfig`` dataclass (``plot_type``,
                                 ``output_dpi``, ``colormap``, ``generate_3d``,
                                 ``generate_report``, ``show_hkl_labels``).
        image_shape:             ``(ny, nx)`` — shape of the image.
        orientation_unique_spots: ``{grain_nr: {...}}``.
        config_obj:              Full ``LaueConfig`` dataclass (only needed by
                                 ``create_analysis_report``).
        result_dir:              Results directory for linking in reports.

    Returns:
        ``{"success": bool}``
    """
    if vis_config is None:
        logger.warning("Visualization config missing.")
        return {"success": False, "error": "Visualization config missing"}

    plot_type = vis_config.plot_type
    success = True

    if plot_type in ("static", "both"):
        r = create_static_visualization(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, image_shape, orientation_unique_spots,
        )
        if not r["success"]:
            logger.warning(f"Static viz failed: {r.get('error')}")
            success = False

    if plot_type in ("interactive", "both"):
        r = create_interactive_visualization(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, image_shape, orientation_unique_spots,
        )
        if not r["success"]:
            logger.warning(f"Interactive viz failed: {r.get('error')}")
            success = False

    if vis_config.generate_3d:
        try:
            create_3d_visualization(output_path, orientations, spots)
        except Exception as e:
            logger.warning(f"3D viz failed: {e}")

    if vis_config.generate_report:
        try:
            create_analysis_report(
                output_path, orientations, spots, labels, filtered_image,
                vis_config, orientation_unique_spots, config_obj, result_dir,
            )
        except Exception as e:
            logger.warning(f"Report failed: {e}")

    return {"success": success}


# ── 2. static visualization ──────────────────────────────────────────────

def create_static_visualization(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    filtered_image: np.ndarray,
    vis_config,
    image_shape: Tuple[int, int],
    orientation_unique_spots: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """Create static PNG of labelled spots and quality map."""
    if not HAS_MPL:
        return {"success": False, "error": "matplotlib not available"}

    logger.info("Creating static visualizations...")
    scalar_x, scalar_y = _figure_size(image_shape)
    fig_label = None

    try:
        fig_label, ax_label = plt.subplots(figsize=(scalar_x, scalar_y))

        display_img = filtered_image.copy().astype(float)
        display_img[display_img <= 0] = 1
        ax_label.imshow(np.log(display_img), cmap="Greens", origin="upper")

        colormap_name = vis_config.colormap if vis_config else "nipy_spectral"
        num_orientations = len(orientations)
        colors = plt.get_cmap(colormap_name, num_orientations) if num_orientations > 0 else None

        if orientations.size > 0 and spots.size > 0 and colors:
            plot_orientation_spots(
                orientations, spots, filtered_image, labels, colors, ax_label,
                show_hkl_labels=vis_config.show_hkl_labels if vis_config else False,
            )

        ax_label.set_title(f"Indexed Spots ({num_orientations} Orientations)")
        ax_label.set_xlabel("X Pixel")
        ax_label.set_ylabel("Y Pixel")
        if num_orientations > 0:
            ax_label.legend(loc="upper right", fontsize="xx-small", markerscale=2)
        plt.tight_layout()

        dpi = vis_config.output_dpi if vis_config else 600
        png_file = f"{output_path}.LabeledImage.png"
        plt.savefig(png_file, dpi=dpi)
        plt.close(fig_label)
        fig_label = None
        logger.info(f"Static labeled visualization saved to {png_file}")

        if orientations.size > 0 and spots.size > 0:
            create_quality_map(
                output_path, orientations, spots, filtered_image,
                vis_config, image_shape, orientation_unique_spots,
            )

        return {"success": True}

    except Exception as e:
        logger.error(f"Error creating static visualization: {e}", exc_info=True)
        if fig_label is not None and plt.fignum_exists(fig_label.number):
            plt.close(fig_label)
        return {"success": False, "error": str(e)}


# ── 3. quality map ───────────────────────────────────────────────────────

def create_quality_map(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
    filtered_image: np.ndarray,
    vis_config,
    image_shape: Tuple[int, int],
    orientation_unique_spots: Optional[Dict[int, Dict]] = None,
) -> None:
    """Create a quality map PNG (Gaussian-blurred per-pixel quality score)."""
    if not HAS_MPL:
        return

    logger.debug("Creating quality map...")
    dpi = vis_config.output_dpi if vis_config else 600
    scalar_x, scalar_y = _figure_size(image_shape)
    fig_qual, ax_qual = plt.subplots(figsize=(scalar_x, scalar_y))

    quality_map = np.zeros_like(filtered_image, dtype=float)
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, axis=0)

    max_q = 0.0
    for orientation in orientations:
        grain_nr = int(orientation[0])
        q = orientation[4]
        uc = 0
        if orientation_unique_spots and grain_nr in orientation_unique_spots:
            uc = orientation_unique_spots[grain_nr].get("unique_label_count", 0)
        eq = q * (1.0 + 0.1 * uc)
        max_q = max(max_q, eq)
        for spot in spots[spots[:, 0] == grain_nr]:
            try:
                x, y = int(spot[5]), int(spot[6])
                if 0 <= y < quality_map.shape[0] and 0 <= x < quality_map.shape[1]:
                    quality_map[y, x] = max(quality_map[y, x], eq)
            except (IndexError, ValueError):
                continue

    qm = ndimg.gaussian_filter(quality_map, sigma=3) if np.any(quality_map > 0) else quality_map

    im = ax_qual.imshow(qm, cmap="viridis", origin="upper", vmin=0, vmax=max_q if max_q > 0 else 1)
    plt.colorbar(im, ax=ax_qual, label="Indexing Quality (Score × Unique Spot Boost)")
    ax_qual.set_title("Orientation Indexing Quality Map")
    ax_qual.set_xlabel("X Pixel")
    ax_qual.set_ylabel("Y Pixel")
    plt.tight_layout()

    qmap_file = f"{output_path}.QualityMap.png"
    plt.savefig(qmap_file, dpi=dpi)
    plt.close(fig_qual)
    logger.info(f"Quality map saved to {qmap_file}")


# ── 4. interactive visualization (Plotly) ─────────────────────────────────

def create_interactive_visualization(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    filtered_image: np.ndarray,
    vis_config,
    image_shape: Tuple[int, int],
    orientation_unique_spots: Optional[Dict[int, Dict]] = None,
) -> Dict[str, Any]:
    """Create a 2-panel interactive Plotly HTML (spots + quality map)."""
    if not HAS_PLOTLY:
        return {"success": False, "error": "plotly not available"}

    logger.info("Creating interactive visualization...")

    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Indexed Diffraction Pattern", "Orientation Quality"],
            horizontal_spacing=0.05,
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            shared_xaxes=True, shared_yaxes=True,
        )

        # Background
        display_img = filtered_image.copy().astype(float)
        display_img[display_img <= 0] = 1
        log_img = np.log(display_img)
        fig.add_trace(
            go.Heatmap(z=log_img, colorscale="Greens", showscale=False, name="Log Intensity"),
            row=1, col=1,
        )

        # Quality map
        quality_map = np.zeros_like(filtered_image, dtype=float)
        max_q = 0.0
        if orientations.size > 0:
            orients_2d = orientations if orientations.ndim == 2 else np.expand_dims(orientations, 0)
            for ori in orients_2d:
                gn = int(ori[0])
                q = ori[4]
                uc = 0
                if orientation_unique_spots and gn in orientation_unique_spots:
                    uc = orientation_unique_spots[gn].get("unique_label_count", 0)
                eq = q * (1.0 + 0.1 * uc)
                max_q = max(max_q, eq)
                for spot in spots[spots[:, 0] == gn]:
                    try:
                        x, y = int(spot[5]), int(spot[6])
                        if 0 <= y < quality_map.shape[0] and 0 <= x < quality_map.shape[1]:
                            quality_map[y, x] = max(quality_map[y, x], eq)
                    except (IndexError, ValueError):
                        continue

        qm = ndimg.gaussian_filter(quality_map, sigma=3) if np.any(quality_map > 0) else quality_map
        fig.add_trace(
            go.Heatmap(
                z=qm, colorscale="Viridis", showscale=True, name="Quality",
                colorbar=dict(title="Quality", x=0.46, y=0.5, len=0.9, thickness=15),
                zmin=0, zmax=max_q if max_q > 0 else 1,
            ),
            row=1, col=2,
        )

        # Spot traces
        palette = px.colors.qualitative.Plotly
        if orientations.size > 0 and spots.size > 0:
            orients_plot = orientations if orientations.ndim == 2 else np.expand_dims(orientations, 0)
            for i, ori in enumerate(orients_plot):
                gn = int(ori[0])
                color = palette[i % len(palette)]
                uc = 0
                if orientation_unique_spots and gn in orientation_unique_spots:
                    uc = orientation_unique_spots[gn].get("unique_label_count", 0)
                os_ = spots[spots[:, 0] == gn]
                if os_.size == 0:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=os_[:, 5], y=os_[:, 6],
                        mode="markers", name=f"Grain {gn} ({uc} unique)",
                        legendgroup=f"grain_{gn}",
                        marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=1.5)),
                        hovertext=[
                            f"Grain: {gn}<br>HKL: ({int(s[2])},{int(s[3])},{int(s[4])})<br>"
                            f"Pos: ({s[5]:.1f}, {s[6]:.1f})<br>Unique: {uc}"
                            for s in os_
                        ],
                        hoverinfo="text",
                    ),
                    row=1, col=1,
                )

        # Layout
        y_max, x_max = image_shape
        fig.update_layout(
            title="Laue Diffraction Analysis (Filtered Results)",
            height=700, width=1600,
            showlegend=True,
            legend=dict(
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                bordercolor="Black", borderwidth=1, font=dict(size=10), title=dict(text="Grains"),
            ),
            margin=dict(l=50, r=150, b=50, t=50),
            hovermode="closest",
        )
        fig.update_xaxes(range=[0, x_max], constrain="domain")
        fig.update_yaxes(range=[y_max, 0], constrain="domain")
        fig.update_xaxes(title_text="X Pixel", row=1, col=1)
        fig.update_yaxes(title_text="Y Pixel", row=1, col=1)
        fig.update_xaxes(title_text="X Pixel", row=1, col=2)
        fig.update_yaxes(title_text="Y Pixel", row=1, col=2)

        html_file = f"{output_path}.interactive.html"
        fig.write_html(html_file, include_plotlyjs="cdn")
        logger.info(f"Interactive visualization saved to {html_file}")
        return {"success": True}

    except Exception as e:
        logger.error(f"Error creating interactive visualization: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── 5. 3-D orientation visualization ─────────────────────────────────────

def create_3d_visualization(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
) -> None:
    """Create a 3-D Plotly HTML of crystal orientation axes."""
    if not HAS_PLOTLY:
        logger.warning("plotly not available; skipping 3D visualization")
        return

    logger.debug("Creating 3D orientation visualization...")
    if orientations.size == 0:
        logger.warning("No orientations; skipping 3D visualization")
        return

    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, axis=0)

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for i, ori in enumerate(orientations):
        gn = int(ori[0])
        color = palette[i % len(palette)]
        try:
            matrix = ori[22:31].reshape(3, 3)
        except (IndexError, ValueError):
            logger.warning(f"Could not extract matrix for Grain {gn}; skipping.")
            continue

        origin = [0, 0, 0]
        scale = 1.0
        for axis, vec, ac in [("X", matrix[0], "red"), ("Y", matrix[1], "green"), ("Z", matrix[2], "blue")]:
            fig.add_trace(go.Scatter3d(
                x=[origin[0], origin[0] + scale * vec[0]],
                y=[origin[1], origin[1] + scale * vec[1]],
                z=[origin[2], origin[2] + scale * vec[2]],
                mode="lines+markers", name=f"Grain {gn} {axis}-axis",
                legendgroup=f"grain_{gn}",
                line=dict(color=ac, width=5), marker=dict(size=3, color=ac),
            ))

        cell = 0.2 * scale
        r1, r2, r3 = matrix[0], matrix[1], matrix[2]
        corners = [
            np.array(origin), r1 * cell, r2 * cell, r3 * cell,
            (r1 + r2) * cell, (r1 + r3) * cell, (r2 + r3) * cell, (r1 + r2 + r3) * cell,
        ]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        for p1i, p2i in edges:
            p1, p2 = corners[p1i], corners[p2i]
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                mode="lines", line=dict(color=color, width=1.5), showlegend=False,
            ))

    fig.update_layout(
        title="3D Visualization of Crystal Orientations (Filtered)",
        scene=dict(xaxis_title="Lab X", yaxis_title="Lab Y", zaxis_title="Lab Z", aspectmode="data"),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        width=800, height=800,
        legend=dict(x=1.05, y=0.5, font=dict(size=10), bordercolor="Black", borderwidth=1),
        margin=dict(l=10, r=150, b=10, t=40),
    )

    html_3d = f"{output_path}.3D.html"
    try:
        fig.write_html(html_3d, include_plotlyjs="cdn")
        logger.info(f"3D visualization saved to {html_3d}")
    except Exception as e:
        logger.error(f"Could not save 3D visualization: {e}")


# ── 6. analysis report ───────────────────────────────────────────────────

def create_analysis_report(
    output_path: str,
    orientations: np.ndarray,
    spots: np.ndarray,
    labels: np.ndarray,
    filtered_image: np.ndarray,
    vis_config,
    orientation_unique_spots: Optional[Dict[int, Dict]] = None,
    config_obj=None,
    result_dir: str = "results",
) -> None:
    """
    Write a comprehensive HTML analysis report.

    Args:
        config_obj: Full ``LaueConfig`` dataclass (optional; if *None* the
                    processing-parameters section is omitted).
    """
    logger.info("Generating analysis report...")
    if orientations.size == 0:
        logger.warning("No orientations; skipping report.")
        return
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, axis=0)

    report_path = f"{output_path}.report.html"
    num_orientations = len(orientations)
    num_spots = len(spots)
    avg_spots = f"{num_spots / num_orientations:.1f}" if num_orientations > 0 else "N/A"
    image_basename = os.path.basename(output_path)

    # ── HTML head ────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laue Diffraction Analysis Report: {image_basename}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; line-height: 1.4; }}
        h1, h2, h3 {{ color: #2c3e50; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        .container {{ max-width: 1100px; margin: 0 auto; background-color: #ecf0f1; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary, .parameters {{ background-color: #ffffff; padding: 15px; border: 1px solid #bdc3c7; border-radius: 5px; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; background-color: #fff; }}
        th, td {{ border: 1px solid #bdc3c7; padding: 10px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9f9; }}
        .matrix-table table, .matrix-table td {{ border: none; padding: 1px 3px; font-size: 0.9em; text-align: right; }}
        .image-container img {{ max-width: 48%; height: auto; border: 1px solid #bdc3c7; margin: 5px; }}
        .chart-container {{ height: 400px; background-color: #fff; padding: 15px; border-radius: 5px; border: 1px solid #bdc3c7; margin-bottom: 20px;}}
        .footer {{ margin-top: 30px; font-size: 0.85em; color: #7f8c8d; text-align: center; }}
        a {{ color: #2980b9; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul {{ padding-left: 20px; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>Laue Diffraction Analysis Report</h1>
        <p><strong>File:</strong> {image_basename}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary">
            <h2>Analysis Summary (Filtered Results)</h2>
            <ul>
                <li>Total orientations found (after filtering): {num_orientations}</li>
                <li>Total spots indexed (for filtered orientations): {num_spots}</li>
                <li>Average spots per filtered orientation: {avg_spots}</li>
            </ul>
        </div>

        <h2>Visualizations</h2>
        <div class="image-container" style="text-align: center;">
            <a href="{image_basename}.LabeledImage.png" target="_blank"><img src="{image_basename}.LabeledImage.png" alt="Indexed Diffraction Pattern"></a>
            <a href="{image_basename}.QualityMap.png" target="_blank"><img src="{image_basename}.QualityMap.png" alt="Orientation Quality Map"></a>
        </div>
        <p style="text-align: center;"><em>Click images to enlarge</em></p>

        <h2>Orientation Summary (Filtered)</h2>
        <table>
            <thead><tr>
                <th>Grain Nr</th><th>Quality</th><th>Total Spots</th><th>Unique Spots</th><th>Orientation Matrix [Lab <- Crystal]</th>
            </tr></thead>
            <tbody>
    """

    for orientation in orientations:
        gn = int(orientation[0])
        quality = orientation[4]
        total = int(orientation[5])
        us = 0
        if orientation_unique_spots and gn in orientation_unique_spots:
            us = orientation_unique_spots[gn].get("unique_label_count", 0)

        mat_html = "<span class='matrix-table'><table>"
        try:
            mat = orientation[22:31].reshape(3, 3)
            for row in mat:
                mat_html += "<tr>" + "".join([f"<td>{x: .4f}</td>" for x in row]) + "</tr>"
        except (IndexError, ValueError):
            mat_html += "<tr><td colspan='3'>Error</td></tr>"
        mat_html += "</table></span>"

        html += f"""
                <tr>
                    <td>{gn}</td><td>{quality:.4f}</td><td>{total}</td><td>{us}</td><td>{mat_html}</td>
                </tr>"""

    html += """
            </tbody>
        </table>
    """

    # ── Spot distribution chart ──────────────────────────────────────
    grain_numbers = [int(o[0]) for o in orientations]
    spo = {}
    uso = {}
    for gn in grain_numbers:
        spo[gn] = int(np.sum(spots[:, 0] == gn))
        uso[gn] = (orientation_unique_spots or {}).get(gn, {}).get("unique_label_count", 0)

    html += f"""
        <h2>Spot Distribution per Filtered Orientation</h2>
        <div class="chart-container">
            <canvas id="spotsChart"></canvas>
        </div>
        <script>
            const ctx = document.getElementById('spotsChart').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: [{", ".join(map(str, grain_numbers))}],
                    datasets: [
                        {{ label: 'Total Spots Matched',
                           data: [{", ".join(str(spo.get(g, 0)) for g in grain_numbers)}],
                           backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1 }},
                        {{ label: 'Unique Spots (Labels)',
                           data: [{", ".join(str(uso.get(g, 0)) for g in grain_numbers)}],
                           backgroundColor: 'rgba(255, 99, 132, 0.6)', borderColor: 'rgba(255, 99, 132, 1)', borderWidth: 1 }}
                    ]
                }},
                options: {{
                    responsive: true, maintainAspectRatio: false, indexAxis: 'x',
                    scales: {{
                         y: {{ beginAtZero: true, title: {{ display: true, text: 'Number of Spots' }} }},
                         x: {{ title: {{ display: true, text: 'Grain Number' }} }}
                    }}
                }}
            }});
        </script>
    """

    # ── Links ────────────────────────────────────────────────────────
    html += """
        <h2>Interactive Visualizations</h2>
        <ul>"""
    for label, suffix in [
        ("Interactive Diffraction Pattern & Quality Map", ".interactive.html"),
        ("Simulation Comparison", ".simulation_comparison.html"),
        ("3D Orientation Visualization", ".3D.html"),
    ]:
        fname = f"{image_basename}{suffix}"
        if os.path.exists(os.path.join(result_dir, fname)):
            html += f'<li><a href="{fname}" target="_blank">{label}</a></li>'
    html += "</ul>"

    # ── Processing parameters table ──────────────────────────────────
    if config_obj is not None:
        html += """
        <div class="parameters">
            <h2>Key Processing Parameters</h2>
            <table><thead><tr><th>Parameter Group</th><th>Parameter</th><th>Value</th></tr></thead><tbody>"""

        cfg = config_obj
        sections = [
            ("Core/Detector", [
                ("Space Group", cfg.space_group), ("Symmetry", cfg.symmetry),
                ("Lattice Parameters", cfg.lattice_parameter),
                ("Detector Size (px)", f"{cfg.nr_px_x} x {cfg.nr_px_y}"),
                ("Pixel Size (mm)", f"{cfg.px_x} x {cfg.px_y}"),
                ("Distance (mm?)", cfg.distance),
            ]),
            ("Image Processing", [
                ("Threshold Method", cfg.image_processing.threshold_method),
                ("Threshold Value (Fixed)", cfg.image_processing.threshold_value),
                ("Threshold Percentile", cfg.image_processing.threshold_percentile),
                ("Min Area", cfg.image_processing.min_area),
                ("Median Radius", cfg.image_processing.filter_radius),
                ("Median Passes", cfg.image_processing.median_passes),
                ("Watershed", cfg.image_processing.watershed_enabled),
                ("Enhance Contrast", cfg.image_processing.enhance_contrast),
                ("Denoise", cfg.image_processing.denoise_image),
                ("Edge Enhance", cfg.image_processing.edge_enhancement),
            ]),
            ("Filtering", [("Min Unique Spots", cfg.min_good_spots)]),
            ("Indexing Executable", [
                ("Processing Type", cfg.processing_type), ("CPUs Used", cfg.num_cpus),
                ("Do Forward Sim?", cfg.do_forward), ("Min Nr Spots (Exec)", cfg.min_nr_spots),
                ("Max Laue Spots (Exec)", cfg.max_laue_spots), ("Max Angle (Exec)", cfg.maxAngle),
            ]),
            ("Python Simulation", [
                ("Enable Sim (Python)", cfg.simulation.enable_simulation),
                ("Skip Percentage", cfg.simulation.skip_percentage),
                ("Simulation Energies", cfg.simulation.energies),
            ]),
        ]

        for group_name, params in sections:
            for j, (pname, pval) in enumerate(params):
                rs = f'<td rowspan="{len(params)}">{group_name}</td>' if j == 0 else ""
                html += f"<tr>{rs}<td>{pname}</td><td>{pval}</td></tr>\n"

        html += """
            </tbody></table>
        </div>"""

    # ── Footer ───────────────────────────────────────────────────────
    html += """
        <div class="footer">
            <p>Report generated by LaueMatching Software</p>
            <p>Contact: Hemant Sharma (hsharma@anl.gov)</p>
        </div>
    </div>
</body>
</html>"""

    try:
        with open(report_path, "w") as f:
            f.write(html)
        logger.info(f"Analysis report saved to {report_path}")
    except Exception as e:
        logger.error(f"Could not write report {report_path}: {e}")


# ── 7. plot orientation spots (static helper) ────────────────────────────

def plot_orientation_spots(
    orientations: np.ndarray,
    spots: np.ndarray,
    filtered_image: np.ndarray,
    labels: np.ndarray,
    colors,
    ax,
    show_hkl_labels: bool = False,
) -> None:
    """Plot filtered spots per orientation on a Matplotlib axis."""
    if orientations.size == 0 or spots.size == 0 or colors is None:
        return
    if orientations.ndim == 1:
        orientations = np.expand_dims(orientations, axis=0)

    for i, ori in enumerate(orientations):
        gn = int(ori[0])
        color = colors(i / len(orientations)) if len(orientations) > 1 else colors(0)
        os_ = spots[spots[:, 0] == gn]
        if os_.size == 0:
            continue
        ax.plot(
            os_[:, 5], os_[:, 6], "o",
            markerfacecolor="none", markersize=4,
            markeredgecolor=color, markeredgewidth=0.5,
            label=f"Grain {gn} ({len(os_)} spots)",
        )
        if show_hkl_labels:
            for spot in os_:
                try:
                    h, k, l = int(spot[2]), int(spot[3]), int(spot[4])
                    x, y = spot[5], spot[6]
                    ax.text(x, y + 5, f"({h}{k}{l})", fontsize=1.5, ha="center", color=color, clip_on=True)
                except (IndexError, ValueError):
                    continue


# ── 8. simulation comparison visualization ───────────────────────────────

def create_simulation_comparison_visualization(
    output_path: str,
    indexed_orientations: np.ndarray,
    simulated_spots: np.ndarray,
    simulated_image: np.ndarray,
    filtered_exp_image: np.ndarray,
) -> None:
    """
    Create a 3-panel Plotly HTML comparing experimental and simulated spots.

    Panels: Experimental Spots | Simulated Spots | Overlay & Missing.

    Args:
        output_path:          Base file path (e.g. ``results/image_001``).
        indexed_orientations: Filtered indexed orientation array.
        simulated_spots:      Simulated spot array from GenerateSimulation.py.
        simulated_image:      Simulated diffraction image array.
        filtered_exp_image:   Filtered thresholded experimental image.
    """
    if not HAS_PLOTLY:
        logger.warning("plotly not available; skipping simulation comparison")
        return

    logger.info("Creating simulation comparison visualization")

    # Load experimental spots from the associated .bin.spots.txt file
    spots_file_path = f"{output_path}.bin.spots.txt"
    try:
        all_exp_spots = np.genfromtxt(spots_file_path, skip_header=1)
        if all_exp_spots.ndim == 1 and all_exp_spots.size > 0:
            all_exp_spots = np.expand_dims(all_exp_spots, axis=0)
        elif all_exp_spots.size == 0:
            all_exp_spots = np.empty((0, 8))

        kept_grain_nrs = (
            set(indexed_orientations[:, 0].astype(int))
            if indexed_orientations.size > 0 else set()
        )
        if all_exp_spots.size > 0:
            exp_spots = all_exp_spots[
                np.isin(all_exp_spots[:, 0].astype(int), list(kept_grain_nrs))
            ]
        else:
            exp_spots = all_exp_spots
    except Exception as e:
        logger.error(f"Error loading exp spots from '{spots_file_path}': {e}")
        return

    if indexed_orientations.size == 0:
        logger.warning("No indexed orientations; skipping simulation comparison.")
        return

    # --- Plotly figure ---
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Experimental Spots", "Simulated Spots", "Overlay & Missing"],
        horizontal_spacing=0.05,
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        shared_xaxes=True, shared_yaxes=True,
    )

    display_exp = filtered_exp_image.copy().astype(float)
    display_exp[display_exp <= 0] = 1
    log_exp = np.log(display_exp)

    fig.add_trace(go.Heatmap(z=log_exp, colorscale="Greens", showscale=False, name="Experimental"), row=1, col=1)
    if simulated_image.size > 0:
        fig.add_trace(go.Heatmap(z=simulated_image, colorscale="Reds", showscale=False, name="Simulated", opacity=0.7), row=1, col=2)
    fig.add_trace(go.Heatmap(z=log_exp, colorscale="gray", showscale=False, name="Exp Overlay", opacity=0.3), row=1, col=3)

    palette = px.colors.qualitative.Plotly
    if indexed_orientations.ndim == 1:
        indexed_orientations = np.expand_dims(indexed_orientations, axis=0)

    missing_spots_dict: Dict[int, List] = {}
    matched_exp_positions: Dict[int, set] = {}

    for i, ori in enumerate(indexed_orientations):
        gn = int(ori[0])
        color = palette[i % len(palette)]

        # Experimental spots for this grain
        grain_exp = exp_spots[exp_spots[:, 0] == gn]
        if grain_exp.size > 0:
            exp_trace = go.Scatter(
                x=grain_exp[:, 5], y=grain_exp[:, 6],
                mode="markers", name=f"Exp Grain {gn}", legendgroup=f"grain_{gn}",
                marker=dict(color=color, size=8, symbol="circle-open", line=dict(width=2, color=color)),
                hovertext=[
                    f"Grain: {gn}<br>HKL: ({int(s[2])},{int(s[3])},{int(s[4])})<br>"
                    f"Pos: ({s[5]:.1f}, {s[6]:.1f})<br>Source: Exp"
                    for s in grain_exp
                ],
                hoverinfo="text",
            )
            fig.add_trace(exp_trace, row=1, col=1)
            fig.add_trace(go.Scatter(exp_trace), row=1, col=3)
            matched_exp_positions[gn] = {
                (round(float(s[5])), round(float(s[6]))) for s in grain_exp
            }

        # Simulated spots for this grain
        use_gn = (
            simulated_spots.size > 0
            and np.max(simulated_spots[:, 2]) >= len(indexed_orientations)
        )
        grain_sim = (
            simulated_spots[simulated_spots[:, 2] == gn]
            if use_gn
            else simulated_spots[simulated_spots[:, 2] == i]
        ) if simulated_spots.size > 0 else np.empty((0,))

        # Deduplicate simulated spots
        unique_sim = []
        seen = set()
        if grain_sim.size > 0:
            for spot in grain_sim:
                key = (round(float(spot[1])), round(float(spot[0])))
                if key not in seen:
                    unique_sim.append(spot)
                    seen.add(key)

        if unique_sim:
            usa = np.array(unique_sim)
            sim_trace = go.Scatter(
                x=usa[:, 1], y=usa[:, 0],
                mode="markers", name=f"Sim Grain {gn}", legendgroup=f"grain_{gn}",
                marker=dict(color=color, size=8, symbol="x", line=dict(width=2, color=color)),
                hovertext=[
                    f"Grain: {gn}<br>Pos: ({s[1]:.1f}, {s[0]:.1f})<br>Source: Sim"
                    for s in usa
                ],
                hoverinfo="text",
            )
            fig.add_trace(sim_trace, row=1, col=2)
            fig.add_trace(go.Scatter(sim_trace), row=1, col=3)

            # Find missing simulated spots
            missing_spots_dict[gn] = []
            threshold_sq = 4.0  # 2.0 ** 2
            exp_pos = matched_exp_positions.get(gn, set())
            for spot in usa:
                sim_pos = (round(float(spot[1])), round(float(spot[0])))
                if not any(
                    (sim_pos[0] - ep[0]) ** 2 + (sim_pos[1] - ep[1]) ** 2 < threshold_sq
                    for ep in exp_pos
                ):
                    missing_spots_dict[gn].append(spot)

    # Missing spots traces
    for gn, missing in missing_spots_dict.items():
        if not missing:
            continue
        ma = np.array(missing)
        try:
            gidx = list(indexed_orientations[:, 0].astype(int)).index(gn)
            color = palette[gidx % len(palette)]
        except ValueError:
            color = "grey"
        fig.add_trace(go.Scatter(
            x=ma[:, 1], y=ma[:, 0],
            mode="markers", name=f"Missing Sim {gn}", legendgroup=f"grain_{gn}",
            marker=dict(color=color, size=10, symbol="diamond-open", line=dict(width=2, color="black")),
            hovertext=[f"Grain: {gn}<br>Pos: ({s[1]:.1f}, {s[0]:.1f})<br>Source: Missing" for s in ma],
            hoverinfo="text", showlegend=False,
        ), row=1, col=3)
        logger.info(f"Grain {gn}: {len(missing)} simulated spots missing experimentally.")

    # Layout
    fig.update_layout(
        title="Experimental vs. Simulated Diffraction Comparison",
        height=700, width=1800,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                    bordercolor="Black", borderwidth=1),
        margin=dict(l=50, r=200, b=50, t=50),
        hovermode="closest",
    )
    y_max, x_max = filtered_exp_image.shape[:2]
    for col in range(1, 4):
        fig.update_xaxes(title_text="X Position (pixels)", row=1, col=col, range=[0, x_max], constrain="domain")
        fig.update_yaxes(title_text="Y Position (pixels)", row=1, col=col, range=[y_max, 0], constrain="domain")

    html_file = f"{output_path}.simulation_comparison.html"
    try:
        fig.write_html(html_file, include_plotlyjs="cdn")
        logger.info(f"Simulation comparison saved to {html_file}")
    except Exception as e:
        logger.error(f"Could not save simulation comparison: {e}")

    # Save unique spot counts
    unique_exp_counts: Dict[int, int] = {}
    if exp_spots.size > 0:
        for gn_val in kept_grain_nrs:
            gs = exp_spots[exp_spots[:, 0] == gn_val]
            unique_exp_counts[gn_val] = len({(int(s[5]), int(s[6])) for s in gs}) if gs.size > 0 else 0

    counts_file = f"{output_path}.unique_spot_counts.txt"
    try:
        with open(counts_file, "w") as f:
            f.write("Grain_Nr\tUnique_Experimental_Spots\n")
            for gn_val, cnt in sorted(unique_exp_counts.items()):
                f.write(f"{gn_val}\t{cnt}\n")
        logger.info(f"Unique experimental spot counts saved to {counts_file}")
    except Exception as e:
        logger.error(f"Could not save unique spot counts: {e}")
