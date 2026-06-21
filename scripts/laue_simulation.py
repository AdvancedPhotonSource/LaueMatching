"""Diffraction simulation step (shells out to GenerateSimulation.py).

Lifted out of RunImage.EnhancedImageProcessor so the orchestrator stays thin
(REFACTOR_PLAN §6.5).  Behaviour-preserving move of the former
``_run_simulation`` method body; ``config`` is the ConfigurationManager.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Any, Dict, List

import numpy as np

import laue_visualization as lv
from laue_index.records import SOLUTION_FORMATS

logger = logging.getLogger("LaueMatching")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON = sys.executable
_RI = SOLUTION_FORMATS["runimage"]

__all__ = ["run_simulation"]


def run_simulation(config, output_path: str, orientations: np.ndarray,
                   centers: List, filtered_thresholded_image: np.ndarray
                   ) -> Dict[str, Any]:
    """Run GenerateSimulation.py for the indexed orientations and load results."""
    logger.info("Running diffraction simulation (GenerateSimulation.py)")
    sim_config = config.get("simulation")
    if not sim_config:
        logger.error("Simulation configuration missing. Cannot run simulation.")
        return {"success": False, "error": "Simulation configuration missing"}
    if orientations.size == 0:
        logger.info("No orientations provided for simulation. Skipping.")
        return {"success": True, "message": "No orientations for simulation"}

    import h5py
    try:
        main_config_file = config.config_file
        sim_orient_input_file = f"{output_path}.indexed_orientations_for_sim.txt"
        with open(sim_orient_input_file, "w") as f:
            if len(orientations.shape) == 1:
                orientations = np.expand_dims(orientations, axis=0)
            for orient in orientations:
                matrix_elements = orient[_RI.om_start:_RI.om_start + 9]
                f.write(" ".join(map(str, matrix_elements)) + "\n")
        logger.debug(f"Created temporary orientation file for simulation: {sim_orient_input_file}")

        sim_output_h5 = f"{output_path}.simulation.h5"
        skip_percentage = sim_config.skip_percentage

        sim_script_path = os.path.join(_SCRIPT_DIR, "GenerateSimulation.py")
        if not os.path.exists(sim_script_path):
            logger.error(f"Simulation script not found: {sim_script_path}")
            return {"success": False, "error": "GenerateSimulation.py not found"}

        sim_cmd = [
            _PYTHON, sim_script_path,
            "-configFile", main_config_file,
            "-orientationFile", sim_orient_input_file,
            "-outputFile", sim_output_h5,
            "-skipPercentage", str(skip_percentage),
        ]
        logger.info(f"Running simulation command: {' '.join(sim_cmd)}")
        sim_stdout_log = f"{output_path}.simulation_stdout.txt"
        sim_stderr_log = f"{output_path}.simulation_stderr.txt"
        process = subprocess.run(sim_cmd, capture_output=True, text=True, check=False)
        with open(sim_stdout_log, "w") as f_out:
            f_out.write(process.stdout)
        with open(sim_stderr_log, "w") as f_err:
            f_err.write(process.stderr)

        if process.returncode != 0:
            logger.error(f"Simulation command failed with exit code {process.returncode}.")
            logger.error(f"Check logs: {sim_stdout_log} and {sim_stderr_log}")
            logger.error(f"Stderr tail:\n{process.stderr[-500:]}")
            return {"success": False, "error": f"Simulation command failed with code {process.returncode}"}

        logger.info(f"Simulation command completed successfully. Output: {sim_output_h5}")

        simulation_results: Dict[str, Any] = {}
        if os.path.exists(sim_output_h5):
            try:
                with h5py.File(sim_output_h5, "r") as h5f:
                    entry_group = "entry1" if "entry1" in h5f else "entry"
                    if f"/{entry_group}/spots" in h5f:
                        simulation_results["simulated_spots"] = np.array(h5f[f"/{entry_group}/spots"][()])
                    if f"/{entry_group}/data/data" in h5f:
                        simulation_results["simulated_images"] = {
                            "simulated_image": np.array(h5f[f"/{entry_group}/data/data"][()])
                        }
                    if f"/{entry_group}/recips" in h5f:
                        simulation_results["recips"] = np.array(h5f[f"/{entry_group}/recips"][()])
                logger.info(f"Loaded simulation results from {sim_output_h5}")

                vis_config = config.get("visualization")
                if vis_config and vis_config.enable_visualization:
                    lv.create_simulation_comparison_visualization(
                        output_path, orientations,
                        simulation_results.get("simulated_spots", np.array([])),
                        simulation_results.get("simulated_images", {}).get("simulated_image", np.array([])),
                        filtered_thresholded_image,
                    )
                simulation_results["success"] = True
            except Exception as e:
                logger.error(f"Error loading simulation results from HDF5 '{sim_output_h5}': {str(e)}")
                simulation_results = {"success": False, "error": f"Failed to load simulation HDF5 results: {str(e)}"}
        else:
            logger.error(f"Simulation output file not found: {sim_output_h5}")
            simulation_results = {"success": False, "error": "Simulation output HDF5 file not found"}

        return simulation_results
    except Exception as e:
        logger.error(f"An unexpected error occurred during simulation setup or execution: {str(e)}")
        return {"success": False, "error": f"Unexpected error during simulation: {e}"}
