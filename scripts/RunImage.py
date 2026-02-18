#!/usr/bin/env python

"""
LaueMatching - Advanced Laue Diffraction Pattern Indexing Tool
Author: Hemant Sharma (hsharma@anl.gov)

This script processes Laue diffraction images to identify crystal orientations.
It supports both CPU and GPU processing modes with enhanced functionality.
"""

import os
import sys
import time
import shutil
import subprocess
import argparse
import glob
import logging
import json
import yaml
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import scipy.ndimage as ndimg
import diplib as dip
import skimage.segmentation
from skimage import exposure, filters, restoration # Added restoration for denoise
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure matplotlib
plt.rcParams['font.size'] = 3

# Get installation path — repo root is one level above scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTALL_PATH = os.path.dirname(SCRIPT_DIR)  # repo root (contains bin/, LIBS/, etc.)
PYTHON_PATH = sys.executable

# Shared streaming utilities (also used by laue_image_server / laue_postprocess)
import laue_stream_utils as lsu
import laue_visualization as lv

# Configuration system (extracted to laue_config.py)
from laue_config import (
    LogLevel, setup_logger,
    ImageProcessingConfig, VisualizationConfig, SimulationConfig, LaueConfig,
    ConfigurationManager, ProgressReporter,
)

# Create global logger
logger = setup_logger()


# --- Image Processor ---

class EnhancedImageProcessor:
    """Enhanced image processing for Laue diffraction patterns."""

    def __init__(self, config: ConfigurationManager):
        """
        Initialize the image processor with configuration parameters.

        Args:
            config: Configuration manager containing processing parameters
        """
        self.config = config
        self.background = None # This will store the computed median background

        # Configure output image settings
        vis_config = config.get("visualization")
        self.outdpi = vis_config.output_dpi if vis_config else 600
        nr_px_x = config.get("nr_px_x", 2048)
        nr_px_y = config.get("nr_px_y", 2048)
        self.scalarX = nr_px_x / self.outdpi if self.outdpi > 0 else 10
        self.scalarY = nr_px_y / self.outdpi if self.outdpi > 0 else 10
        plt.rcParams['figure.figsize'] = [self.scalarX, self.scalarY]

        # Initialize background if available
        self._load_background()

    def _load_background(self) -> None:
        """Load background image if it exists, otherwise initialize empty array."""
        self.background = lsu.load_background(
            self.config.get("background_file", ""),
            self.config.get("nr_px_x", 2048),
            self.config.get("nr_px_y", 2048),
        )

    def compute_background(self, image: np.ndarray) -> np.ndarray:
        """
        Compute background by applying median filter to the image.

        Args:
            image: Input image array (numpy format, assume Y,X)

        Returns:
            Background image array (numpy format, Y,X)
        """
        logger.info("Computing background...")
        img_config = self.config.get("image_processing")
        if not img_config:
             logger.error("Image processing configuration not found. Cannot compute background.")
             return np.zeros_like(image)

        self.background = lsu.compute_background(
            image,
            filter_radius=img_config.filter_radius,
            median_passes=img_config.median_passes,
        )

        # Save the background for future use
        background_file_path = self.config.get("background_file")
        if background_file_path:
            try:
                self.background.tofile(background_file_path)
                logger.info(f"Computed background saved to {background_file_path}")
            except Exception as e:
                logger.error(f"Error saving computed background to {background_file_path}: {e}")

        return self.background

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply optional image enhancements based on configuration."""
        img_config = self.config.get("image_processing")
        if not img_config:
            return image
        return lsu.enhance_image(
            image,
            denoise=getattr(img_config, 'denoise_image', False),
            denoise_strength=getattr(img_config, 'denoise_strength', 1.0),
            contrast=getattr(img_config, 'enhance_contrast', False),
            edge=getattr(img_config, 'edge_enhancement', False),
        )

    def apply_threshold(self, image: np.ndarray, override_thresh: int = 0) -> Tuple[np.ndarray, float]:
        """Apply thresholding based on configuration or override value."""
        img_config = self.config.get("image_processing")
        if not img_config:
            logger.error("Image processing configuration not found. Cannot apply threshold.")
            return image.copy(), 0.0

        if override_thresh > 0:
            return lsu.apply_threshold(image, method="fixed", fixed_value=float(override_thresh))

        return lsu.apply_threshold(
            image,
            method=img_config.threshold_method,
            fixed_value=img_config.threshold_value,
            percentile=img_config.threshold_percentile,
        )


    def correct_image(self, image: np.ndarray, override_thresh: int = 0, background_already_subtracted: bool = False) -> Tuple[np.ndarray, float]:
        """
        Apply background correction (if not already done), enhancement,
        and thresholding to input image.

        Args:
            image: Input image array (can be raw or already background subtracted)
            override_thresh: Optional threshold override value
            background_already_subtracted: Flag indicating if background was subtracted before calling

        Returns:
            Tuple of (corrected image after enhancement and thresholding, threshold used)
        """
        # Check if we need to compute background (only if not already subtracted and background is zero/not loaded)
        if not background_already_subtracted and np.count_nonzero(self.background) == 0:
             logger.info("Background not available, computing from current (raw) image.")
             # This assumes the input 'image' is the raw image in this case
             self.compute_background(image)

        # Apply background correction if not already done AND background is available
        if not background_already_subtracted and np.count_nonzero(self.background) > 0:
            logger.info("Applying background correction (subtracting median)")
            # Ensure types are suitable for subtraction (e.g., float or double)
            corrected = image.astype(np.double) - self.background.astype(np.double)
            # Optional: Prevent negative values after subtraction
            corrected = np.maximum(corrected, 0)
        elif background_already_subtracted:
            logger.info("Background subtraction already performed, skipping.")
            corrected = image.astype(np.double) # Ensure input is double/float
        else: # Background not subtracted and not available/computable
             logger.warning("Background subtraction skipped (background not available).")
             corrected = image.astype(np.double) # Ensure input is double/float


        # Apply enhancements to the background-corrected image
        enhanced = self.enhance_image(corrected) # Returns float

        # Apply thresholding to the enhanced, background-corrected image
        thresholded, threshold_value_used = self.apply_threshold(enhanced, override_thresh) # Returns float

        # Return the image after enhancement and thresholding, cast to uint16 for compatibility
        return thresholded.astype(np.uint16), threshold_value_used

    @staticmethod
    def find_connected_components(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Find connected components in the thresholded image."""
        return lsu.find_connected_components(image)


    def filter_small_components(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bounding_boxes: np.ndarray,
        areas: np.ndarray,
        nlabels: int
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """Filter out components smaller than the minimum area and calculate centers of mass."""
        img_config = self.config.get("image_processing")
        min_area = img_config.min_area if img_config else 10
        return lsu.filter_small_components(
            image, labels, bounding_boxes, areas, nlabels, min_area=min_area
        )

    @staticmethod
    def calculate_gaussian_width(centers: List, pixel_size: float, distance: float, orient_spacing: float) -> float:
        """Calculate optimal Gaussian blur sigma based on spot spacing."""
        return lsu.calculate_gaussian_sigma(centers, pixel_size, distance, orient_spacing)


    def process_image(self, image_path: str, override_thresh: int = 0) -> Dict[str, Any]:
        """
        Process a single image file.

        Args:
            image_path: Path to the input image file
            override_thresh: Optional threshold override value

        Returns:
            Dictionary of processing results
        """
        logger.info(f"Processing image: {image_path}")
        start_time = time.time()

        # Create result directory if it doesn't exist
        result_dir = self.config.get("result_dir", "results")
        os.makedirs(result_dir, exist_ok=True)

        # Initialize progress reporter (9 steps)
        progress = ProgressReporter(9, f"Processing {os.path.basename(image_path)}")

        # --- Step 1: Load the image ---
        try:
            with h5py.File(image_path, 'r') as h_file:
                 # Assuming data is typically under /entry/data/data or similar NeXus paths
                 data_path = '/entry/data/data' # Common path
                 if data_path not in h_file:
                      # Try alternative common paths
                      possible_paths = ['/entry1/data/data', '/entry/data/raw_data', '/data']
                      for path in possible_paths:
                          if path in h_file:
                              data_path = path
                              break
                      else:
                          raise KeyError(f"Could not find image data in standard HDF5 paths for {image_path}")
                 # Load data and ensure correct shape (Y, X)
                 image_data = np.array(h_file[data_path][()])
                 # Ensure it's 2D
                 if image_data.ndim > 2:
                      logger.warning(f"Image data has {image_data.ndim} dimensions. Taking first slice.")
                      image_data = image_data[0, :, :]
                 elif image_data.ndim < 2:
                     raise ValueError(f"Image data has invalid dimensions: {image_data.ndim}")

                 # Ensure Y, X convention if possible (check attributes or assume common format)
                 # Let's assume loaded shape is correct for now (Y, X)
                 raw_image = np.copy(image_data) # Keep a pristine copy

        except Exception as e:
            logger.error(f"Error loading image file '{image_path}': {str(e)}")
            progress.complete("Failed at loading")
            return {"success": False, "error": f"Error loading image: {str(e)}"}

        progress.update(1, "Image loaded")

        # --- Step 2: Calculate/Load Background and Perform Subtraction ---
        # Ensure background is ready (load or compute if needed)
        if np.count_nonzero(self.background) == 0 and not os.path.exists(self.config.get("background_file", "")):
             logger.info("Background not loaded and file doesn't exist, computing from current raw image.")
             self.compute_background(raw_image) # Compute from raw image if needed
        elif np.count_nonzero(self.background) == 0 and os.path.exists(self.config.get("background_file", "")):
             self._load_background() # Try loading again if it was zero initially

        # Perform subtraction if background is available
        if np.count_nonzero(self.background) > 0:
             logger.info("Calculating background-subtracted image (raw - median).")
             background_subtracted = raw_image.astype(np.double) - self.background.astype(np.double)
             background_subtracted = np.maximum(background_subtracted, 0) # Ensure non-negative
        else:
             logger.warning("Background (median) image is all zeros or unavailable. Proceeding without subtraction.")
             background_subtracted = raw_image.astype(np.double) # Use raw image if no background

        progress.update(1, "Background processed")

        # --- Step 3: Correct the image (Enhance + Threshold the background-subtracted data) ---
        # Pass the already subtracted image to correct_image
        corrected_image_post_subtraction, threshold_value_used = self.correct_image(
            background_subtracted,
            override_thresh=override_thresh,
            background_already_subtracted=True # Signal that subtraction is done
        )
        # 'corrected_image_post_subtraction' holds the image after enhancement and thresholding
        thresholded_image = corrected_image_post_subtraction.astype(np.uint16) # Ensure correct type

        progress.update(1, "Image corrected & thresholded")

        # --- Prepare output paths ---
        base_output_name = os.path.splitext(os.path.basename(image_path))[0] # Get base name without ext
        output_path = os.path.join(result_dir, base_output_name) # Use base name for output files
        output_h5 = f"{output_path}.output.h5" # Changed from .bin.output.h5

        # --- Step 4: Process connected components on the thresholded image ---
        with h5py.File(output_h5, 'w') as hf_out: # Start with 'w' for a clean file
            # --- Store initial data in HDF5 ---
            data_group = hf_out.require_group('/entry/data')
            data_group.create_dataset('raw_data', data=raw_image)
            data_group.create_dataset('background_median', data=self.background)
            data_group.create_dataset('background_subtracted', data=background_subtracted)
            data_group.create_dataset('cleaned_data_threshold', data=thresholded_image, dtype=np.uint16)
            data_group['cleaned_data_threshold'].attrs['threshold_value'] = threshold_value_used
            logger.debug(f"Initial data saved to {output_h5}")

            # --- Continue processing ---
            labels, bboxes, areas, nlabels = self.find_connected_components(thresholded_image)
            # Handle case where no components are found (except background)
            if nlabels <= 1:
                 logger.warning("No connected components found after thresholding. Stopping processing for this image.")
                 progress.complete("No components found")
                 # Add empty datasets for consistency?
                 data_group.create_dataset('cleaned_data_threshold_labels_unfiltered', data=labels)
                 data_group.create_dataset('cleaned_data_threshold_filtered', data=thresholded_image) # No filtering done
                 data_group.create_dataset('cleaned_data_threshold_filtered_labels', data=labels) # No filtering done
                 data_group.create_dataset('component_centers', data=np.empty((0, 4)))
                 data_group.create_dataset('input_blurred', data=thresholded_image.astype(np.double))
                 return {"success": True, "message": "No components found", "output_h5": output_h5}

            data_group.create_dataset('cleaned_data_threshold_labels_unfiltered', data=labels, dtype=np.int32)
            progress.update(1, "Connected components found")

            # --- Step 5: Filter small components ---
            filtered_thresholded_image, filtered_labels, centers = self.filter_small_components(
                thresholded_image, labels, bboxes, areas, nlabels
            )
            # 'filtered_thresholded_image' is the thresholded image with small components removed

            data_group.create_dataset('cleaned_data_threshold_filtered', data=filtered_thresholded_image, dtype=np.uint16)
            data_group.create_dataset('cleaned_data_threshold_filtered_labels', data=filtered_labels, dtype=np.int32)
            # Store component centers properly
            if centers:
                 centers_array = np.array([[float(c[0]), float(c[1][0]), float(c[1][1]), float(c[2])] for c in centers], dtype=np.float64)
            else:
                 centers_array = np.empty((0, 4), dtype=np.float64)
            data_group.create_dataset('component_centers', data=centers_array)
            progress.update(1, "Small components filtered")

            # Check if any components remain after filtering
            if not centers:
                 logger.warning("No components remained after filtering small areas. Stopping processing.")
                 progress.complete("No components left after filtering")
                 # Save blurred image as just the filtered_thresholded (which is likely all zero)
                 data_group.create_dataset('input_blurred', data=filtered_thresholded_image.astype(np.double))
                 return {"success": True, "message": "No components left after filtering", "output_h5": output_h5}


            # --- Step 6: Calculate Gaussian blur width and apply blur ---
            gauss_sigma = self.calculate_gaussian_width(
                centers,
                self.config.get("px_x", 0.2),
                self.config.get("distance", 0.513),
                self.config.get("orientation_spacing", 0.4)
            )
            # Apply Gaussian blur to the *filtered thresholded* image intensities
            # Use float type for blurring
            blurred_image = ndimg.gaussian_filter(filtered_thresholded_image.astype(np.double), gauss_sigma)
            data_group.create_dataset('input_blurred', data=blurred_image)

            # Save blurred image for indexing executable (using .bin extension as before)
            indexing_input_file = f"{output_path}.bin"
            try:
                 blurred_image.astype(np.double).tofile(indexing_input_file)
                 logger.info(f"Image for indexing saved to {indexing_input_file}")
            except Exception as e:
                 logger.error(f"Error saving blurred image for indexing: {e}")
                 progress.complete("Failed saving indexing input")
                 return {"success": False, "error": f"Failed to save indexing input: {e}"}

            progress.update(1, "Image blurred")

            # --- Step 7: Perform watershed segmentation if enabled ---
            watershed_mask = filtered_thresholded_image > 0 # Mask based on filtered thresholded image
            if self.config.get("image_processing").watershed_enabled:
                if np.any(watershed_mask): # Check if mask is not empty
                     logger.info("Performing watershed segmentation...")
                     # Use negative blurred image as the landscape
                     # Use filtered labels as markers
                     try:
                          watershed_labels = skimage.segmentation.watershed(
                              -blurred_image,         # Energy landscape (inverted intensity)
                              markers=filtered_labels,  # Seed markers from connected components
                              mask=watershed_mask,      # Apply watershed only within the mask
                              connectivity=2            # 8-connectivity
                          )
                          final_labels = watershed_labels
                          max_labels = np.max(watershed_labels)
                          logger.info(f'Watershed segmentation found {max_labels} regions')
                          data_group.create_dataset('watershed_labels', data=watershed_labels, dtype=np.int32)
                     except Exception as e:
                          logger.error(f"Watershed segmentation failed: {e}. Using filtered labels instead.")
                          final_labels = filtered_labels
                          max_labels = np.max(final_labels) if final_labels.size > 0 else 0
                else:
                    logger.warning("Image is empty after filtering small components. Skipping watershed.")
                    final_labels = filtered_labels # Fallback to filtered labels
                    max_labels = np.max(final_labels) if final_labels.size > 0 else 0
            else:
                final_labels = filtered_labels
                max_labels = np.max(final_labels) if final_labels.size > 0 else 0
                logger.info('Watershed segmentation was disabled')

            progress.update(1, "Segmentation completed")

            # --- Step 8: Run indexing and process results ---
            # Pass the *filtered thresholded* image for visualization context if needed
            indexing_results = self._run_indexing(
                output_path, final_labels, max_labels, blurred_image, centers, filtered_thresholded_image
            )

            if "success" in indexing_results and not indexing_results["success"]:
                progress.complete("Failed at indexing stage")
                # Store text files even on failure if they exist
                self._store_txt_files_in_h5(output_path, hf_out)
                return indexing_results
            else:
                 # Indexing succeeded (or files were missing but didn't cause error return)
                 # Now process the results, including filtering based on unique spots
                 processed_indexing_results = self._process_indexing_results(
                     output_path, final_labels, max_labels, filtered_thresholded_image # Pass filtered_thresholded
                 )
                 # Check if processing the results failed
                 if not processed_indexing_results["success"]:
                      progress.complete("Failed processing indexing results")
                      self._store_txt_files_in_h5(output_path, hf_out)
                      return processed_indexing_results


            # --- Step 9: Run simulation if enabled ---
            sim_group = hf_out.require_group('/entry/simulation') # Ensure group exists
            # Use the *filtered* orientations from processed_indexing_results
            orientations_for_simulation = processed_indexing_results.get("orientations", np.array([]))

            if self.config.get("simulation").enable_simulation and orientations_for_simulation.size > 0:
                simulation_results = self._run_simulation(
                    output_path, orientations_for_simulation, centers, filtered_thresholded_image # Pass filtered_thresholded
                )

                if "success" in simulation_results and simulation_results["success"]:
                    # Store simulation results in H5 file
                    if "simulated_spots" in simulation_results:
                        if 'simulated_spots' in sim_group: del sim_group['simulated_spots']
                        sim_group.create_dataset('simulated_spots', data=simulation_results["simulated_spots"])
                    if "simulated_images" in simulation_results:
                        for img_name, img_data in simulation_results["simulated_images"].items():
                            if img_name in sim_group: del sim_group[img_name]
                            sim_group.create_dataset(img_name, data=img_data)
                    if "recips" in simulation_results:
                        if 'recips' in sim_group: del sim_group['recips']
                        sim_group.create_dataset('recips', data=simulation_results["recips"])

                progress.update(1, "Simulation completed")
            else:
                if not self.config.get("simulation").enable_simulation:
                     logger.info("Simulation disabled.")
                elif orientations_for_simulation.size == 0:
                     logger.info("Skipping simulation as no orientations remained after filtering.")
                progress.update(1, "Simulation skipped")

            # --- Final HDF5 Tasks ---
            # Store text file logs/outputs
            self._store_txt_files_in_h5(output_path, hf_out)
            # Ensure binary data headers/columns are stored correctly (called within _create_h5_output and here for simulation)
            self._store_binary_headers_in_h5(output_path, hf_out)

            progress.complete("Processing completed")

            # Return processing results (based on filtered orientations/spots)
            return {
                "success": True,
                "image_path": image_path,
                "output_path": output_path,
                "output_h5": output_h5,
                "centers": centers,
                "final_labels": final_labels,
                "indexing_results": processed_indexing_results, # Contains filtered results primarily
                "processing_time": time.time() - start_time
            } # End of `with h5py.File(...)`


    def _store_txt_files_in_h5(self, output_path: str, h5_file) -> None:
        """Delegate to lsu.store_txt_files_in_h5."""
        lsu.store_txt_files_in_h5(output_path, h5_file)

    def _store_binary_headers_in_h5(self, output_path: str, h5_file) -> None:
        """Delegate to lsu.store_binary_headers_in_h5."""
        lsu.store_binary_headers_in_h5(output_path, h5_file)


    def _run_indexing(
        self,
        output_path: str, # Base path for output files, e.g., results/image_001
        labels: np.ndarray,
        nlabels: int,
        blurred_image: np.ndarray, # The input to the executable
        centers: List,
        filtered_thresholded_image: np.ndarray # Used for result processing context
    ) -> Dict[str, Any]:
        """
        Run the Laue indexing executable.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            labels: Final labels array (e.g., from watershed)
            nlabels: Number of labels in final_labels
            blurred_image: Blurred image input to the executable
            centers: List of center points from filtered components
            filtered_thresholded_image: Filtered thresholded image for context

        Returns:
            Dictionary with status { "success": bool, "error": str (optional) }
            Note: Result *processing* happens in _process_indexing_results
        """
        compute_type = self.config.get("processing_type", "CPU").upper()
        ncpus = self.config.get("num_cpus", 1)

        # Find executable path relative to repo root (one level above scripts/)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        repo_root = os.path.dirname(script_dir)
        build_dir = os.path.join(repo_root, 'bin')

        # Choose the appropriate executable
        do_forward = self.config.get("do_forward", False)
        if compute_type == 'GPU' and not do_forward:
             executable_name = 'LaueMatchingGPU'
        else:
             executable_name = 'LaueMatchingCPU'
             if compute_type == 'GPU' and do_forward:
                 logger.warning("GPU requested but DoFwd is enabled. Using CPU implementation (LaueMatchingCPU).")
             elif compute_type != 'CPU':
                 logger.warning(f"Processing type '{compute_type}' not recognized or incompatible. Using CPU implementation.")

        executable_path = os.path.join(build_dir, executable_name)

        # Check if executable exists
        if not os.path.exists(executable_path):
             logger.error(f"Indexing executable not found at: {executable_path}")
             logger.error("Please ensure the code is compiled (e.g., run 'make' in the build directory).")
             return {"success": False, "error": "Indexing executable not found"}

        # --- Prepare required input files ---
        config_file = self.config.config_file
        orient_db_file = self.config.get("orientation_file", "orientations.bin")
        hkl_file = self.config.get("hkl_file", "hkls.bin")
        indexing_input_image = f"{output_path}.bin" # The blurred image saved to file

        # Ensure orientation database exists (copy default if needed)
        if not os.path.exists(orient_db_file):
            default_orient_db = os.path.join(INSTALL_PATH, '100MilOrients.bin') # Default name/location
            if os.path.exists(default_orient_db):
                 logger.info(f"Orientation database '{orient_db_file}' not found. Copying default from '{default_orient_db}'.")
                 try:
                      shutil.copy2(default_orient_db, orient_db_file)
                 except Exception as e:
                      logger.error(f"Failed to copy default orientation database: {e}")
                      return {"success": False, "error": "Orientation database missing and copy failed"}
            else:
                 logger.error(f"Orientation database '{orient_db_file}' not found, and default DB '{default_orient_db}' is also missing.")
                 return {"success": False, "error": "Orientation database missing"}

        # Generate HKL file if it doesn't exist
        if not os.path.exists(hkl_file):
            logger.info(f"HKL file '{hkl_file}' not found, generating...")
            hkl_gen_result = self._generate_hkl_file()
            if not hkl_gen_result["success"]:
                return {"success": False, "error": f"Failed to generate HKL file: {hkl_gen_result.get('error', 'Unknown')}"}

        # --- Set up environment for executable (LD_LIBRARY_PATH) ---
        env = dict(os.environ)
        lib_path_nlopt = os.path.join(repo_root, 'LIBS', 'NLOPT', 'lib')
        lib_path_nlopt64 = os.path.join(repo_root, 'LIBS', 'NLOPT', 'lib64')
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        # Prepend NLopt paths
        env['LD_LIBRARY_PATH'] = f"{lib_path_nlopt}:{lib_path_nlopt64}:{current_ld_path}"

        # --- Construct and Run the Indexing Command ---
        indexing_cmd = [
             executable_path,
             config_file,
             orient_db_file,
             hkl_file,
             indexing_input_image,
             str(ncpus)
        ]
        logger.info(f'Running indexing command: {" ".join(indexing_cmd)}')
        stdout_log = f'{output_path}.LaueMatching_stdout.txt'
        stderr_log = f'{output_path}.LaueMatching_stderr.txt'

        try:
             # Use subprocess.run for better control
             process = subprocess.run(
                 indexing_cmd,
                 env=env,
                 capture_output=True, # Capture stdout/stderr
                 text=True,           # Decode as text
                 check=False           # Don't raise exception on non-zero exit code immediately
             )

             # Save stdout/stderr regardless of exit code
             with open(stdout_log, 'w') as f_out:
                 f_out.write(process.stdout)
             with open(stderr_log, 'w') as f_err:
                 f_err.write(process.stderr)

             # Check return code
             if process.returncode == 0:
                 logger.info(f"Indexing command completed successfully (exit code 0). Output saved to {stdout_log}")
                 return {"success": True}
             else:
                 logger.error(f"Indexing command failed with exit code {process.returncode}.")
                 logger.error(f"Check logs for details: {stdout_log} and {stderr_log}")
                 logger.error(f"Stderr tail:\n{process.stderr[-500:]}") # Show last part of stderr
                 return {
                     "success": False,
                     "error": f"Indexing command failed with code {process.returncode}"
                 }

        except FileNotFoundError:
             logger.error(f"Executable not found at {executable_path} when trying to run.")
             return {"success": False, "error": "Indexing executable not found during execution"}
        except Exception as e:
             logger.error(f"An unexpected error occurred while running indexing: {str(e)}")
             # Save any partial output if possible
             if 'process' in locals() and hasattr(process, 'stdout'):
                 with open(stdout_log, 'a') as f_out: f_out.write(f"\n\nERROR DURING EXECUTION: {e}\n{process.stdout}")
             if 'process' in locals() and hasattr(process, 'stderr'):
                 with open(stderr_log, 'a') as f_err: f_err.write(f"\n\nERROR DURING EXECUTION: {e}\n{process.stderr}")
             return {"success": False, "error": f"Unexpected error during indexing execution: {e}"}


    def _run_simulation(
        self,
        output_path: str, # Base path e.g. results/image_001
        orientations: np.ndarray, # Filtered orientations
        centers: List,
        filtered_thresholded_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run diffraction simulation (GenerateSimulation.py) for the indexed orientations.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            orientations: Filtered orientation data array
            centers: List of center points
            filtered_thresholded_image: Filtered thresholded experimental image for context

        Returns:
            Dictionary with simulation results
        """
        logger.info("Running diffraction simulation (GenerateSimulation.py)")
        sim_config = self.config.get("simulation")
        if not sim_config:
             logger.error("Simulation configuration missing. Cannot run simulation.")
             return {"success": False, "error": "Simulation configuration missing"}

        if orientations.size == 0:
             logger.info("No orientations provided for simulation. Skipping.")
             return {"success": True, "message": "No orientations for simulation"}

        try:
            # --- Prepare inputs for GenerateSimulation.py ---
            # 1. Configuration file (use the main one)
            main_config_file = self.config.config_file

            # 2. Orientation file (create temporary text file)
            sim_orient_input_file = f"{output_path}.indexed_orientations_for_sim.txt"
            with open(sim_orient_input_file, 'w') as f:
                # Write header? Assume GenerateSimulation doesn't need one.
                if len(orientations.shape) == 1: # Handle single orientation case
                    orientations = np.expand_dims(orientations, axis=0)

                for orient in orientations:
                    # Extract orientation matrix (columns 22-30) and write space-separated
                    matrix_elements = orient[22:31]
                    f.write(" ".join(map(str, matrix_elements)) + "\n")
            logger.debug(f"Created temporary orientation file for simulation: {sim_orient_input_file}")

            # 3. Output file base name for simulation (HDF5)
            sim_output_h5 = f"{output_path}.simulation.h5" # Explicit H5 extension

            # 4. Skip percentage
            skip_percentage = sim_config.skip_percentage

            # --- Construct and Run the Simulation Command ---
            sim_script_path = os.path.join(SCRIPT_DIR, 'GenerateSimulation.py')
            if not os.path.exists(sim_script_path):
                 logger.error(f"Simulation script not found: {sim_script_path}")
                 return {"success": False, "error": "GenerateSimulation.py not found"}

            sim_cmd = [
                PYTHON_PATH, # Use the same python executable
                sim_script_path,
                "-configFile", main_config_file,
                "-orientationFile", sim_orient_input_file,
                "-outputFile", sim_output_h5,
                "-skipPercentage", str(skip_percentage)
            ]

            logger.info(f"Running simulation command: {' '.join(sim_cmd)}")
            sim_stdout_log = f'{output_path}.simulation_stdout.txt'
            sim_stderr_log = f'{output_path}.simulation_stderr.txt' # Capture stderr too

            process = subprocess.run(
                sim_cmd,
                capture_output=True,
                text=True,
                check=False # Check manually
            )

            # Save simulation stdout/stderr
            with open(sim_stdout_log, 'w') as f_out:
                f_out.write(process.stdout)
            with open(sim_stderr_log, 'w') as f_err:
                 f_err.write(process.stderr) # Save stderr

            # Check return code
            if process.returncode != 0:
                logger.error(f"Simulation command failed with exit code {process.returncode}.")
                logger.error(f"Check logs: {sim_stdout_log} and {sim_stderr_log}")
                logger.error(f"Stderr tail:\n{process.stderr[-500:]}")
                # Clean up temporary orientation file
                # if os.path.exists(sim_orient_input_file): os.remove(sim_orient_input_file)
                return {"success": False, "error": f"Simulation command failed with code {process.returncode}"}

            logger.info(f"Simulation command completed successfully. Output: {sim_output_h5}")

            # --- Load simulation results from HDF5 ---
            simulation_results = {}
            if os.path.exists(sim_output_h5):
                try:
                    with h5py.File(sim_output_h5, 'r') as h5f:
                        # Adjust paths based on GenerateSimulation.py output structure
                        # Assuming structure like /entry1/spots, /entry1/data/data, /entry1/recips
                        entry_group = 'entry1' # Or determine dynamically if possible
                        if entry_group not in h5f:
                             entry_group = 'entry' # Fallback
                        if f'/{entry_group}/spots' in h5f:
                            simulation_results["simulated_spots"] = np.array(h5f[f'/{entry_group}/spots'][()])
                        if f'/{entry_group}/data/data' in h5f:
                             simulation_results["simulated_images"] = {
                                 "simulated_image": np.array(h5f[f'/{entry_group}/data/data'][()])
                             }
                        if f'/{entry_group}/recips' in h5f:
                            simulation_results["recips"] = np.array(h5f[f'/{entry_group}/recips'][()])

                    logger.info(f"Loaded simulation results from {sim_output_h5}")

                    # Create comparison visualization if enabled
                    vis_config = self.config.get("visualization")
                    if vis_config and vis_config.enable_visualization:
                        self._create_simulation_comparison_visualization(
                            output_path,
                            orientations, # Pass filtered orientations
                            simulation_results.get("simulated_spots", np.array([])),
                            simulation_results.get("simulated_images", {}).get("simulated_image", np.array([])),
                            filtered_thresholded_image # Pass filtered experimental image
                        )

                    simulation_results["success"] = True

                except Exception as e:
                    logger.error(f"Error loading simulation results from HDF5 '{sim_output_h5}': {str(e)}")
                    simulation_results = {"success": False, "error": f"Failed to load simulation HDF5 results: {str(e)}"}
            else:
                 logger.error(f"Simulation output file not found: {sim_output_h5}")
                 simulation_results = {"success": False, "error": "Simulation output HDF5 file not found"}

            # Clean up temporary orientation file
            # if os.path.exists(sim_orient_input_file): os.remove(sim_orient_input_file)

            return simulation_results

        except Exception as e:
            logger.error(f"An unexpected error occurred during simulation setup or execution: {str(e)}")
            # Clean up temporary orientation file if it exists
            # if 'sim_orient_input_file' in locals() and os.path.exists(sim_orient_input_file):
            #     os.remove(sim_orient_input_file)
            return {"success": False, "error": f"Unexpected error during simulation: {e}"}


    def _create_simulation_comparison_visualization(
        self,
        output_path: str,
        indexed_orientations: np.ndarray,
        simulated_spots: np.ndarray,
        simulated_image: np.ndarray,
        filtered_exp_image: np.ndarray,
    ) -> None:
        """Delegate to lv.create_simulation_comparison_visualization."""
        lv.create_simulation_comparison_visualization(
            output_path, indexed_orientations, simulated_spots,
            simulated_image, filtered_exp_image,
        )


    def _generate_hkl_file(self) -> Dict[str, Any]:
        """
        Generate HKL file using the GenerateHKLs.py script.

        Returns:
            Dictionary with generation result { "success": bool, "error": str (optional) }
        """
        hkl_file = self.config.get("hkl_file", "hkls.bin")
        sg_num = self.config.get("space_group")
        sym = self.config.get("symmetry")
        lat_c = self.config.get("lattice_parameter")
        r_arr = self.config.get("r_array")
        p_arr = self.config.get("p_array")
        n_px_x = self.config.get("nr_px_x")
        n_px_y = self.config.get("nr_px_y")
        dx = self.config.get("px_x")
        dy = self.config.get("px_y")
        elo = self.config.get("elo")
        ehi = self.config.get("ehi")

        logger.info("Attempting to generate HKL file...")
        genhkl_script = os.path.join(SCRIPT_DIR, 'GenerateHKLs.py')
        if not os.path.exists(genhkl_script):
             logger.error(f"GenerateHKLs.py script not found at {genhkl_script}")
             return {"success": False, "error": "GenerateHKLs.py not found"}

        cmd = [
            PYTHON_PATH, genhkl_script,
            '-resultFileName', hkl_file,
            '-sgnum', str(sg_num),
            '-sym', sym,
            '-latticeParameter', lat_c,
            '-RArray', r_arr,
            '-PArray', p_arr,
            '-NumPxX', str(n_px_x),
            '-NumPxY', str(n_px_y),
            '-dx', str(dx),
            '-dy', str(dy),
            '-Elo', str(elo), # Add energy limits
            '-Ehi', str(ehi)
        ]

        logger.debug(f"GenerateHKLs command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False # Check manually
            )

            if result.returncode == 0:
                 # Check if file was actually created and is not empty
                 if os.path.exists(hkl_file) and os.path.getsize(hkl_file) > 0:
                     logger.info(f"HKL file generated successfully: {hkl_file}")
                     return {"success": True}
                 else:
                     logger.error(f"GenerateHKLs script ran but did not create a valid HKL file '{hkl_file}'.")
                     logger.error(f"Stdout:\n{result.stdout}")
                     logger.error(f"Stderr:\n{result.stderr}")
                     return {"success": False, "error": "HKL file generation failed (script ran but output invalid)"}
            else:
                logger.error(f"Error generating HKL file (exit code {result.returncode}).")
                logger.error(f"Stdout:\n{result.stdout}")
                logger.error(f"Stderr:\n{result.stderr}")
                return {"success": False, "error": f"HKL generation script failed with code {result.returncode}"}

        except Exception as e:
            logger.error(f"An unexpected error occurred while running GenerateHKLs.py: {str(e)}")
            return {"success": False, "error": f"Unexpected error during HKL generation: {e}"}


    def _process_indexing_results(
        self,
        output_path: str, # Base path, e.g., results/image_001
        final_labels: np.ndarray,
        max_labels: int,
        filtered_thresholded_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process indexing results: read, calculate unique spots, filter, save, visualize.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            final_labels: Final labels array (e.g., from watershed)
            max_labels: Number of unique labels in final_labels
            filtered_thresholded_image: Filtered thresholded image for context

        Returns:
            Dictionary with processed results (filtered data primary)
        """
        solutions_file = f'{output_path}.bin.solutions.txt'
        spots_file = f'{output_path}.bin.spots.txt'

        # --- Read solutions and spots files ---
        try:
            # Read orientations
            if not os.path.exists(solutions_file):
                 raise FileNotFoundError(f"Solutions file not found: {solutions_file}")
            with open(solutions_file, 'r') as f:
                 header_gr = f.readline().strip() # Read header
            orientations_unfiltered = np.genfromtxt(solutions_file, skip_header=1)
            if orientations_unfiltered.size == 0:
                 logger.warning(f"Solutions file '{solutions_file}' is empty or contains no valid data.")
                 orientations_unfiltered = np.empty((0, 31)) # Assume typical width if empty
            elif len(orientations_unfiltered.shape) == 1:
                 orientations_unfiltered = np.expand_dims(orientations_unfiltered, axis=0)

            # Read spots
            if not os.path.exists(spots_file):
                 raise FileNotFoundError(f"Spots file not found: {spots_file}")
            with open(spots_file, 'r') as f:
                 header_sp = f.readline().strip() # Read header
            spots_unfiltered = np.genfromtxt(spots_file, skip_header=1)
            if spots_unfiltered.size == 0:
                 logger.warning(f"Spots file '{spots_file}' is empty or contains no valid data.")
                 # Infer columns from header if possible, else assume default
                 num_spot_cols = len(header_sp.split()) if header_sp else 8
                 spots_unfiltered = np.empty((0, num_spot_cols))
            elif len(spots_unfiltered.shape) == 1:
                 spots_unfiltered = np.expand_dims(spots_unfiltered, axis=0)

            logger.info(f"Read indexing results: {len(orientations_unfiltered)} orientations, {len(spots_unfiltered)} spots (before filtering)")

        except FileNotFoundError as e:
             logger.error(f"Indexing result file not found: {e}. Cannot process results.")
             return {"success": False, "error": f"Indexing output file not found: {e}"}
        except Exception as e:
             logger.error(f"Error reading indexing result files: {str(e)}")
             return {"success": False, "error": f"Failed to read indexing results: {str(e)}"}

        # --- Calculate unique spots per orientation ---
        orientation_unique_spots = self._calculate_unique_spots_per_orientation(
             orientations_unfiltered, spots_unfiltered, final_labels
        )

        # --- Sort orientations by quality score ---
        orientations_sorted = self._sort_orientations_by_quality(
             orientations_unfiltered, orientation_unique_spots # Pass unique spots info if needed for sorting
        )

        # --- Filter Orientations based on Min Unique Spots ---
        min_unique_spots_required = self.config.get("min_good_spots", 2) # Use MinGoodSpots from config
        logger.info(f"Filtering orientations with less than {min_unique_spots_required} unique spots...")

        indices_to_keep = []
        kept_grain_nrs = set()
        filtered_out_count = 0

        if orientations_sorted.size > 0:
            if len(orientations_sorted.shape) == 1: # Handle single orientation case
                orientations_sorted = np.expand_dims(orientations_sorted, axis=0)

            for i, orientation in enumerate(orientations_sorted):
                grain_nr = int(orientation[0])
                unique_spot_data = orientation_unique_spots.get(grain_nr)

                if unique_spot_data:
                    # Use unique_label_count which counts distinct segmented regions
                    unique_spot_count = unique_spot_data["unique_label_count"]
                    if unique_spot_count >= min_unique_spots_required:
                        indices_to_keep.append(i)
                        kept_grain_nrs.add(grain_nr)
                    else:
                        logger.debug(f"Filtering out Grain Nr {grain_nr} (Unique Spots: {unique_spot_count} < {min_unique_spots_required})")
                        filtered_out_count += 1
                else:
                    # Should not happen if calculation included all, but handle defensively
                    logger.warning(f"Grain Nr {grain_nr} from sorted orientations not found in unique spot results. Filtering it out.")
                    filtered_out_count += 1

            filtered_orientations = orientations_sorted[indices_to_keep]
        else:
             filtered_orientations = np.empty((0, orientations_sorted.shape[1])) # Keep shape if empty

        # --- Filter Spots based on kept Grain Numbers ---
        if spots_unfiltered.size > 0 and kept_grain_nrs:
            filtered_spots = spots_unfiltered[np.isin(spots_unfiltered[:, 0].astype(int), list(kept_grain_nrs))]
        else:
            filtered_spots = np.empty((0, spots_unfiltered.shape[1])) # Keep shape if empty


        logger.info(f"Filtered out {filtered_out_count} orientations. Kept {len(filtered_orientations)} orientations.")
        logger.info(f"Filtered spots: {len(filtered_spots)} spots remain.")

        # --- Save Filtered Orientations to Text File ---
        filtered_solutions_file = f'{output_path}.bin.solutions_filtered.txt'
        try:
            np.savetxt(filtered_solutions_file, filtered_orientations,
                      header=header_gr, comments='') # Use original header
            logger.info(f"Filtered orientations saved to {filtered_solutions_file}")
        except Exception as e:
             logger.error(f"Error saving filtered solutions file: {e}")


        # --- Save Results to HDF5 ---
        # This also adds headers/columns attributes to the datasets
        self._create_h5_output(output_path, orientations_sorted, filtered_orientations,
                               spots_unfiltered, filtered_spots, orientation_unique_spots)

        # --- Visualize Filtered Results ---
        visualization_results = {"success": False, "message": "Visualization skipped or failed"}
        vis_config = self.config.get("visualization")
        if vis_config and vis_config.enable_visualization:
            if filtered_orientations.size > 0: # Only visualize if there are orientations left
                visualization_results = self._visualize_results(
                    output_path, filtered_orientations, filtered_spots, final_labels,
                    filtered_thresholded_image, # Pass the image used for spot finding
                    orientation_unique_spots # Pass unique spot info for labels/tooltips
                )
                if not visualization_results["success"]:
                    logger.warning(f"Visualization failed: {visualization_results.get('error', 'Unknown error')}")
            else:
                 logger.info("Skipping visualization as no orientations remained after filtering.")
                 visualization_results = {"success": True, "message": "Visualization skipped (no orientations after filtering)"}
        else:
            logger.info("Visualization disabled, skipping")
            visualization_results = {"success": True, "message": "Visualization disabled"}

        # --- Return Processed Results ---
        return {
            "success": True,
            "orientations": filtered_orientations, # Primary result: filtered
            "spots": filtered_spots,               # Primary result: filtered
            "unfiltered_orientations": orientations_sorted, # Reference: original sorted
            "unfiltered_spots": spots_unfiltered,           # Reference: original
            "unique_spots_per_orientation": orientation_unique_spots, # Refers to original grain numbers
            "visualization": visualization_results
        }


    def _calculate_unique_spots_per_orientation(
        self,
        orientations: np.ndarray,
        spots: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        Calculate the number of unique spots/labels for each orientation,
        prioritizing assignments based on orientation quality score.
        """
        result = lsu.calculate_unique_spots(orientations, spots, labels)
        logger.info(f"Calculated unique spots/labels for {len(result)} orientations based on quality.")
        return result


    def _sort_orientations_by_quality(
        self,
        orientations: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> np.ndarray:
        """Sort orientations by quality score (col 4) descending. Delegates to lsu."""
        sorted_arr = lsu.sort_orientations_by_quality(orientations)
        if sorted_arr.size > 0:
            logger.info(f"Sorted {len(sorted_arr)} orientations by quality score (column 4) descending.")
        return sorted_arr


    def _create_h5_output(
        self,
        output_path: str,
        orientations_unfiltered: np.ndarray,
        filtered_orientations: np.ndarray,
        spots_unfiltered: np.ndarray,
        filtered_spots: np.ndarray,
        orientation_unique_spots: Dict[int, Dict]
    ) -> None:
        """Create/Update HDF5 file. Delegates to lsu.create_h5_output."""
        lsu.create_h5_output(
            output_path, orientations_unfiltered, filtered_orientations,
            spots_unfiltered, filtered_spots, orientation_unique_spots
        )


    def _visualize_results(
        self,
        output_path: str,
        orientations: np.ndarray,
        spots: np.ndarray,
        labels: np.ndarray,
        filtered_image: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> Dict[str, Any]:
        """Delegate to lv.visualize_results."""
        vis_config = self.config.get("visualization")
        return lv.visualize_results(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, filtered_image.shape,
            orientation_unique_spots=orientation_unique_spots,
            config_obj=self.config.config,
            result_dir=self.config.get("result_dir", "results"),
        )

    def _create_static_visualization(
        self, output_path, orientations, spots, labels, filtered_image,
        orientation_unique_spots=None,
    ) -> Dict[str, Any]:
        """Delegate to lv.create_static_visualization."""
        vis_config = self.config.get("visualization")
        return lv.create_static_visualization(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, filtered_image.shape, orientation_unique_spots,
        )

    def _create_quality_map(
        self, output_path, orientations, spots, filtered_image,
        orientation_unique_spots=None,
    ) -> None:
        """Delegate to lv.create_quality_map."""
        vis_config = self.config.get("visualization")
        lv.create_quality_map(
            output_path, orientations, spots, filtered_image,
            vis_config, filtered_image.shape, orientation_unique_spots,
        )

    def _create_interactive_visualization(
        self, output_path, orientations, spots, labels, filtered_image,
        orientation_unique_spots=None,
    ) -> Dict[str, Any]:
        """Delegate to lv.create_interactive_visualization."""
        vis_config = self.config.get("visualization")
        return lv.create_interactive_visualization(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, filtered_image.shape, orientation_unique_spots,
        )

    def _create_3d_visualization(
        self, output_path, orientations, spots,
    ) -> None:
        """Delegate to lv.create_3d_visualization."""
        lv.create_3d_visualization(output_path, orientations, spots)

    def _create_analysis_report(
        self, output_path, orientations, spots, labels, filtered_image,
        orientation_unique_spots=None,
    ) -> None:
        """Delegate to lv.create_analysis_report."""
        vis_config = self.config.get("visualization")
        lv.create_analysis_report(
            output_path, orientations, spots, labels, filtered_image,
            vis_config, orientation_unique_spots,
            config_obj=self.config.config,
            result_dir=self.config.get("result_dir", "results"),
        )

    def _plot_orientation_spots(
        self, orientations, spots, filtered_image, labels, colors, ax,
    ) -> None:
        """Delegate to lv.plot_orientation_spots."""
        vis_config = self.config.get("visualization")
        show = vis_config.show_hkl_labels if vis_config else False
        lv.plot_orientation_spots(
            orientations, spots, filtered_image, labels, colors, ax,
            show_hkl_labels=show,
        )


# --- Command Line Interface ---

def parse_arguments():
    """Parse command line arguments with enhanced interface."""
    parser = argparse.ArgumentParser(
        description='LaueMatching - Advanced Laue Diffraction Pattern Indexing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process single image using config.txt, 4 cores:
    %(prog)s process -c config.txt -i image_001.h5 -n 4

  Process multiple images using glob, GPU if possible:
    %(prog)s process -c config.json -i "data/scan_*.h5" -g

  Process using 95th percentile threshold override:
    %(prog)s process -c config.yaml -i img.h5 --threshold-percentile 95.0

  Process using fixed threshold value override:
    %(prog)s process -c config.txt -i img.h5 -t 500

  Generate default text config file:
    %(prog)s config -o config_new.txt

  Generate JSON config file:
    %(prog)s config -o config_new.json -t json

  Validate existing config:
    %(prog)s config -v existing_config.yaml

  View results interactively from HDF5 output:
    %(prog)s view -i results/image_001.output.h5

  Generate HTML report from HDF5 output:
    %(prog)s report -i results/image_001.output.h5 -o image_report.html
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)

    # --- Process Command ---
    process_parser = subparsers.add_parser('process',
        help='Process Laue diffraction images',
        description='Process single or multiple Laue diffraction images using specified configuration.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    process_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file (txt, json, or yaml)')
    process_parser.add_argument('-i', '--image', type=str, required=True, help='Input image file path or glob pattern (e.g., "images/*.h5")')
    process_parser.add_argument('-n', '--ncpus', type=int, default=4, help='Number of CPU cores for the indexing executable')
    process_parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for indexing executable if available and compatible')
    process_parser.add_argument('-t', '--threshold', type=float, default=0, # Changed to float for flexibility
                              help='Override threshold *value*. Takes precedence over method/percentile. (Default: 0, uses config method)')
    process_parser.add_argument('--threshold-percentile', type=float, default=None,
                                help='Override threshold percentile (e.g., 95.0). Sets method to percentile if --threshold is not used.')
    process_parser.add_argument('-a', '--nfiles', type=int, default=0,
                              help='Max number of files to process if using legacy number pattern (e.g., img_001.h5). 0 means process all found by glob/single file.')
    process_parser.add_argument('-o', '--output', type=str, default=None,
                              help='Output directory (overrides ResultDir in config file)')
    process_parser.add_argument('--dry-run', action='store_true', help='Load config and find files, but do not process images')
    process_parser.add_argument('--no-viz', action='store_true', help='Disable all visualization generation')
    process_parser.add_argument('--no-sim', action='store_true', help='Disable the python simulation step (GenerateSimulation.py)')


    # --- Config Command ---
    config_parser = subparsers.add_parser('config',
        help='Generate or validate configuration file',
        description='Generate a new configuration file with default values or validate an existing one.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument('-o', '--output', type=str, help='Path to write the new configuration file')
    config_group.add_argument('-v', '--validate', type=str, help='Path to an existing configuration file to validate')
    config_parser.add_argument('-t', '--type', type=str, choices=['txt', 'json', 'yaml'], default='txt',
                            help='Format for generated configuration file (used with -o)')

    # --- View Command ---
    view_parser = subparsers.add_parser('view',
        help='View processing results interactively',
        description='Launch interactive Plotly viewer for processed HDF5 results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    view_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the processed HDF5 output file (e.g., image_001.output.h5)')


    # --- Report Command ---
    report_parser = subparsers.add_parser('report',
        help='Generate analysis report',
        description='Generate a detailed HTML report from processed HDF5 results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    report_parser.add_argument('-i', '--input', type=str, required=True, help='Path to the processed HDF5 output file (e.g., image_001.output.h5)')
    report_parser.add_argument('-o', '--output', type=str, default=None,
                             help='Output path for the HTML report (default: input filename + .report.html)')
    report_parser.add_argument('-t', '--template', type=str, default='default',
                             help='Report template to use (currently only "default")')

    # --- Common Arguments ---
    parser.add_argument('--loglevel', choices=[level.name for level in LogLevel], default='INFO',
                       help='Set the logging level')
    parser.add_argument('--logfile', type=str, default=None, help='Redirect logs to a file instead of console')

    args = parser.parse_args()

    # --- Post-parsing Validation ---
    if args.command == 'process' and args.threshold > 0 and args.threshold_percentile is not None:
         parser.warning("Both --threshold and --threshold-percentile provided. --threshold value override will be used.")

    # Output path for report command
    if args.command == 'report' and args.output is None:
        args.output = os.path.splitext(args.input)[0] + '.report.html'


    return args

# --- Main Execution Functions ---

def process_images(args):
    """
    Process images based on command line arguments.
    """
    # Set up logging based on args
    log_level = LogLevel[args.loglevel]
    global logger # Use the global logger instance
    logger = setup_logger(level=log_level, log_file=args.logfile)

    # Load configuration
    try:
        config_manager = ConfigurationManager(args.config)
    except SystemExit: # Raised by ConfigurationManager on fatal error
         logger.critical("Exiting due to configuration loading errors.")
         return [] # Indicate failure

    # Override configuration from command line arguments
    if args.gpu:
        if config_manager.get("do_forward", False):
            logger.warning("GPU option specified but do_forward is enabled in configuration. Using CPU instead.")
            config_manager.set("processing_type", "CPU")
        else:
            config_manager.set("processing_type", "GPU")
    else:
         config_manager.set("processing_type", "CPU") # Explicitly set CPU if -g not used

    config_manager.set("num_cpus", args.ncpus)
    if args.output:
        config_manager.set("result_dir", args.output)
    if args.no_viz:
        config_manager.config.visualization.enable_visualization = False
    if args.no_sim:
        config_manager.config.simulation.enable_simulation = False

    # Handle threshold overrides - Value override (-t) takes precedence
    if args.threshold > 0:
         # The override_thresh parameter in process_image handles this.
         logger.info(f"Command line threshold value override provided: {args.threshold}. This will be used.")
         # We pass args.threshold directly to process_image later
    elif args.threshold_percentile is not None:
         # Only apply percentile override if value override was NOT given
         logger.info(f"Command line threshold percentile override: {args.threshold_percentile}. Setting method to 'percentile'.")
         config_manager.config.image_processing.threshold_method = "percentile"
         config_manager.config.image_processing.threshold_percentile = args.threshold_percentile


    # Create result directory
    result_dir = config_manager.get("result_dir", "results")
    try:
        os.makedirs(result_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create output directory '{result_dir}': {e}")
        return []

    # --- Find Image Files ---
    if '*' in args.image or '?' in args.image:
        image_files_raw = glob.glob(args.image)
        if not image_files_raw:
            logger.error(f"No files found matching glob pattern: {args.image}")
            return []
        image_files = sorted(image_files_raw)
    elif args.nfiles > 1 : # Legacy mode check
        # Check if the input file looks like a pattern base
        logger.warning(f"Using legacy file number pattern mode (-a {args.nfiles}). Glob pattern is recommended.")
        image_files = get_image_files(args.image, args.nfiles)
        if not image_files:
             logger.error(f"Could not generate file list using legacy pattern: base={args.image}, nfiles={args.nfiles}")
             return []
    else:
        # Single file input
        if not os.path.exists(args.image):
             logger.error(f"Input image file not found: {args.image}")
             return []
        image_files = [args.image]

    # Limit number of files if -a/nfiles used with glob
    if args.nfiles > 0 and ('*' in args.image or '?' in args.image):
         logger.info(f"Limiting processing to first {args.nfiles} files found by glob pattern.")
         image_files = image_files[:args.nfiles]


    logger.info(f"Found {len(image_files)} images to process in directory: {os.path.dirname(image_files[0]) if image_files else 'N/A'}")
    logger.info(f"Output will be saved in: {result_dir}")
    logger.info(f"Using {config_manager.get('processing_type')} for indexing with {config_manager.get('num_cpus')} cores.")

    # Dry run check
    if args.dry_run:
        logger.info("Dry run requested. Configuration loaded and files found. No processing will occur.")
        logger.info("Files identified:")
        for f in image_files: logger.info(f"  - {f}")
        return []

    # --- Initialize Image Processor ---
    try:
        processor = EnhancedImageProcessor(config_manager)
    except Exception as e:
         logger.error(f"Failed to initialize image processor: {e}")
         return []

    # --- Process Images ---
    results = []
    overall_start_time = time.time()
    for i, image_file in enumerate(image_files):
        logger.info(f"--- Processing image {i+1}/{len(image_files)}: {os.path.basename(image_file)} ---")

        # Reset doFwd after first image if it was initially True (specific legacy behavior?)
        # This seems counter-intuitive, maybe remove? For now, keep original logic.
        if i > 0 and config_manager.get("do_forward"):
             logger.debug("Resetting DoFwd to 0 for subsequent images.")
             config_manager.set("do_forward", False)
             # config_manager.write_config() # Avoid writing config repeatedly

        # Process the image, passing the threshold override value
        result = processor.process_image(image_file, int(args.threshold)) # Pass threshold override
        results.append(result)

        # Log result summary for this image
        if result and result.get("success", False):
            processing_time = result.get("processing_time", -1)
            orient_count = len(result.get("indexing_results", {}).get("orientations", []))
            logger.info(f"Successfully processed {os.path.basename(image_file)} in {processing_time:.2f}s. Found {orient_count} orientations (after filtering).")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Processor returned None'
            logger.error(f"Failed to process image: {os.path.basename(image_file)}: {error_msg}")

    # --- Log Overall Summary ---
    overall_end_time = time.time()
    successful_count = sum(1 for r in results if r and r.get("success", False))
    total_time = overall_end_time - overall_start_time
    avg_time = total_time / len(image_files) if image_files else 0

    logger.info("="*40 + " Processing Summary " + "="*40)
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {len(image_files) - successful_count}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average time per image: {avg_time:.2f} seconds")
    logger.info("="*98)

    return results


def generate_config(args):
    """
    Generate or validate a configuration file.
    """
    log_level = LogLevel[args.loglevel]
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)

    if args.validate:
        # Validate existing configuration file
        try:
            logger.info(f"Validating configuration file: {args.validate}")
            config_manager = ConfigurationManager(args.validate) # Loading implicitly validates
            logger.info(f"Configuration file '{args.validate}' is valid.")
        except SystemExit:
             logger.error(f"Configuration validation failed for '{args.validate}'. See previous errors.")
             sys.exit(1) # Exit if validation fails
        except Exception as e:
            logger.error(f"Unexpected error during configuration validation: {str(e)}")
            sys.exit(1)
    elif args.output:
        # Generate new configuration file
        try:
            logger.info(f"Generating new configuration file: {args.output} (Format: {args.type})")
            # Create a config manager with the target filename to use its write methods
            config_manager = ConfigurationManager(args.output)
            # Set its config object to default LaueConfig
            config_manager.config = LaueConfig()
            # Write using the specified format (write_config handles format based on filename ext)
            config_manager.config_file = args.output # Ensure correct filename is used
            config_manager.write_config()
            logger.info(f"Default configuration file generated successfully: {args.output}")

        except Exception as e:
            logger.error(f"Error generating configuration file '{args.output}': {str(e)}")
            sys.exit(1)


def _load_h5_datasets(h5_file_path: str):
    """Load image, labels, orientations, spots, and unique-spot info from HDF5."""
    with h5py.File(h5_file_path, 'r') as hf:
        # Best available image
        image = None
        for key in ('/entry/data/cleaned_data_threshold_filtered',
                    '/entry/data/cleaned_data_threshold',
                    '/entry/data/background_subtracted',
                    '/entry/data/raw_data'):
            if key in hf:
                image = np.array(hf[key][()])
                break
        if image is None:
            raise ValueError("No suitable image dataset found in HDF5 file.")

        # Best available labels
        labels = None
        for key in ('/entry/data/watershed_labels',
                    '/entry/data/cleaned_data_threshold_filtered_labels',
                    '/entry/data/cleaned_data_threshold_labels_unfiltered'):
            if key in hf:
                labels = np.array(hf[key][()])
                break
        if labels is None:
            labels = np.zeros_like(image, dtype=np.int32)

        # Orientations (prefer filtered)
        orientations = np.array([])
        for key in ('/entry/results/filtered_orientations', '/entry/results/orientations'):
            if key in hf:
                orientations = np.array(hf[key][()])
                break

        # Spots (prefer filtered)
        spots = np.array([])
        for key in ('/entry/results/filtered_spots', '/entry/results/spots'):
            if key in hf:
                spots = np.array(hf[key][()])
                break

        # Unique spot counts
        unique_spots_data = None
        if '/entry/results/unique_spots_per_orientation' in hf:
            uca = np.array(hf['/entry/results/unique_spots_per_orientation'][()])
            if uca.size > 0:
                unique_spots_data = {int(r[0]): {"unique_label_count": int(r[1])} for r in uca}

    return image, labels, orientations, spots, unique_spots_data


def view_results(args):
    """
    View processing results interactively from HDF5 file.
    """
    log_level = LogLevel[args.loglevel]
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)

    h5_file_path = args.input
    if not os.path.exists(h5_file_path):
        logger.error(f"Input HDF5 file not found: {h5_file_path}")
        sys.exit(1)

    logger.info(f"Generating interactive view for: {h5_file_path}")

    try:
        image, labels, orientations, spots, unique_spots_data = _load_h5_datasets(h5_file_path)
        output_base_path = os.path.splitext(h5_file_path)[0]

        vis_config = VisualizationConfig()  # defaults
        viz_result = lv.create_interactive_visualization(
            output_base_path, orientations, spots, labels, image,
            vis_config, image.shape, unique_spots_data,
        )

        if viz_result.get("success"):
            html_path = f"{output_base_path}.interactive.html"
            logger.info(f"Interactive visualization saved to: {html_path}")
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
            except Exception as e:
                logger.warning(f"Could not auto-open browser: {e}")
                logger.info(f"Open manually: {html_path}")
        else:
            logger.error("Failed to generate interactive visualization.")

    except Exception as e:
        logger.error(f"Error viewing results: {e}", exc_info=True)
        sys.exit(1)


def generate_report(args):
    """
    Generate an analysis report from HDF5 file.
    """
    log_level = LogLevel[args.loglevel]
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)

    h5_file_path = args.input
    report_output_path = args.output

    if not os.path.exists(h5_file_path):
        logger.error(f"Input HDF5 file not found: {h5_file_path}")
        sys.exit(1)

    logger.info(f"Generating analysis report for: {h5_file_path}")
    logger.info(f"Report will be saved to: {report_output_path}")

    try:
        image, labels, orientations, spots, unique_spots_data = _load_h5_datasets(h5_file_path)
        report_base_path = os.path.splitext(report_output_path)[0]

        vis_config = VisualizationConfig(report_template=args.template)
        result_dir = os.path.dirname(report_output_path) or "results"

        lv.create_analysis_report(
            report_base_path, orientations, spots, labels, image,
            vis_config, unique_spots_data, result_dir=result_dir,
        )
        logger.info(f"Analysis report generated: {report_output_path}")

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        sys.exit(1)


def get_image_files(image_file_base: str, n_files: int) -> List[str]:
    """
    Get list of image files to process using legacy number pattern.
    DEPRECATED: Use glob patterns instead.
    """
    if n_files <= 0:
        return []

    # Try to parse pattern like 'stem_number.ext' or 'stem_number'
    try:
        base_dir = os.path.dirname(image_file_base)
        filename = os.path.basename(image_file_base)
        parts = filename.split('_')
        if len(parts) < 2: raise ValueError("Filename does not contain '_' for number separation.")

        numeric_part_with_ext = parts[-1]
        file_stem = os.path.join(base_dir, '_'.join(parts[:-1]))

        ext_parts = numeric_part_with_ext.split('.')
        if len(ext_parts) > 1:
            start_num = int(ext_parts[0])
            ext = '.' + '.'.join(ext_parts[1:])
            num_format = f"{{:0{len(ext_parts[0])}d}}" # Preserve leading zeros based on example
        else:
            start_num = int(numeric_part_with_ext)
            ext = ""
            num_format = "{}" # No leading zeros if no extension found after number

        image_files = []
        for i in range(n_files):
            current_num_str = num_format.format(start_num + i)
            image_files.append(f"{file_stem}_{current_num_str}{ext}")

        logger.debug(f"Generated legacy file list: {image_files}")
        # Optionally check if these files actually exist?
        return image_files

    except (ValueError, IndexError) as e:
        logger.error(f"Could not parse legacy file pattern '{image_file_base}': {e}")
        return []


def main():
    """Main function to run the Laue matching process."""
    print("--- LaueMatching - Advanced Laue Diffraction Pattern Indexing Tool ---")
    print("--- Author: Hemant Sharma (hsharma@anl.gov) ---")

    start_time = time.time()
    args = parse_arguments()

    # Execute command based on subparser selected
    if args.command == 'process':
        process_images(args)
    elif args.command == 'config':
        generate_config(args)
    elif args.command == 'view':
        view_results(args)
    elif args.command == 'report':
        generate_report(args)
    else:
        # This should not happen due to 'required=True' in add_subparsers
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)

    end_time = time.time()
    logger.info(f"Command '{args.command}' finished in {end_time - start_time:.2f} seconds.")
    print(f"--- Command '{args.command}' finished in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()