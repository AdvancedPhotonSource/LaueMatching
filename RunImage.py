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

# Get installation path for relative imports
INSTALL_PATH = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = sys.executable

class LogLevel(Enum):
    """Log level enum for configuration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logger(
    name: str = "LaueMatching",
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and configure logger with file and/or console output.

    Args:
        name: Name of the logger
        level: Logging level
        log_file: Optional path to log file
        console_output: Whether to output logs to console
        format_string: Format string for log messages

    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)8s | %(module)s:%(lineno)d | %(message)s'

    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(name)
    logger.setLevel(level.value)

    # Clear any existing handlers
    logger.handlers = []

    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Create global logger
logger = setup_logger()

# --- Configuration Dataclasses ---

@dataclass
class ImageProcessingConfig:
    """Image processing configuration parameters."""
    threshold_method: str = "adaptive"  # adaptive, otsu, fixed, or percentile
    threshold_value: float = 0.0       # Used only if threshold_method is 'fixed'
    threshold_percentile: float = 90.0 # Used only if threshold_method is 'percentile'
    min_area: int = 10
    filter_radius: int = 101
    median_passes: int = 5
    watershed_enabled: bool = True
    gaussian_factor: float = 0.25 # Note: This seems unused, gaussian width is calculated dynamically
    enhance_contrast: bool = False
    denoise_image: bool = False
    denoise_strength: float = 1.0
    edge_enhancement: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""
    output_dpi: int = 600
    colormap: str = "nipy_spectral"
    plot_type: str = "interactive"  # static, interactive, or both
    plot_format: str = "html"  # png, pdf, html
    generate_3d: bool = False
    generate_report: bool = True
    report_template: str = "default"
    show_hkl_labels: bool = False
    enable_visualization: bool = True  # Added parameter to make visualization optional

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SimulationConfig:
    """Configuration parameters for diffraction simulation."""
    enable_simulation: bool = True
    skip_percentage: float = 0.0  # Default to 0 (no skipping)
    orientation_file: str = "orientations.txt" # File containing orientations *for* simulation
    energies: str = "5 30"  # Energy range in keV (Elo Ehi)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LaueConfig:
    """Main configuration class for Laue matching."""
    # Core parameters
    space_group: int = 225
    symmetry: str = "F"
    lattice_parameter: str = "0.3615 0.3615 0.3615 90 90 90"
    r_array: str = "-1.2 -1.2 -1.2" # Initial reference orientation vector
    p_array: str = "0.02 0.002 0.513" # Detector angles (alpha, beta) and distance (gamma)
    min_good_spots: int = 5       # Min unique spots for filtering orientations
    max_laue_spots: int = 7       # Used by LaueMatchingCPU/GPU executable
    min_nr_spots: int = 5         # Used by LaueMatchingCPU/GPU executable

    # File paths
    result_dir: str = "results"
    orientation_file: str = "orientations.bin" # Input orientation database file
    hkl_file: str = "hkls.bin"         # Generated HKL file
    background_file: str = "median.bin"    # Computed median background file
    forward_file: str = "forward.bin"      # Forward simulation output file? (Used by executable?)

    # Detector parameters
    px_x: float = 0.2
    px_y: float = 0.2
    nr_px_x: int = 2048
    nr_px_y: int = 2048
    orientation_spacing: float = 0.4 # For dynamic Gaussian width calc
    distance: float = 0.513          # Sample-detector distance (mm?) - sync with p_array[2]
    min_intensity: float = 50.0      # Min intensity threshold (likely deprecated by new methods)
    elo: float = 5.0                 # Min energy for HKL generation
    ehi: float = 30.0                # Max energy for HKL generation
    maxAngle: float = 2.0            # Max angular deviation allowed by executable

    # Processing parameters
    do_forward: bool = True          # Enable forward simulation inside executable?
    processing_type: str = "CPU"     # CPU or GPU for executable
    num_cpus: int = 60               # Number of CPUs for executable

    # Enhanced configuration sections
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)  # Added simulation config

    # Additional parameters
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to dictionary format."""
        config_dict = {k: v for k, v in self.__dict__.items()
                      if not isinstance(v, (ImageProcessingConfig, VisualizationConfig, SimulationConfig))}

        config_dict["image_processing"] = self.image_processing.to_dict()
        config_dict["visualization"] = self.visualization.to_dict()
        config_dict["simulation"] = self.simulation.to_dict()

        # Handle enum conversion
        if "log_level" in config_dict:
            config_dict["log_level"] = config_dict["log_level"].name

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LaueConfig':
        """Create configuration from dictionary."""
        # Create a copy to avoid modifying the input
        config = config_dict.copy()

        # Handle nested configurations
        img_config = config.pop("image_processing", {})
        vis_config = config.pop("visualization", {})
        sim_config = config.pop("simulation", {})

        # Handle enum conversion
        if "log_level" in config:
            config["log_level"] = LogLevel[config["log_level"]]

        # Create instance - Ensure only valid fields are passed
        valid_fields = {f.name for f in field(cls) if f.init}
        filtered_config = {k: v for k, v in config.items() if k in valid_fields}
        instance = cls(**filtered_config)

        # Update nested configurations
        if img_config:
            instance.image_processing = ImageProcessingConfig(**img_config)
        if vis_config:
            instance.visualization = VisualizationConfig(**vis_config)
        if sim_config:
            instance.simulation = SimulationConfig(**sim_config)

        return instance


# --- Configuration Manager ---

class ConfigurationManager:
    """Manages configuration for the Laue matching process."""

    def __init__(self, config_file: str):
        """
        Initialize configuration manager with parameters from config file.

        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = LaueConfig() # Initialize with defaults first
        self._load_config()

    def _load_config(self) -> None:
        """Parse the configuration file and load parameters."""
        if not os.path.exists(self.config_file):
            logger.error(f"Configuration file {self.config_file} not found.")
            # Optionally create a default one? Or just exit.
            # self.write_config() # Write default if not found? Careful with overwrites.
            # logger.info(f"Created default config file at {self.config_file}")
            sys.exit(1)

        try:
            file_ext = os.path.splitext(self.config_file)[1].lower()

            if file_ext == '.json':
                self._load_from_json()
            elif file_ext in ('.yaml', '.yml'):
                self._load_from_yaml()
            else:
                self._load_from_text()

            logger.info(f"Configuration loaded from {self.config_file}")

            # Sync potentially inconsistent parameters
            self._sync_parameters()

        except Exception as e:
            logger.error(f"Error reading or parsing configuration file '{self.config_file}': {str(e)}")
            sys.exit(1)

    def _load_from_json(self) -> None:
        """Load configuration from JSON file."""
        with open(self.config_file, 'r') as f:
            config_dict = json.load(f)
            self.config = LaueConfig.from_dict(config_dict)

    def _load_from_yaml(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            self.config = LaueConfig.from_dict(config_dict)

    def _load_from_text(self) -> None:
        """Load configuration from classic text format."""
        with open(self.config_file, 'r') as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):
             line_content = line.strip()
             if line_content and not line_content.startswith('#'):
                try:
                    self._parse_classic_config_line(line_content)
                except Exception as e:
                    logger.error(f"Error parsing line {line_num + 1} in {self.config_file}: '{line_content}' - {str(e)}")
                    # Decide whether to continue or exit
                    # sys.exit(1)

    def _parse_classic_config_line(self, line: str) -> None:
        """
        Parse a single configuration line from classic format.

        Args:
            line: A line from the configuration file
        """
        # Strip inline comments: everything after '#' is a comment
        if '#' in line:
            line = line[:line.index('#')].strip()
        parts = line.split()
        if not parts: return
        key = parts[0]
        num_parts = len(parts)

        # Helper to get value or log error
        def get_val(index, expected_type=str, expected_parts=2):
            if num_parts >= expected_parts:
                try:
                    return expected_type(parts[index])
                except (ValueError, IndexError):
                    logger.error(f"Invalid value format for {key} on line: '{line}'. Expected {expected_type}.")
                    # Optionally raise error or return default
                    raise ValueError(f"Invalid format for {key}")
            else:
                logger.error(f"Missing value for {key} on line: '{line}'. Expected {expected_parts-1} value(s).")
                raise ValueError(f"Missing value for {key}")

        img_proc = self.config.image_processing # Shortcut
        vis_conf = self.config.visualization   # Shortcut
        sim_conf = self.config.simulation     # Shortcut

        try:
            if key == 'SpaceGroup':
                self.config.space_group = get_val(1, int)
            elif key == 'Symmetry':
                sym = get_val(1, str)
                if sym not in 'FICAR' or len(sym) != 1:
                    logger.error('Invalid value for Symmetry, must be one character from F,I,C,A,R')
                    raise ValueError('Invalid Symmetry')
                self.config.symmetry = sym
            elif key == 'LatticeParameter':
                 if num_parts == 7:
                    self.config.lattice_parameter = ' '.join(parts[1:7])
                 else:
                     logger.error(f"Incorrect number of values for LatticeParameter. Expected 6, got {num_parts-1}.")
                     raise ValueError("Incorrect LatticeParameter format")
            elif key == 'R_Array':
                if num_parts == 4:
                    self.config.r_array = ' '.join(parts[1:4])
                else:
                     logger.error(f"Incorrect number of values for R_Array. Expected 3, got {num_parts-1}.")
                     raise ValueError("Incorrect R_Array format")
            elif key == 'P_Array':
                 if num_parts == 4:
                    self.config.p_array = ' '.join(parts[1:4])
                    self.config.distance = float(parts[3]) # Distance is part of P_Array line
                 else:
                     logger.error(f"Incorrect number of values for P_Array. Expected 3, got {num_parts-1}.")
                     raise ValueError("Incorrect P_Array format")
            # --- Thresholding Params ---
            elif key == 'ThresholdMethod':
                method = get_val(1, str).lower()
                if method in ["adaptive", "otsu", "fixed", "percentile"]:
                    img_proc.threshold_method = method
                else:
                    logger.warning(f"Unknown ThresholdMethod '{parts[1]}'. Using default '{img_proc.threshold_method}'.")
            elif key == 'Threshold': # Specific value for 'fixed' method
                img_proc.threshold_value = get_val(1, float)
            elif key == 'ThresholdPercentile': # Specific value for 'percentile' method
                 img_proc.threshold_percentile = get_val(1, float)
            # --- End Thresholding Params ---
            elif key == 'MinIntensity':
                self.config.min_intensity = get_val(1, float)
            elif key == 'Elo':
                self.config.elo = get_val(1, float)
            elif key == 'Ehi':
                self.config.ehi = get_val(1, float)
            elif key == 'PxX':
                self.config.px_x = get_val(1, float)
            elif key == 'PxY':
                self.config.px_y = get_val(1, float)
            elif key == 'OrientationSpacing':
                self.config.orientation_spacing = get_val(1, float)
            elif key == 'WatershedImage':
                img_proc.watershed_enabled = bool(get_val(1, int))
            elif key == 'NrPxX':
                self.config.nr_px_x = get_val(1, int)
            elif key == 'NrPxY':
                self.config.nr_px_y = get_val(1, int)
            elif key == 'FilterRadius':
                img_proc.filter_radius = get_val(1, int)
            elif key == 'NMeadianPasses':
                img_proc.median_passes = get_val(1, int)
            elif key == 'MinArea':
                img_proc.min_area = get_val(1, int)
            elif key == 'MinGoodSpots':
                self.config.min_good_spots = get_val(1, int)
            elif key == 'MinNrSpots':
                self.config.min_nr_spots = get_val(1, int)
            elif key == 'MaxAngle':
                self.config.maxAngle = get_val(1, float)
            elif key == 'MaxNrLaueSpots':
                self.config.max_laue_spots = get_val(1, int)
            elif key == 'ResultDir':
                self.config.result_dir = get_val(1, str)
            elif key == 'OrientationFile':
                self.config.orientation_file = get_val(1, str)
            elif key == 'HKLFile':
                self.config.hkl_file = get_val(1, str)
            elif key == 'BackgroundFile':
                self.config.background_file = get_val(1, str)
            elif key == 'ForwardFile':
                self.config.forward_file = get_val(1, str)
            elif key == 'DoFwd':
                self.config.do_forward = bool(get_val(1, int))
            elif key == 'EnableVisualization':
                vis_conf.enable_visualization = bool(get_val(1, int))
            elif key == 'EnableSimulation':
                sim_conf.enable_simulation = bool(get_val(1, int))
            elif key == 'SkipPercentage':
                sim_conf.skip_percentage = get_val(1, float)
            elif key == 'SimulationEnergies':
                if num_parts == 3:
                    sim_conf.energies = ' '.join(parts[1:3])
                else:
                     logger.error(f"Incorrect number of values for SimulationEnergies. Expected 2, got {num_parts-1}.")
                     raise ValueError("Incorrect SimulationEnergies format")
            # --- Image Processing Enhancements ---
            elif key == 'EnhanceContrast':
                img_proc.enhance_contrast = bool(get_val(1, int))
            elif key == 'DenoiseImage':
                img_proc.denoise_image = bool(get_val(1, int))
            elif key == 'DenoiseStrength':
                img_proc.denoise_strength = get_val(1, float)
            elif key == 'EdgeEnhancement':
                img_proc.edge_enhancement = bool(get_val(1, int))
            # --- Keys used by other scripts (GenerateSimulation.py, etc.) ---
            elif key in ('AStar', 'SimulationSmoothingWidth'):
                pass  # Recognized but not used by RunImage.py
            # --- Ignore truly unknown keys ---
            else:
                logger.warning(f"Ignoring unknown configuration key '{key}' on line: '{line}'")

        except ValueError as e:
             # Logged inside helper or above
             # Decide whether to re-raise, exit, or continue
             # For now, let's re-raise to stop execution on bad config
             raise e

    def _sync_parameters(self):
        """Ensure consistency between related parameters."""
        # Sync distance from P_Array[2]
        try:
            p_array_vals = self.config.p_array.split()
            if len(p_array_vals) == 3:
                self.config.distance = float(p_array_vals[2])
            else:
                logger.warning(f"Could not sync distance from P_Array '{self.config.p_array}'. Using existing value {self.config.distance}.")
        except (ValueError, IndexError):
             logger.warning(f"Could not parse P_Array '{self.config.p_array}' to sync distance. Using existing value {self.config.distance}.")

        # Sync simulation energies with Elo/Ehi if SimulationEnergies isn't explicitly set in text file?
        # This is tricky because they might be intentionally different.
        # Let's assume the dedicated SimulationEnergies takes precedence if present.
        if self.config.simulation.energies == SimulationConfig().energies: # If still default
            self.config.simulation.energies = f"{self.config.elo} {self.config.ehi}"
            logger.debug(f"Synced SimulationEnergies from Elo/Ehi to '{self.config.simulation.energies}'")


    def write_config(self) -> None:
        """Write current configuration to file."""
        file_ext = os.path.splitext(self.config_file)[1].lower()

        # Before writing, ensure parameters are synced
        self._sync_parameters()

        try:
            if file_ext == '.json':
                self._write_to_json()
            elif file_ext in ('.yaml', '.yml'):
                self._write_to_yaml()
            else:
                self._write_to_text()

            logger.info(f"Configuration saved to {self.config_file}")

        except Exception as e:
            logger.error(f"Error writing configuration to {self.config_file}: {str(e)}")
            # sys.exit(1) # Avoid exiting if just writing fails

    def _write_to_json(self) -> None:
        """Write configuration to JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)

    def _write_to_yaml(self) -> None:
        """Write configuration to YAML file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def _write_to_text(self) -> None:
        """Write configuration to classic text format."""
        with open(self.config_file, 'w') as f:
            # --- Comments ---
            f.write("# LaueMatching Configuration File\n")
            f.write("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

            # --- Core Crystal Parameters ---
            f.write("# --- Crystal Parameters ---\n")
            f.write(f"SpaceGroup         {self.config.space_group}\n")
            f.write(f"Symmetry           {self.config.symmetry}\n")
            f.write(f"LatticeParameter   {self.config.lattice_parameter}\n")
            f.write(f"R_Array            {self.config.r_array}\n")
            f.write(f"P_Array            {self.config.p_array}\n\n") # Includes distance implicitly

            # --- Detector Parameters ---
            f.write("# --- Detector Parameters ---\n")
            f.write(f"NrPxX              {self.config.nr_px_x}\n")
            f.write(f"NrPxY              {self.config.nr_px_y}\n")
            f.write(f"PxX                {self.config.px_x}\n")
            f.write(f"PxY                {self.config.px_y}\n")
            # Distance is implicit in P_Array
            f.write(f"OrientationSpacing {self.config.orientation_spacing}\n\n")

            # --- HKL Generation Parameters ---
            f.write("# --- HKL Generation Parameters ---\n")
            f.write(f"Elo                {self.config.elo}\n")
            f.write(f"Ehi                {self.config.ehi}\n\n")

            # --- Indexing Parameters ---
            f.write("# --- Indexing Parameters (Executable) ---\n")
            f.write(f"MinNrSpots         {self.config.min_nr_spots}\n")
            f.write(f"MaxNrLaueSpots     {self.config.max_laue_spots}\n")
            f.write(f"MaxAngle           {self.config.maxAngle}\n")
            f.write(f"MinIntensity       {self.config.min_intensity} # (May be deprecated by threshold methods)\n\n")

            # --- Filtering Parameters ---
            f.write("# --- Orientation Filtering (Python) ---\n")
            f.write(f"MinGoodSpots       {self.config.min_good_spots} # Min unique spots to keep orientation\n\n")

            # --- Image Processing Parameters ---
            f.write("# --- Image Processing (Python) ---\n")
            img_proc = self.config.image_processing
            f.write(f"ThresholdMethod     {img_proc.threshold_method} # options: adaptive, otsu, fixed, percentile\n")
            f.write(f"Threshold           {img_proc.threshold_value} # Used only if ThresholdMethod is 'fixed'\n")
            f.write(f"ThresholdPercentile {img_proc.threshold_percentile} # Used only if ThresholdMethod is 'percentile'\n")
            f.write(f"MinArea             {img_proc.min_area}\n")
            f.write(f"FilterRadius        {img_proc.filter_radius}\n")
            f.write(f"NMeadianPasses      {img_proc.median_passes}\n")
            f.write(f"WatershedImage      {int(img_proc.watershed_enabled)}\n")
            f.write(f"EnhanceContrast     {int(img_proc.enhance_contrast)}\n")
            f.write(f"DenoiseImage        {int(img_proc.denoise_image)}\n")
            f.write(f"DenoiseStrength     {img_proc.denoise_strength}\n")
            f.write(f"EdgeEnhancement     {int(img_proc.edge_enhancement)}\n\n")

            # --- File Paths ---
            f.write("# --- File Paths ---\n")
            f.write(f"ResultDir          {self.config.result_dir}\n")
            f.write(f"OrientationFile    {self.config.orientation_file} # Input orientation database\n")
            f.write(f"HKLFile            {self.config.hkl_file}\n")
            f.write(f"BackgroundFile     {self.config.background_file}\n")
            f.write(f"ForwardFile        {self.config.forward_file} # Output from executable forward sim?\n\n")

            # --- Processing Control ---
            f.write("# --- Processing Control ---\n")
            f.write(f"DoFwd              {int(self.config.do_forward)} # Enable forward sim in executable?\n")
            # Processing type (CPU/GPU) and Num CPUs usually set via command line
            # f.write(f"ProcessingType     {self.config.processing_type}\n")
            # f.write(f"NumCPUs            {self.config.num_cpus}\n")

            # --- Visualization Parameters ---
            f.write("# --- Visualization Parameters (Python) ---\n")
            vis_config = self.config.visualization
            f.write(f"EnableVisualization {int(vis_config.enable_visualization)}\n")
            # Add other vis params if needed in text format
            # f.write(f"OutputDPI          {vis_config.output_dpi}\n")
            # f.write(f"Colormap           {vis_config.colormap}\n")
            # f.write(f"PlotType           {vis_config.plot_type}\n")
            # f.write(f"PlotFormat         {vis_config.plot_format}\n")
            # f.write(f"Generate3D         {int(vis_config.generate_3d)}\n")
            # f.write(f"GenerateReport     {int(vis_config.generate_report)}\n")
            # f.write(f"ShowHKLLabels      {int(vis_config.show_hkl_labels)}\n")
            f.write("\n")

            # --- Simulation Parameters ---
            f.write("# --- Simulation Parameters (Python GenerateSimulation.py) ---\n")
            sim_config = self.config.simulation
            f.write(f"EnableSimulation   {int(sim_config.enable_simulation)}\n")
            f.write(f"SkipPercentage     {sim_config.skip_percentage}\n")
            f.write(f"SimulationEnergies {sim_config.energies}\n")
            # f.write(f"SimulationOrientationFile {sim_config.orientation_file}\n") # Input file FOR simulation
            f.write("\n")


    def get(self, key: str, default=None):
        """Get a configuration parameter value."""
        # Try direct attribute first
        if hasattr(self.config, key):
            return getattr(self.config, key)
        # Check nested configs
        if hasattr(self.config.image_processing, key):
            return getattr(self.config.image_processing, key)
        if hasattr(self.config.visualization, key):
            return getattr(self.config.visualization, key)
        if hasattr(self.config.simulation, key):
            return getattr(self.config.simulation, key)
        return default

    def set(self, key: str, value) -> None:
        """Set a configuration parameter value."""
        # Try direct attribute first
        if hasattr(self.config, key):
            # Basic type conversion attempt if types don't match
            try:
                current_value = getattr(self.config, key)
                if type(value) != type(current_value):
                     setattr(self.config, key, type(current_value)(value))
                else:
                     setattr(self.config, key, value)
                # Special case: update distance if p_array is set
                if key == 'p_array':
                    self._sync_parameters()
            except Exception as e:
                 logger.error(f"Error setting config key '{key}' with value '{value}': {e}")

        # Check nested configs
        elif hasattr(self.config.image_processing, key):
            try:
                current_value = getattr(self.config.image_processing, key)
                if type(value) != type(current_value):
                     setattr(self.config.image_processing, key, type(current_value)(value))
                else:
                     setattr(self.config.image_processing, key, value)
            except Exception as e:
                 logger.error(f"Error setting image_processing config key '{key}' with value '{value}': {e}")
        elif hasattr(self.config.visualization, key):
            try:
                current_value = getattr(self.config.visualization, key)
                if type(value) != type(current_value):
                     setattr(self.config.visualization, key, type(current_value)(value))
                else:
                    setattr(self.config.visualization, key, value)
            except Exception as e:
                 logger.error(f"Error setting visualization config key '{key}' with value '{value}': {e}")
        elif hasattr(self.config.simulation, key):
            try:
                current_value = getattr(self.config.simulation, key)
                if type(value) != type(current_value):
                     setattr(self.config.simulation, key, type(current_value)(value))
                else:
                     setattr(self.config.simulation, key, value)
            except Exception as e:
                 logger.error(f"Error setting simulation config key '{key}' with value '{value}': {e}")
        else:
            logger.warning(f"Attempted to set unknown configuration parameter: {key}")

    def load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Environment variables should be prefixed with LAUE_
        Nested config keys can be specified like LAUE_IMAGE_PROCESSING_MIN_AREA
        """
        prefix = 'LAUE_'
        for env_key, value in os.environ.items():
            if env_key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key_parts = env_key[len(prefix):].lower().split('_')
                target_obj = self.config
                key_to_set = config_key_parts[-1]
                processed = False

                # Handle nested structure like image_processing_min_area
                if len(config_key_parts) > 1:
                    nested_key = '_'.join(config_key_parts[:-1])
                    if hasattr(self.config, nested_key) and isinstance(getattr(self.config, nested_key), (ImageProcessingConfig, VisualizationConfig, SimulationConfig)):
                        target_obj = getattr(self.config, nested_key)
                    else:
                        # If not a direct nested object, maybe it's a top-level key like min_good_spots
                        key_to_set = '_'.join(config_key_parts)
                        target_obj = self.config # Reset target to top level

                # Check if the final key exists on the target object
                if hasattr(target_obj, key_to_set):
                    try:
                        current_value = getattr(target_obj, key_to_set)
                        # Convert value to appropriate type
                        if isinstance(current_value, bool):
                             setattr(target_obj, key_to_set, value.lower() in ('true', '1', 'yes'))
                        elif isinstance(current_value, int):
                             setattr(target_obj, key_to_set, int(value))
                        elif isinstance(current_value, float):
                             setattr(target_obj, key_to_set, float(value))
                        elif isinstance(current_value, LogLevel): # Handle LogLevel Enum
                             try:
                                 setattr(target_obj, key_to_set, LogLevel[value.upper()])
                             except KeyError:
                                 logger.warning(f"Invalid LogLevel '{value}' from env var {env_key}")
                        else:
                             setattr(target_obj, key_to_set, value)
                        logger.debug(f"Loaded config from env: {key_to_set} = {getattr(target_obj, key_to_set)}")
                        processed = True
                    except Exception as e:
                         logger.warning(f"Could not set config key '{key_to_set}' from env var {env_key}: {e}")

                if not processed:
                     logger.warning(f"Ignoring environment variable {env_key}: Cannot map to configuration parameter.")
        # Resync parameters after potentially loading from env
        self._sync_parameters()


# --- Progress Reporter ---

class ProgressReporter:
    """Reports progress of multi-step operations."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress reporter.

        Args:
            total_steps: Total number of steps
            description: Description of the operation
        """
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.progress_bar = tqdm(total=total_steps, desc=description, unit="step", ncols=100) # Set width

    def update(self, step_increment: int = 1, status: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            step_increment: Number of steps to increment
            status: Optional status message
        """
        self.current_step += step_increment
        current_time = time.time()

        # Update progress bar
        self.progress_bar.update(step_increment)

        if status:
            self.progress_bar.set_description(f"{self.description}: {status}")

        # Calculate statistics
        elapsed = current_time - self.start_time
        if self.current_step > 0 and self.total_steps > 0: # Avoid division by zero
             percentage = min(100.0 * self.current_step / self.total_steps, 100.0)
             # Estimate remaining only if not completed
             if self.current_step < self.total_steps:
                remaining = elapsed * (self.total_steps - self.current_step) / self.current_step
                # Log progress every 10% or at least 5 seconds
                if (percentage % 10 < (percentage - step_increment * 100.0 / self.total_steps) % 10 or
                        current_time - self.last_update_time > 5):
                    self.last_update_time = current_time
                    logger.info(f"Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps}), "
                               f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
             else: # Log completion time if exactly total_steps reached
                 if self.current_step == self.total_steps:
                     logger.info(f"Progress: 100.0% ({self.current_step}/{self.total_steps}), "
                                 f"Total Elapsed: {elapsed:.1f}s")


    def complete(self, status: str = "Completed") -> None:
        """
        Mark progress as complete.

        Args:
            status: Final status message
        """
        # Ensure the bar reaches 100% even if called early
        remaining_steps = self.total_steps - self.current_step
        if remaining_steps > 0:
            self.progress_bar.update(remaining_steps)
        self.current_step = self.total_steps # Mark as complete internally

        self.progress_bar.set_description(f"{self.description}: {status}")
        self.progress_bar.close()

        elapsed = time.time() - self.start_time
        logger.info(f"Operation '{self.description}' {status.lower()} in {elapsed:.2f} seconds")


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
        background_file = self.config.get("background_file")
        nPxX = self.config.get("nr_px_x", 2048)
        nPxY = self.config.get("nr_px_y", 2048)

        if background_file and os.path.exists(background_file):
            try:
                # Check file size
                expected_size = nPxX * nPxY * np.dtype(np.double).itemsize
                actual_size = os.path.getsize(background_file)
                if actual_size != expected_size:
                    logger.warning(f"Background file '{background_file}' size mismatch. Expected {expected_size}, got {actual_size}. May be corrupted or dimensions changed.")
                    self.background = np.zeros((nPxY, nPxX)) # Use Y,X convention for numpy array shape
                    return # Don't attempt to load

                self.background = np.fromfile(background_file, dtype=np.double).reshape((nPxY, nPxX)) # Y, X
                logger.info(f"Background file loaded: {background_file}")
            except Exception as e:
                logger.error(f"Error loading background file '{background_file}': {str(e)}")
                self.background = np.zeros((nPxY, nPxX))
        else:
            logger.info("Background file not found or specified. Will compute if needed.")
            self.background = np.zeros((nPxY, nPxX))

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

        filter_radius = img_config.filter_radius
        median_passes = img_config.median_passes

        if filter_radius <= 0 or median_passes <= 0:
            logger.warning(f"Invalid median filter parameters (radius={filter_radius}, passes={median_passes}). Returning zero background.")
            self.background = np.zeros_like(image, dtype=np.double)
            return self.background

        # Convert numpy array (Y, X) to diplib image (X, Y) ? No, diplib seems to handle numpy arrays directly
        try:
            background_dip = dip.Image(image) # diplib should handle Y,X -> X,Y implicitly if needed
            for i in range(median_passes):
                logger.debug(f"Median filter pass {i+1}/{median_passes} with radius {filter_radius}")
                # Diplib MedianFilter takes shape param, assumes isotropic if scalar
                background_dip = dip.MedianFilter(background_dip, filter_radius)

            # Convert back to numpy array (Y, X)
            background_arr = np.array(background_dip).astype(np.double)
            self.background = background_arr # Store the computed background

            # Save the background for future use
            background_file_path = self.config.get("background_file")
            if background_file_path:
                try:
                     background_arr.tofile(background_file_path)
                     logger.info(f"Computed background saved to {background_file_path}")
                except Exception as e:
                     logger.error(f"Error saving computed background to {background_file_path}: {e}")
            return background_arr

        except Exception as e:
            logger.error(f"Error during diplib median filtering: {e}")
            self.background = np.zeros_like(image, dtype=np.double)
            return self.background

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply optional image enhancements based on configuration.

        Args:
            image: Input image array (numpy format, Y,X)

        Returns:
            Enhanced image array (numpy format, Y,X)
        """
        img_config = self.config.get("image_processing")
        if not img_config: return image # No config, no enhancement

        enhanced = image.copy().astype(np.float32) # Work with float for intermediate steps

        # --- Apply enhancements in a reasonable order ---

        # 1. Denoising (applied first to avoid enhancing noise)
        if img_config.denoise_image:
            logger.info("Applying denoising")
            try:
                # Non-local means expects float image in range [0, 1] or [-1, 1] if signed
                # Need to normalize image first
                img_min, img_max = enhanced.min(), enhanced.max()
                if img_max > img_min:
                    normalized_img = (enhanced - img_min) / (img_max - img_min)
                    # Parameters for nl_means might need tuning
                    denoised_normalized = restoration.denoise_nl_means(
                        normalized_img,
                        h=img_config.denoise_strength, # Controls filter strength
                        fast_mode=True,
                        patch_size=5,        # Size of patches used for comparison
                        patch_distance=7,    # Max distance between patches
                        channel_axis=None    # Grayscale image
                    )
                    # Rescale back to original range
                    enhanced = (denoised_normalized * (img_max - img_min)) + img_min
                else:
                    logger.warning("Image is constant, skipping denoising.")

            except Exception as e:
                logger.error(f"Error during denoising: {e}")


        # 2. Contrast Enhancement (applied after denoising)
        if img_config.enhance_contrast:
            logger.info("Applying contrast enhancement (CLAHE)")
            try:
                 # CLAHE expects uint8 or uint16. Convert temporarily.
                 # Normalize to 0-1 first, then scale to uint16 range
                 img_min, img_max = enhanced.min(), enhanced.max()
                 if img_max > img_min:
                      normalized_img = (enhanced - img_min) / (img_max - img_min)
                      img_uint16 = (normalized_img * 65535).astype(np.uint16)
                      # Apply CLAHE
                      enhanced_uint16 = exposure.equalize_adapthist(img_uint16, clip_limit=0.03)
                      # Convert back to float and original range
                      enhanced_normalized = enhanced_uint16.astype(np.float32) / 65535.0
                      enhanced = (enhanced_normalized * (img_max - img_min)) + img_min
                 else:
                      logger.warning("Image is constant, skipping contrast enhancement.")
            except Exception as e:
                 logger.error(f"Error during contrast enhancement: {e}")

        # 3. Edge Enhancement (applied last)
        if img_config.edge_enhancement:
            logger.info("Applying edge enhancement (Unsharp Masking)")
            try:
                # Use unsharp masking for edge enhancement
                # Gaussian blur the image
                blurred = filters.gaussian(enhanced, sigma=1.0)
                # Subtract blurred from original, scale, and add back
                # Factor 0.5 can be adjusted
                enhanced = enhanced + 0.5 * (enhanced - blurred)
                # Ensure values stay within reasonable bounds (e.g., no negatives if original was positive)
                enhanced = np.maximum(enhanced, 0) # Example if original image was non-negative
            except Exception as e:
                 logger.error(f"Error during edge enhancement: {e}")


        return enhanced # Return as float

    def apply_threshold(self, image: np.ndarray, override_thresh: int = 0) -> Tuple[np.ndarray, float]:
        """
        Apply thresholding based on configuration or override value.

        Args:
            image: Input image array (should be background corrected and potentially enhanced)
            override_thresh: Optional threshold override value (takes highest priority)

        Returns:
            Tuple of (thresholded image, threshold used)
        """
        img_config = self.config.get("image_processing")
        if not img_config:
            logger.error("Image processing configuration not found. Cannot apply threshold.")
            return image.copy(), 0.0 # Return original, threshold 0

        threshold_method = img_config.threshold_method
        threshold = 0.0 # Initialize threshold value

        # Determine the threshold value based on method or override
        if override_thresh > 0:
            threshold = float(override_thresh)
            logger.info(f"Using override threshold value: {threshold}")
            threshold_method = "override" # Mark method as overridden
        elif threshold_method == "percentile":
            percentile_value = img_config.threshold_percentile
            if not 0 < percentile_value < 100:
                 logger.warning(f"Threshold percentile {percentile_value} is outside valid range (0, 100). Clamping to 90.")
                 percentile_value = 90.0
            try:
                # Calculate percentile on the flattened array of the input image
                threshold = np.percentile(image.ravel(), percentile_value)
                logger.info(f"Using percentile threshold ({percentile_value}th percentile): {threshold:.2f}")
            except IndexError: # Handle empty image case
                 logger.warning("Cannot calculate percentile on empty image. Using threshold=0.")
                 threshold = 0.0
            except Exception as e:
                 logger.error(f"Error calculating percentile threshold: {e}. Using threshold=0.")
                 threshold = 0.0
        elif threshold_method == "adaptive":
            # Original script logic based on std dev bins
            std_dev = np.std(image)
            threshold_adaptive = 60.0 * (1.0 + std_dev // 60.0) # Use float division/calculation
            threshold = max(threshold_adaptive, 1) # Ensure threshold is at least 1
            logger.info(f"Using adaptive threshold (std dev bins): {threshold:.2f} (based on std dev {std_dev:.2f})")
            # Alternative: mean + N * std_dev
            # mean_val = np.mean(image)
            # n_std_dev = 3.0
            # threshold_alt = mean_val + n_std_dev * std_dev
            # threshold = max(threshold_alt, 1)
            # logger.info(f"Using adaptive threshold (mean + {n_std_dev}*std): {threshold:.2f} (mean={mean_val:.2f}, std={std_dev:.2f})")
        elif threshold_method == "otsu":
            try:
                 # Otsu may work better on background-subtracted before enhancement
                 # Apply here on the input 'image' (which might be enhanced)
                 threshold = filters.threshold_otsu(image)
                 logger.info(f"Using Otsu threshold: {threshold:.2f}")
            except Exception as e:
                 logger.error(f"Error calculating Otsu threshold: {e}. Using threshold=0.")
                 threshold = 0.0
        elif threshold_method == "fixed":
            threshold = img_config.threshold_value
            logger.info(f"Using fixed threshold value: {threshold}")
        else:
            # Fallback or unknown method
            threshold = img_config.threshold_value # Fallback to fixed value
            logger.warning(f"Unknown threshold method '{threshold_method}'. Falling back to fixed threshold value: {threshold}")

        # Apply threshold: create a copy, set values below threshold to zero
        thresholded_image = image.copy()
        thresholded_image[image <= threshold] = 0

        return thresholded_image, threshold


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
        """
        Find connected components in the thresholded image.

        Args:
            image: Thresholded input image (uint16 or similar)

        Returns:
            Tuple of (labels, bounding boxes, areas, number of labels (including background))
        """
        # Convert to binary image (8-bit required for connectedComponentsWithStats)
        # Ensure threshold is appropriate (e.g., > 0 for uint images)
        binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)

        # Find connected components
        # output = (num_labels, labels_matrix, stats_matrix, centroids_matrix)
        # stats = [left, top, width, height, area]
        try:
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)
            # Extract areas and bounding boxes from stats (remove background label 0 stats)
            areas = stats[1:, cv2.CC_STAT_AREA]
            bounding_boxes = stats[1:, :cv2.CC_STAT_HEIGHT+1] # left, top, width, height

            logger.info(f"Found {nlabels - 1} connected components (excluding background)")

            return labels, bounding_boxes, areas, nlabels

        except Exception as e:
            logger.error(f"Error finding connected components: {e}")
            # Return empty results in case of error
            return np.zeros_like(image, dtype=np.int32), np.empty((0, 4)), np.empty((0,)), 1


    def filter_small_components(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        bounding_boxes: np.ndarray, # Note: Index corresponds to label-1
        areas: np.ndarray,          # Note: Index corresponds to label-1
        nlabels: int
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Filter out components smaller than the minimum area and calculate centers of mass.

        Args:
            image: Input thresholded image array
            labels: Component labels array (output from find_connected_components)
            bounding_boxes: Bounding boxes array (for labels 1 to nlabels-1)
            areas: Component areas array (for labels 1 to nlabels-1)
            nlabels: Number of labels (including background)

        Returns:
            Tuple of (filtered image, filtered labels, list of centers)
            centers format: [label_idx, (center_x, center_y), area]
        """
        img_config = self.config.get("image_processing")
        if not img_config:
             logger.error("Image processing config missing, cannot filter components.")
             return image, labels, []

        min_area = img_config.min_area
        centers = [] # Store [label, (cx, cy), area] for kept components

        # Create copies to modify
        filtered_image = image.copy()
        filtered_labels = labels.copy()

        filtered_count = 0
        kept_count = 0
        # Iterate through labels 1 to nlabels-1
        for label_idx in range(1, nlabels):
            # Adjust index for areas and bounding_boxes arrays (which are 0-based for labels 1+)
            array_idx = label_idx - 1
            if areas[array_idx] >= min_area:
                kept_count += 1
                # Get the bounding box
                x, y, w, h = bounding_boxes[array_idx]

                # Extract the region corresponding to this label ONLY within the bounding box
                # Create a mask for the current label within the bbox
                label_mask_in_bbox = (labels[y:y+h, x:x+w] == label_idx)
                # Get image intensities only for this label
                region_intensities = image[y:y+h, x:x+w] * label_mask_in_bbox

                # Compute center of mass using the intensities as weights
                try:
                    com = ndimg.center_of_mass(region_intensities) # Returns (row, col) relative to bbox
                    # Convert COM coordinates to be relative to the full image
                    center_y = com[0] + y
                    center_x = com[1] + x

                    # Store label index, center coordinates (x, y), and area
                    centers.append([label_idx, (center_x, center_y), areas[array_idx]])
                except Exception as e:
                     logger.warning(f"Could not calculate center of mass for label {label_idx}: {e}. Skipping component.")
                     # Optionally filter this component out too
                     filtered_image[labels == label_idx] = 0
                     filtered_labels[labels == label_idx] = 0
                     kept_count -= 1 # Decrement kept count
                     filtered_count += 1


            else:
                # Remove small components by setting their pixels to 0
                filtered_count += 1
                filtered_image[labels == label_idx] = 0
                filtered_labels[labels == label_idx] = 0 # Also remove label

        logger.info(f"Filtered out {filtered_count} small components (area < {min_area}), kept {kept_count}")
        return filtered_image, filtered_labels, centers

    @staticmethod
    def calculate_gaussian_width(centers: List, pixel_size: float, distance: float, orient_spacing: float) -> int:
        """
        Calculate optimal Gaussian blur width based on spot spacing.

        Args:
            centers: List of center points [label, (cx, cy), area]
            pixel_size: Pixel size in mm
            distance: Sample-to-detector distance in mm
            orient_spacing: Orientation spacing in degrees

        Returns:
            Gaussian blur width in pixels (integer >= 1)
        """
        if not centers or len(centers) < 2:
            logger.warning("Not enough centers (< 2) to calculate optimal Gaussian width, using default width=3.")
            return 3 # Return a reasonable default

        # Extract just the (cx, cy) coordinates
        center_coords = [c[1] for c in centers]

        # Find minimum distance between centers
        min_distance_sq = float('inf')
        for i in range(len(center_coords)):
            for j in range(i + 1, len(center_coords)):
                c1 = center_coords[i]
                c2 = center_coords[j]
                dist_sq = (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
                min_distance_sq = min(min_distance_sq, dist_sq)

        min_pixel_distance = np.sqrt(min_distance_sq)

        # Calculate expected separation based on orientation spacing (angular separation -> pixel separation)
        # delta_theta = np.radians(orient_spacing) # Angle between spots
        # Using small angle approximation or tan: delta_pos_mm = distance * tan(delta_theta)
        # Better: use half angle if orient_spacing is between different orientations
        # delta_pos_pixels = (distance * np.tan(np.radians(orient_spacing / 2.0)) * 2.0 ) / pixel_size
        # Simpler estimate: pixel separation corresponding to angular separation at distance
        if pixel_size <= 0 or distance <= 0:
             logger.warning("Invalid pixel size or distance for Gaussian width calculation. Using min_pixel_distance.")
             delta_pos = min_pixel_distance # Fallback
        else:
             delta_pos = (distance * np.tan(np.radians(orient_spacing))) / pixel_size

        # Use a fraction (e.g., 0.25) of the *minimum* observed spacing or the *expected* spacing
        # This aims to blur enough to connect parts of a spot without merging adjacent spots
        relevant_distance = min(min_pixel_distance, delta_pos) if delta_pos > 0 else min_pixel_distance
        # Gaussian sigma is related to FWHM. Width is often interpreted as sigma.
        # Let sigma be a fraction of the minimum separation.
        # The original factor was 0.25 * ceil(distance). Let's use sigma = 0.25 * distance.
        blur_sigma = 0.25 * relevant_distance

        # Ensure sigma is at least 1 pixel
        blur_sigma = max(blur_sigma, 1.0)

        # Convert sigma to an integer width parameter if needed (ndimg.gaussian_filter takes sigma directly)
        # If an integer width was needed, ceil(blur_sigma) might be appropriate.
        # Since ndimg takes sigma, we can just use it. Let's return integer ceil for logging consistency.
        blur_width_int = int(np.ceil(blur_sigma))

        logger.info(f"Calculated Gaussian sigma: {blur_sigma:.2f} pixels (min distance: {min_pixel_distance:.1f} px, expected delta: {delta_pos:.1f} px)")
        # Return the sigma value (float) for ndimage.gaussian_filter
        return blur_sigma


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
        """
        Store the contents of specified generated txt files in the H5 file,
        and add their headers as attributes to the corresponding datasets.

        Args:
            output_path: Base path for output files (e.g., 'results/image_001')
            h5_file: Open H5 file handle to store data
        """
        # List of expected txt files and their HDF5 dataset paths
        txt_files_map = {
            f"{output_path}.bin.solutions.txt": '/entry/results/solutions_text',
            f"{output_path}.bin.solutions_filtered.txt": '/entry/results/solutions_filtered_text', # Added filtered
            f"{output_path}.bin.spots.txt": '/entry/results/spots_text',
            f"{output_path}.bin.LaueMatching_stdout.txt": '/entry/logs/stdout',
            f"{output_path}.bin.LaueMatching_stderr.txt": '/entry/logs/stderr',
            f"{output_path}.simulation_stdout.txt": '/entry/logs/simulation_stdout', # Added sim stdout
            f"{output_path}.bin.unique_spot_counts.txt": '/entry/results/unique_spot_counts_text' # Added unique counts text
        }

        # Ensure parent groups exist
        for txt_file_path, dataset_path in txt_files_map.items():
            group_path = os.path.dirname(dataset_path)
            if group_path != '/': # Avoid trying to create root group
                 h5_file.require_group(group_path)

        # Read and store each text file
        for txt_file_path, dataset_path in txt_files_map.items():
            try:
                if os.path.exists(txt_file_path):
                    with open(txt_file_path, 'r') as f:
                        lines = f.readlines()
                        header = lines[0].strip() if lines else ""
                        content = "".join(lines)

                    # Delete existing dataset before creating (safer in append mode)
                    if dataset_path in h5_file:
                         del h5_file[dataset_path]
                    # Create dataset with the content as bytes
                    dataset = h5_file.create_dataset(dataset_path, data=np.bytes_(content))
                    # Add header as an attribute
                    if header:
                        dataset.attrs['header'] = header

                    logger.debug(f"Stored '{txt_file_path}' in H5 dataset '{dataset_path}'")
                # else:
                #     logger.debug(f"Text file not found, skipping H5 storage: {txt_file_path}")

            except Exception as e:
                logger.warning(f"Error storing text file '{txt_file_path}' in H5: {str(e)}")

    def _store_binary_headers_in_h5(self, output_path: str, h5_file) -> None:
        """
        Store the headers from the text files as attributes for corresponding binary datasets.
        Handles both filtered and unfiltered datasets.

        Args:
            output_path: Base path for output files (e.g., 'results/image_001')
            h5_file: Open H5 file handle to store data
        """
        # Map HDF5 datasets to their corresponding text files containing headers
        # Uses the *original* solutions/spots files for headers
        binary_datasets_map = {
            '/entry/results/orientations': f"{output_path}.bin.solutions.txt",
            '/entry/results/filtered_orientations': f"{output_path}.bin.solutions.txt",
            '/entry/results/spots': f"{output_path}.bin.spots.txt",
            '/entry/results/filtered_spots': f"{output_path}.bin.spots.txt",
        }

        # Add headers as attributes
        for dataset_path, header_file_path in binary_datasets_map.items():
            try:
                if dataset_path in h5_file and os.path.exists(header_file_path):
                    with open(header_file_path, 'r') as f:
                        header = f.readline().strip() # Read only the first line

                    if header:
                        dataset = h5_file[dataset_path]
                        dataset.attrs['header'] = header
                        # Split header into column names, handling potential extra spaces
                        columns = [col.strip() for col in header.split() if col.strip()]
                        dataset.attrs['columns'] = columns
                        logger.debug(f"Added header and {len(columns)} columns attribute to {dataset_path}")
                    else:
                         logger.debug(f"Header file '{header_file_path}' was empty. No attributes added to {dataset_path}.")
                # else:
                #      if dataset_path not in h5_file:
                #          logger.debug(f"Dataset not found, skipping header addition: {dataset_path}")
                #      if not os.path.exists(header_file_path):
                #          logger.debug(f"Header file not found, skipping header addition: {header_file_path}")


            except Exception as e:
                logger.warning(f"Error adding header attribute to {dataset_path} using {header_file_path}: {str(e)}")

        # Add header/columns attribute to unique spots dataset
        unique_spots_path = '/entry/results/unique_spots_per_orientation'
        if unique_spots_path in h5_file:
            try:
                unique_spots_dataset = h5_file[unique_spots_path]
                unique_spots_dataset.attrs['header'] = "Grain_Nr Unique_Spots"
                unique_spots_dataset.attrs['columns'] = ['Grain_Nr', 'Unique_Spots']
                logger.debug("Added header attribute and columns to unique spots dataset")
            except Exception as e:
                logger.warning(f"Error adding header attributes to {unique_spots_path}: {str(e)}")

        # Add header/columns attribute to simulated spots dataset
        simulated_spots_path = '/entry/simulation/simulated_spots'
        if simulated_spots_path in h5_file:
            try:
                simulated_spots_dataset = h5_file[simulated_spots_path]
                # Add header/columns only if they don't exist
                if 'header' not in simulated_spots_dataset.attrs:
                    simulated_spots_dataset.attrs['header'] = "X Y GrainID Matched H K L Energy"
                    simulated_spots_dataset.attrs['columns'] = ['X', 'Y', 'GrainID', 'Matched', 'H', 'K', 'L', 'Energy']
                    logger.debug("Added header attribute and columns to simulated spots dataset")
                # else:
                #     logger.debug("Header attribute already exists for simulated spots dataset.")
            except Exception as e:
                logger.warning(f"Error adding header attributes to {simulated_spots_path}: {str(e)}")


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

        # Find executable path relative to script location
        script_dir = os.path.dirname(os.path.realpath(__file__))
        build_dir = os.path.join(script_dir, 'bin')

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
        lib_path_nlopt = os.path.join(script_dir, 'LIBS', 'NLOPT', 'lib')
        lib_path_nlopt64 = os.path.join(script_dir, 'LIBS', 'NLOPT', 'lib64')
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
            sim_script_path = os.path.join(INSTALL_PATH, 'GenerateSimulation.py')
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
        indexed_orientations: np.ndarray, # Filtered orientations
        simulated_spots: np.ndarray,
        simulated_image: np.ndarray,
        filtered_exp_image: np.ndarray # Filtered thresholded experimental image
    ) -> None:
        """
        Create visualization comparing simulated and matched spots.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            indexed_orientations: Filtered indexed orientation data array
            simulated_spots: Simulated diffraction spots array from GenerateSimulation.py
            simulated_image: Simulated diffraction image array from GenerateSimulation.py
            filtered_exp_image: Filtered thresholded experimental image
        """
        logger.info("Creating simulation comparison visualization")

        # Load experimental spots corresponding to the *filtered* orientations
        spots_file_path = f'{output_path}.bin.spots.txt'
        try:
            all_exp_spots = np.genfromtxt(spots_file_path, skip_header=1)
            if len(all_exp_spots.shape) == 1 and all_exp_spots.size > 0: # Handle single spot case
                 all_exp_spots = np.expand_dims(all_exp_spots, axis=0)
            elif all_exp_spots.size == 0:
                 all_exp_spots = np.empty((0, 8)) # Assume 8 columns if empty

            # Filter these spots to only include those from the indexed_orientations
            kept_grain_nrs = set(indexed_orientations[:, 0].astype(int)) if indexed_orientations.size > 0 else set()
            if all_exp_spots.size > 0:
                 exp_spots = all_exp_spots[np.isin(all_exp_spots[:, 0].astype(int), list(kept_grain_nrs))]
            else:
                 exp_spots = all_exp_spots # Keep empty array if it was empty

        except Exception as e:
            logger.error(f"Error loading experimental spots from '{spots_file_path}': {str(e)}")
            return

        # --- Basic Checks ---
        if indexed_orientations.size == 0:
            logger.warning("No indexed orientations provided for simulation comparison. Skipping visualization.")
            return
        if simulated_spots.size == 0:
             logger.warning("No simulated spots provided for comparison. Visualization may be incomplete.")
        if simulated_image.size == 0:
             logger.warning("No simulated image provided for comparison. Visualization may be incomplete.")
        if filtered_exp_image.size == 0:
             logger.warning("No experimental image provided for comparison. Visualization may be incomplete.")


        # --- Create Plotly Figure ---
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Experimental Spots", "Simulated Spots", "Overlay & Missing"],
            horizontal_spacing=0.05,
            specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
            shared_xaxes=True, shared_yaxes=True
        )

        # --- Prepare Experimental Image Display ---
        display_exp_img = filtered_exp_image.copy().astype(float)
        display_exp_img[display_exp_img <= 0] = 1 # Avoid log(0) or log(negative)
        log_exp_img = np.log(display_exp_img)

        # Add experimental image heatmap to all subplots as background/overlay
        fig.add_trace(go.Heatmap(z=log_exp_img, colorscale='Greens', showscale=False, name='Experimental'), row=1, col=1)
        # Add simulated image heatmap
        if simulated_image.size > 0:
             fig.add_trace(go.Heatmap(z=simulated_image, colorscale='Reds', showscale=False, name='Simulated', opacity=0.7), row=1, col=2)
        # Add gray overlay for third plot
        fig.add_trace(go.Heatmap(z=log_exp_img, colorscale='gray', showscale=False, name='Exp Overlay', opacity=0.3), row=1, col=3)


        # --- Plot Spots ---
        orientation_colors = px.colors.qualitative.Plotly # Use a standard Plotly palette
        if len(indexed_orientations.shape) == 1: # Ensure 2D
            indexed_orientations = np.expand_dims(indexed_orientations, axis=0)

        missing_spots_dict = {} # Store missing simulated spots per grain
        matched_exp_spots_dict = {} # Store matched experimental spot positions per grain

        # Plot spots for each orientation
        for i, orientation in enumerate(indexed_orientations):
            grain_nr = int(orientation[0])
            color = orientation_colors[i % len(orientation_colors)]

            # --- Plot Experimental Spots for this Grain ---
            grain_exp_spots = exp_spots[exp_spots[:, 0] == grain_nr]
            if grain_exp_spots.size > 0:
                exp_trace = go.Scatter(
                    x=grain_exp_spots[:, 5], y=grain_exp_spots[:, 6],
                    mode='markers', name=f"Exp Grain {grain_nr}", legendgroup=f'grain_{grain_nr}',
                    marker=dict(color=color, size=8, symbol='circle-open', line=dict(width=2, color=color)),
                    hovertext=[f"Grain: {grain_nr}<br>HKL: ({int(s[2])},{int(s[3])},{int(s[4])})<br>Pos: ({s[5]:.1f}, {s[6]:.1f})<br>Source: Exp" for s in grain_exp_spots],
                    hoverinfo="text"
                )
                fig.add_trace(exp_trace, row=1, col=1)
                fig.add_trace(go.Scatter(exp_trace), row=1, col=3) # Add to overlay plot too

                # Store positions for matching later
                matched_exp_spots_dict[grain_nr] = set((round(float(s[5])), round(float(s[6]))) for s in grain_exp_spots)


            # --- Plot Simulated Spots for this Grain ---
            # Check if simulation uses grain numbers or indices (heuristic)
            use_grain_numbers = simulated_spots.size > 0 and np.max(simulated_spots[:, 2]) >= len(indexed_orientations)

            if use_grain_numbers:
                grain_sim_spots = simulated_spots[simulated_spots[:, 2] == grain_nr]
            else:
                grain_sim_spots = simulated_spots[simulated_spots[:, 2] == i] # Use index i

            # Remove positional duplicates from simulated spots for this grain
            unique_sim_spots_for_grain = []
            seen_sim_positions = set()
            if grain_sim_spots.size > 0:
                  for spot in grain_sim_spots:
                       # GenerateSimulation.py output: Y=col0, X=col1, GrainID=col2, Matched=col3
                       pos_key = (round(float(spot[1])), round(float(spot[0]))) # X, Y
                       if pos_key not in seen_sim_positions:
                            unique_sim_spots_for_grain.append(spot)
                            seen_sim_positions.add(pos_key)

            if unique_sim_spots_for_grain:
                unique_spots_array = np.array(unique_sim_spots_for_grain)
                sim_trace = go.Scatter(
                    x=unique_spots_array[:, 1], y=unique_spots_array[:, 0], # X=col1, Y=col0
                    mode='markers', name=f"Sim Grain {grain_nr}", legendgroup=f'grain_{grain_nr}',
                    marker=dict(color=color, size=8, symbol='x', line=dict(width=2, color=color)),
                    hovertext=[f"Grain: {grain_nr}<br>Pos: ({s[1]:.1f}, {s[0]:.1f})<br>Source: Sim" for s in unique_spots_array],
                    hoverinfo="text"
                )
                fig.add_trace(sim_trace, row=1, col=2)
                fig.add_trace(go.Scatter(sim_trace), row=1, col=3) # Add to overlay plot too

                # --- Find Missing Simulated Spots ---
                missing_spots_dict[grain_nr] = []
                proximity_threshold = 2.0 # Pixels
                exp_positions = matched_exp_spots_dict.get(grain_nr, set())

                for spot in unique_spots_array:
                    sim_pos = (round(float(spot[1])), round(float(spot[0]))) # X=col1, Y=col0
                    is_matched = False
                    min_dist_sq = float('inf')
                    # Check proximity to any experimental spot for this grain
                    for exp_pos in exp_positions:
                         dist_sq = (sim_pos[0] - exp_pos[0])**2 + (sim_pos[1] - exp_pos[1])**2
                         if dist_sq < proximity_threshold**2:
                              is_matched = True
                              break
                         min_dist_sq = min(min_dist_sq, dist_sq)

                    if not is_matched:
                        missing_spots_dict[grain_nr].append(spot)


        # --- Plot Missing Spots ---
        for grain_nr, missing_list in missing_spots_dict.items():
            if not missing_list: continue
            missing_array = np.array(missing_list)
            # Find original index for color consistency
            try:
                 grain_idx = list(indexed_orientations[:, 0].astype(int)).index(grain_nr)
                 color = orientation_colors[grain_idx % len(orientation_colors)]
            except ValueError:
                 color = 'grey' # Fallback color

            missing_trace = go.Scatter(
                 x=missing_array[:, 1], y=missing_array[:, 0], # X=col1, Y=col0
                 mode='markers', name=f"Missing Sim {grain_nr}", legendgroup=f'grain_{grain_nr}',
                 marker=dict(color=color, size=10, symbol='diamond-open', line=dict(width=2, color='black')),
                 hovertext=[f"Grain: {grain_nr}<br>Pos: ({s[1]:.1f}, {s[0]:.1f})<br>Source: Missing Sim" for s in missing_array],
                 hoverinfo="text",
                 showlegend=False # Avoid cluttering legend too much
            )
            fig.add_trace(missing_trace, row=1, col=3)
            logger.info(f"Grain {grain_nr}: Found {len(missing_list)} simulated spots missing experimentally.")


        # --- Final Layout Updates ---
        fig.update_layout(
            title="Experimental vs. Simulated Diffraction Comparison",
            height=700, width=1800,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01, bordercolor="Black", borderwidth=1),
            margin=dict(l=50, r=200, b=50, t=50) # Adjust margins
        )

        img_shape = filtered_exp_image.shape # Y, X
        y_max, x_max = img_shape[0], img_shape[1]

        for col in range(1, 4):
             fig.update_xaxes(title_text="X Position (pixels)", row=1, col=col, range=[0, x_max], constrain='domain')
             fig.update_yaxes(title_text="Y Position (pixels)", row=1, col=col, range=[y_max, 0], constrain='domain') # Invert Y axis for image convention

        fig.update_layout(hovermode='closest') # Removed dragmode='zoom' for now

        # --- Save Visualization ---
        html_file = f"{output_path}.simulation_comparison.html"
        html_standalone_file = f"{output_path}.simulation_comparison_standalone.html"
        try:
             fig.write_html(html_file, include_plotlyjs='cdn')
             logger.info(f"Simulation comparison visualization saved to {html_file}")
             # Save standalone version
             # fig.write_html(html_standalone_file, include_plotlyjs=True, full_html=True) # Requires more disk space
             # logger.info(f"Standalone simulation comparison saved to {html_standalone_file}")
        except Exception as e:
             logger.error(f"Could not save simulation comparison visualization: {str(e)}")

        # --- Save Unique Spot Counts (based on experimental data) ---
        # This calculation seems slightly redundant if done in _process_results, but recalculate here for safety
        unique_exp_spot_counts = {}
        if exp_spots.size > 0:
             for grain_nr_val in kept_grain_nrs:
                  grain_spots_data = exp_spots[exp_spots[:, 0] == grain_nr_val]
                  unique_positions = set()
                  if grain_spots_data.size > 0:
                       unique_positions = set((int(s[5]), int(s[6])) for s in grain_spots_data)
                  unique_exp_spot_counts[grain_nr_val] = len(unique_positions)

        unique_counts_file = f"{output_path}.unique_spot_counts.txt"
        try:
            with open(unique_counts_file, 'w') as f:
                f.write("Grain_Nr\tUnique_Experimental_Spots\n")
                for grain_nr_val, count in sorted(unique_exp_spot_counts.items()):
                    f.write(f"{grain_nr_val}\t{count}\n")
            logger.info(f"Unique experimental spot counts saved to {unique_counts_file}")
        except Exception as e:
             logger.error(f"Could not save unique spot counts file: {e}")


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
        genhkl_script = os.path.join(INSTALL_PATH, 'GenerateHKLs.py')
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
        labels: np.ndarray # Should be the final labels used for segmentation (e.g., watershed)
    ) -> Dict[int, Dict]:
        """
        Calculate the number of unique spots/labels for each orientation,
        prioritizing assignments based on orientation quality score.

        Args:
            orientations: Orientation data array (potentially unsorted)
            spots: Spot data array
            labels: Label image from segmentation (e.g., watershed output)

        Returns:
            Dictionary: { grain_nr: { "spots": set((x,y)), "count": int,
                                      "total_intensity": float, "unique_labels": set(int),
                                      "unique_label_count": int } }
        """
        unique_spots_results = {}
        if orientations.size == 0:
            return unique_spots_results # Return empty if no orientations

        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)

        # Sort orientations by quality score (column 4) descending
        # Store original index along with orientation and quality
        orient_with_quality = [(i, orient, orient[4]) for i, orient in enumerate(orientations)]
        orient_with_quality.sort(key=lambda x: x[2], reverse=True) # Sort by quality descending

        # Keep track of labels and pixel positions already assigned to a higher-quality orientation
        assigned_labels = set()
        assigned_positions = set()

        # Process spots for each orientation in quality order
        for _, orientation, quality in orient_with_quality: # Use sorted list
            grain_nr = int(orientation[0])

            # Initialize result for this grain
            unique_spots_results[grain_nr] = {
                "spots": set(), "count": 0, "total_intensity": 0.0,
                "unique_labels": set(), "unique_label_count": 0
            }

            # Find all spots belonging to this grain number
            orientation_spots = spots[spots[:, 0] == grain_nr]
            if orientation_spots.size == 0:
                continue # Skip if no spots for this grain

            current_grain_unique_positions = set()
            current_grain_unique_labels = set()
            current_grain_total_intensity = 0.0

            # Iterate through spots associated with this orientation
            for spot in orientation_spots:
                # Get spot pixel coordinates (assuming col 5=x, col 6=y)
                try:
                    x, y = int(spot[5]), int(spot[6])
                    pos_tuple = (x, y)
                except (IndexError, ValueError):
                     logger.warning(f"Could not parse position for spot in grain {grain_nr}. Skipping spot.")
                     continue

                # --- Check if position or label is already assigned ---
                # 1. Check position: If already assigned to higher quality grain, skip
                if pos_tuple in assigned_positions:
                    continue

                # 2. Get label at this position from the final segmentation map
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                    label_at_spot = labels[y, x]
                    # Check if this label (if valid > 0) is already assigned
                    if label_at_spot > 0 and label_at_spot in assigned_labels:
                        continue # Skip spot if its label belongs to a better orientation

                    # --- Assign spot/label to current orientation ---
                    current_grain_unique_positions.add(pos_tuple)
                    assigned_positions.add(pos_tuple) # Mark position as assigned

                    if label_at_spot > 0:
                         current_grain_unique_labels.add(label_at_spot)
                         assigned_labels.add(label_at_spot) # Mark label as assigned

                    # Add intensity (assuming column 7)
                    try:
                         current_grain_total_intensity += float(spot[7])
                    except (IndexError, ValueError):
                         current_grain_total_intensity += 0.0 # Or handle missing intensity

                else:
                    # Spot position is outside label image bounds - assign position but not label
                    logger.warning(f"Spot at ({x},{y}) for grain {grain_nr} is outside label image bounds ({labels.shape}).")
                    current_grain_unique_positions.add(pos_tuple)
                    assigned_positions.add(pos_tuple)
                    # Cannot assign a label

            # Store results for this grain
            unique_spots_results[grain_nr]["spots"] = current_grain_unique_positions
            unique_spots_results[grain_nr]["count"] = len(current_grain_unique_positions)
            unique_spots_results[grain_nr]["total_intensity"] = current_grain_total_intensity
            unique_spots_results[grain_nr]["unique_labels"] = current_grain_unique_labels
            unique_spots_results[grain_nr]["unique_label_count"] = len(current_grain_unique_labels)

        logger.info(f"Calculated unique spots/labels for {len(unique_spots_results)} orientations based on quality.")
        return unique_spots_results


    def _sort_orientations_by_quality(
        self,
        orientations: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None # Optional info
    ) -> np.ndarray:
        """
        Sort orientations primarily by quality score (column 4) descending.
        Original GrainNr (column 0) is preserved with the sorted data.

        Args:
            orientations: Orientation data array
            orientation_unique_spots: Optional dictionary of unique spots per orientation (not used for sorting itself)

        Returns:
            Sorted orientation data array (copy)
        """
        if orientations.size == 0 or len(orientations.shape) == 1:
            return orientations # Return as is if empty or 1D

        # Create tuples of (original_index, quality_score)
        quality_scores = []
        for i, orientation in enumerate(orientations):
            try:
                quality_score = orientation[4] # Assuming column 4 is quality
                quality_scores.append((i, quality_score))
            except IndexError:
                logger.warning(f"Could not access quality score (column 4) for orientation index {i}. Assigning score 0.")
                quality_scores.append((i, 0.0))

        # Sort indices based on quality score (descending)
        sorted_indices = [idx for idx, score in sorted(quality_scores, key=lambda x: x[1], reverse=True)]

        # Create sorted array using the sorted indices
        sorted_orientations = orientations[sorted_indices].copy()

        logger.info(f"Sorted {len(sorted_orientations)} orientations by quality score (column 4) descending.")
        return sorted_orientations


    def _create_h5_output(
        self,
        output_path: str, # Base path, e.g. results/image_001
        orientations_unfiltered: np.ndarray,
        filtered_orientations: np.ndarray,
        spots_unfiltered: np.ndarray,
        filtered_spots: np.ndarray,
        orientation_unique_spots: Dict[int, Dict]
    ) -> None:
        """
        Create/Update HDF5 file with orientation and spot data (unfiltered and filtered).
        Also stores unique spot counts per orientation.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            orientations_unfiltered: Original sorted orientation data array
            filtered_orientations: Filtered orientation data array (>= min unique spots)
            spots_unfiltered: Original spot data array
            filtered_spots: Filtered spot data array (corresponding to filtered orientations)
            orientation_unique_spots: Dictionary of unique spots per orientation
        """
        output_h5 = f"{output_path}.output.h5" # Use consistent naming

        # Create unique spot count array: [Grain_Nr, Unique_Label_Count]
        unique_counts_list = []
        if orientation_unique_spots:
            for grain_nr, data in orientation_unique_spots.items():
                 unique_counts_list.append([grain_nr, data.get("unique_label_count", 0)])
        unique_counts_array = np.array(unique_counts_list, dtype=np.int32) if unique_counts_list else np.empty((0, 2), dtype=np.int32)

        try:
            with h5py.File(output_h5, 'a') as hf: # Open in append mode ('a')
                results_group = hf.require_group('/entry/results')

                # Define datasets to save
                datasets_to_save = {
                    'orientations': orientations_unfiltered,
                    'filtered_orientations': filtered_orientations,
                    'spots': spots_unfiltered,
                    'filtered_spots': filtered_spots,
                    'unique_spots_per_orientation': unique_counts_array
                }

                # Save each dataset, overwriting if it exists
                for name, data in datasets_to_save.items():
                     dataset_path = f'/entry/results/{name}'
                     if dataset_path in hf:
                          del hf[dataset_path]
                     hf.create_dataset(dataset_path, data=data)

                logger.info(f"Saved/Updated orientation and spot data in {output_h5}")

                # Add headers/columns attributes using the separate method
                self._store_binary_headers_in_h5(output_path, hf)

        except Exception as e:
            logger.error(f"Error creating/updating H5 output in '{output_h5}': {str(e)}")


    def _visualize_results(
        self,
        output_path: str, # Base path, e.g. results/image_001
        orientations: np.ndarray, # Filtered orientations
        spots: np.ndarray, # Filtered spots
        labels: np.ndarray, # Final segmentation labels
        filtered_image: np.ndarray, # Filtered thresholded image
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate various visualizations based on the filtered indexing results.

        Args:
            output_path: Base path for output files (e.g., results/image_001)
            orientations: Filtered orientation data array
            spots: Filtered spot data array
            labels: Final segmentation labels array
            filtered_image: Filtered thresholded image used for spot finding
            orientation_unique_spots: Dictionary of unique spots per orientation (for context)

        Returns:
            Dictionary with visualization success status
        """
        vis_config = self.config.get("visualization")
        if not vis_config:
            logger.warning("Visualization configuration missing, cannot generate plots.")
            return {"success": False, "error": "Visualization config missing"}

        plot_type = vis_config.plot_type
        success_flag = True # Assume success unless something fails

        # --- Static Visualization (TIF/PNG) ---
        if plot_type in ["static", "both"]:
            static_result = self._create_static_visualization(
                output_path, orientations, spots, labels, filtered_image, orientation_unique_spots
            )
            if not static_result["success"]:
                 logger.warning(f"Failed to create static visualization: {static_result.get('error', 'Unknown')}")
                 success_flag = False


        # --- Interactive Visualization (HTML) ---
        if plot_type in ["interactive", "both"]:
            interactive_result = self._create_interactive_visualization(
                output_path, orientations, spots, labels, filtered_image, orientation_unique_spots
            )
            if not interactive_result["success"]:
                logger.warning(f"Failed to create interactive visualization: {interactive_result.get('error', 'Unknown')}")
                success_flag = False

        # --- 3D Orientation Visualization (HTML) ---
        if vis_config.generate_3d:
            try:
                logger.info("Generating 3D orientation visualization")
                self._create_3d_visualization(output_path, orientations, spots)
            except Exception as e:
                logger.warning(f"Failed to create 3D visualization: {str(e)}")
                # Don't mark overall success as false for optional plot failure

        # --- Analysis Report (HTML) ---
        if vis_config.generate_report:
            try:
                logger.info("Generating analysis report")
                self._create_analysis_report(
                    output_path, orientations, spots, labels, filtered_image, orientation_unique_spots
                )
            except Exception as e:
                logger.warning(f"Failed to create analysis report: {str(e)}")
                # Don't mark overall success as false for optional report failure

        return {"success": success_flag}


    def _create_static_visualization(
        self,
        output_path: str, # Base path, e.g. results/image_001
        orientations: np.ndarray, # Filtered
        spots: np.ndarray, # Filtered
        labels: np.ndarray,
        filtered_image: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Create static visualizations (PNG/TIF) of the filtered indexing results.

        Returns: Dictionary with success status.
        """
        logger.info("Creating static visualizations...")
        try:
            # --- Plot 1: Labeled Image ---
            fig_label, ax_label = plt.subplots(figsize=(self.scalarX, self.scalarY))

            # Display background image (log of intensity)
            display_img = filtered_image.copy().astype(float)
            display_img[display_img <= 0] = 1 # Avoid log(0)
            ax_label.imshow(np.log(display_img), cmap='Greens', origin='upper') # Use origin='upper' for standard image display

            # Get colormap
            vis_config = self.config.get("visualization")
            colormap_name = vis_config.colormap if vis_config else "nipy_spectral"
            num_orientations = len(orientations)
            if num_orientations > 0:
                 colors = plt.get_cmap(colormap_name, num_orientations)
            else:
                 colors = None # No colors needed if no orientations

            # Plot spots for each *filtered* orientation
            if orientations.size > 0 and spots.size > 0 and colors:
                 # We pass filtered orientations/spots, so plotting logic uses them directly
                 self._plot_orientation_spots(orientations, spots, filtered_image, labels, colors, ax_label)

            ax_label.set_title(f"Indexed Spots ({num_orientations} Orientations)")
            ax_label.set_xlabel("X Pixel")
            ax_label.set_ylabel("Y Pixel")
            if num_orientations > 0 :
                 ax_label.legend(loc='upper right', fontsize='xx-small', markerscale=2) # Increase marker size in legend
            plt.tight_layout()

            # Save figure
            dpi = vis_config.output_dpi if vis_config else 600
            png_file = f'{output_path}.LabeledImage.png'
            tif_file = f'{output_path}.LabeledImage.tif'
            plt.savefig(png_file, dpi=dpi)
            # plt.savefig(tif_file, dpi=dpi) # TIF saving might require extra libraries (e.g., tifffile)
            plt.close(fig_label)
            logger.info(f"Static labeled visualization saved to {png_file}") # and {tif_file}")

            # --- Plot 2: Quality Map ---
            if orientations.size > 0 and spots.size > 0:
                 self._create_quality_map(output_path, orientations, spots, filtered_image, orientation_unique_spots)

            return {"success": True}

        except Exception as e:
            logger.error(f"Error creating static visualization: {str(e)}", exc_info=True)
            # Ensure plot is closed if error occurred
            if 'fig_label' in locals() and plt.fignum_exists(fig_label.number): plt.close(fig_label)
            return {"success": False, "error": str(e)}


    def _create_quality_map(
        self,
        output_path: str, # Base path
        orientations: np.ndarray, # Filtered
        spots: np.ndarray, # Filtered
        filtered_image: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> None:
        """
        Create a quality map visualization based on filtered results.
        """
        logger.debug("Creating quality map...")
        vis_config = self.config.get("visualization")
        dpi = vis_config.output_dpi if vis_config else 600

        # Setup figure
        fig_qual, ax_qual = plt.subplots(figsize=(self.scalarX, self.scalarY))

        # Create empty quality map
        quality_map = np.zeros_like(filtered_image, dtype=float)

        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)

        # Populate quality map
        max_quality_recorded = 0.0
        for orientation in orientations:
            grain_nr = int(orientation[0])
            quality_score = orientation[4] # Column 4: Quality score from executable
            unique_count = 0
            if orientation_unique_spots and grain_nr in orientation_unique_spots:
                 unique_count = orientation_unique_spots[grain_nr].get("unique_label_count", 0)

            # Define enhanced quality (e.g., boost by unique spots)
            # Simple boost: quality * (1 + factor * unique_count)
            enhanced_quality = quality_score * (1.0 + 0.1 * unique_count)
            max_quality_recorded = max(max_quality_recorded, enhanced_quality)

            # Get spots for this orientation (from the filtered spots array)
            orientation_spots = spots[spots[:, 0] == grain_nr]

            for spot in orientation_spots:
                try:
                    x, y = int(spot[5]), int(spot[6])
                    if 0 <= y < quality_map.shape[0] and 0 <= x < quality_map.shape[1]:
                        # Assign the maximum quality score found at this pixel
                        quality_map[y, x] = max(quality_map[y, x], enhanced_quality)
                except (IndexError, ValueError):
                     continue # Skip if spot data is malformed


        # Apply Gaussian blur to smooth the map
        if np.any(quality_map > 0): # Only blur if there's data
             quality_map_blurred = ndimg.gaussian_filter(quality_map, sigma=3)
        else:
             quality_map_blurred = quality_map # Keep as zeros if no data

        # Display quality map
        im = ax_qual.imshow(quality_map_blurred, cmap='viridis', origin='upper', vmin=0, vmax=max_quality_recorded if max_quality_recorded > 0 else 1) # Set vmax
        plt.colorbar(im, ax=ax_qual, label='Indexing Quality (Score * Unique Spot Boost)')
        ax_qual.set_title('Orientation Indexing Quality Map')
        ax_qual.set_xlabel("X Pixel")
        ax_qual.set_ylabel("Y Pixel")
        plt.tight_layout()

        # Save figure
        quality_map_file = f'{output_path}.QualityMap.png'
        plt.savefig(quality_map_file, dpi=dpi)
        plt.close(fig_qual)
        logger.info(f"Quality map saved to {quality_map_file}")


    def _create_interactive_visualization(
        self,
        output_path: str, # Base path
        orientations: np.ndarray, # Filtered
        spots: np.ndarray, # Filtered
        labels: np.ndarray,
        filtered_image: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Create interactive visualization using Plotly based on filtered results.

        Returns: Dictionary with success status.
        """
        logger.info("Creating interactive visualization...")
        vis_config = self.config.get("visualization")

        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Indexed Diffraction Pattern", "Orientation Quality"],
                horizontal_spacing=0.05, # Reduced spacing
                specs=[[{"type": "xy"}, {"type": "xy"}]],
                shared_xaxes=True, shared_yaxes=True
            )

            # --- Background Image ---
            display_img = filtered_image.copy().astype(float)
            display_img[display_img <= 0] = 1 # Avoid log(0)
            log_img = np.log(display_img)
            fig.add_trace(go.Heatmap(z=log_img, colorscale='Greens', showscale=False, name='Log Intensity'), row=1, col=1)

            # --- Quality Map ---
            quality_map = np.zeros_like(filtered_image, dtype=float)
            max_quality_recorded = 0.0
            if orientations.size > 0: # Ensure orientations is not empty
                 if len(orientations.shape) == 1: # Ensure 2D
                     orientations_2d = np.expand_dims(orientations, axis=0)
                 else:
                     orientations_2d = orientations

                 for orientation in orientations_2d:
                     grain_nr = int(orientation[0])
                     quality_score = orientation[4]
                     unique_count = 0
                     if orientation_unique_spots and grain_nr in orientation_unique_spots:
                          unique_count = orientation_unique_spots[grain_nr].get("unique_label_count", 0)
                     enhanced_quality = quality_score * (1.0 + 0.1 * unique_count)
                     max_quality_recorded = max(max_quality_recorded, enhanced_quality)

                     orientation_spots = spots[spots[:, 0] == grain_nr]
                     for spot in orientation_spots:
                          try:
                              x, y = int(spot[5]), int(spot[6])
                              if 0 <= y < quality_map.shape[0] and 0 <= x < quality_map.shape[1]:
                                   quality_map[y, x] = max(quality_map[y, x], enhanced_quality)
                          except (IndexError, ValueError): continue

            # Blur quality map
            quality_map_blurred = ndimg.gaussian_filter(quality_map, sigma=3) if np.any(quality_map > 0) else quality_map

            fig.add_trace(go.Heatmap(
                 z=quality_map_blurred, colorscale='Viridis', showscale=True, name='Quality',
                 colorbar=dict(title="Quality", x=0.46, y=0.5, len=0.9, thickness=15), # Adjusted position
                 zmin=0, zmax=max_quality_recorded if max_quality_recorded > 0 else 1 # Set color range
                 ), row=1, col=2)


            # --- Plot Spots ---
            orientation_colors = px.colors.qualitative.Plotly # Use standard palette
            num_orientations = len(orientations)

            if orientations.size > 0 and spots.size > 0 :
                if len(orientations.shape) == 1: # Ensure 2D
                     orientations_plot = np.expand_dims(orientations, axis=0)
                else:
                     orientations_plot = orientations

                for i, orientation in enumerate(orientations_plot):
                    grain_nr = int(orientation[0])
                    color = orientation_colors[i % len(orientation_colors)]
                    unique_count = 0
                    if orientation_unique_spots and grain_nr in orientation_unique_spots:
                         unique_count = orientation_unique_spots[grain_nr].get("unique_label_count", 0)

                    orientation_spots = spots[spots[:, 0] == grain_nr]
                    if orientation_spots.size == 0: continue

                    trace = go.Scatter(
                        x=orientation_spots[:, 5], y=orientation_spots[:, 6],
                        mode='markers', name=f"Grain {grain_nr} ({unique_count} unique)",
                        legendgroup=f'grain_{grain_nr}', # Group legend items
                        marker=dict(color=color, size=7, symbol='circle-open', line=dict(width=1.5)),
                        hovertext=[f"Grain: {grain_nr}<br>HKL: ({int(s[2])},{int(s[3])},{int(s[4])})<br>Pos: ({s[5]:.1f}, {s[6]:.1f})<br>Unique Count: {unique_count}" for s in orientation_spots],
                        hoverinfo="text"
                    )
                    fig.add_trace(trace, row=1, col=1) # Add spots only to the first plot


            # --- Layout ---
            img_shape = filtered_image.shape # Y, X
            y_max, x_max = img_shape[0], img_shape[1]
            fig.update_layout(
                title="Laue Diffraction Analysis (Filtered Results)",
                height=700, width=1600, # Adjusted width
                showlegend=True,
                legend=dict( orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
                             bordercolor="Black", borderwidth=1, font=dict(size=10), title=dict(text="Grains") ),
                margin=dict(l=50, r=150, b=50, t=50), # Adjusted right margin for legend
                hovermode='closest'
            )
            # Set axes range and direction (inverted Y for image convention)
            fig.update_xaxes(range=[0, x_max], constrain='domain')
            fig.update_yaxes(range=[y_max, 0], constrain='domain')
            # Add titles to axes
            fig.update_xaxes(title_text="X Pixel", row=1, col=1)
            fig.update_yaxes(title_text="Y Pixel", row=1, col=1)
            fig.update_xaxes(title_text="X Pixel", row=1, col=2)
            fig.update_yaxes(title_text="Y Pixel", row=1, col=2)


            # --- Save HTML ---
            html_file = f"{output_path}.interactive.html"
            fig.write_html(html_file, include_plotlyjs='cdn')
            logger.info(f"Interactive visualization saved to {html_file}")

            return {"success": True}

        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}


    def _create_3d_visualization(
        self,
        output_path: str, # Base path
        orientations: np.ndarray, # Filtered
        spots: np.ndarray # Filtered (not used here, but kept for signature consistency)
    ) -> None:
        """
        Create 3D visualization of filtered crystal orientations using Plotly.
        """
        logger.debug("Creating 3D orientation visualization...")

        if orientations.size == 0:
            logger.warning("No orientations found, skipping 3D visualization")
            return

        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        # Plot orientation vectors for each grain
        for i, orientation in enumerate(orientations):
            grain_nr = int(orientation[0])
            color = colors[i % len(colors)]

            # Extract orientation matrix (columns 22-30)
            try:
                 matrix = orientation[22:31].reshape(3, 3)
                 r1 = matrix[0, :] # Crystal X in lab frame
                 r2 = matrix[1, :] # Crystal Y in lab frame
                 r3 = matrix[2, :] # Crystal Z in lab frame
            except (IndexError, ValueError):
                 logger.warning(f"Could not extract 3x3 matrix for Grain {grain_nr}. Skipping 3D plot.")
                 continue

            origin = [0, 0, 0]
            scale = 1.0 # Length of axes vectors

            # Add crystal axes traces
            axes_data = {'X': r1, 'Y': r2, 'Z': r3}
            axes_colors = {'X': 'red', 'Y': 'green', 'Z': 'blue'}
            for axis, vector in axes_data.items():
                 fig.add_trace(go.Scatter3d(
                      x=[origin[0], origin[0] + scale * vector[0]],
                      y=[origin[1], origin[1] + scale * vector[1]],
                      z=[origin[2], origin[2] + scale * vector[2]],
                      mode='lines+markers', name=f"Grain {grain_nr} {axis}-axis", legendgroup=f'grain_{grain_nr}',
                      line=dict(color=axes_colors[axis], width=5), marker=dict(size=3, color=axes_colors[axis])
                 ))

            # Optional: Add a small representation of the unit cell (cube for simplicity)
            cell_scale = 0.2 * scale
            corners = [np.array(origin), r1*cell_scale, r2*cell_scale, r3*cell_scale,
                      (r1+r2)*cell_scale, (r1+r3)*cell_scale, (r2+r3)*cell_scale, (r1+r2+r3)*cell_scale]
            edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
            for p1_idx, p2_idx in edges:
                 p1, p2 = corners[p1_idx], corners[p2_idx]
                 fig.add_trace(go.Scatter3d(
                      x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                      mode='lines', line=dict(color=color, width=1.5), showlegend=False
                 ))


        # Update layout
        fig.update_layout(
            title="3D Visualization of Crystal Orientations (Filtered)",
            scene=dict(xaxis_title="Lab X", yaxis_title="Lab Y", zaxis_title="Lab Z", aspectmode='data'), # Use 'data' aspect
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)), # Adjust camera angle
            width=800, height=800,
            legend=dict(x=1.05, y=0.5, font=dict(size=10), bordercolor="Black", borderwidth=1),
            margin=dict(l=10, r=150, b=10, t=40) # Adjust margins
        )

        # Save as HTML
        html_3d_file = f"{output_path}.3D.html"
        try:
             fig.write_html(html_3d_file, include_plotlyjs='cdn')
             logger.info(f"3D visualization saved to {html_3d_file}")
        except Exception as e:
             logger.error(f"Could not save 3D visualization: {e}")


    def _create_analysis_report(
        self,
        output_path: str, # Base path
        orientations: np.ndarray, # Filtered
        spots: np.ndarray, # Filtered
        labels: np.ndarray,
        filtered_image: np.ndarray,
        orientation_unique_spots: Optional[Dict[int, Dict]] = None
    ) -> None:
        """
        Create comprehensive HTML analysis report based on filtered results.
        """
        logger.info("Generating analysis report...")
        vis_config = self.config.get("visualization")
        if not vis_config:
             logger.warning("Visualization config missing, cannot generate report.")
             return

        if orientations.size == 0:
            logger.warning("No orientations found (after filtering), skipping analysis report generation.")
            return

        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)

        report_path = f"{output_path}.report.html"
        template = vis_config.report_template # Currently only 'default' is implemented

        # --- Gather Report Data ---
        num_orientations = len(orientations)
        num_spots = len(spots)
        avg_spots = f"{num_spots / num_orientations:.1f}" if num_orientations > 0 else "N/A"
        image_basename = os.path.basename(output_path)

        # --- Start HTML ---
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

        # Add table rows for each orientation
        for i, orientation in enumerate(orientations):
            grain_nr = int(orientation[0])
            quality = orientation[4]
            num_spots_total = int(orientation[5]) # Total spots matched by executable
            unique_spots = 0
            if orientation_unique_spots and grain_nr in orientation_unique_spots:
                 unique_spots = orientation_unique_spots[grain_nr].get("unique_label_count", 0)

            # Format matrix
            matrix_html = "<span class='matrix-table'><table>"
            try:
                matrix = orientation[22:31].reshape(3, 3)
                for row in matrix:
                     matrix_html += "<tr>" + "".join([f"<td>{x: .4f}</td>" for x in row]) + "</tr>"
            except (IndexError, ValueError):
                 matrix_html += "<tr><td colspan='3'>Error</td></tr>"
            matrix_html += "</table></span>"

            html += f"""
                <tr>
                    <td>{grain_nr}</td><td>{quality:.4f}</td><td>{num_spots_total}</td><td>{unique_spots}</td><td>{matrix_html}</td>
                </tr>"""

        html += """
            </tbody>
        </table>
        """

        # --- Spot Distribution Chart ---
        spots_per_orientation = {}
        unique_spots_per_orientation = {}
        grain_numbers = [int(orient[0]) for orient in orientations]

        for grain_nr in grain_numbers:
            # Use filtered spots for counting
            spots_for_this_grain = spots[spots[:, 0] == grain_nr]
            spots_per_orientation[grain_nr] = len(spots_for_this_grain)
            # Get unique count from pre-calculated dict
            if orientation_unique_spots and grain_nr in orientation_unique_spots:
                unique_spots_per_orientation[grain_nr] = orientation_unique_spots[grain_nr].get("unique_label_count", 0)
            else:
                unique_spots_per_orientation[grain_nr] = 0

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
                           data: [{", ".join(map(str, [spots_per_orientation.get(gn, 0) for gn in grain_numbers]))}],
                           backgroundColor: 'rgba(54, 162, 235, 0.6)', borderColor: 'rgba(54, 162, 235, 1)', borderWidth: 1 }},
                        {{ label: 'Unique Spots (Labels)',
                           data: [{", ".join(map(str, [unique_spots_per_orientation.get(gn, 0) for gn in grain_numbers]))}],
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

        # --- Links to Other Visualizations ---
        html += """
        <h2>Interactive Visualizations</h2>
        <ul>"""
        interactive_file = f"{image_basename}.interactive.html"
        simulation_file = f"{image_basename}.simulation_comparison.html"
        threed_file = f"{image_basename}.3D.html"

        result_dir = self.config.get("result_dir", "results")
        if os.path.exists(os.path.join(result_dir, interactive_file)):
             html += f'<li><a href="{interactive_file}" target="_blank">Interactive Diffraction Pattern & Quality Map</a></li>'
        if os.path.exists(os.path.join(result_dir, simulation_file)):
             html += f'<li><a href="{simulation_file}" target="_blank">Simulation Comparison</a></li>'
        if vis_config.generate_3d and os.path.exists(os.path.join(result_dir, threed_file)):
             html += f'<li><a href="{threed_file}" target="_blank">3D Orientation Visualization</a></li>'
        html += """</ul>"""

        # --- Processing Parameters ---
        html += """
        <div class="parameters">
            <h2>Key Processing Parameters</h2>
            <table><thead><tr><th>Parameter Group</th><th>Parameter</th><th>Value</th></tr></thead><tbody>"""

        cfg = self.config.config  # Access the underlying LaueConfig object
        core_params = [
            ("Space Group", cfg.space_group), ("Symmetry", cfg.symmetry),
            ("Lattice Parameters", cfg.lattice_parameter), ("Detector Size (px)", f"{cfg.nr_px_x} x {cfg.nr_px_y}"),
            ("Pixel Size (mm)", f"{cfg.px_x} x {cfg.px_y}"), ("Distance (mm?)", cfg.distance)
        ]
        img_proc = cfg.image_processing
        img_params = [
            ("Threshold Method", img_proc.threshold_method), ("Threshold Value (Fixed)", img_proc.threshold_value),
            ("Threshold Percentile", img_proc.threshold_percentile), ("Min Area", img_proc.min_area),
            ("Median Radius", img_proc.filter_radius), ("Median Passes", img_proc.median_passes),
            ("Watershed", img_proc.watershed_enabled), ("Enhance Contrast", img_proc.enhance_contrast),
            ("Denoise", img_proc.denoise_image), ("Edge Enhance", img_proc.edge_enhancement)
        ]
        filt_params = [("Min Unique Spots", cfg.min_good_spots)]
        exec_params = [
             ("Processing Type", cfg.processing_type), ("CPUs Used", cfg.num_cpus),
             ("Do Forward Sim?", cfg.do_forward), ("Min Nr Spots (Exec)", cfg.min_nr_spots),
             ("Max Laue Spots (Exec)", cfg.max_laue_spots), ("Max Angle (Exec)", cfg.maxAngle)
        ]
        sim_params = [
            ("Enable Sim (Python)", cfg.simulation.enable_simulation),
            ("Skip Percentage", cfg.simulation.skip_percentage),
            ("Simulation Energies", cfg.simulation.energies)
        ]

        def add_param_rows(group_name, params):
            rows = ""
            for i, (name, value) in enumerate(params):
                rowspan_cell = '<td rowspan="{}">{}</td>'.format(len(params), group_name) if i == 0 else ''
                rows += f"<tr>{rowspan_cell}<td>{name}</td><td>{value}</td></tr>\n"
            return rows

        html += add_param_rows("Core/Detector", core_params)
        html += add_param_rows("Image Processing", img_params)
        html += add_param_rows("Filtering", filt_params)
        html += add_param_rows("Indexing Executable", exec_params)
        html += add_param_rows("Python Simulation", sim_params)

        html += """
            </tbody></table>
        </div>"""

        # --- Footer ---
        html += """
        <div class="footer">
            <p>Report generated by LaueMatching Software</p>
            <p>Contact: Hemant Sharma (hsharma@anl.gov)</p>
        </div>
    </div>
</body>
</html>"""

        # --- Write HTML File ---
        try:
            with open(report_path, 'w') as f:
                f.write(html)
            logger.info(f"Analysis report saved to {report_path}")
        except Exception as e:
             logger.error(f"Could not write analysis report {report_path}: {e}")


    def _plot_orientation_spots(
        self,
        orientations: np.ndarray, # Filtered
        spots: np.ndarray, # Filtered
        filtered_image: np.ndarray,
        labels: np.ndarray,
        colors, # Colormap object
        ax # Matplotlib axis
    ) -> None:
        """
        Helper function to plot filtered spots for each orientation on a static plot.
        """
        if orientations.size == 0 or spots.size == 0 or colors is None:
             return # Nothing to plot

        # Sort orientations *again* just for consistent legend order? Or assume input is sorted?
        # Assume input orientations are already sorted by quality
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)

        # Track labels plotted to avoid duplicates if using label-based filtering (not strictly needed here)
        # labels_plotted = set()

        for i, orientation in enumerate(orientations):
            grain_nr = int(orientation[0])
            color = colors(i / len(orientations)) if len(orientations) > 1 else colors(0) # Get color from map

            # Get spots for this orientation (already filtered)
            orientation_spots = spots[spots[:, 0] == grain_nr]
            if orientation_spots.size == 0:
                 continue

            # --- Plot the spots ---
            # Use circle markers for clarity
            ax.plot(
                orientation_spots[:, 5], # X coordinates
                orientation_spots[:, 6], # Y coordinates
                'o',                     # Circle marker
                markerfacecolor='none',
                markersize=4,            # Adjust size
                markeredgecolor=color,
                markeredgewidth=0.5,
                label=f'Grain {grain_nr} ({len(orientation_spots)} spots)' # Legend entry
            )

            # Optionally add HKL labels
            vis_config = self.config.get("visualization")
            show_labels = vis_config.show_hkl_labels if vis_config else False
            if show_labels:
                for spot in orientation_spots:
                    try:
                        h, k, l = int(spot[2]), int(spot[3]), int(spot[4])
                        x, y = spot[5], spot[6]
                        ax.text(x, y + 5, f"({h}{k}{l})", fontsize=1.5, ha='center', color=color, clip_on=True) # Adjust offset/size
                    except (IndexError, ValueError):
                         continue # Skip malformed spots


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

    logger.info(f"Attempting to generate interactive view for: {h5_file_path}")

    try:
        # Create a temporary minimal config (needed for EnhancedImageProcessor init)
        # The actual processing params don't matter here, only visualization settings if needed
        temp_config = LaueConfig()
        temp_config_manager = ConfigurationManager("temp_viewer_config.txt") # Dummy path
        temp_config_manager.config = temp_config
        # Ensure visualization is enabled for the viewer instance
        temp_config_manager.config.visualization.enable_visualization = True

        processor = EnhancedImageProcessor(temp_config_manager)

        # Load required data from HDF5 file
        with h5py.File(h5_file_path, 'r') as hf:
            # Determine which image to display (filtered thresholded usually best)
            image_to_display = None
            if '/entry/data/cleaned_data_threshold_filtered' in hf:
                image_to_display = np.array(hf['/entry/data/cleaned_data_threshold_filtered'][()])
            elif '/entry/data/cleaned_data_threshold' in hf:
                image_to_display = np.array(hf['/entry/data/cleaned_data_threshold'][()])
            elif '/entry/data/background_subtracted' in hf:
                 image_to_display = np.array(hf['/entry/data/background_subtracted'][()])
            elif '/entry/data/raw_data' in hf:
                 image_to_display = np.array(hf['/entry/data/raw_data'][()])
            if image_to_display is None:
                 raise ValueError("Could not find a suitable image dataset in HDF5 file.")

            # Load labels (final if available, else filtered, else unfiltered)
            labels = None
            if '/entry/data/watershed_labels' in hf:
                labels = np.array(hf['/entry/data/watershed_labels'][()])
            elif '/entry/data/cleaned_data_threshold_filtered_labels' in hf:
                labels = np.array(hf['/entry/data/cleaned_data_threshold_filtered_labels'][()])
            elif '/entry/data/cleaned_data_threshold_labels_unfiltered' in hf:
                labels = np.array(hf['/entry/data/cleaned_data_threshold_labels_unfiltered'][()])
            if labels is None:
                 labels = np.zeros_like(image_to_display, dtype=np.int32) # Fallback to empty labels

            # Load filtered orientations and spots (these are the primary results)
            orientations = np.array([])
            if '/entry/results/filtered_orientations' in hf:
                 orientations = np.array(hf['/entry/results/filtered_orientations'][()])
            elif '/entry/results/orientations' in hf: # Fallback to unfiltered if filtered not present
                 orientations = np.array(hf['/entry/results/orientations'][()])

            spots = np.array([])
            if '/entry/results/filtered_spots' in hf:
                 spots = np.array(hf['/entry/results/filtered_spots'][()])
            elif '/entry/results/spots' in hf: # Fallback to unfiltered
                 spots = np.array(hf['/entry/results/spots'][()])

            # Load unique spot counts for context
            unique_spots_data = None
            if '/entry/results/unique_spots_per_orientation' in hf:
                unique_counts_array = np.array(hf['/entry/results/unique_spots_per_orientation'][()])
                if unique_counts_array.size > 0:
                    unique_spots_data = {}
                    for row in unique_counts_array:
                        grain_nr, count = int(row[0]), int(row[1])
                        # Recreate structure expected by visualization functions
                        unique_spots_data[grain_nr] = {"unique_label_count": count}


        # Define base path for the output HTML file
        output_base_path = os.path.splitext(h5_file_path)[0]

        # Generate the interactive visualization HTML
        logger.info("Generating interactive Plotly HTML...")
        viz_result = processor._create_interactive_visualization(
            output_base_path,
            orientations,
            spots,
            labels,
            image_to_display,
            unique_spots_data
        )

        if viz_result.get("success"):
            html_path = f"{output_base_path}.interactive.html"
            logger.info(f"Interactive visualization saved to: {html_path}")
            # Try to open in browser
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(html_path)}")
                logger.info("Attempting to open the HTML file in your default web browser...")
            except Exception as e:
                logger.warning(f"Could not automatically open browser: {e}")
                logger.info(f"Please open the file manually: {html_path}")
        else:
             logger.error("Failed to generate interactive visualization.")

    except Exception as e:
        logger.error(f"Error occurred during result viewing: {str(e)}", exc_info=True)
        sys.exit(1)


def generate_report(args):
    """
    Generate an analysis report from HDF5 file.
    """
    log_level = LogLevel[args.loglevel]
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)

    h5_file_path = args.input
    report_output_path = args.output # Already defaulted if None

    if not os.path.exists(h5_file_path):
        logger.error(f"Input HDF5 file not found: {h5_file_path}")
        sys.exit(1)

    logger.info(f"Generating analysis report for: {h5_file_path}")
    logger.info(f"Report will be saved to: {report_output_path}")

    try:
        # Create temporary minimal config
        temp_config = LaueConfig()
        temp_config.visualization.report_template = args.template # Set template from args
        temp_config_manager = ConfigurationManager("temp_report_config.txt") # Dummy path
        temp_config_manager.config = temp_config
        # Ensure visualization is enabled for the report instance if it needs plots
        temp_config_manager.config.visualization.enable_visualization = True

        processor = EnhancedImageProcessor(temp_config_manager)

        # Load required data from HDF5 file
        with h5py.File(h5_file_path, 'r') as hf:
            # Load primary results (filtered)
            orientations = np.array([])
            if '/entry/results/filtered_orientations' in hf:
                 orientations = np.array(hf['/entry/results/filtered_orientations'][()])
            elif '/entry/results/orientations' in hf:
                 orientations = np.array(hf['/entry/results/orientations'][()])

            spots = np.array([])
            if '/entry/results/filtered_spots' in hf:
                 spots = np.array(hf['/entry/results/filtered_spots'][()])
            elif '/entry/results/spots' in hf:
                 spots = np.array(hf['/entry/results/spots'][()])

            # Load supporting data (image, labels, unique counts)
            image_to_display = None # Find best available image for context
            if '/entry/data/cleaned_data_threshold_filtered' in hf: image_to_display = np.array(hf['/entry/data/cleaned_data_threshold_filtered'][()])
            elif '/entry/data/cleaned_data_threshold' in hf: image_to_display = np.array(hf['/entry/data/cleaned_data_threshold'][()])
            elif '/entry/data/background_subtracted' in hf: image_to_display = np.array(hf['/entry/data/background_subtracted'][()])
            if image_to_display is None: image_to_display = np.zeros((100,100)) # Placeholder if no image

            labels = None # Find best available labels
            if '/entry/data/watershed_labels' in hf: labels = np.array(hf['/entry/data/watershed_labels'][()])
            elif '/entry/data/cleaned_data_threshold_filtered_labels' in hf: labels = np.array(hf['/entry/data/cleaned_data_threshold_filtered_labels'][()])
            elif '/entry/data/cleaned_data_threshold_labels_unfiltered' in hf: labels = np.array(hf['/entry/data/cleaned_data_threshold_labels_unfiltered'][()])
            if labels is None: labels = np.zeros_like(image_to_display, dtype=np.int32)

            unique_spots_data = None
            if '/entry/results/unique_spots_per_orientation' in hf:
                 unique_counts_array = np.array(hf['/entry/results/unique_spots_per_orientation'][()])
                 if unique_counts_array.size > 0:
                      unique_spots_data = {int(row[0]): {"unique_label_count": int(row[1])} for row in unique_counts_array}


        # Define base path for report generation (for finding linked images)
        # Assumes report is saved relative to the images/plots
        report_base_path = os.path.splitext(report_output_path)[0]


        # Generate the report HTML
        logger.info("Generating report HTML content...")
        processor._create_analysis_report(
            report_base_path, # Use report base path here
            orientations,
            spots,
            labels,
            image_to_display,
            unique_spots_data
        )

        logger.info(f"Analysis report generated successfully: {report_output_path}") # Log the actual output path

    except Exception as e:
        logger.error(f"Error occurred during report generation: {str(e)}", exc_info=True)
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