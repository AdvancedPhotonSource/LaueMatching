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
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import threading
from datetime import datetime
import concurrent.futures
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
import h5py
import cv2
import scipy.ndimage as ndimg
import diplib as dip
import skimage.segmentation
from skimage import exposure, filters, morphology, feature
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ipywidgets as widgets
from IPython.display import display, HTML

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

@dataclass
class ImageProcessingConfig:
    """Image processing configuration parameters."""
    threshold_method: str = "adaptive"  # adaptive, otsu, or fixed
    threshold_value: float = 0.0
    min_area: int = 10
    filter_radius: int = 101
    median_passes: int = 5
    watershed_enabled: bool = True
    gaussian_factor: float = 0.25
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LaueConfig:
    """Main configuration class for Laue matching."""
    # Core parameters
    space_group: int = 225
    symmetry: str = "F"
    lattice_parameter: str = "3.615 3.615 3.615 90 90 90"
    r_array: str = "4 6 2.0"
    p_array: str = "130 130 100"
    min_good_spots: int = 10
    max_laue_spots: int = 100
    
    # File paths
    result_dir: str = "results"
    orientation_file: str = "orientations.bin"
    hkl_file: str = "hkls.bin"
    background_file: str = "median.bin"
    forward_file: str = "forward.bin"
    
    # Detector parameters
    px_x: float = 0.2
    px_y: float = 0.2
    nr_px_x: int = 2048
    nr_px_y: int = 2048
    orientation_spacing: float = 0.5
    distance: float = 100.0
    
    # Processing parameters
    do_forward: bool = True
    processing_type: str = "CPU"
    num_cpus: int = 4
    
    # Enhanced configuration sections
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Additional parameters
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to dictionary format."""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not isinstance(v, (ImageProcessingConfig, VisualizationConfig))}
        
        config_dict["image_processing"] = self.image_processing.to_dict()
        config_dict["visualization"] = self.visualization.to_dict()
        
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
        
        # Handle enum conversion
        if "log_level" in config:
            config["log_level"] = LogLevel[config["log_level"]]
            
        # Create instance
        instance = cls(**config)
        
        # Update nested configurations
        if img_config:
            instance.image_processing = ImageProcessingConfig(**img_config)
        if vis_config:
            instance.visualization = VisualizationConfig(**vis_config)
            
        return instance


class ConfigurationManager:
    """Manages configuration for the Laue matching process."""
    
    def __init__(self, config_file: str):
        """
        Initialize configuration manager with parameters from config file.
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = LaueConfig()
        self._load_config()
        
    def _load_config(self) -> None:
        """Parse the configuration file and load parameters."""
        try:
            file_ext = os.path.splitext(self.config_file)[1].lower()
            
            if file_ext == '.json':
                self._load_from_json()
            elif file_ext in ('.yaml', '.yml'):
                self._load_from_yaml()
            else:
                self._load_from_text()
                
            logger.info(f"Configuration loaded from {self.config_file}")
                
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_file} not found.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading configuration file: {str(e)}")
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
                
        for line in lines:
            if line.strip() and not line.startswith('#'):
                self._parse_classic_config_line(line)
                
    def _parse_classic_config_line(self, line: str) -> None:
        """
        Parse a single configuration line from classic format.
        
        Args:
            line: A line from the configuration file
        """
        if line.startswith('SpaceGroup'):
            self.config.space_group = int(line.split()[1])
        elif line.startswith('Symmetry'):
            sym = line.split()[1]
            if sym not in 'FICAR' or len(sym) != 1:
                logger.error('Invalid value for sym, must be one character from F,I,C,A,R')
                sys.exit(1)
            self.config.symmetry = sym
        elif line.startswith('LatticeParameter'):
            self.config.lattice_parameter = ' '.join(line.split()[1:7])
        elif line.startswith('R_Array'):
            self.config.r_array = ' '.join(line.split()[1:4])
        elif line.startswith('P_Array'):
            self.config.p_array = ' '.join(line.split()[1:4])
            self.config.distance = float(line.split()[3])
        elif line.startswith('Threshold'):
            self.config.image_processing.threshold_value = float(line.split()[1])
        elif line.startswith('MinIntensity'):
            min_intensity = float(line.split()[1])
        elif line.startswith('PxX'):
            self.config.px_x = float(line.split()[1])
        elif line.startswith('PxY'):
            self.config.px_y = float(line.split()[1])
        elif line.startswith('OrientationSpacing'):
            self.config.orientation_spacing = float(line.split()[1])
        elif line.startswith('WatershedImage'):
            self.config.image_processing.watershed_enabled = bool(int(line.split()[1]))
        elif line.startswith('NrPxX'):
            self.config.nr_px_x = int(line.split()[1])
        elif line.startswith('NrPxY'):
            self.config.nr_px_y = int(line.split()[1])
        elif line.startswith('FilterRadius'):
            self.config.image_processing.filter_radius = int(line.split()[1])
        elif line.startswith('NMeadianPasses'):
            self.config.image_processing.median_passes = int(line.split()[1])
        elif line.startswith('MinArea'):
            self.config.image_processing.min_area = int(line.split()[1])
        elif line.startswith('MinGoodSpots'):
            self.config.min_good_spots = int(line.split()[1])
        elif line.startswith('MaxNrLaueSpots'):
            self.config.max_laue_spots = int(line.split()[1])
        elif line.startswith('ResultDir'):
            self.config.result_dir = line.split()[1]
        elif line.startswith('OrientationFile'):
            self.config.orientation_file = line.split()[1]
        elif line.startswith('HKLFile'):
            self.config.hkl_file = line.split()[1]
        elif line.startswith('BackgroundFile'):
            self.config.background_file = line.split()[1]
        elif line.startswith('ForwardFile'):
            self.config.forward_file = line.split()[1]
        elif line.startswith('DoFwd'):
            self.config.do_forward = bool(int(line.split()[1]))
            
    def write_config(self) -> None:
        """Write current configuration to file."""
        file_ext = os.path.splitext(self.config_file)[1].lower()
        
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
            sys.exit(1)
            
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
            f.write(f"SpaceGroup {self.config.space_group}\n")
            f.write(f"Symmetry {self.config.symmetry}\n")
            f.write(f"LatticeParameter {self.config.lattice_parameter}\n")
            f.write(f"R_Array {self.config.r_array}\n")
            f.write(f"P_Array {self.config.p_array}\n")
            f.write(f"Threshold {self.config.image_processing.threshold_value}\n")
            f.write(f"PxX {self.config.px_x}\n")
            f.write(f"PxY {self.config.px_y}\n")
            f.write(f"OrientationSpacing {self.config.orientation_spacing}\n")
            f.write(f"WatershedImage {int(self.config.image_processing.watershed_enabled)}\n")
            f.write(f"NrPxX {self.config.nr_px_x}\n")
            f.write(f"NrPxY {self.config.nr_px_y}\n")
            f.write(f"FilterRadius {self.config.image_processing.filter_radius}\n")
            f.write(f"NMeadianPasses {self.config.image_processing.median_passes}\n")
            f.write(f"MinArea {self.config.image_processing.min_area}\n")
            f.write(f"MinGoodSpots {self.config.min_good_spots}\n")
            f.write(f"MaxNrLaueSpots {self.config.max_laue_spots}\n")
            f.write(f"ResultDir {self.config.result_dir}\n")
            f.write(f"OrientationFile {self.config.orientation_file}\n")
            f.write(f"HKLFile {self.config.hkl_file}\n")
            f.write(f"BackgroundFile {self.config.background_file}\n")
            f.write(f"ForwardFile {self.config.forward_file}\n")
            f.write(f"DoFwd {int(self.config.do_forward)}\n")
            
    def get(self, key: str, default=None):
        """Get a configuration parameter value."""
        if hasattr(self.config, key):
            return getattr(self.config, key)
        return default
        
    def set(self, key: str, value) -> None:
        """Set a configuration parameter value."""
        if hasattr(self.config, key):
            setattr(self.config, key, value)
        else:
            logger.warning(f"Unknown configuration parameter: {key}")
            
    def load_from_env(self) -> None:
        """
        Load configuration from environment variables.
        
        Environment variables should be prefixed with LAUE_
        """
        for key, value in os.environ.items():
            if key.startswith('LAUE_'):
                # Remove prefix and convert to lowercase
                config_key = key[5:].lower()
                
                if hasattr(self.config, config_key):
                    # Convert value to appropriate type
                    current_value = getattr(self.config, config_key)
                    if isinstance(current_value, bool):
                        setattr(self.config, config_key, value.lower() in ('true', '1', 'yes'))
                    elif isinstance(current_value, int):
                        setattr(self.config, config_key, int(value))
                    elif isinstance(current_value, float):
                        setattr(self.config, config_key, float(value))
                    else:
                        setattr(self.config, config_key, value)

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
        self.progress_bar = tqdm(total=total_steps, desc=description, unit="step")
        
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
        if self.current_step > 0 and self.current_step < self.total_steps:
            remaining = elapsed * (self.total_steps - self.current_step) / self.current_step
            percentage = 100.0 * self.current_step / self.total_steps
            
            # Log progress every 10% or at least 5 seconds
            if (percentage % 10 < (percentage - step_increment * 100.0 / self.total_steps) % 10 or 
                    current_time - self.last_update_time > 5):
                self.last_update_time = current_time
                logger.info(f"Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps}), "
                           f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
                
    def complete(self, status: str = "Completed") -> None:
        """
        Mark progress as complete.
        
        Args:
            status: Final status message
        """
        self.progress_bar.update(self.total_steps - self.current_step)
        self.progress_bar.set_description(f"{self.description}: {status}")
        self.progress_bar.close()
        
        elapsed = time.time() - self.start_time
        logger.info(f"Operation completed in {elapsed:.2f} seconds")


###########################################
# 8. Image Processing Enhancements Implementation #
###########################################

class EnhancedImageProcessor:
    """Enhanced image processing for Laue diffraction patterns."""
    
    def __init__(self, config: ConfigurationManager):
        """
        Initialize the image processor with configuration parameters.
        
        Args:
            config: Configuration manager containing processing parameters
        """
        self.config = config
        self.background = None
        
        # Configure output image settings
        vis_config = config.get("visualization")
        self.outdpi = vis_config.output_dpi
        self.scalarX = config.get("nr_px_x") / self.outdpi
        self.scalarY = config.get("nr_px_y") / self.outdpi
        plt.rcParams['figure.figsize'] = [self.scalarX, self.scalarY]
        
        # Initialize background if available
        self._load_background()
        
    def _load_background(self) -> None:
        """Load background image if it exists, otherwise initialize empty array."""
        background_file = self.config.get("background_file")
        nPxX = self.config.get("nr_px_x")
        nPxY = self.config.get("nr_px_y")
        
        if os.path.exists(background_file):
            try:
                self.background = np.fromfile(background_file, dtype=np.double).reshape((nPxX, nPxY))
                logger.info(f"Background file loaded: {background_file}")
            except Exception as e:
                logger.error(f"Error loading background file: {str(e)}")
                self.background = np.zeros((nPxX, nPxY))
        else:
            self.background = np.zeros((nPxX, nPxY))
            
    def compute_background(self, image: np.ndarray) -> np.ndarray:
        """
        Compute background by applying median filter to the image.
        
        Args:
            image: Input image array
            
        Returns:
            Background image array
        """
        logger.info("Computing background...")
        img_config = self.config.get("image_processing")
        background = dip.Image(image)
        for i in range(img_config.median_passes):
            logger.debug(f"Median filter pass {i+1}/{img_config.median_passes}")
            background = dip.MedianFilter(background, img_config.filter_radius)
            
        # Save the background for future use
        background_arr = np.array(background).astype(np.double)
        background_arr.tofile(self.config.get("background_file"))
        self.background = background_arr
        return background_arr
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply optional image enhancements based on configuration.
        
        Args:
            image: Input image array
            
        Returns:
            Enhanced image array
        """
        img_config = self.config.get("image_processing")
        enhanced = image.copy()
        
        # Apply contrast enhancement if enabled
        if img_config.enhance_contrast:
            logger.info("Applying contrast enhancement")
            # Use CLAHE (Contrast Limited Adaptive Histogram Equalization)
            enhanced = exposure.equalize_adapthist(enhanced)
            
        # Apply denoising if enabled
        if img_config.denoise_image:
            logger.info("Applying denoising")
            # Use non-local means denoising
            enhanced = skimage.restoration.denoise_nl_means(
                enhanced, 
                h=img_config.denoise_strength,
                fast_mode=True,
                patch_size=5,
                patch_distance=7
            )
            
        # Apply edge enhancement if enabled
        if img_config.edge_enhancement:
            logger.info("Applying edge enhancement")
            # Use unsharp masking for edge enhancement
            blurred = filters.gaussian(enhanced, sigma=1.0)
            enhanced = enhanced + 0.5 * (enhanced - blurred)
            
        return enhanced
        
    def apply_threshold(self, image: np.ndarray, override_thresh: int = 0) -> Tuple[np.ndarray, float]:
        """
        Apply adaptive or fixed thresholding based on configuration.
        
        Args:
            image: Input image array
            override_thresh: Optional threshold override value
            
        Returns:
            Tuple of (thresholded image, threshold used)
        """
        img_config = self.config.get("image_processing")
        threshold_method = img_config.threshold_method
        
        if override_thresh:
            # Use override value
            threshold = override_thresh
            logger.info(f"Using override threshold: {threshold}")
        elif threshold_method == "adaptive":
            # Compute threshold based on standard deviation
            std_dev = np.std(image)
            threshold = 60 * (1 + std_dev // 60)
            logger.info(f"Using adaptive threshold: {threshold} (based on std dev {std_dev:.2f})")
        elif threshold_method == "otsu":
            # Use Otsu's method to find optimal threshold
            threshold = filters.threshold_otsu(image)
            logger.info(f"Using Otsu threshold: {threshold}")
        else:
            # Use fixed threshold from configuration
            threshold = img_config.threshold_value
            logger.info(f"Using fixed threshold: {threshold}")
            
        # Apply threshold
        binary = image > threshold
        thresholded = image.copy()
        thresholded[~binary] = 0
        
        return thresholded, threshold
        
    def correct_image(self, image: np.ndarray, override_thresh: int = 0) -> Tuple[np.ndarray, float]:
        """
        Apply background correction, enhancement, and thresholding to input image.
        
        Args:
            image: Raw input image array
            override_thresh: Optional threshold override value
            
        Returns:
            Tuple of (corrected image, threshold used)
        """
        # Check if we need to compute background
        if np.all(self.background == 0):
            self.compute_background(image)
            
        # Apply background correction
        logger.info("Applying background correction")
        corrected = image.astype(np.double) - self.background
        
        # Apply enhancements
        corrected = self.enhance_image(corrected)
        
        # Apply thresholding
        thresholded, threshold = self.apply_threshold(corrected, override_thresh)
        
        return thresholded.astype(np.uint16), threshold
        
    @staticmethod
    def find_connected_components(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Find connected components in the image.
        
        Args:
            image: Thresholded input image
            
        Returns:
            Tuple of (labels, bounding boxes, areas, number of labels)
        """
        # Convert to binary image
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        
        # Find connected components
        output = cv2.connectedComponentsWithStats(binary, 8, cv2.CV_32S)
        
        nlabels = output[0]
        areas = output[2][:, -1]
        bounding_boxes = output[2][:, :-1]  # left, top, width, height
        labels = output[1]
        
        logger.info(f"Found {nlabels-1} connected components")
        
        return labels, bounding_boxes, areas, nlabels
        
    def filter_small_components(
        self, 
        image: np.ndarray, 
        labels: np.ndarray, 
        bounding_boxes: np.ndarray, 
        areas: np.ndarray, 
        nlabels: int
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Filter out components smaller than the minimum area.
        
        Args:
            image: Input image array
            labels: Component labels array
            bounding_boxes: Bounding boxes array
            areas: Component areas array
            nlabels: Number of labels
            
        Returns:
            Tuple of (filtered image, filtered labels, center points)
        """
        min_area = self.config.get("image_processing").min_area
        centers = []
        
        # Keep track of the filtered image and labels
        filtered_image = image.copy()
        filtered_labels = labels.copy()
        
        filtered_count = 0
        for label_idx in range(1, nlabels):
            if areas[label_idx] > min_area:
                # Get the bounding box
                x, y, w, h = bounding_boxes[label_idx]
                
                # Extract the region for this component
                region = image[y:y+h, x:x+w]
                
                # Compute center of mass
                com = ndimg.center_of_mass(region)
                center_x = com[1] + x
                center_y = com[0] + y
                
                # Store label, center coordinates, and area
                centers.append([label_idx, (center_x, center_y), areas[label_idx]])
            else:
                # Remove small components
                filtered_count += 1
                y, x, w, h = bounding_boxes[label_idx]
                filtered_image[y:y+h, x:x+w] = 0
                filtered_labels[y:y+h, x:x+w] = 0
                
        logger.info(f"Filtered out {filtered_count} small components, kept {len(centers)}")
        return filtered_image, filtered_labels, centers
        
    @staticmethod
    def calculate_gaussian_width(centers: List, pixel_size: float, distance: float, orient_spacing: float) -> int:
        """
        Calculate optimal Gaussian blur width based on spot spacing.
        
        Args:
            centers: List of center points for each component
            pixel_size: Pixel size in mm
            distance: Sample-to-detector distance in mm
            orient_spacing: Orientation spacing in degrees
            
        Returns:
            Gaussian blur width in pixels
        """
        if not centers or len(centers) < 2:
            logger.warning("Not enough centers to calculate optimal Gaussian width, using default")
            return 5  # Default value if not enough centers
            
        # Find minimum distance between centers
        min_distance = float('inf')
        for i in range(len(centers) - 1):
            center1 = centers[i][1]
            for j in range(i + 1, len(centers)):
                center2 = centers[j][1]
                dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                min_distance = min(min_distance, dist)
                
        # Calculate expected separation based on orientation spacing
        delta_pos = np.tan(np.radians(orient_spacing / 2)) * distance / pixel_size
        
        # Use the smaller of the two values and apply scaling factor
        blur_width = int(np.ceil(0.25 * np.ceil(min(delta_pos, min_distance))))
        logger.info(f"Calculated Gaussian width: {blur_width} pixels (min distance: {min_distance:.1f} px, delta_pos: {delta_pos:.1f} px)")
        return max(blur_width, 1)  # Ensure width is at least 1

    def process_image(self, image_path: str, override_thresh: int = 0) -> Dict[str, Any]:
        """
        Process a single image file.
        
        Args:
            image_path: Path to the input image file
            override_thresh: Optional threshold override
            
        Returns:
            Dictionary of processing results
        """
        logger.info(f"Processing image: {image_path}")
        start_time = time.time()
        
        # Create result directory if it doesn't exist
        result_dir = self.config.get("result_dir")
        os.makedirs(result_dir, exist_ok=True)
        
        # Initialize progress reporter
        progress = ProgressReporter(7, f"Processing {os.path.basename(image_path)}")
        
        # Step 1: Load the image
        try:
            with h5py.File(image_path, 'r') as h_file:
                image_data = np.array(h_file['/entry1/data/data'][()])
                raw_image = np.copy(image_data)
        except Exception as e:
            logger.error(f"Error loading image file: {str(e)}")
            return {"success": False, "error": str(e)}
            
        progress.update(1, "Image loaded")
        
        # Step 2: Correct the image and apply threshold
        corrected_image, threshold = self.correct_image(image_data, override_thresh)
        progress.update(1, "Image corrected")
        
        # Prepare output file path
        output_path = os.path.join(result_dir, os.path.basename(image_path))
        output_h5 = f"{output_path}.bin.output.h5"
        
        # Create output HDF5 file
        with h5py.File(output_h5, 'w') as hf_out:
            # Store raw and corrected data
            hf_out.create_dataset('/entry/data/raw_data', data=raw_image)
            hf_out.create_dataset('/entry/data/cleaned_data_threshold', data=corrected_image)
            
            # Step 3: Process connected components
            labels, bboxes, areas, nlabels = self.find_connected_components(corrected_image)
            hf_out.create_dataset('/entry/data/cleaned_data_threshold_labels_unfiltered', data=labels)
            progress.update(1, "Connected components found")
            
            # Step 4: Filter small components
            filtered_image, filtered_labels, centers = self.filter_small_components(
                corrected_image, labels, bboxes, areas, nlabels
            )
            
            hf_out.create_dataset('/entry/data/cleaned_data_threshold_filtered', data=filtered_image)
            hf_out.create_dataset('/entry/data/cleaned_data_threshold_filtered_labels', data=filtered_labels)
            progress.update(1, "Small components filtered")
            
            # Step 5: Calculate Gaussian blur width
            gauss_width = self.calculate_gaussian_width(
                centers, 
                self.config.get("px_x"), 
                self.config.get("distance"),
                self.config.get("orientation_spacing")
            )
            
            # Apply Gaussian blur
            blurred_image = ndimg.gaussian_filter(filtered_image, gauss_width)
            hf_out.create_dataset('/entry/data/input_blurred', data=blurred_image)
            
            # Save blurred image for indexing
            blurred_image.astype(np.double).tofile(f"{output_path}.bin")
            progress.update(1, "Image blurred")
            
            # Step 6: Perform watershed segmentation if enabled
            if self.config.get("image_processing").watershed_enabled:
                watershed_labels = skimage.segmentation.watershed(
                    -blurred_image, mask=blurred_image, connectivity=2
                )
                final_labels = watershed_labels
                max_labels = np.max(watershed_labels)
                logger.info(f'Watershed segmentation found {max_labels} regions')
            else:
                final_labels = filtered_labels
                max_labels = nlabels
                logger.info('Watershed segmentation was not used')
            
            progress.update(1, "Segmentation completed")
            
            # Step 7: Run indexing and process results
            indexing_results = self._run_indexing(
                output_path, final_labels, max_labels, blurred_image, centers, filtered_image
            )
            
            if "success" in indexing_results and not indexing_results["success"]:
                progress.complete("Failed at indexing stage")
                return indexing_results
                
            progress.complete("Processing completed")
            
            # Return processing results
            return {
                "success": True,
                "image_path": image_path,
                "output_path": output_path,
                "output_h5": output_h5,
                "centers": centers,
                "final_labels": final_labels,
                "indexing_results": indexing_results,
                "processing_time": time.time() - start_time
            }
            
    def _run_indexing(
        self, 
        output_path: str, 
        labels: np.ndarray, 
        nlabels: int, 
        blurred_image: np.ndarray,
        centers: List,
        filtered_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run the Laue indexing process on the prepared image.
        
        Args:
            output_path: Path to the output files
            labels: Final labels array
            nlabels: Number of labels
            blurred_image: Blurred image for indexing
            centers: List of center points
            filtered_image: Filtered image
            
        Returns:
            Dictionary with indexing results
        """
        # Run the indexing executable
        compute_type = self.config.get("processing_type")
        ncpus = self.config.get("num_cpus")
        
        file_path = os.path.dirname(os.path.realpath(__file__))
        env = dict(os.environ)
        lib_path = os.environ.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f'{file_path}/LIBS/NLOPT/lib:{file_path}/LIBS/NLOPT/lib64:{lib_path}'
        
        config_file = self.config.config_file
        orient_file = self.config.get("orientation_file")
        hkl_file = self.config.get("hkl_file")
        
        # Ensure orientation file exists
        if not os.path.exists(orient_file):
            logger.info(f"Orientation file not found, copying from {INSTALL_PATH}/100MilOrients.bin")
            shutil.copy2(f'{INSTALL_PATH}/100MilOrients.bin', orient_file)
            
        # Generate HKL file if it doesn't exist
        if not os.path.exists(hkl_file):
            logger.info(f"HKL file not found, generating")
            result = self._generate_hkl_file()
            if not result["success"]:
                return result
            
        # Choose the appropriate executable
        if compute_type == 'CPU':
            executable = f'{file_path}/build/LaueMatchingCPU'
        else:
            executable = f'{file_path}/build/LaueMatchingGPU'
            
        # Run the indexing command
        indexing_cmd = f'{executable} {config_file} {orient_file} {hkl_file} {output_path}.bin {ncpus}'
        logger.info(f'Running indexing command: {indexing_cmd}')
        
        try:
            result = subprocess.run(
                indexing_cmd, 
                shell=True, 
                env=env, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                check=True,
                text=True
            )
            
            # Save stdout to file
            with open(f'{output_path}.LaueMatching_stdout.txt', 'w') as stdout_file:
                stdout_file.write(result.stdout)
                
            logger.info(f"Indexing completed successfully")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running indexing command: {e}")
            logger.error(f"Command output: {e.stdout}")
            
            # Save stdout to file even on error
            with open(f'{output_path}.LaueMatching_stderr.txt', 'w') as stderr_file:
                stderr_file.write(e.stdout)
                
            return {
                "success": False, 
                "error": f"Indexing command failed with code {e.returncode}"
            }
                
        # Process indexing results
        return self._process_indexing_results(output_path, labels, nlabels, filtered_image)
        
    def _generate_hkl_file(self) -> Dict[str, Any]:
        """
        Generate HKL file using the GenerateHKLs.py script.
        
        Returns:
            Dictionary with generation result
        """
        hkl_file = self.config.get("hkl_file")
        sg_num = self.config.get("space_group")
        sym = self.config.get("symmetry")
        lat_c = self.config.get("lattice_parameter")
        r_arr = self.config.get("r_array")
        p_arr = self.config.get("p_array")
        n_px_x = self.config.get("nr_px_x")
        n_px_y = self.config.get("nr_px_y")
        dx = self.config.get("px_x")
        dy = self.config.get("px_y")
        
        logger.info("Generating HKL file...")
        cmd = f'{PYTHON_PATH} {INSTALL_PATH}/GenerateHKLs.py '
        cmd += f'-resultFileName {hkl_file} -sgnum {sg_num} -sym {sym} '
        cmd += f'-latticeParameter {lat_c} -RArray {r_arr} -PArray {p_arr} '
        cmd += f'-NumPxX {n_px_x} -NumPxY {n_px_y} -dx {dx} -dy {dy}'
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            logger.info("HKL file generated successfully")
            return {"success": True}
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating HKL file: {e}")
            logger.error(f"Command output: {e.stdout}")
            return {"success": False, "error": f"HKL generation failed with code {e.returncode}"}
            
    def _process_indexing_results(
        self, 
        output_path: str, 
        labels: np.ndarray, 
        nlabels: int,
        filtered_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process and visualize the indexing results.
        
        Args:
            output_path: Path to the output files
            labels: Final labels array
            nlabels: Number of labels
            filtered_image: Filtered image
            
        Returns:
            Dictionary with processing results
        """
        # Read solutions and spots
        try:
            with open(f'{output_path}.bin.solutions.txt') as f:
                header_gr = f.readline()
                
            orientations = np.genfromtxt(f'{output_path}.bin.solutions.txt', skip_header=1)
            
            with open(f'{output_path}.bin.spots.txt') as f:
                header_sp = f.readline()
                
            spots = np.genfromtxt(f'{output_path}.bin.spots.txt', skip_header=1)
            
            logger.info(f"Read indexing results: {orientations.shape[0] if len(orientations.shape) > 1 else 1} orientations")
            
        except Exception as e:
            logger.error(f"Error reading indexing results: {str(e)}")
            return {"success": False, "error": f"Failed to read indexing results: {str(e)}"}
            
        # Create visualization
        visualization_results = self._visualize_results(
            output_path, orientations, spots, labels, filtered_image
        )
        
        if not visualization_results["success"]:
            return visualization_results
            
        # Return results
        return {
            "success": True,
            "orientations": orientations,
            "spots": spots,
            "visualization": visualization_results
        }
        

###########################################
# 9. Visualization Improvements Implementation #
###########################################

    def _visualize_results(
        self, 
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        labels: np.ndarray,
        filtered_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Visualize the indexing results with enhanced plotting options.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
            labels: Final labels array
            filtered_image: Filtered image
            
        Returns:
            Dictionary with visualization results
        """
        vis_config = self.config.get("visualization")
        plot_type = vis_config.plot_type
        
        # Create standard static visualization for H5 output
        static_result = self._create_static_visualization(
            output_path, orientations, spots, labels, filtered_image
        )
        
        if not static_result["success"]:
            return static_result
            
        # Create interactive visualization if requested
        if plot_type in ["interactive", "both"]:
            interactive_result = self._create_interactive_visualization(
                output_path, orientations, spots, labels, filtered_image
            )
            
            if not interactive_result["success"]:
                logger.warning(f"Failed to create interactive visualization: {interactive_result['error']}")
                
        # Create 3D visualization if requested
        if vis_config.generate_3d:
            try:
                logger.info("Generating 3D visualization")
                self._create_3d_visualization(output_path, orientations, spots)
            except Exception as e:
                logger.warning(f"Failed to create 3D visualization: {str(e)}")
                
        # Generate report if requested
        if vis_config.generate_report:
            try:
                logger.info("Generating analysis report")
                self._create_analysis_report(
                    output_path, orientations, spots, labels, filtered_image
                )
            except Exception as e:
                logger.warning(f"Failed to create analysis report: {str(e)}")
                
        return {"success": True}
        
    def _create_static_visualization(
        self, 
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        labels: np.ndarray,
        filtered_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create static visualizations of the indexing results.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
            labels: Final labels array
            filtered_image: Filtered image
            
        Returns:
            Dictionary with visualization results
        """
        try:
            # Setup figure
            plt.figure(figsize=(self.scalarX, self.scalarY))
            ax = plt.gca()
            
            # Display background image (log of intensity)
            display_img = filtered_image.copy()
            display_img[display_img == 0] = 1  # Avoid log(0)
            ax.imshow(np.log(display_img), cmap='Greens')
            
            # Get colormap for different orientations
            min_good_spots = self.config.get("min_good_spots")
            colormap_name = self.config.get("visualization").colormap
            max_orientations = np.max(labels) // min_good_spots if np.max(labels) > 0 else 10
            colors = plt.get_cmap(colormap_name, max_orientations)
            
            # Plot spots for each orientation
            self._plot_orientation_spots(orientations, spots, filtered_image, labels, colors, ax)
            
            # Add legend and save the figure
            plt.legend(loc='upper right', fontsize='x-small')
            plt.tight_layout()
            
            # Save figure at specified DPI
            dpi = self.config.get("visualization").output_dpi
            plt.savefig(f'{output_path}.bin.LabeledImage.tif', dpi=dpi)
            plt.close()
            
            logger.info(f"Static visualization saved to {output_path}.bin.LabeledImage.tif")
            
            # Create additional visualization: Orientation quality map
            self._create_quality_map(output_path, orientations, spots, filtered_image)
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error creating static visualization: {str(e)}")
            return {"success": False, "error": str(e)}
        
    def _create_quality_map(
        self, 
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        filtered_image: np.ndarray
    ) -> None:
        """
        Create a quality map visualization showing the confidence of orientation indexing.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
            filtered_image: Filtered image
        """
        # Skip if no orientations
        if orientations.size == 0:
            logger.warning("No orientations found, skipping quality map")
            return
            
        # Setup figure
        plt.figure(figsize=(self.scalarX, self.scalarY))
        
        # Create empty quality map
        quality_map = np.zeros_like(filtered_image, dtype=float)
        
        # For each orientation, add quality score to the map at spot locations
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)
            
        for orientation in orientations:
            # Get orientation quality score
            quality_score = orientation[4]  # Assuming column 4 is the quality score
            
            # Get spots for this orientation
            orientation_spots = spots[spots[:, 0] == orientation[0]]
            
            # Add quality score to map at spot locations
            for spot in orientation_spots:
                x, y = int(spot[5]), int(spot[6])
                if 0 <= x < quality_map.shape[1] and 0 <= y < quality_map.shape[0]:
                    quality_map[y, x] = max(quality_map[y, x], quality_score)
                    
        # Apply Gaussian blur to make the map smoother
        quality_map = ndimg.gaussian_filter(quality_map, 3)
        
        # Display quality map
        plt.imshow(quality_map, cmap='plasma')
        plt.colorbar(label='Indexing Quality')
        plt.title('Orientation Indexing Quality Map')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{output_path}.bin.QualityMap.png', dpi=self.config.get("visualization").output_dpi)
        plt.close()
        
        logger.info(f"Quality map saved to {output_path}.bin.QualityMap.png")
        
    def _create_interactive_visualization(
        self, 
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        labels: np.ndarray,
        filtered_image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create interactive visualization using Plotly.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
            labels: Final labels array
            filtered_image: Filtered image
            
        Returns:
            Dictionary with visualization results
        """
        try:
            # Create a figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["Indexed Diffraction Pattern", "Orientation Quality"],
                horizontal_spacing=0.1
            )
            
            # Display background image (log of intensity)
            display_img = filtered_image.copy()
            display_img[display_img == 0] = 1  # Avoid log(0)
            log_img = np.log(display_img)
            
            # Add image to first subplot
            fig.add_trace(
                go.Heatmap(
                    z=log_img,
                    colorscale='Greens',
                    showscale=False,
                ),
                row=1, col=1
            )
            
            # Create empty quality map for second subplot
            quality_map = np.zeros_like(filtered_image, dtype=float)
            
            # Process orientations
            if len(orientations.shape) == 1:
                orientations = np.expand_dims(orientations, axis=0)
                
            # Create colormap
            min_good_spots = self.config.get("min_good_spots")
            orientation_colors = px.colors.qualitative.Dark24
            
            # Dictionary to store traces by orientation
            orientation_traces = {}
            
            # Plot spots for each orientation
            for i, orientation in enumerate(orientations):
                orientation_id = int(orientation[0])
                color = orientation_colors[i % len(orientation_colors)]
                
                # Get orientation quality score
                quality_score = orientation[4]
                
                # Get spots for this orientation
                orientation_spots = spots[spots[:, 0] == orientation_id]
                
                # Skip if no spots
                if len(orientation_spots) == 0:
                    continue
                    
                # Create scatter trace for spots
                x_coords = orientation_spots[:, 5]
                y_coords = orientation_spots[:, 6]
                
                # Add trace for this orientation
                name = f"ID {i}"
                trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=6,
                        line=dict(width=1, color='black')
                    ),
                    name=name,
                    hovertext=[
                        f"Orientation: {i}<br>"
                        f"HKL: {int(spot[2])},{int(spot[3])},{int(spot[4])}<br>"
                        f"Position: ({spot[5]:.1f}, {spot[6]:.1f})"
                        for spot in orientation_spots
                    ],
                    hoverinfo="text"
                )
                
                fig.add_trace(trace, row=1, col=1)
                
                # Add to quality map
                for spot in orientation_spots:
                    x, y = int(spot[5]), int(spot[6])
                    if 0 <= x < quality_map.shape[1] and 0 <= y < quality_map.shape[0]:
                        quality_map[y, x] = max(quality_map[y, x], quality_score)
                        
            # Smooth quality map and add to second subplot
            quality_map = ndimg.gaussian_filter(quality_map, 3)
            
            fig.add_trace(
                go.Heatmap(
                    z=quality_map,
                    colorscale='Plasma',
                    colorbar=dict(title="Quality"),
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Laue Diffraction Analysis",
                height=800,
                width=1200,
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="X Position (pixels)", row=1, col=1)
            fig.update_yaxes(title_text="Y Position (pixels)", row=1, col=1)
            fig.update_xaxes(title_text="X Position (pixels)", row=1, col=2)
            fig.update_yaxes(title_text="Y Position (pixels)", row=1, col=2)
            
            # Save as HTML
            fig.write_html(f"{output_path}.bin.interactive.html")
            logger.info(f"Interactive visualization saved to {output_path}.bin.interactive.html")
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error creating interactive visualization: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _create_3d_visualization(
        self,
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray
    ) -> None:
        """
        Create 3D visualization of crystal orientations.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
        """
        # Skip if no orientations
        if orientations.size == 0:
            logger.warning("No orientations found, skipping 3D visualization")
            return
            
        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)
            
        # Create figure
        fig = go.Figure()
        
        # Define colors
        colors = px.colors.qualitative.Plotly
        
        # Extract quaternions or rotation matrices
        # Assuming columns 7-10 are quaternion components (w,x,y,z)
        for i, orientation in enumerate(orientations):
            # Extract quaternion (assuming columns 7-10 are quaternion components)
            q_w, q_x, q_y, q_z = orientation[7:11]
            
            # Convert quaternion to rotation matrix for visualization
            # Simplified here - actual implementation would depend on the data format
            
            # Create coordinate system visualization (3 axes)
            color = colors[i % len(colors)]
            
            # Create x-axis
            x_axis_x = [0, 1]
            x_axis_y = [0, 0]
            x_axis_z = [0, 0]
            
            # Create y-axis
            y_axis_x = [0, 0]
            y_axis_y = [0, 1]
            y_axis_z = [0, 0]
            
            # Create z-axis
            z_axis_x = [0, 0]
            z_axis_y = [0, 0]
            z_axis_z = [0, 1]
            
            # Add traces
            fig.add_trace(go.Scatter3d(
                x=x_axis_x, y=x_axis_y, z=x_axis_z,
                mode='lines',
                line=dict(color='red', width=5),
                name=f"Orientation {i} X-axis"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=y_axis_x, y=y_axis_y, z=y_axis_z,
                mode='lines',
                line=dict(color='green', width=5),
                name=f"Orientation {i} Y-axis"
            ))
            
            fig.add_trace(go.Scatter3d(
                x=z_axis_x, y=z_axis_y, z=z_axis_z,
                mode='lines',
                line=dict(color='blue', width=5),
                name=f"Orientation {i} Z-axis"
            ))
            
        # Update layout
        fig.update_layout(
            title="3D Visualization of Crystal Orientations",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            width=800,
            height=800
        )
        
        # Save as HTML
        fig.write_html(f"{output_path}.bin.3D.html")
        logger.info(f"3D visualization saved to {output_path}.bin.3D.html")

    def _create_analysis_report(
        self,
        output_path: str, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        labels: np.ndarray,
        filtered_image: np.ndarray
    ) -> None:
        """
        Create comprehensive HTML analysis report.
        
        Args:
            output_path: Path to the output files
            orientations: Orientation data array
            spots: Spot data array
            labels: Final labels array
            filtered_image: Filtered image
        """
        # Skip if no orientations
        if orientations.size == 0:
            logger.warning("No orientations found, skipping analysis report")
            return
            
        # Ensure orientations is 2D
        if len(orientations.shape) == 1:
            orientations = np.expand_dims(orientations, axis=0)
            
        # Get report template
        template = self.config.get("visualization").report_template
        
        # Create HTML report
        html_content = []
        
        # Add header
        html_content.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Laue Diffraction Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333366; }
                .container { max-width: 1200px; margin: 0 auto; }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .image-container { margin: 20px 0; }
                .image-container img { max-width: 100%; }
                .chart-container { height: 400px; margin: 20px 0; }
                .footer { margin-top: 30px; font-size: 0.8em; color: #666; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Laue Diffraction Analysis Report</h1>
                <p class="timestamp">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        """)
        
        # Add summary section
        html_content.append("""
                <div class="summary">
                    <h2>Analysis Summary</h2>
                    <p>File analyzed: """ + os.path.basename(output_path) + """</p>
                    <p>Total orientations found: """ + str(len(orientations)) + """</p>
                    <p>Total spots indexed: """ + str(len(spots)) + """</p>
                    <p>Average spots per orientation: """ + f"{len(spots) / len(orientations):.1f}" + """</p>
                </div>
        """)
        
        # Add image section
        html_content.append("""
                <h2>Diffraction Images</h2>
                <div class="image-container">
                    <h3>Indexed Diffraction Pattern</h3>
                    <img src=""" + f'"{os.path.basename(output_path)}.bin.LabeledImage.tif"' + """ alt="Indexed Diffraction Pattern">
                </div>
                <div class="image-container">
                    <h3>Orientation Quality Map</h3>
                    <img src=""" + f'"{os.path.basename(output_path)}.bin.QualityMap.png"' + """ alt="Orientation Quality Map">
                </div>
        """)
        
        # Add orientation summary table
        html_content.append("""
                <h2>Orientation Summary</h2>
                <table>
                    <tr>
                        <th>ID</th>
                        <th>Quality</th>
                        <th>Spots</th>
                        <th>Quaternion (w,x,y,z)</th>
                        <th>Euler Angles (°)</th>
                    </tr>
        """)
        
        for i, orientation in enumerate(orientations):
            # Extract orientation parameters (adjust column indices as needed)
            orientation_id = int(orientation[0])
            quality = orientation[4]
            num_spots = orientation[5]
            
            # Extract quaternion components (assuming w,x,y,z format)
            qw, qx, qy, qz = orientation[7:11]
            
            # Convert quaternion to Euler angles (ZXZ convention)
            # This is a simplified conversion - adjust as needed based on your convention
            try:
                # Convert quaternion to Euler angles (in degrees)
                euler_z1 = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
                euler_x = np.degrees(np.arcsin(2*(qw*qy - qz*qx)))
                euler_z2 = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
                euler_angles = f"{euler_z1:.2f}, {euler_x:.2f}, {euler_z2:.2f}"
            except:
                euler_angles = "N/A"
            
            # Add table row
            html_content.append(f"""
                    <tr>
                        <td>{i}</td>
                        <td>{quality:.4f}</td>
                        <td>{int(num_spots)}</td>
                        <td>{qw:.4f}, {qx:.4f}, {qy:.4f}, {qz:.4f}</td>
                        <td>{euler_angles}</td>
                    </tr>
            """)
            
        html_content.append("</table>")
        
        # Add spot distribution visualization (simple histogram of spots per orientation)
        spots_per_orientation = {}
        for spot in spots:
            orientation_id = int(spot[0])
            if orientation_id not in spots_per_orientation:
                spots_per_orientation[orientation_id] = 0
            spots_per_orientation[orientation_id] += 1
            
        html_content.append("""
                <h2>Spot Distribution</h2>
                <div class="chart-container">
                    <canvas id="spotsChart"></canvas>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <script>
                    const ctx = document.getElementById('spotsChart');
                    new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: [""" + ", ".join([f"'{i}'" for i in range(len(orientations))]) + """],
                            datasets: [{
                                label: 'Spots per Orientation',
                                data: [""" + ", ".join([str(spots_per_orientation.get(int(orientation[0]), 0)) for orientation in orientations]) + """],
                                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Number of Spots'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Orientation ID'
                                    }
                                }
                            }
                        }
                    });
                </script>
        """)
        
        # Add links to interactive visualizations
        html_content.append("""
                <h2>Interactive Visualizations</h2>
                <p>Click the links below to open interactive visualizations:</p>
                <ul>
                    <li><a href=""" + f'"{os.path.basename(output_path)}.bin.interactive.html"' + """ target="_blank">Interactive Diffraction Pattern</a></li>
        """)
        
        if self.config.get("visualization").generate_3d:
            html_content.append(f"""
                    <li><a href="{os.path.basename(output_path)}.bin.3D.html" target="_blank">3D Orientation Visualization</a></li>
            """)
            
        html_content.append("""
                </ul>
        """)
        
        # Add processing parameters section
        html_content.append("""
                <h2>Processing Parameters</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
        """)
        
        # Add key processing parameters from config
        parameters = [
            ("Space Group", self.config.get("space_group")),
            ("Symmetry", self.config.get("symmetry")),
            ("Lattice Parameters", self.config.get("lattice_parameter")),
            ("Detector Size (pixels)", f"{self.config.get('nr_px_x')} × {self.config.get('nr_px_y')}"),
            ("Pixel Size (mm)", f"{self.config.get('px_x')} × {self.config.get('px_y')}"),
            ("Sample-Detector Distance (mm)", self.config.get("distance")),
            ("Threshold Method", self.config.get("image_processing").threshold_method),
            ("Threshold Value", self.config.get("image_processing").threshold_value),
            ("Min Good Spots", self.config.get("min_good_spots")),
            ("Watershed Enabled", "Yes" if self.config.get("image_processing").watershed_enabled else "No"),
            ("Processing Type", self.config.get("processing_type")),
            ("CPUs Used", self.config.get("num_cpus"))
        ]
        
        for param, value in parameters:
            html_content.append(f"""
                    <tr>
                        <td>{param}</td>
                        <td>{value}</td>
                    </tr>
            """)
            
        html_content.append("""
                </table>
        """)
        
        # Add footer
        html_content.append("""
                <div class="footer">
                    <p>Generated by LaueMatching Software</p>
                    <p>Contact: Hemant Sharma at hsharma@anl.gov</p>
                </div>
            </div>
        </body>
        </html>
        """)
        
        # Write HTML file
        report_path = f"{output_path}.bin.report.html"
        with open(report_path, 'w') as f:
            f.write("".join(html_content))
            
        logger.info(f"Analysis report saved to {report_path}")
        
    def _plot_orientation_spots(
        self, 
        orientations: np.ndarray, 
        spots: np.ndarray, 
        filtered_image: np.ndarray, 
        labels: np.ndarray, 
        colors, 
        ax
    ) -> None:
        """
        Plot spots for each orientation.
        
        Args:
            orientations: Orientation data array
            spots: Spot data array
            filtered_image: Filtered image
            labels: Final labels array
            colors: Color map
            ax: Matplotlib axis
        """
        # Sort orientations by quality score (column 4)
        if len(orientations.shape) == 2:
            orientations = orientations[np.argsort(-orientations[:, 4])]
        else:
            orientations = np.expand_dims(orientations, axis=0)
            
        # Track which labels have been found
        labels_found = []
        
        # Process each orientation
        for i, orientation in enumerate(orientations):
            # Get spots for this orientation
            orientation_spots = spots[spots[:, 0] == orientation[0]]
            
            # Keep only spots with intensity in filtered image
            good_spots = []
            for spot in orientation_spots:
                x, y = int(spot[5]), int(spot[6])
                if 0 <= x < filtered_image.shape[1] and 0 <= y < filtered_image.shape[0]:
                    if filtered_image[y, x] > 0:
                        good_spots.append(spot)
                        
            if not good_spots:
                continue
                
            # Convert to numpy array
            good_spots = np.array(good_spots)
            
            # Skip if not enough good spots
            min_good_spots = self.config.get("min_good_spots")
            if len(good_spots) < min_good_spots:
                continue
                
            # Check if spots are in regions that have already been found
            good_spot_indices = []
            for j, spot in enumerate(good_spots):
                x, y = int(spot[5]), int(spot[6])
                if 0 <= x < labels.shape[1] and 0 <= y < labels.shape[0]:
                    label = labels[y, x]
                    if label > 0 and label not in labels_found:
                        good_spot_indices.append(j)
                        labels_found.append(label)
                        
            if len(good_spot_indices) < min_good_spots:
                continue
                
            # Plot the spots
            good_spots_filtered = good_spots[good_spot_indices]
            ax.plot(
                good_spots_filtered[:, 5], 
                good_spots_filtered[:, 6], 
                'ks', 
                markerfacecolor='none', 
                ms=3, 
                markeredgecolor=colors(i),
                markeredgewidth=0.3,
                label=f'ID {i}'
            )
            
            # Optionally add HKL labels
            if hasattr(self.config.get("visualization"), "show_hkl_labels") and self.config.get("visualization").show_hkl_labels:
                for spot in good_spots_filtered:
                    h, k, l = int(spot[2]), int(spot[3]), int(spot[4])
                    x, y = spot[5], spot[6]
                    ax.text(
                        x, y + 2, 
                        f"({h},{k},{l})", 
                        fontsize=2, 
                        ha='center', 
                        color=colors(i)
                    )


#############################################
# 12. Command Line Interface Implementation #
#############################################

def parse_arguments():
    """Parse command line arguments with enhanced interface."""
    parser = argparse.ArgumentParser(
        description='LaueMatching - Advanced Laue Diffraction Pattern Indexing Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a single image:
    python RunImage.py process -c config.txt -i image_001.h5 -n 4
    
  Process multiple images:
    python RunImage.py process -c config.txt -i "data/*.h5" -n 4
    
  Generate a configuration file:
    python RunImage.py config -o config.json -t json
    
  View results interactively:
    python RunImage.py view -i results/image_001.h5.bin.output.h5
    
  Generate a report:
    python RunImage.py report -i results/image_001.h5.bin.output.h5 -o report.html

Command Summary:
  process - Process Laue diffraction images
    Required: -c/--config, -i/--image
    Optional: -n/--ncpus, -g/--gpu, -t/--threshold, -o/--output, --dry-run
    
  config  - Generate or validate configuration files
    Required: -o/--output (for generation), -v/--validate (for validation)
    Optional: -t/--type (when generating)
    Note: Use either generation OR validation mode, not both
    
  view    - View processing results interactively
    Required: -i/--input
    
  report  - Generate analysis reports
    Required: -i/--input
    Optional: -o/--output, -t/--template
        """
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process command
    process_parser = subparsers.add_parser('process', 
        help='Process Laue diffraction images',
        description='Process single or multiple Laue diffraction images using specified configuration',
        epilog="""
Examples:
  Basic processing:
    python RunImage.py process -c config.txt -i image_001.h5
  
  With GPU acceleration:
    python RunImage.py process -c config.txt -i image_001.h5 -g
    
  Process multiple files with 8 CPU cores:
    python RunImage.py process -c config.txt -i "data/*.h5" -n 8
    
  Test configuration without processing:
    python RunImage.py process -c config.txt -i image_001.h5 --dry-run
        """
    )
    process_parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file (required)')
    process_parser.add_argument('-i', '--image', type=str, required=True, help='Image file or glob pattern (required)')
    process_parser.add_argument('-n', '--ncpus', type=int, default=4, help='Number of CPU cores to use (default: 4)')
    process_parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for processing if available')
    process_parser.add_argument('-t', '--threshold', type=int, default=0, 
                              help='Override threshold value from configuration (default: 0, no override)')
    process_parser.add_argument('-a', '--nfiles', type=int, default=1, 
                              help='Number of files to process (default: 1, process one file only)')
    process_parser.add_argument('-o', '--output', type=str, 
                              help='Output directory (overrides config; default: results/)')
    process_parser.add_argument('--dry-run', action='store_true', 
                              help='Validate configuration without processing images')
    
    # Config command
    config_parser = subparsers.add_parser('config', 
        help='Generate or validate configuration file',
        description='Generate a new configuration file or validate an existing one',
        epilog="""
Examples:
  Generate default configuration:
    python RunImage.py config -o config.txt
    
  Generate JSON configuration:
    python RunImage.py config -o config.json -t json
    
  Validate existing configuration:
    python RunImage.py config -v existing_config.txt
        """
    )
    config_parser.add_argument('-o', '--output', type=str, 
                             help='Output configuration file (required for generation)')
    config_parser.add_argument('-t', '--type', type=str, choices=['txt', 'json', 'yaml'], default='txt', 
                            help='Configuration file format (default: txt)')
    config_parser.add_argument('-v', '--validate', type=str, 
                             help='Validate existing configuration file (incompatible with -o/--output)')
    
    # View command
    view_parser = subparsers.add_parser('view', 
        help='View processing results interactively',
        description='Launch interactive viewer for processed results',
        epilog="""
Examples:
  View a single result file:
    python RunImage.py view -i results/image_001.h5.bin.output.h5
        """
    )
    view_parser.add_argument('-i', '--input', type=str, required=True, 
                           help='Processed H5 file to view (required)')
    
    # Report command
    report_parser = subparsers.add_parser('report', 
        help='Generate analysis report',
        description='Generate detailed report from processed results',
        epilog="""
Examples:
  Generate default HTML report:
    python RunImage.py report -i results/image_001.h5.bin.output.h5 -o report.html
    
  Generate report with custom template:
    python RunImage.py report -i results/image_001.h5.bin.output.h5 -o report.pdf -t publication
        """
    )
    report_parser.add_argument('-i', '--input', type=str, required=True, 
                             help='Processed H5 file (required)')
    report_parser.add_argument('-o', '--output', type=str, 
                             help='Output report file (default: report.html)')
    report_parser.add_argument('-t', '--template', type=str, default='default', 
                             help='Report template to use (default: default)')
    
    # Add common arguments
    parser.add_argument('--loglevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                       help='Logging level (default: INFO)')
    parser.add_argument('--logfile', type=str, 
                       help='Log to file instead of console')
    
    args = parser.parse_args()
    
    # Additional validation
    if args.command == 'config' and args.output and args.validate:
        parser.error("Cannot use both -o/--output and -v/--validate together")
    
    if args.command == 'config' and not (args.output or args.validate):
        parser.error("Config command requires either -o/--output or -v/--validate")
    
    # Validate arguments
    if not args.command:
        parser.print_help()
        sys.exit(1)
        
    return args

def process_images(args):
    """
    Process images based on command line arguments.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_level = getattr(LogLevel, args.loglevel)
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)
    
    # Load configuration
    config_manager = ConfigurationManager(args.config)
    
    # Override configuration from command line arguments
    if args.gpu:
        config_manager.set("processing_type", "GPU")
    config_manager.set("num_cpus", args.ncpus)
    if args.output:
        config_manager.set("result_dir", args.output)
        
    # Create result directory
    result_dir = config_manager.get("result_dir")
    os.makedirs(result_dir, exist_ok=True)
    
    # Initialize image processor
    processor = EnhancedImageProcessor(config_manager)
    
    # Get image files
    if '*' in args.image or '?' in args.image:
        image_files = sorted(glob.glob(args.image))
        if not image_files:
            logger.error(f"No files found matching pattern: {args.image}")
            sys.exit(1)
    else:
        # Check if this is a single file or a pattern like FILESTEM_XX.h5
        if args.nfiles > 1:
            # Legacy multi-file handling
            image_files = get_image_files(args.image, args.nfiles)
        else:
            image_files = [args.image]
            
    logger.info(f"Found {len(image_files)} images to process")
    
    # Dry run check
    if args.dry_run:
        logger.info("Dry run completed, configuration is valid")
        return
        
    # Process each image
    results = []
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file}")
        
        # Reset doFwd after first image
        if i > 0:
            config_manager.set("do_forward", 0)
            config_manager.write_config()
            
        # Process the image
        result = processor.process_image(image_file, args.threshold)
        results.append(result)
        
        # Log result
        if result["success"]:
            logger.info(f"Successfully processed image: {image_file}")
        else:
            logger.error(f"Failed to process image: {image_file}: {result.get('error', 'Unknown error')}")
            
    # Log summary
    successful = sum(1 for r in results if r.get("success", False))
    logger.info(f"Processing complete: {successful}/{len(image_files)} images processed successfully")
    
    return results


def generate_config(args):
    """
    Generate a configuration file.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_level = getattr(LogLevel, args.loglevel)
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)
    
    # Create default configuration
    config = LaueConfig()
    
    # If validating an existing config, load it first
    if args.validate:
        try:
            temp_config = ConfigurationManager(args.validate)
            config = temp_config.config
            logger.info(f"Configuration validated: {args.validate}")
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            sys.exit(1)
    
    # Determine output format
    output_format = args.type.lower()
    
    # Create configuration manager with the config
    config_manager = ConfigurationManager(args.output)
    config_manager.config = config
    
    # Write configuration file
    try:
        if output_format == 'json':
            with open(args.output, 'w') as f:
                json.dump(config.to_dict(), f, indent=4)
        elif output_format == 'yaml':
            with open(args.output, 'w') as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            config_manager.write_config()
            
        logger.info(f"Configuration file generated: {args.output}")
        
    except Exception as e:
        logger.error(f"Error generating configuration file: {str(e)}")
        sys.exit(1)


def view_results(args):
    """
    View processing results interactively.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_level = getattr(LogLevel, args.loglevel)
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    logger.info(f"Opening viewer for: {args.input}")
    
    # This would normally launch an interactive viewer
    # For this implementation, we'll just recreate interactive html
    try:
        # Create temporary config
        config = LaueConfig()
        config_manager = ConfigurationManager("temp_config.txt")
        config_manager.config = config
        
        # Create processor
        processor = EnhancedImageProcessor(config_manager)
        
        # Load data from H5 file
        with h5py.File(args.input, 'r') as h5_file:
            # Load filtered image
            if '/entry/data/cleaned_data_threshold_filtered' in h5_file:
                filtered_image = np.array(h5_file['/entry/data/cleaned_data_threshold_filtered'])
            else:
                filtered_image = np.array(h5_file['/entry/data/cleaned_data_threshold'])
                
            # Load labels
            if '/entry/data/cleaned_data_threshold_filtered_labels' in h5_file:
                labels = np.array(h5_file['/entry/data/cleaned_data_threshold_filtered_labels'])
            else:
                labels = np.zeros_like(filtered_image)
                
            # Load orientations and spots
            orientations = np.array(h5_file['/entry/results/filtered_orientations'])
            spots = np.array(h5_file['/entry/results/filtered_spots'])
            
        # Generate interactive visualization
        output_path = os.path.splitext(args.input)[0]
        processor._create_interactive_visualization(
            output_path, orientations, spots, labels, filtered_image
        )
        
        # Open the interactive HTML (platform-dependent)
        html_path = f"{output_path}.interactive.html"
        logger.info(f"Interactive visualization saved to: {html_path}")
        
        # Try to open in browser
        try:
            import webbrowser
            webbrowser.open(html_path)
        except:
            logger.info(f"Please open {html_path} in your web browser")
            
    except Exception as e:
        logger.error(f"Error viewing results: {str(e)}")
        sys.exit(1)


def generate_report(args):
    """
    Generate an analysis report.
    
    Args:
        args: Command line arguments
    """
    # Set up logging
    log_level = getattr(LogLevel, args.loglevel)
    global logger
    logger = setup_logger(level=log_level, log_file=args.logfile)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
        
    # Create temporary config
    config = LaueConfig()
    config.visualization.report_template = args.template
    config_manager = ConfigurationManager("temp_config.txt")
    config_manager.config = config
    
    # Create processor
    processor = EnhancedImageProcessor(config_manager)
    
    try:
        # Load data from H5 file
        with h5py.File(args.input, 'r') as h5_file:
            # Load filtered image
            if '/entry/data/cleaned_data_threshold_filtered' in h5_file:
                filtered_image = np.array(h5_file['/entry/data/cleaned_data_threshold_filtered'])
            else:
                filtered_image = np.array(h5_file['/entry/data/cleaned_data_threshold'])
                
            # Load labels
            if '/entry/data/cleaned_data_threshold_filtered_labels' in h5_file:
                labels = np.array(h5_file['/entry/data/cleaned_data_threshold_filtered_labels'])
            else:
                labels = np.zeros_like(filtered_image)
                
            # Load orientations and spots
            orientations = np.array(h5_file['/entry/results/filtered_orientations'])
            spots = np.array(h5_file['/entry/results/filtered_spots'])
            
        # Determine output path
        if args.output:
            output_path = os.path.splitext(args.output)[0]
        else:
            output_path = os.path.splitext(args.input)[0]
            
        # Generate report
        processor._create_analysis_report(
            output_path, orientations, spots, labels, filtered_image
        )
        
        logger.info(f"Report generated: {output_path}.report.html")
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        sys.exit(1)


def get_image_files(image_file: str, n_files: int) -> List[str]:
    """
    Get list of image files to process.
    
    Args:
        image_file: Base image file name
        n_files: Number of files to process
        
    Returns:
        List of image file paths
    """
    if n_files == 1:
        return [image_file]
        
    # Parse file name pattern
    file_parts = image_file.split('_')
    file_stem = '_'.join(file_parts[:-1])
    last_part = file_parts[-1]
    
    # Get extension and number
    ext_parts = last_part.split('.')
    if len(ext_parts) > 1:
        ext = f".{'.'.join(ext_parts[1:])}"
        start_num = int(ext_parts[0])
    else:
        ext = ""
        start_num = int(last_part)
        
    # Generate file names
    image_files = []
    for i in range(start_num, start_num + n_files):
        image_files.append(f"{file_stem}_{i}{ext}")
        
    logger.info(f"Files to be processed: {image_files}")
    return image_files


def main():
    """Main function to run the Laue matching process."""
    print("LaueMatching - Advanced Laue Diffraction Pattern Indexing Tool")
    print("Contact: Hemant Sharma at hsharma@anl.gov")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Execute command
    if args.command == 'process':
        process_images(args)
    elif args.command == 'config':
        generate_config(args)
    elif args.command == 'view':
        view_results(args)
    elif args.command == 'report':
        generate_report(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()