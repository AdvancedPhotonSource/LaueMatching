#!/usr/bin/env python
"""
laue_config.py — Configuration system for LaueMatching

Contains all configuration dataclasses, the ConfigurationManager, and the
ProgressReporter.  Extracted from RunImage.py so that it can be reused by
the streaming pipeline scripts as well.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None  # YAML support is optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # tqdm is optional — ProgressReporter degrades gracefully

# REFACTOR_PLAN §6.4: one declarative schema drives config parse + write
# (replaces the hand-synced elif chain + write block).  Ensure laue_index is
# importable (repo root one level above scripts/).
_INSTALL_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _INSTALL_PATH not in sys.path:
    sys.path.insert(0, _INSTALL_PATH)
from laue_index import config_schema as _schema  # noqa: E402


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

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
    _logger = logging.getLogger(name)
    _logger.setLevel(level.value)

    # Clear any existing handlers
    _logger.handlers = []

    # Add file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

    return _logger


# Module-level logger (used by ConfigurationManager)
logger = logging.getLogger("LaueMatching")


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

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
    enable_visualization: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SimulationConfig:
    """Configuration parameters for diffraction simulation."""
    enable_simulation: bool = True
    skip_percentage: float = 0.0
    orientation_file: str = "orientations.txt"
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
    r_array: str = "-1.2 -1.2 -1.2"
    p_array: str = "0.02 0.002 0.513"
    min_good_spots: int = 5
    max_laue_spots: int = 7
    min_nr_spots: int = 5
    # Twin/CSL-aware robust orientation filter (default on). When True, real
    # Sigma3 twins are not deleted by the unique-spot dedup; set False for the
    # legacy unique-spot-only filter.
    robust_filter: bool = True
    # Per-thread orientation batch size for the indexer.  Bounds memory:
    # peak RAM ~= numProcs * batch_size * (1 + 2*max_laue_spots) * 2 bytes.
    batch_size: int = 1_000_000

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
    orientation_spacing: float = 0.4
    distance: float = 0.513
    min_intensity: float = 50.0
    elo: float = 5.0
    ehi: float = 30.0
    maxAngle: float = 2.0

    # Processing parameters
    do_forward: bool = True
    processing_type: str = "CPU"
    num_cpus: int = 60

    # Enhanced configuration sections
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Additional parameters
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None

    # Optional IndexFile metadata (used by scripts/laue_indexfile.py)
    xtal_file: str = ""           # path to a CIF/xml crystal description (optional)
    structure_desc: str = ""      # short structure tag, e.g. "Ni", "Cu"
    atom_description: str = ""    # raw ``AtomDesctiption`` line contents (sic)
    write_indexfile: bool = True  # emit .indexing.txt alongside output HDF5 (runtime flag)

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
        config = config_dict.copy()

        # Handle nested configurations
        img_config = config.pop("image_processing", {})
        vis_config = config.pop("visualization", {})
        sim_config = config.pop("simulation", {})

        # Handle enum conversion
        if "log_level" in config:
            config["log_level"] = LogLevel[config["log_level"]]

        # Create instance - Ensure only valid fields are passed
        valid_fields = {f.name for f in dataclass_fields(cls) if f.init}
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


# ---------------------------------------------------------------------------
# Configuration Manager
# ---------------------------------------------------------------------------

class ConfigurationManager:
    """Manages configuration for the Laue matching process."""

    def __init__(self, config_file: str):
        """
        Initialize configuration manager with parameters from config file.

        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = LaueConfig()  # Initialize with defaults first
        self._load_config()

    def _load_config(self) -> None:
        """Parse the configuration file and load parameters."""
        if not os.path.exists(self.config_file):
            logger.error(f"Configuration file {self.config_file} not found.")
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
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
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

    def _parse_classic_config_line(self, line: str) -> None:
        """Parse one classic-format config line via the declarative schema
        (REFACTOR_PLAN §6.4 — replaces the hand-written elif chain)."""
        if not _schema.parse_line(self.config, line):
            parts = line.split()
            key = parts[0] if parts else line
            logger.warning(f"Ignoring unknown configuration key '{key}' on line: '{line}'")

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

        # Sync simulation energies with Elo/Ehi if SimulationEnergies isn't explicitly set
        if self.config.simulation.energies == SimulationConfig().energies:
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

    def _write_to_json(self) -> None:
        """Write configuration to JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=4)

    def _write_to_yaml(self) -> None:
        """Write configuration to YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config.to_dict(), f, default_flow_style=False)

    def _write_to_text(self) -> None:
        """Write configuration to classic text format from the declarative
        schema (REFACTOR_PLAN §6.4 — replaces the parallel f.write block)."""
        with open(self.config_file, 'w') as f:
            f.write(_schema.render_text(
                self.config,
                header_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

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
                        # If not a direct nested object, maybe it's a top-level key
                        key_to_set = '_'.join(config_key_parts)
                        target_obj = self.config

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
                        elif isinstance(current_value, LogLevel):
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


# ---------------------------------------------------------------------------
# Progress Reporter
# ---------------------------------------------------------------------------

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
        if tqdm is not None:
            self.progress_bar = tqdm(total=total_steps, desc=description, unit="step", ncols=100)
        else:
            self.progress_bar = None

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
        if self.progress_bar is not None:
            self.progress_bar.update(step_increment)
            if status:
                self.progress_bar.set_description(f"{self.description}: {status}")

        # Calculate statistics
        elapsed = current_time - self.start_time
        if self.current_step > 0 and self.total_steps > 0:
             percentage = min(100.0 * self.current_step / self.total_steps, 100.0)
             if self.current_step < self.total_steps:
                remaining = elapsed * (self.total_steps - self.current_step) / self.current_step
                if (percentage % 10 < (percentage - step_increment * 100.0 / self.total_steps) % 10 or
                        current_time - self.last_update_time > 5):
                    self.last_update_time = current_time
                    logger.info(f"Progress: {percentage:.1f}% ({self.current_step}/{self.total_steps}), "
                               f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
             else:
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
        if remaining_steps > 0 and self.progress_bar is not None:
            self.progress_bar.update(remaining_steps)
        self.current_step = self.total_steps

        if self.progress_bar is not None:
            self.progress_bar.set_description(f"{self.description}: {status}")
            self.progress_bar.close()

        elapsed = time.time() - self.start_time
        logger.info(f"Operation '{self.description}' {status.lower()} in {elapsed:.2f} seconds")
