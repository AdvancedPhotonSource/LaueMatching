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


def _sd(key: str):
    """Default for *key* from config_schema.SCHEMA: the ONE place defaults live
    (0.8). This dataclass used to carry its own, and several disagreed with the
    schema and with the streaming parser (NMeadianPasses 5 vs 1,
    EnableSimulation/EnableVisualization True vs 0, PxX 0.2 not metres)."""
    return _schema.SCHEMA_BY_KEY[key].default


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
    threshold_method: str = _sd("ThresholdMethod")  # adaptive, otsu, fixed, or percentile
    threshold_value: float = _sd("Threshold")       # Used only if threshold_method is 'fixed'
    threshold_percentile: float = _sd("ThresholdPercentile") # Used only if threshold_method is 'percentile'
    min_area: int = _sd("MinArea")
    # Cap (px) on the automatic matching-blur sigma, 0 = none. Applied by
    # laue_index.preprocess (streaming) and RunImage. See config_schema.
    gauss_sigma_max: float = _sd("GaussSigmaMax")
    # 0 = choose automatically (laue_index.workers). See config_schema.
    preprocess_workers: int = _sd("PreprocessWorkers")
    # Detector positions whose spots must not count as evidence (a known substrate,
    # or the spots an accepted orientation already explains between iterative
    # passes). Consumed in laue_index.preprocess.  NOTE: the FIELD LIST of this
    # dataclass is hand-maintained (only the defaults come from
    # config_schema.SCHEMA, via _sd) -- a key present in the schema but missing
    # here is parsed and then silently dropped, so the run proceeds with the
    # exclusion quietly not applied.  Keep the two in step.
    exclude_spots_file: str = _sd("ExcludeSpotsFile")
    exclude_spots_dir: str = _sd("ExcludeSpotsDir")
    filter_radius: int = _sd("FilterRadius")
    median_passes: int = _sd("NMeadianPasses")
    watershed_enabled: bool = _sd("WatershedImage")
    gaussian_factor: float = _sd("GaussianFactor")
    enhance_contrast: bool = _sd("EnhanceContrast")
    denoise_image: bool = _sd("DenoiseImage")
    denoise_strength: float = _sd("DenoiseStrength")
    edge_enhancement: bool = _sd("EdgeEnhancement")

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
    enable_visualization: bool = _sd("EnableVisualization")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SimulationConfig:
    """Configuration parameters for diffraction simulation."""
    enable_simulation: bool = _sd("EnableSimulation")
    skip_percentage: float = _sd("SkipPercentage")
    orientation_file: str = "orientations.txt"
    energies: str = _sd("SimulationEnergies")  # Energy range in keV (Elo Ehi)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class LaueConfig:
    """Main configuration class for Laue matching."""
    # Core parameters
    space_group: int = _sd("SpaceGroup")
    symmetry: str = _sd("Symmetry")
    lattice_parameter: str = _sd("LatticeParameter")
    r_array: str = _sd("R_Array")
    p_array: str = _sd("P_Array")
    # Exclusive-spot floor for the orientation filter: WINNER-TAKE-ALL across
    # the frame's orientations (laue_index.filtering.calculate_unique_spots),
    # not a count of distinct observed peaks.
    min_good_spots: int = _sd("MinGoodSpots")
    max_laue_spots: int = _sd("MaxNrLaueSpots")
    min_nr_spots: int = _sd("MinNrSpots")
    # Twin/CSL-aware robust orientation filter (default on). When True, a real
    # Sigma3 twin is not deleted just because the winner-take-all assignment
    # gave its shared reflections to the parent; set False for the legacy
    # filter (exclusive-spot count only).
    robust_filter: bool = _sd("RobustFilter")
    # Per-thread orientation batch size for the indexer.  Bounds memory:
    # peak RAM ~= numProcs * batch_size * (1 + 2*max_laue_spots) * 2 bytes.
    batch_size: int = _sd("BatchSize")

    # File paths
    result_dir: str = _sd("ResultDir")
    orientation_file: str = _sd("OrientationFile")
    hkl_file: str = _sd("HKLFile")
    background_file: str = _sd("BackgroundFile")
    forward_file: str = _sd("ForwardFile")

    # Detector parameters
    px_x: float = _sd("PxX")
    px_y: float = _sd("PxY")
    nr_px_x: int = _sd("NrPxX")
    nr_px_y: int = _sd("NrPxY")
    orientation_spacing: float = _sd("OrientationSpacing")
    distance: float = 0.513
    min_intensity: float = _sd("MinIntensity")
    elo: float = _sd("Elo")
    ehi: float = _sd("Ehi")
    maxAngle: float = _sd("MaxAngle")
    # Read by the C binaries straight from the params file; mirrored here so
    # config_schema validates them (FRACTIONS in [0, 1)) and a rewrite keeps them.
    min_spot_intensity: float = _sd("MinSpotIntensity")
    tol_lat_c: str = _sd("tol_LatC")
    tol_c_over_a: float = _sd("tol_c_over_a")

    # Processing parameters
    do_forward: bool = _sd("DoFwd")
    processing_type: str = "CPU"
    num_cpus: int = 60

    # Enhanced configuration sections
    image_processing: ImageProcessingConfig = field(default_factory=ImageProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # Additional parameters
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[str] = None

    # Optional IndexFile metadata (used by laue_indexfile.py, alongside this file)
    xtal_file: str = _sd("XtalFile")           # path to a CIF/xml crystal description (optional)
    structure_desc: str = _sd("StructureDesc")      # short structure tag, e.g. "Ni", "Cu"
    atom_description: str = _sd("AtomDescription")    # raw ``AtomDesctiption`` line contents (sic)
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

            # Whole-file checks that one line cannot decide (tol_LatC depends
            # on tol_c_over_a, which may come later in the file).
            _schema.validate_tolerances(self.config)

            # Sync potentially inconsistent parameters
            self._sync_parameters()

        except Exception as e:
            logger.error(f"Error reading or parsing configuration file '{self.config_file}': {str(e)}")
            sys.exit(1)

    def _check_required_dict(self, config_dict) -> None:
        """REQUIRED_KEYS for a JSON/YAML config, which is keyed by field name."""
        present = set()
        for key in _schema.REQUIRED_KEYS:
            prm = _schema.SCHEMA_BY_KEY[key]
            d = config_dict if prm.target == "config" else (config_dict or {}).get(prm.target, {})
            if isinstance(d, dict) and prm.field in d:
                present.add(key)
        missing = _schema.missing_required(present)
        if missing:
            raise _schema.FatalConfigError(
                _schema.missing_required_message(missing, self.config_file))

    def _load_from_json(self) -> None:
        """Load configuration from JSON file."""
        with open(self.config_file, 'r') as f:
            config_dict = json.load(f)
            self._check_required_dict(config_dict)
            self.config = LaueConfig.from_dict(config_dict)

    def _load_from_yaml(self) -> None:
        """Load configuration from YAML file."""
        if yaml is None:
            raise ImportError("PyYAML is required for YAML config files: pip install pyyaml")
        with open(self.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            self._check_required_dict(config_dict)
            self.config = LaueConfig.from_dict(config_dict)

    def _load_from_text(self) -> None:
        """Load configuration from classic text format."""
        with open(self.config_file, 'r') as f:
            lines = f.readlines()

        present = set()
        for line_num, line in enumerate(lines):
             line_content = line.strip()
             if line_content and not line_content.startswith('#'):
                key_tok = line_content.split('#', 1)[0].split()
                if key_tok:
                    present.add(_schema._ALIASES.get(key_tok[0], key_tok[0]))
                try:
                    self._parse_classic_config_line(line_content)
                except _schema.FatalConfigError as e:
                    # No safe default (config_schema.FATAL_KEYS: SpaceGroup,
                    # Symmetry, LatticeParameter, P_Array, R_Array, Elo, Ehi):
                    # stop here -- _load_config exits non-zero
                    # -- instead of running on the built-in Ni geometry.
                    raise _schema.FatalConfigError(
                        f"line {line_num + 1} of {self.config_file}: {e}") from e
                except Exception as e:
                    logger.error(f"Error parsing line {line_num + 1} in {self.config_file}: '{line_content}' - {str(e)}")
        # Keys with no default (config_schema.REQUIRED_KEYS): all missing ones
        # named at once; _load_config exits non-zero.
        missing = _schema.missing_required(present)
        if missing:
            raise _schema.FatalConfigError(
                _schema.missing_required_message(missing, self.config_file))

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
