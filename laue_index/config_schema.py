"""Declarative configuration schema — one table drives parse + write + docs.

REFACTOR_PLAN §5 / §6.4.  The legacy config had a hand-written
``_parse_classic_config_line`` elif chain AND a parallel ``_write_to_text``
block kept in sync by hand (pain point #4: we tripped on max_angle vs maxAngle).
This module is the single declarative source: a list of :class:`Param` rows that
both the parser and the writer iterate, so a key can never drift between them.

The engine operates on any object exposing the target attributes (the legacy
``LaueConfig`` dataclass and its nested ``image_processing`` / ``visualization``
/ ``simulation`` sub-objects), so this stays independent of that class.

Behaviour note: parsing reproduces the legacy parsed *values* exactly (pinned by
tests/test_char_config.py::config_laueconfig_todict).  The written text is
regenerated with *consistent* formatting (the old block mixed pad widths); it
remains key-value and order/comment-insensitive for the C parser, and the
round-trip stays idempotent.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("LaueMatching")

__all__ = ["Param", "SCHEMA", "SCHEMA_BY_KEY", "parse_line", "render_text"]


@dataclass(frozen=True)
class Param:
    key: str               # text-file key, e.g. "MaxAngle"
    field: str             # attribute name on the target object
    type: type             # int | float | str | bool
    default: object
    target: str = "config"  # "config" | "image_processing" | "visualization" | "simulation"
    section: str = ""      # write grouping / docs
    doc: str = ""          # inline write comment + docs
    nvals: int = 1         # >1: join this many tokens into a space-separated string
    write: bool = True     # emit in render_text()
    kind: str = "scalar"   # scalar | multi | symmetry | threshold_method | atom_desc


# Section order = write order.
_CRYSTAL = "Crystal Parameters"
_DET = "Detector Parameters"
_HKL = "HKL Generation Parameters"
_IDX = "Indexing Parameters (Executable)"
_FILT = "Orientation Filtering (Python)"
_IMG = "Image Processing (Python)"
_PATHS = "File Paths"
_CTRL = "Processing Control"
_VIS = "Visualization Parameters (Python)"
_SIM = "Simulation Parameters (Python GenerateSimulation.py)"
_META = "IndexFile Metadata"

SCHEMA = [
    # --- Crystal ---
    Param("SpaceGroup", "space_group", int, 225, "config", _CRYSTAL),
    Param("Symmetry", "symmetry", str, "F", "config", _CRYSTAL, kind="symmetry"),
    Param("LatticeParameter", "lattice_parameter", str, "0.3615 0.3615 0.3615 90 90 90",
          "config", _CRYSTAL, nvals=6, kind="multi"),
    Param("R_Array", "r_array", str, "-1.2 -1.2 -1.2", "config", _CRYSTAL, nvals=3, kind="multi"),
    Param("P_Array", "p_array", str, "0.02 0.002 0.513", "config", _CRYSTAL, nvals=3, kind="multi"),
    # --- Detector ---
    Param("NrPxX", "nr_px_x", int, 2048, "config", _DET),
    Param("NrPxY", "nr_px_y", int, 2048, "config", _DET),
    Param("PxX", "px_x", float, 0.2, "config", _DET),
    Param("PxY", "px_y", float, 0.2, "config", _DET),
    Param("OrientationSpacing", "orientation_spacing", float, 0.4, "config", _DET),
    # --- HKL ---
    Param("Elo", "elo", float, 5.0, "config", _HKL),
    Param("Ehi", "ehi", float, 30.0, "config", _HKL),
    # --- Indexing ---
    Param("MinNrSpots", "min_nr_spots", int, 5, "config", _IDX),
    Param("MaxNrLaueSpots", "max_laue_spots", int, 7, "config", _IDX),
    Param("BatchSize", "batch_size", int, 1_000_000, "config", _IDX,
          doc="Indexer batch size; bounds peak RAM"),
    Param("MaxAngle", "maxAngle", float, 2.0, "config", _IDX),
    Param("MinIntensity", "min_intensity", float, 50.0, "config", _IDX,
          doc="(May be deprecated by threshold methods)"),
    # --- Filtering ---
    Param("MinGoodSpots", "min_good_spots", int, 5, "config", _FILT,
          doc="Min unique spots to keep orientation"),
    Param("RobustFilter", "robust_filter", bool, True, "config", _FILT,
          doc="1=twin/CSL-aware filter (keep Sigma3 twins), 0=legacy unique-spot only"),
    # --- Image Processing ---
    Param("ThresholdMethod", "threshold_method", str, "adaptive", "image_processing", _IMG,
          doc="options: adaptive, otsu, fixed, percentile", kind="threshold_method"),
    Param("Threshold", "threshold_value", float, 0.0, "image_processing", _IMG,
          doc="Used only if ThresholdMethod is 'fixed'"),
    Param("ThresholdPercentile", "threshold_percentile", float, 90.0, "image_processing", _IMG,
          doc="Used only if ThresholdMethod is 'percentile'"),
    Param("MinArea", "min_area", int, 10, "image_processing", _IMG),
    Param("FilterRadius", "filter_radius", int, 101, "image_processing", _IMG),
    Param("NMeadianPasses", "median_passes", int, 1, "image_processing", _IMG),
    Param("WatershedImage", "watershed_enabled", bool, True, "image_processing", _IMG),
    Param("EnhanceContrast", "enhance_contrast", bool, False, "image_processing", _IMG),
    Param("DenoiseImage", "denoise_image", bool, False, "image_processing", _IMG),
    Param("DenoiseStrength", "denoise_strength", float, 1.0, "image_processing", _IMG),
    Param("EdgeEnhancement", "edge_enhancement", bool, False, "image_processing", _IMG),
    # --- File Paths ---
    Param("ResultDir", "result_dir", str, "results", "config", _PATHS),
    Param("OrientationFile", "orientation_file", str, "orientations.bin", "config", _PATHS,
          doc="Input orientation database"),
    Param("HKLFile", "hkl_file", str, "hkls.bin", "config", _PATHS),
    Param("BackgroundFile", "background_file", str, "median.bin", "config", _PATHS),
    Param("ForwardFile", "forward_file", str, "forward.bin", "config", _PATHS,
          doc="Output from executable forward sim?"),
    # --- Processing Control ---
    Param("DoFwd", "do_forward", bool, True, "config", _CTRL,
          doc="Enable forward sim in executable?"),
    # --- Visualization ---
    Param("EnableVisualization", "enable_visualization", bool, False, "visualization", _VIS),
    # --- Simulation ---
    Param("EnableSimulation", "enable_simulation", bool, False, "simulation", _SIM),
    Param("SkipPercentage", "skip_percentage", float, 0.0, "simulation", _SIM),
    Param("SimulationEnergies", "energies", str, "5.0 30.0", "simulation", _SIM,
          nvals=2, kind="multi"),
    # --- IndexFile metadata (parse-only; not written) ---
    Param("XtalFile", "xtal_file", str, "", "config", _META, write=False),
    Param("StructureDesc", "structure_desc", str, "", "config", _META, write=False),
    Param("AtomDescription", "atom_description", str, "", "config", _META,
          write=False, kind="atom_desc"),
]

# Alias keys (historical mis-spelling) -> canonical Param.
_ALIASES = {"AtomDesctiption": "AtomDescription"}
# Recognised but unused keys (consumed silently, like the legacy parser).
_IGNORED = {"AStar", "SimulationSmoothingWidth"}

SCHEMA_BY_KEY = {p.key: p for p in SCHEMA}


def _coerce(value: str, typ: type):
    if typ is bool:
        return bool(int(value))
    return typ(value)


def _target(config, param: Param):
    return config if param.target == "config" else getattr(config, param.target)


def parse_line(config, line: str) -> bool:
    """Apply one classic-format config line to *config*. Returns True if handled.

    Reproduces the legacy ``_parse_classic_config_line`` semantics (value
    coercion, multi-value join, Symmetry / ThresholdMethod validation,
    P_Array->distance handled by the caller's _sync, AtomDescription rest-of-line,
    ignored keys), raising ValueError on malformed required values.
    """
    if "#" in line:
        line = line[:line.index("#")].strip()
    parts = line.split()
    if not parts:
        return True
    key = parts[0]
    n = len(parts)

    if key in _IGNORED:
        return True
    canonical = _ALIASES.get(key, key)
    param = SCHEMA_BY_KEY.get(canonical)
    if param is None:
        return False  # caller logs "unknown key"

    if param.kind == "atom_desc":
        setattr(config, param.field, line.split(None, 1)[1] if n > 1 else "")
        return True

    if param.kind == "multi":
        if n != param.nvals + 1:
            logger.error(f"Incorrect number of values for {key}. "
                         f"Expected {param.nvals}, got {n - 1}.")
            raise ValueError(f"Incorrect {key} format")
        setattr(_target(config, param), param.field, " ".join(parts[1:param.nvals + 1]))
        return True

    # scalar-ish: need at least one value
    if n < 2:
        logger.error(f"Missing value for {key} on line: '{line}'.")
        raise ValueError(f"Missing value for {key}")

    if param.kind == "symmetry":
        sym = parts[1]
        if sym not in 'FICARPB' or len(sym) != 1:
            logger.error('Invalid value for Symmetry, must be one character from F,I,C,A,R,P,B')
            raise ValueError('Invalid Symmetry')
        config.symmetry = sym
        return True

    if param.kind == "threshold_method":
        method = parts[1].lower()
        if method in ("adaptive", "otsu", "fixed", "percentile"):
            setattr(_target(config, param), param.field, method)
        else:
            logger.warning(f"Unknown ThresholdMethod '{parts[1]}'. Using default "
                           f"'{getattr(_target(config, param), param.field)}'.")
        return True

    try:
        value = _coerce(parts[1], param.type)
    except (ValueError, IndexError):
        logger.error(f"Invalid value format for {key} on line: '{line}'. "
                     f"Expected {param.type}.")
        raise ValueError(f"Invalid format for {key}")
    setattr(_target(config, param), param.field, value)
    return True


def render_text(config, header_timestamp: str | None = None) -> str:
    """Render *config* to the classic text format from the schema (consistent
    formatting; key-value, C-parseable, round-trip idempotent)."""
    pad = max(len(p.key) for p in SCHEMA if p.write) + 1
    lines = ["# LaueMatching Configuration File"]
    if header_timestamp:
        lines.append(f"# Generated on: {header_timestamp}")
    lines.append("")

    current_section = None
    for p in SCHEMA:
        if not p.write:
            continue
        if p.section != current_section:
            if current_section is not None:
                lines.append("")
            lines.append(f"# --- {p.section} ---")
            current_section = p.section
        raw = getattr(_target(config, p), p.field)
        val = int(raw) if p.type is bool else raw
        line = f"{p.key:<{pad}}{val}"
        if p.doc:
            line += f" # {p.doc}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines) + "\n"
