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

__all__ = ["Param", "SCHEMA", "SCHEMA_BY_KEY", "parse_line", "render_text",
           "FatalConfigError", "FATAL_KEYS", "validate_tolerances",
           "parse_space_group", "SYMMETRY_LETTERS", "SYMMETRY_RULE"]


class FatalConfigError(ValueError):
    """A key with no safe default carries a value that cannot be parsed.

    Raised instead of the per-line ValueError so ConfigurationManager does not
    log it and carry on with the built-in default (SpaceGroup 225, the Ni
    lattice, a 0.513 m detector distance): a run on those is a run on someone
    else's geometry that looks like it worked.
    """


# Keys whose built-in default is another experiment's value, so a malformed line
# (e.g. the templates' literal ``__SET_ME__``) must stop the run, not fall back.
# Elo/Ehi: the band's 5/30 keV default is not this beamline's either; the C
# reads them with an unchecked sscanf, so an unparseable value silently keeps
# 5/30 there.
FATAL_KEYS = frozenset({"SpaceGroup", "Symmetry", "LatticeParameter",
                        "P_Array", "R_Array", "Elo", "Ehi"})


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
    kind: str = "scalar"   # scalar | multi | fraction | symmetry | threshold_method | atom_desc


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
    Param("MinSpotIntensity", "min_spot_intensity", float, 0.0, "config", _IDX,
          doc="Fit stage: a predicted spot counts only if the blurred image "
              "exceeds this at its pixel. 0 = the C default"),
    # Crystal-fit tolerances. FRACTIONS, not percent: the C forms the bounds as
    # value * (1 -/+ tol), so 1.0 would put the lower bound at zero. Validated
    # here like the C does (validateCrystalFitTolerances in
    # LaueMatchingHeaders.h): >= 1 (or negative / NaN) is rejected, > 0.1 warns.
    # The C reads these keys from the params file itself; the rows exist so the
    # Python side validates them, keeps them on a rewrite, and records them in
    # provenance instead of logging "unknown configuration key".
    Param("tol_LatC", "tol_lat_c", str, "0 0 0 0 0 0", "config", _IDX,
          doc="Lattice-fit tolerance per a b c alpha beta gamma, as FRACTIONS "
              "(0.01 = 1%); 0 holds that parameter fixed. Ignored by the C "
              "when tol_c_over_a is non-zero",
          nvals=6, kind="fraction"),
    Param("tol_c_over_a", "tol_c_over_a", float, 0.0, "config", _IDX,
          doc="c/a fit at constant volume, as a FRACTION (0.01 = 1%); "
              "0 disables. Overrides tol_LatC",
          kind="fraction"),
    # --- Filtering ---
    Param("MinGoodSpots", "min_good_spots", int, 5, "config", _FILT,
          doc="Min EXCLUSIVE spots to keep an orientation: winner-take-all "
              "across the frame's orientations (a spot claimed by a better "
              "orientation does not count), not distinct observed peaks"),
    # Default 1 on the RunImage path. The STREAMING path treats an absent key as
    # 0 (0.7.1 behaviour) and says so at startup; an explicit key is honoured on
    # both. See laue_postprocess._robust_in_force.
    Param("RobustFilter", "robust_filter", bool, True, "config", _FILT,
          doc="1=twin/CSL-aware filter (keep Sigma3 twins), 0=legacy exclusive "
              "(winner-take-all) spot count only"),
    # --- Image Processing ---
    Param("ThresholdMethod", "threshold_method", str, "adaptive", "image_processing", _IMG,
          doc="options: adaptive, otsu, fixed, percentile", kind="threshold_method"),
    Param("Threshold", "threshold_value", float, 0.0, "image_processing", _IMG,
          doc="Used only if ThresholdMethod is 'fixed'"),
    Param("ThresholdPercentile", "threshold_percentile", float, 90.0, "image_processing", _IMG,
          doc="Used only if ThresholdMethod is 'percentile'"),
    Param("MinArea", "min_area", int, 10, "image_processing", _IMG),
    Param("GaussSigmaMax", "gauss_sigma_max", float, 0.0, "image_processing", _IMG,
          doc="Cap (px) on the automatic matching-blur sigma; 0 = no cap. "
              "Applied on both the streaming and the RunImage path"),
    Param("PreprocessWorkers", "preprocess_workers", int, 0, "image_processing", _IMG,
          doc="Upper bound on parallel preprocessing worker processes. 0 (default) "
              "lets laue_index.workers.choose_preprocess_workers decide from the "
              "usable CPU count and the memory this frame size needs per worker. "
              "Set it only to cap the pool below what the machine could feed; it "
              "cannot raise the count above the memory budget. The environment "
              "variable LAUE_PREPROCESS_WORKERS overrides both."),
    Param("ExcludeSpotsDir", "exclude_spots_dir", str, "", "image_processing", _IMG,
          doc="Directory of PER-FRAME exclusion lists for iterative indexing: one "
              "'<frame-stem>.txt' of 'x y [radius]' rows per frame, holding the spots "
              "an already-accepted orientation explained on THAT frame. A missing file "
              "means nothing to exclude there, which is normal once a frame is "
              "exhausted. Applied on top of ExcludeSpotsFile, not instead of it, so a "
              "static substrate list and a per-frame residual list compose."),
    Param("ExcludeSpotsFile", "exclude_spots_file", str, "", "image_processing", _IMG,
          doc="Detector positions whose spots must NOT count as evidence. .npy bool "
              "mask (NrPxY,NrPxX), or a text file of 'x y [radius]' rows. Whole "
              "connected components centred inside are dropped AFTER component "
              "filtering, so they never reach the matcher. Use for a known substrate, "
              "or between passes of iterative indexing. Do NOT try to do this by "
              "editing the images: removing a reflection promotes its neighbourhood "
              "to local maxima and manufactures a ring of false peaks (measured "
              "2.7-3.4x for weak spots, 21x for a saturated one)."),
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


# Crystal-fit tolerance limits, mirroring validateCrystalFitTolerances in
# LaueMatchingHeaders.h: a fraction >= 1 cannot be an elastic refinement bound
# (it is almost always a percent written where a fraction was meant); above 0.1
# is legal but suspicious.
FRACTION_MAX = 1.0
FRACTION_WARN = 0.1


def _check_fractions(key: str, values) -> None:
    """Reject a tolerance that is not a fraction in [0, 1); warn above 0.1."""
    for i, v in enumerate(values):
        tag = key if len(values) == 1 else f"{key}[{i}]"
        if not (0.0 <= v < FRACTION_MAX):          # also rejects NaN
            logger.error(
                f"{tag} = {v:g} is not a valid FRACTION: it must satisfy "
                f"0 <= tol < 1. If you meant one percent, write 0.01, not 1.0.")
            raise ValueError(f"{tag} must be a fraction in [0, 1)")
        if v > FRACTION_WARN:
            logger.warning(
                f"{tag} = {v:g} is a FRACTION: +-{100.0 * v:g}%. "
                f"If you meant {v:g}%, write {v / 100.0:g}.")


# Symmetry is not read by the C binaries at all; its consumer is GenerateHKLs,
# which accepts exactly one UPPERCASE letter. Both Python parsers therefore
# refuse a lowercase letter rather than guess.
SYMMETRY_LETTERS = "FICARPB"
SYMMETRY_RULE = ("Symmetry must be ONE uppercase letter from F I C A R P B. It "
                 "is case-sensitive: the C does not read it, and GenerateHKLs, "
                 "which does, rejects lowercase.")


def parse_space_group(token: str) -> int:
    """SpaceGroup as the C reads it (sscanf %d), minus its silent truncation:
    an integral float such as ``225.0`` is accepted, ``225.5`` is not."""
    try:
        v = int(token)
    except ValueError:
        f = float(token)                      # raises for non-numbers
        if not f.is_integer():
            raise ValueError(f"SpaceGroup {token} is not an integer")
        v = int(f)
    if not 1 <= v <= 230:
        raise ValueError(f"SpaceGroup {v} outside 1-230")
    return v


def _check_token_count(key: str, got: int, need: int) -> None:
    """Token-count rule shared with the C (sscanf reads the first N): too few is
    an error, extra trailing tokens are ignored with a warning."""
    if got < need:
        logger.error(f"Incorrect number of values for {key}. "
                     f"Expected {need}, got {got}.")
        raise ValueError(f"{key} needs {need} values, got {got}")
    if got > need:
        logger.warning(f"{key}: {got} values given, using the first {need} "
                       f"(as the C does); the rest are ignored.")


def validate_tolerances(config) -> None:
    """Whole-file tolerance check, mirroring validateCrystalFitTolerances.

    When tol_c_over_a is non-zero the C ignores tol_LatC and does not validate
    it; so here a NOTE is logged and the value is kept as written. Otherwise an
    out-of-range tol_LatC is logged as an error and not kept (reset to zeros),
    as a bad tol_c_over_a line is at parse time.
    """
    raw = str(getattr(config, "tol_lat_c", "0 0 0 0 0 0"))
    vals = [float(v) for v in raw.split()]
    if float(getattr(config, "tol_c_over_a", 0.0) or 0.0) != 0.0:
        if any(v != 0.0 for v in vals):
            logger.info(f"NOTE: tol_c_over_a is set, so it overrides tol_LatC; "
                        f"tol_LatC ({raw}) is ignored and not validated.")
        return
    try:
        _check_fractions("tol_LatC", vals)
    except ValueError:
        config.tol_lat_c = SCHEMA_BY_KEY["tol_LatC"].default


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
    ignored keys), raising ValueError on malformed values. For a key in
    :data:`FATAL_KEYS` the error is a :class:`FatalConfigError`, which callers
    must not swallow.
    """
    try:
        return _parse_line(config, line)
    except FatalConfigError:
        raise
    except ValueError as exc:
        body = line[:line.index("#")] if "#" in line else line
        parts = body.split()
        key = _ALIASES.get(parts[0], parts[0]) if parts else ""
        if key in FATAL_KEYS:
            raise FatalConfigError(
                f"{key} has an invalid value {' '.join(parts[1:])!r} ({exc}). "
                f"{key} has no safe default -- set it for this experiment.") from exc
        raise


def _parse_line(config, line: str) -> bool:
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

    if param.kind == "fraction":
        _check_token_count(key, n - 1, param.nvals)
        try:
            vals = [float(v) for v in parts[1:param.nvals + 1]]
        except ValueError:
            logger.error(f"Invalid value format for {key} on line: '{line}'. "
                         f"Expected {param.nvals} float(s).")
            raise ValueError(f"Invalid format for {key}")
        # tol_LatC's range is checked after the whole file is read
        # (validate_tolerances): the C ignores it when tol_c_over_a is set, and
        # that line may come later in the file.
        if key != "tol_LatC":
            _check_fractions(key, vals)
        value = " ".join(parts[1:param.nvals + 1]) if param.nvals > 1 else vals[0]
        setattr(_target(config, param), param.field, value)
        return True

    if param.kind == "multi":
        _check_token_count(key, n - 1, param.nvals)
        # Every multi-value key is numeric. Stored as the original tokens (the
        # C re-reads the file), but a non-number -- e.g. a template's
        # __SET_ME__ -- must not be accepted as a lattice or a detector pose.
        try:
            [float(v) for v in parts[1:param.nvals + 1]]
        except ValueError:
            logger.error(f"Non-numeric value for {key} on line: '{line}'.")
            raise ValueError(f"Non-numeric {key}")
        setattr(_target(config, param), param.field, " ".join(parts[1:param.nvals + 1]))
        return True

    # scalar-ish: need at least one value
    if n < 2:
        logger.error(f"Missing value for {key} on line: '{line}'.")
        raise ValueError(f"Missing value for {key}")

    if param.kind == "symmetry":
        sym = parts[1]
        if len(sym) != 1 or sym not in SYMMETRY_LETTERS:
            logger.error(SYMMETRY_RULE)
            raise ValueError(f"Invalid Symmetry {sym!r}")
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
        if key == "SpaceGroup":
            value = parse_space_group(parts[1])
        else:
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
