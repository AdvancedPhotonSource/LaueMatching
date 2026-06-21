# LaueMatching — Refactor Plan & Handoff

**Audience:** a fresh dev/chat picking up a structural refactor of the Python
side of LaueMatching. Self-contained. Written 2026-06-21 after a debugging
session that fixed two real robustness bugs and exposed the structural issues
below. **Do characterization tests FIRST (§6) so the refactor is provably
behavior-preserving.**

---

## 0. TL;DR
The library works but has grown organically around the C indexer. The Python
orchestration has (a) the *same* filtering logic in 3 places, (b) a god-object
`RunImage.py`, (c) positional-column "magic numbers" for parsing C output,
(d) camelCase/snake_case config drift with hand-synced parse/write blocks,
(e) loose `scripts/` rather than a `laue_torch`-style package. (The big
orientation binary is already handled well — gitignored + downloaded by
`build.sh` — keep that.) Target: a small set of
**typed pipeline stages** with **strategy objects** for the two things that
actually vary (thresholding, orientation filtering), one **config schema**, and
**typed records** instead of column indices.

---

## 1. What LaueMatching is (orientation for a newcomer)
White-beam Laue pattern indexer.
- **C core** (`src/LaueMatchingCPU.c`, `LaueMatchingGPU.cu`, `LaueMatchingGPUStream.cu`, shared `LaueMatchingHeaders.h`): brute-force orientation search of a precomputed orientation list (`100MilOrients.bin`, row-major 3×3 float64, 72 B/record, no header) against detected spots, then nlopt refinement. Invoked as:
  `LaueMatchingCPU <config.txt> <orientation_file> <hkls.csv> <blurred_image.bin> <ncpus>`
  Writes `<image>.bin.solutions.txt` (all refined solutions) + `.indexing.txt`.
- **Python orchestration** (`scripts/`):
  - `RunImage.py` — main `process` entry. Per image: load → background → threshold → connected-components → small-component filter → Gaussian blur → write `<out>.bin` (the blurred image) → run C binary → parse solutions → **unique-spot filter** → write `<out>.output.h5` (`filtered_orientations`, `filtered_spots`, intermediates) → viz.
  - `laue_stream_utils.py` ("extracted from RunImage.py for reuse") — the reusable funcs: `apply_threshold`, `compute_background`, `calculate_unique_spots`, `filter_orientations`, `filter_orientations_robust` (new), `sort_orientations_by_quality`, H5 I/O.
  - `laue_postprocess.py` — a **second** filtering path (`process_image_orientations`) used for the stream/batch case; calls the `lsu` funcs.
  - `laue_config.py` — `LaueConfig` dataclass + text-param parse + write.
  - others: `laue_orchestrator.py`, `laue_stream_*`, `laue_visualization.py`, `laue_indexfile.py`, `GenerateOrientations/HKLs/Simulation.py`, `ImageCleanup.py`.

---

## 1.5 Target conventions — mirror `laue_torch` / `laue_jax` / `jax_cpfem`
The newer in-repo packages already establish a clean, consistent architecture.
The legacy `scripts/` should be brought into the **same style** (not coupled —
see the constraint box). Conventions observed in `laue_torch` (most mature;
`laue_jax` mirrors it for a second backend):

- **Flat package with a curated `__init__.py`** that does explicit imports and
  defines `__all__` (the public API surface). See `laue_torch/__init__.py`.
- **Single-responsibility modules**, e.g.:
  - `geometry.py` — pure functions only (`rodrigues_to_matrix`, `reciprocal_matrix`,
    `voigt_to_symmetric`, …), `Tensor -> Tensor`, no state, no I/O.
  - `io.py` — a typed **`@dataclass LaueParams`** (fields with units in comments)
    + `parse_params(path) -> LaueParams` + a `.to_tensors(dtype, device)` method
    that converts to the compute representation; plus `load_orientations`,
    `generate_hkls`.
  - `forward.py` — the model as a **class** (`LaueForwardModel`) taking static
    config in `__init__` and variables in `forward(...)`, returning a typed
    **`@dataclass ForwardAux`** (never a bare tuple).
  - `cli.py` — `argparse` `main(argv) -> int`, `ArgumentDefaultsHelpFormatter`,
    explicit flags; thin wrapper over the library.
- **Typed records, not column indices** — `ForwardAux` is the pattern our
  `Solution` record (§5) should follow.
- **Explicit dtype/device handling** threaded through, not globals.
- **Modern type hints** (`str | Path`, `list[str] | None`, `-> Tensor`).
- **Tests colocated** in `package/tests/test_*.py` with descriptive names;
  cross-backend behavior pinned by a parity test (`laue_jax/tests/test_parity_torch_jax.py`).
- Heavier/experimental code segregated into `experiments/`, `realdata/`,
  `examples/`, `report/` subpackages — kept out of the core modules.

> ⚠️ **Constraint (do not break):** `laue_torch`, `laue_jax`, `jax_cpfem` are
> **unpublished and paper-tied** — they will keep moving until their papers are
> submitted. The legacy indexer is the published, user-facing tool. So:
> **adopt their conventions/structure, but do NOT `import` from them** (no
> coupling the shipped indexer to still-moving, unreleased code). Where the same
> pure math is genuinely shared (orientation/CSL/geometry helpers), duplicate it
> consciously now; once the paper packages publish, a *future* step can extract a
> common `laue_core`/`geometry` leaf that all of them depend on (the same
> shared-leaf pattern used elsewhere). Flag such duplications with a
> `# TODO(unify-after-publish)` marker so they are easy to find later.

---

## 2. Pain points (evidence from the 2026-06-21 session)

1. **Filtering logic duplicated in 3 places.** The unique-spot filter is inlined
   in `RunImage._process_indexing_results` (~L1014), *and* in
   `lsu.filter_orientations`, *and* driven by `laue_postprocess`. The Σ3-twin
   bug had to be fixed by wiring the new filter into RunImage's inline copy
   **separately** from `lsu`. → **single source of truth.**

2. **`RunImage.py` is a god-object.** One class does config, background,
   threshold, segmentation, components, blur, `.bin` I/O, C subprocess, parse,
   filter, H5 output, viz. Threshold logic is welded to file I/O — to unit-test
   it we had to reimplement the pipeline in numpy. → **typed pipeline stages.**

3. **Positional column magic.** "OM = cols 22:31", "NMatches = col 5",
   "grain = col 0 *or* 1", with `n_cols > 30 → stream format` heuristics in
   `RunImage`, `laue_postprocess`, and the new filter. Fragile; caused real
   friction threading `om_start_col`. → **typed `Solution` record parsed once.**

4. **Config name drift.** `maxAngle` (camel) vs `min_nr_spots` (snake) vs
   `space_group`. Parse is a long `elif` chain; write is a parallel `f.write`
   block kept in sync by hand (we tripped on `max_angle` vs `maxAngle`).
   → **one declarative schema table → auto parse/write/validate/docs.**

5. **Two filtering pipelines** (`RunImage` inline + `laue_postprocess`) → drift
   risk; the extraction into `lsu` is incomplete.

6. **Big-binary handling is ALREADY correct — keep it.** `100MilOrients.bin`
   (~6.7 GB) is *gitignored* (`.gitignore:27`) and downloaded on first build by
   `build.sh` from the GitHub release `v1.0-data` (4 split parts → concatenated;
   `SKIP_DOWNLOAD=1` for CI; provenance sidecar via
   `scripts/annotate_orientation_db.py`). The refactor must **preserve this
   download-on-install path** (the new package's installer/`cli.py` should locate
   or trigger it). The only stray item is chiltepin-local `scripts.bak_*` backup
   dirs on that deployment (not in the repo) — a deployment-hygiene note, not a
   repo issue; prefer `git`-based sync over manual file copies (we hit diverged
   installs during deploy).

7. **No Python test harness** existed before this session. → `tests/` with
   small fixtures (started: `tests/test_robustness_fixes.py`).

---

## 3. Target architecture — a `laue_index` package mirroring `laue_torch`
Turn the loose `scripts/` into a proper package laid out like `laue_torch`:

```
laue_index/
  __init__.py        # curated public API + __all__ (mirror laue_torch/__init__.py)
  io.py              # @dataclass IndexParams (units in comments) + parse_params()
                     #   + .to_tensors()/.to_dict(); load_orientations(); generate_hkls()
  geometry.py        # PURE funcs: rodrigues/quat/sixd->matrix, reciprocal_matrix,
                     #   cubic_proper_ops, disorientation, is_csl_related, CSL table
  records.py         # @dataclass Solution (+ parse_solutions); the ForwardAux analogue
  preprocess.py      # Preprocessor + ThresholdStrategy classes (§4a)
  filtering.py       # OrientationFilter classes (§4b) — single source of truth
  indexer.py         # thin C-binary wrapper (build argv, run, return raw solutions)
  pipeline.py        # one orchestrator: Preprocess -> Index(C) -> Postprocess -> Write
  output.py          # H5 / indexfile writers (thin I/O adapters)
  cli.py             # argparse main(argv)->int  (mirror laue_torch/cli.py)
  tests/             # colocated test_*.py with fixtures
  experiments/ …     # (optional) segregate one-off scripts here, as laue_torch does
```

Data flow (each stage typed, no hidden file I/O — I/O confined to `io.py`/`output.py`):
```
IndexParams ─┐
raw image ─► Preprocessor ─► blurred .bin ─► Indexer(C) ─► list[Solution] ─► PostProcessor ─► list[Solution] ─► OutputWriter
              (ThresholdStrategy)              (parse_solutions→records)        (OrientationFilter)
```
- `RunImage.py` + `laue_postprocess.py` collapse into **one** `pipeline.py`
  orchestrator (stream path reuses the same stages) — kills duplication #1, #5.
- The two things that vary are **strategy objects** (§4), selected from `IndexParams`.
- C output parsed once into typed `Solution` records (§5) — kills column-magic #3.
- Mirrors `laue_torch`: `io.py`(params dataclass) · `geometry.py`(pure) ·
  `records.py`≈`ForwardAux` · class-based model/strategies · `cli.py` · `tests/`.
- **Stays independent** of `laue_torch/laue_jax/jax_cpfem` (constraint box §1.5);
  shared pure math is duplicated with `# TODO(unify-after-publish)` until a
  common leaf can be extracted post-submission.

---

## 4. Strategy pattern (the core ask)

### 4a. ThresholdStrategy
```python
from typing import Protocol, Tuple
import numpy as np

class ThresholdStrategy(Protocol):
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (thresholded_image, threshold_value). `image` is background-
        subtracted (>=0)."""

class NoiseFloorThreshold:
    """Robust per-frame noise floor: thresh = median(nz) + k*1.4826*MAD(nz).
    Recovers faint frames (the 2026-06-21 fix); replaces the old
    max(60*(1+std//60),1) which was a ~fixed 240 that gutted faint patterns."""
    def __init__(self, k: float = 4.0): self.k = k
    def __call__(self, image):
        nz = image[image > 0]
        if nz.size:
            med = float(np.median(nz)); mad = 1.4826*float(np.median(np.abs(nz-med)))
            thr = max(med + self.k*mad, 1.0)
        else:
            thr = 1.0
        out = image.copy(); out[image <= thr] = 0
        return out, float(thr)

class PercentileThreshold:
    def __init__(self, pct: float = 99.5): self.pct = pct
    def __call__(self, image):
        thr = float(np.percentile(image.ravel(), self.pct))
        out = image.copy(); out[image <= thr] = 0
        return out, thr

# also: OtsuThreshold, FixedThreshold(value)
THRESHOLD_STRATEGIES = {"adaptive": NoiseFloorThreshold, "percentile": PercentileThreshold,
                        "otsu": OtsuThreshold, "fixed": FixedThreshold}
```
(The current `lsu.apply_threshold` already dispatches on a `method` string — this
just promotes each branch to a class so it is independently testable/extensible.)

### 4b. OrientationFilter
```python
class OrientationFilter(Protocol):
    def __call__(self, solutions: "list[Solution]") -> "list[Solution]": ...

class LegacyUniqueSpotFilter:
    """Keep solutions with >= min_unique winner-take-all unique spots. (Deletes
    real Sigma3 twins — kept only for back-compat / RobustFilter 0.)"""
    def __init__(self, min_unique: int = 2): ...

class RobustCSLAwareFilter:
    """2026-06-21 fix. (B.1) merge near-duplicates by symmetry-reduced
    disorientation < max_angle; (B.2) keep distinct solutions clearing an
    evidence floor (min_total_spots); (A) EXEMPT CSL/twin-related solutions
    (Sigma3 60deg/<111>, table also Sigma9/Sigma11) from the unique-spot test
    so real twins survive. Cubic-only CSL today; generalise via point-group
    symmetry ops keyed on the config's space group."""
    def __init__(self, min_unique=2, min_total_spots=5, max_angle_deg=5.0,
                 csl_sigmas=(3,), csl_tol_deg=3.0, symmetry="cubic"): ...
```
Reference implementation already exists: `lsu.filter_orientations_robust`
+ helpers `_cubic_proper_ops`, `_disorientation_deg_axis`, `is_csl_related`,
`_CSL_TABLE` (in `laue_stream_utils.py`). The refactor should lift these into a
`filtering.py` module behind the `OrientationFilter` interface and have **both**
RunImage and laue_postprocess call it (kills duplication #1 and #5).

### 4c. Selection (config-driven)
```python
threshold = THRESHOLD_STRATEGIES[cfg.threshold_method](**cfg.threshold_kwargs)
ofilter   = RobustCSLAwareFilter(...) if cfg.robust_filter else LegacyUniqueSpotFilter(...)
```
`RobustFilter` config flag already added (default on). Generalise CSL beyond
cubic by selecting symmetry ops from `cfg.space_group`.

---

## 5. Typed records & config schema
Mirror `laue_torch`: `Solution` is the analogue of `forward.ForwardAux` (a typed
`@dataclass` result, never a bare tuple/array); `IndexParams` is the analogue of
`io.LaueParams` (typed fields w/ units, `parse_params(path) -> IndexParams`, and a
`.to_tensors()`/`.to_dict()` conversion method). The **only** deliberate
extension beyond `laue_torch` is the declarative config `SCHEMA` below: the
legacy param set is far larger than `LaueParams` and currently drifts between a
hand-written parse `elif` chain and a `write` block — the SCHEMA is the scalable
form of `laue_torch`'s `parse_params` idea (one table → parse + write + docs).

### Solution record (replaces column indices)
```python
@dataclass
class Solution:
    grain_nr: int
    n_matches: int
    quality: float           # NMatches*sqrt(Intensity)
    orientation: np.ndarray  # (3,3)
    recip: np.ndarray        # (3,3)
    lattice: np.ndarray      # (6,)
    misorientation_post_refine: float
    # provenance: source row, unique_label_count, etc.

def parse_solutions(path_or_array, fmt: str) -> list[Solution]: ...
```
One parser handles RunImage (grain=col0, OM=22:31, NMatches=5) vs stream
(grain=col1, +1 offsets). Everything downstream uses fields. Kills #3.

### Config schema (declarative)
```python
@dataclass(frozen=True)
class Param:
    name: str          # text-file key, e.g. "MaxAngle"
    field: str         # dataclass attr, e.g. "max_angle"
    type: type
    default: object
    doc: str = ""

SCHEMA = [
    Param("SpaceGroup", "space_group", int, 225),
    Param("MaxAngle", "max_angle", float, 2.0, "Refinement cone (deg)"),
    Param("MinNrSpots", "min_nr_spots", int, 5),
    Param("MinGoodSpots", "min_good_spots", int, 5, "Min unique spots (legacy filter)"),
    Param("RobustFilter", "robust_filter", bool, True, "Twin/CSL-aware filter"),
    ...
]
# parse(): one loop over SCHEMA. write(): one loop over SCHEMA. docs: generated.
```
Pick ONE naming convention (snake_case fields) and let the schema map to the
text-file keys. Kills #4.

---

## 6. Migration plan (incremental, behavior-preserving)
0. **Characterization tests FIRST.** Capture current outputs on a handful of
   real frames + a known solution set, as golden files. Fixtures already on
   disk: `_seed_test_local/` bins, `frame101_filtertest.npz`,
   `tests/test_robustness_fixes.py`. Add: end-to-end RunImage on 2-3 frames →
   snapshot `filtered_orientations`.
1. Extract `parse_solutions` + `Solution` record; switch RunImage/postprocess to
   it (no behavior change). Delete column-index code.
2. Lift filtering into `filtering.py` behind `OrientationFilter`; **delete the
   inline RunImage copy and the laue_postprocess duplicate** — both call the one
   module. (Robust + Legacy already exist in `lsu`.)
3. Lift thresholding into `thresholds.py` behind `ThresholdStrategy`.
4. Introduce the config `SCHEMA`; replace the parse `elif` chain + write block.
5. Split RunImage into `Preprocessor / Indexer / PostProcessor / OutputWriter`;
   RunImage becomes a thin orchestrator; stream path reuses the stages.
6. Hygiene/CI: **preserve** the `build.sh` orientation-DB download-on-install
   (gitignored binary + release parts + provenance sidecar) — wire it to the new
   package's install/`cli.py`; ensure `tests/` runs in CI with `SKIP_DOWNLOAD=1`;
   clean stray `scripts.bak_*` off the chiltepin deployment and sync via `git`.
Run the characterization tests after every step.

---

## 7. Recent fixes to fold in (already on `master` working tree + deployed to chiltepin 2026-06-21)
- **Twin/CSL-aware filter** — `lsu.filter_orientations_robust` (+ helpers), wired
  into `RunImage` with `RobustFilter` config (default on, legacy fallback).
  Fixes: the unique-spot dedup deleted real Σ3 twins (the brighter partner
  claimed the shared coincident reflections).
- **Noise-floor adaptive threshold** — `lsu.apply_threshold` adaptive branch.
  Fixes: old `max(60*(1+std//60),1)` ≈ 240 fixed floor gutted faint frames.
- Tests: `tests/test_robustness_fixes.py` (4 pass). These two fixes are the
  **first two strategies** — the refactor should make them pluggable, not the
  only options.
- Deployed files (chiltepin `/home/beams/S1IDUSER/opt/LaueMatching/scripts/`,
  originals backed up `*.predeploy_2026-06-21`): `laue_stream_utils.py`,
  `RunImage.py`, `laue_config.py`. NOTE chiltepin had small divergent changes
  (HKL arg-unpacking in RunImage; `batch_size`, Symmetry `FICARPB` in config) —
  preserved during deploy. Keep local `~/opt/LaueMatching` and chiltepin in sync
  via git after the refactor rather than file copies.

---

## 8. Context / related docs
- Scientific context that drove these fixes: `laue_torch/report/two_paper_split/NI_AUDIT_HANDOFF.md`
  (Ni nano-indent Laue audit). The filter/threshold bugs were found while
  validating per-frame parent orientations for a chain-kernel ODF fit.
- The C side (`src/`) was not touched and is out of scope here, but note: the
  orientation file format and the 5-arg CLI are the integration contract the
  Python wrapper must preserve.
