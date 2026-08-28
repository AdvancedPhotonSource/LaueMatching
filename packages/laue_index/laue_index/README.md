# laue_index

The Python orchestration for LaueMatching, packaged as typed pipeline stages
around the C/CUDA indexer (REFACTOR_PLAN).  Mirrors the in-repo `laue_torch`
conventions (curated `__init__`, single-responsibility modules, typed records)
but stays **independent** of the paper-tied packages (`laue_torch` / `laue_jax`
/ `jax_cpfem`) — shared pure math is duplicated with `# TODO(unify-after-publish)`.

## Modules
| Module | Responsibility |
|---|---|
| `records.py` | `Solution` typed record + `parse_solutions(source, fmt)` + `SolutionFormat` column maps (runimage / stream) — replaces positional column "magic numbers". |
| `geometry.py` | Pure CSL/disorientation helpers (`disorientation_deg_axis`, `is_csl_related`, cubic ops, CSL table). |
| `filtering.py` | `calculate_unique_spots`, `filter_orientations`, `filter_orientations_robust` + `OrientationFilter` strategies (`LegacyUniqueSpotFilter`, `RobustCSLAwareFilter`). Single source of truth. |
| `thresholds.py` | `ThresholdStrategy` classes (`NoiseFloorThreshold`/adaptive, `Percentile`, `Otsu`, `Fixed`) + `apply_threshold` dispatch. |
| `preprocess.py` | Image pipeline (background → threshold → components → blur) + `Preprocessor`. |
| `indexer.py` | Thin wrapper around the C indexing binary (`run_indexer`). |
| `postprocess.py` | `PostProcessor`: unique-spots → sort → filter → spot-filter. |
| `output.py` | HDF5 result writer. |
| `config_schema.py` | One declarative `SCHEMA` table driving config parse + write. |
| `cli.py` | `laue-index` console entry: `parse` (summarise a solutions table) and `filter` (re-run post-processing on existing C output, no re-indexing). |

## Relationship to `scripts/`
The legacy `scripts/laue_stream_utils.py` is now a **thin shim** that re-exports
from this package, so `RunImage.py`, `laue_postprocess.py`, and
`laue_image_server.py` keep working unchanged while the implementation lives
here.  `RunImage.py` is the orchestrator wiring the stages together.

## Testing
`pytest` (from the repo root) runs the unit suite — golden-anchored
characterization tests pin behaviour through the refactor.  The full end-to-end
test (`tests/test_char_e2e.py`) is opt-in: set `LAUE_E2E=1` with the orientation
DB, the built C binary, and a prebuilt forward cache present; otherwise it skips,
so CI is safe with `SKIP_DOWNLOAD=1` (no 6.7 GB database needed for units).
