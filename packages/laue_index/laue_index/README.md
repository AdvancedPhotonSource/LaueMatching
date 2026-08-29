# laue_index

The Python orchestration for LaueMatching, packaged as typed pipeline stages
around the C/CUDA indexer (REFACTOR_PLAN).  Mirrors the in-repo `laue_torch`
conventions (curated `__init__`, single-responsibility modules, typed records)
but stays **independent** of the paper-tied packages (`laue_torch` / `laue_jax`
/ `jax_cpfem`) — shared pure math is duplicated with `# TODO(unify-after-publish)`.

## Install

```bash
pip install 'laue-index[run]'                # everything the pipeline needs
pip install laue-index                       # library + the CPU indexer (numpy only)
LAUEMATCHING_CUDA=1 pip install laue-index   # + the CUDA binaries (needs nvcc)
```

```bash
laue-index fetch-db --dest ~/laue            # the 6.7 GB orientation database
export LAUEMATCHING_ORIENT_DB=~/laue/100MilOrients.bin
laue-index run process -c params.txt -i frame.h5 -n 8
```

The C indexer is compiled on your machine at install time (this is an sdist, not
a wheel — a binary is tied to the toolkit and GPU architectures that built it).
Without a C compiler and OpenMP the install still succeeds; only the binary is
missing. The CUDA build is opt-in because a toolkit that cannot compile these
sources fails at *build* time, which would take the working CPU binary down with
it.

```python
from laue_index import indexer
indexer.available()             # is the CPU indexer usable?
indexer.available("GPU")        # is LaueMatchingGPU usable?
indexer.binary_path("GPU")      # where it came from
```

Set `LAUEMATCHING_BIN` to a binary — or to a directory holding them — to use one
you built elsewhere or downloaded from a
[release](https://github.com/AdvancedPhotonSource/LaueMatching/releases). It
takes precedence over everything else, and the not-found error names every path
it tried.

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
| `cli.py` | `laue-index` console entry: `run` (the whole image→index pipeline), `fetch-db` (the orientation database), `parse` (summarise a solutions table), `filter` (re-run post-processing on existing C output, no re-indexing), `calibrate`. |
| `pipeline/` | The orchestrators themselves — `RunImage`, the streaming daemon driver and image server, the HKL/simulation generators. They import each other flat, so import them from one place: `from laue_index.pipeline import add_to_path`. |

## Relationship to `scripts/`
The orchestrators used to live in the repo's `scripts/` and are now in
`pipeline/` here, so the package ships something that can actually run a frame.
`scripts/` keeps a one-line shim per entry point — `python scripts/RunImage.py …`
and the shell pipeline behave exactly as before from a checkout.
`laue_stream_utils` remains a thin re-export of this package's stages, so
`RunImage`, `laue_postprocess` and `laue_image_server` are unchanged by any of it.

## Testing
`pytest` (from the repo root) runs the unit suite — golden-anchored
characterization tests pin behaviour through the refactor.  The full end-to-end
test (`tests/test_char_e2e.py`) is opt-in: set `LAUE_E2E=1` with the orientation
DB, the built C binary, and a prebuilt forward cache present; otherwise it skips,
so CI is safe with `SKIP_DOWNLOAD=1` (no 6.7 GB database needed for units).
