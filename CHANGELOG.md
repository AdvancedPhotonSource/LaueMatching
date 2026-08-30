# Changelog

Version history for the LaueMatching C/CUDA indexer and its pipeline. The Python
distributions in `packages/` version independently and are released to PyPI; see
each package's `pyproject.toml` for its current version.


## v2.2 (unreleased)

- **Repository layout rearranged (2026-08-29).** No behaviour change; every move
  was a `git mv`, so history follows the files.
  - The C/CUDA source now lives in **one** place, `packages/laue_index/c_src/`.
    The repo-root `src/` copy is gone, and with it `utils/sync_vendored_c.py` and
    its CI step: the root `CMakeLists.txt` reaches down into the package, exactly
    as it already did for `cmake/LaueCudaArch.cmake`. The duplication existed
    because a pip sdist cannot reach up to a repo-root `src/`; putting the single
    copy inside the package satisfies that without making a one-copy edit
    possible.
  - The doc set moved `scripts/pipeline/laue/` → **`manuals/laue/`**, matching
    MIDAS's `manuals/ff-hedm/` and `manuals/nf-hedm/`.
  - The campaign launcher and its analysis scripts moved
    `scripts/pipeline/` → **`pipeline/`**. **If you have a deployed checkout, the
    launcher is now `pipeline/run_laue.sh`** — the old path no longer exists.
  - `scripts/` now holds only the entry-point shims onto
    `laue_index/pipeline/`, which is all it had left to be.
  - Version history moved out of `README.md` into this file.

- **Fixed: SIGSEGV at the end of every cold-cache run.** A run that *wrote* the
  forward-simulation cache — the first run on any machine, with the orientation
  database under `/dev/shm` — exited with a segmentation fault **after** writing
  complete and correct output, and the pipeline reported the image as failed.
  The cleanup's `if (orientsMapped) munmap(…) else free(orients)` pair had a
  second `if` inserted between its halves (by the C hardening below, ironically),
  re-parenting the `else`: writing the cache leaves `outArr` NULL, so `free()`
  ran on memory `munmap`'d one line earlier. Runs that *read* an existing cache
  took the other branch, which is why every test stayed green — they all
  supplied a prebuilt cache. Now covered by `test_cold_forward_cache.py`, which
  fails against the unfixed binary.
- **`pip install laue-index` is now the whole thing.** The orchestrators moved
  into the package (`laue_index/pipeline/`), so `laue-index run process -c … -i …`
  indexes a frame with no checkout; `laue-index fetch-db` downloads the 6.7 GB
  orientation database; `scripts/` keeps a shim for each entry point so existing
  invocations are unchanged. `LAUEMATCHING_CUDA=1 pip install laue-index` also
  compiles `LaueMatchingGPU` and `LaueMatchingGPUStream` — opt-in, because a
  toolkit that cannot build them would otherwise fail the whole install.
- **Optimizer: BOBYQA and NLopt removed.** Refinement is a vendored Nelder–Mead
  simplex; `Optimizer BOBYQA` in a config is accepted, noted, and ignored. On
  198 paired synthetic seeds Nelder–Mead was better on every statistic (median
  0.0041° vs 0.0054°, p95 3.3× tighter, max 25× tighter) at identical
  wall-clock. With no external optimizer to fetch, the C compiles at
  `pip install` time and builds offline.
- **CUDA builds for every architecture the toolkit supports, plus PTX.** The old
  hardcoded `70;80;86;90` failed outright on CUDA 13, which dropped Volta
  (`nvcc fatal : Unsupported gpu architecture 'compute_70'`), and covered
  nothing newer than Hopper. Building for the *local* card is not the fix
  either: PTX JIT works forward, never backward, so a binary built on a newer
  GPU than it runs on finds **zero** orientations and exits **0**. The build now
  asks `nvcc --list-gpu-arch` and covers all of them, with PTX for the newest.
- **Fixed: an arch mismatch was silent.** `cudaErrorNoKernelImageForDevice` is
  reported by the kernel *launch*, and only the following synchronize was
  checked — so the kernel never ran, nothing complained, and the run reported no
  grains. Both CUDA binaries now check the launch; the streaming daemon would
  otherwise have served zero matches for every frame of a scan.
- **Fixed: the forward cache was validated only by the CPU binary.** The CUDA
  binaries accepted any file that existed, so a 0-byte leftover — which the C
  itself creates when a run is interrupted mid-write — was mapped as a 12.2 GB
  cache and took SIGBUS on first touch. One check, in the shared header, called
  by all three.
- **Provenance tracking**: every generated artifact (HKL CSV, simulation HDF5,
  per-image indexing HDF5, orchestrator run directory) now carries a git
  commit, config snapshot, and weak fingerprints of its input files.
  See [docs/provenance.md](docs/provenance.md).
- **IndexFile text output**: on by default — each indexed image emits a
  Tischler-style `.indexing.txt` alongside the HDF5 (`--no-indexfile` to
  disable). See [docs/indexfile-format.md](docs/indexfile-format.md).
- **`scripts/GenerateOrientations.py`**: reproduce the orientation database
  at any spacing / crystal system using `orix`. Emits the **full SO(3)**,
  not the fundamental zone — the oversampling is load-bearing for the
  indexer's spurious-match filter. Writes a `.meta.json` sidecar with full
  provenance.
- **`scripts/annotate_orientation_db.py`**: writes a retroactive sidecar
  next to the existing `100MilOrients.bin`. Hooked into `build.sh`.
- **`GenerateHKLs.py` -Ehi flag**: the max-energy cutoff is no longer
  silently hardcoded to 30 keV.
- **`laue_index` package**: the Python orchestration is restructured into typed
  pipeline stages (records / geometry / filtering / thresholds / preprocess /
  indexer / postprocess / output / config_schema / cli).  Positional column
  "magic numbers" are replaced by a typed `Solution` record; the orientation
  filter (incl. the twin/CSL-aware variant) lives in one place; thresholding is
  pluggable; one declarative schema drives config parse **and** write.
  `laue_stream_utils.py` is now a thin re-export shim and `RunImage.py` is a
  thin orchestrator over the stages.  pip-installable
  (`pip install -e packages/laue_index`, `laue-index` CLI).  Behaviour-preserving,
  guarded by a golden-anchored characterization test suite under
  `packages/laue_index/tests/`.
- **C / CUDA hardening**: `size_t` indexing (large-DB overflow safety), full
  malloc/`mmap`/`fread` checks, single end-of-run `fsync` (was per-write
  `O_SYNC`), `snprintf` bounds, kernel pixel-bounds + spot-count clamps, and a
  cap on the O(N²) duplicate-merge.  The GPU now scores on full-precision
  intensity (uint8 quantization dropped) for parity with the CPU.  Streaming
  daemon: clean-shutdown thread join, `recv` timeout, `sigaction`.  Built and
  functionally validated on H200 (GPU and streaming-daemon runs reproduce the
  CPU result).
- **Robustness fixes**: twin/CSL-aware orientation filter (keeps real Σ3 twins),
  adaptive noise-floor threshold (recovers faint frames), and a fixed
  stream/RunImage column-format heuristic.

## v2.1 (2026-03-03)

- **GPU Kernel Optimizations**: 2.3× faster GPU matching:
  - Float32 kernel with `__ldg()` texture cache reads.
  - `atomicAdd` compact output: eliminates 800 MB D2H transfer.
  - **uint8 image quantization**: image shrinks from 16 MB to 4 MB, fits in L2 cache. Kernel time drops from 273 ms to 108 ms.
  - Nonzero-preserving quantization ensures spot counts remain exact.
- **CPU uint8 Matching**: `LaueMatchingCPU.c` uses uint8 quantized image for the `doFwd=0` matching path, improving L3 cache sharing across 96 threads.
- **Parallel Preprocessing**: `laue_image_server.py` uses `ProcessPoolExecutor` (up to 8 workers) for multi-process frame preprocessing.
- **Async Pipeline**: 3-stage architecture (submit → consumer → sender) fully decouples preprocessing from TCP sending.
- **KDTree Sigma**: `calculate_gaussian_sigma` uses `scipy.spatial.cKDTree` (O(n log n)) instead of O(n²) brute-force.
- **Reduced Log Verbosity**: Orchestrator result listing replaced with single-line summary.

## v2.0 (2026-02-18)

- **Streaming Pipeline**: New `LaueMatchingGPUStream` CUDA daemon + Python orchestrator for multi-image processing over TCP.
- **Float32 Wire Protocol**: Image transfer uses float32 (16 MB/frame for 2048×2048) instead of float64, halving bandwidth with no precision loss in GPU matching.
- **Pipelined Image Server**: Producer-consumer threading overlaps H5 loading/preprocessing with TCP sending.
- **Progress Bar**: Real-time tqdm progress bar with throughput (img/s) and ETA.
- **Graceful Daemon Shutdown**: Handles unresponsive GPU processes without crashing the pipeline.
- **Scripts Reorganization**: All Python scripts moved to `scripts/` directory with comprehensive `scripts/README.md`.
- **Module Decomposition**: Decomposed `RunImage.py` (3,553 → 1,673 lines) into reusable modules:
  - `laue_config.py` (782 lines) — configuration dataclasses and parameter file parser.
  - `laue_stream_utils.py` (1,108 lines) — image I/O, preprocessing, TCP wire protocol, orientation sorting/filtering.
  - `laue_visualization.py` (937 lines) — 8 standalone visualization functions (Plotly interactive, simulation comparison, reports, etc.).
- **Post-Processing**: `laue_postprocess.py` now sorts filtered orientations by quality and supports optional per-image interactive visualization.
- **Streaming Utilities**: `laue_image_server.py` for TCP image sending with live progress tracking; `laue_orchestrator.py` for full pipeline management.

## v1.0 (2026-02-17)

- **Code Refactor**: Consolidated ~700 lines of duplicated code into shared `LaueMatchingHeaders.h`.
- **Bug Fixes**:
  - Fixed c/a ratio fitting (was integer division `1/3`).
  - Fixed negative pixel handling (uint16_t underflow).
  - Fixed trigonal symmetry definition (consistent between CPU/GPU).
  - Fixed memory leaks and file descriptor handling.
  - Fixed GPU unique-solution indexing bug.
- **Build System**: Improved CMake configuration with working strict warning flags.
- **Performance**: Hoisted memory allocations out of critical loops; added `gpuErrchk` macro for CUDA error handling.
