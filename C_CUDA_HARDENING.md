# C / CUDA hardening — findings & status

Scope expansion beyond REFACTOR_PLAN (which scoped the C side out, §8).  Three
independent reviews of `src/LaueMatchingCPU.c`, `LaueMatchingGPU.cu`,
`LaueMatchingGPUStream.cu`, `LaueMatchingHeaders.h`.  The headline: the CPU
batching/`size_t`/alloc-check hardening (commit "Batch the CPU forward-sim loop")
was never ported to the GPU variants, and there is a shared-header scaling bomb.

Validation: CPU + headers build locally (`./build.sh cpu`) and pass the
deterministic end-to-end golden (`tests/test_char_e2e.py`, DoFwd=0).  CUDA has no
local toolchain — `.cu` changes are built/tested on **sentosa** (H200, CUDA 12.9).

## DONE — CPU.c (behaviour-preserving, e2e-validated)
- Initialise `nrPxX/nrPxY/pxX/pxY/pArr/rArr/LatticeParameter`; validate
  `NrPxX/NrPxY > 0` and `LatticeParameter[a] != 0` before use (was: garbage
  alloc / divide-by-zero on a missing/misspelled key).
- `numProcs = atoi(argv[5])` validated `>= 1` (was: div-by-zero / `num_threads(0)`).
- `Optimizer` `sscanf(aline,"%s %s",dummy,dummy)` self-aliasing → second buffer.
- NULL checks: `image_u8`, `imageF`, `ExtraInfo` fopen, result-collection
  (`mA/rowNrs/FinOrientArr/dArr/bsArr/bsScoreArr`).
- Pixel loops `int pxNr` → `size_t` over `nPixels`; `fread` size cast `size_t`.
- Cached forward mmap: check `open() >= 0` and `outArr != MAP_FAILED`; `munmap`
  the previously-leaked mapping at cleanup (length tracked since `maxNrSpots`
  is tripled later).
- Removed unused `k, l` (the two leftover `-Wunused-variable` warnings).

## DONE — LaueMatchingHeaders.h (shared; CPU-built + e2e-validated)
- All 10 pixel-index sites `(int)((int)py*nrPxX+(int)px)` → `size_t` arithmetic
  (overflow-safe for >46k² detectors).
- `writeCalcOverlap`: per-line budget 200→256 B and `sprintf` → `snprintf` with
  remaining-space accounting (heap-overflow guard).
- NULL checks in `mergeDuplicateOrientations` (`doneArr/quats/misoDist`) and the
  per-iteration fit (`outArrThisFit/validIdx`) — fail loudly, no segfault.

## TODO — GPU `.cu` (build + test on sentosa)
CRITICAL
- **`matchIdx` is `int` → overflows at 100M orientations** (`i + chunkOffset`
  > INT_MAX) → out-of-bounds orientation reads. The exact bug the CPU `size_t`
  change fixed. Make match/orientation indices `size_t`/`int64` end-to-end
  (both GPU.cu and GPUStream.cu, incl. forward-sim thread partitioning).
HIGH
- Silent `MAX_MATCHES` truncation: `atomicAdd` counts past the cap; host clamps
  and processes a racy subset / stale pinned-buffer tail. Warn + handle.
- Missing malloc/cudaMalloc/mmap/fread checks throughout (sibling has them).
- Forward writers still `open(..., O_SYNC)` — the per-write-sync pattern the CPU
  dropped for one end-of-run `fsync`. Drop O_SYNC + single fsync.
- Kernel lacks `px/py` bounds + `nrSpots` clamp before image load (stale/corrupt
  forward cache → illegal address).
- GPUStream daemon: detached client threads never joined → race with
  `queue_destroy`/mutex-destroy on shutdown; no `recv` timeout (a stalled client
  pins a handler); `signal()` → `sigaction`.

## DEFERRED — behaviour-changing (need their own validation, not hardening)
- `mergeDuplicateOrientations` O(N²) pairwise matrix: bound `nrResults` before
  the merge (changes results for pathological high-match frames; the NULL check
  above already prevents the segfault).
- GPU scores `sum·√count` on **uint8-quantized** intensity vs CPU's
  full-precision `count·√sum` family — unify the score/precision (changes GPU
  results; validate on sentosa against the CPU golden).
