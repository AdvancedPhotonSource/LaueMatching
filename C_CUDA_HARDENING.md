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

## DONE — GPU `.cu` (built clean on sentosa, CUDA 12.9 / H200; behaviour-preserving)
- Match/orientation indices `int` → `size_t` end-to-end in both `.cu`
  (kernel sig, device/host buffers, memcpy sizes, downstream `rowNrs`), plus the
  GPUStream forward-sim thread partitioning.  NOTE: at the current 1e8-orientation
  DB an `int` *index* (max 1e8) is below INT_MAX, so this is **defensive**
  (needed for >2.1e9-orientation DBs); the byte-offset products that *do*
  overflow `int` at 1e8 (`orientNr*stride ≈ 1e11`) were already `size_t` in the
  kernels.  Safe widening regardless.
- `MAX_MATCHES` truncation now warns to stderr before clamping; results read with
  the clamped count only.
- malloc/cudaMalloc/mmap/fread/MAP_FAILED checks added throughout both files.
- Forward writers: `O_SYNC` dropped, single `fsync()` before close (CPU parity).
- `compare` kernel: `nrSpots` clamped to `nrMaxSpots`; `px/py` bounds-checked
  before the image load (added `nrPxY` kernel param).
- GPU.cu: `nrPxX/nrPxY/LatticeParameter` init + validation + `numProcs>=1` guard.
- GPUStream daemon: client handler threads tracked + joined before
  `queue_destroy` (shutdown race); `SO_RCVTIMEO` + `EINTR` retry on `recv`;
  `signal()` → `sigaction` (no SA_RESTART); usage text float (was double).

## VALIDATION (sentosa, CUDA 12.9 / H200 sm_90)
- CPU.c + headers: local build clean, e2e DoFwd=0 golden byte-identical.
- GPU.cu + GPUStream.cu: build clean (no warnings).
- GPU functional run on the full 100M-orientation DB (frame 1040, 388 s):
  3180 initial -> 145 unique; the two strong grains match the CPU golden
  exactly by DB row + match count — row 76986273 (19 matches) and row
  95915314 (18 matches).  Confirms the size_t/bounds hardening is correct at
  scale and the GPU agrees with the CPU on indexed orientations.

## DONE since — score unify + streaming daemon (validated on sentosa)
- GPU score now full-precision float (uint8 quantization dropped, CPU parity).
  100M-DB run still matches the CPU golden grains exactly (rows 76986273/19,
  95915314/18).
- LaueMatchingGPUStream daemon functional test: loaded 100M cache (51 s), bound
  port 60517 (4 CUDA streams), received a TCP frame, indexed it (GPU 101 ms,
  0.425 s/frame total) to the SAME golden grains, and shut down cleanly on
  SIGTERM ("exited cleanly" — sigaction + handler-join path).

## DEFERRED — behaviour-changing (need their own validation, not hardening)
- `mergeDuplicateOrientations` O(N²) pairwise matrix: bound `nrResults` before
  the merge (changes results for pathological high-match frames; the NULL check
  above already prevents the segfault).
- GPU scores `sum·√count` on **uint8-quantized** intensity vs CPU's
  full-precision `count·√sum` family — unify the score/precision (changes GPU
  results; validate on sentosa against the CPU golden).
