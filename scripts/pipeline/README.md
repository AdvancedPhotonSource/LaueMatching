# LaueMatching many-grain pipeline

Index and validate **hundreds of overlapping grains per frame** from pink/white-beam Laue data,
live on the beamline's own GPUs, and (for a two-phase hcp/bcc alloy) reconstruct the prior-β grain from its α
variants. This directory is self-contained: a run script, annotated parameter templates, and the
analysis scripts.

Everything is driven by **one parameter file per crystallographic phase** plus a **data folder**.
Nothing here is specific to a single experiment — change only the geometry, lattice, and energy
values in the parameter file to run on a different instrument or material.

---

## Quick start — this beamline (34-ID-E)

Everything is already installed and configured; the command is one line:

```bash
cd $LAUE_WORK     # working dir (params, database, launcher)
./run_laue.sh  /path/to/DATA_FOLDER                   # index alpha+beta live as frames land
touch /path/to/DATA_FOLDER/STOP_LAUE                  # stop
```

There is **one** launcher — `scripts/pipeline/run_laue.sh` — and the `run_laue.sh` in the working
directory is a symlink to it. Its **CONFIG block at the top** holds the values you would otherwise
set by hand; on this beamline they are already filled in:

| CONFIG value | Set to (34-ID-E) |
|---|---|
| `SCRIPTS` — LaueMatching install | `/home/beams/EPIX34ID/opt/LaueMatching/scripts` |
| `PY` — Python environment | `/home/beams/EPIX34ID/conda-envs/laue_rt/bin/python` (the `laue_rt` conda env) |
| `WORK` — working dir | `$LAUE_WORK` (parameter files, database, results) |
| `ALPHA_CONFIG` / `BETA_CONFIG` | `$WORK/params/params_Ti_alpha.txt` / `..._beta.txt` |

**To change `WORK`, `PY`, or the config paths, edit that CONFIG block** — that is the one place they
live. (You can also override any of them for a single run as an environment variable, e.g.
`WORK=/somewhere ./run_laue.sh DATA_FOLDER`.)

The analysis scripts in `analysis/` are run with the same Python, e.g.
`/home/beams/EPIX34ID/conda-envs/laue_rt/bin/python analysis/parentbeta_reconstruct.py 30`.

---

## 0. Prerequisites (once)

- LaueMatching installed with a CUDA GPU (the indexer runs on the GPU; the refinement stage uses CPU cores).
- A **refined detector geometry** (the `geoN_*.xml` from your calibration) → the `P_Array` / `R_Array`
  values in the parameter file.
- The **crystal(s)**: space group + lattice parameters.
- Built **once per material** with the packaged tools in `../`:
  - `GenerateOrientations.py` → the 100-million-orientation database (`100MilOrients.bin`), shared by all phases.
  - `GenerateHKLs.py` → the allowed-reflection list per phase (`valid_hkls_<phase>.csv`).
  - `GenerateSimulation.py` → the forward-spot cache per phase (`forward_<phase>.bin`).

Copy `params_alpha.template.txt` / `params_beta.template.txt`, fill in the marked values, and save as
`params_alpha.txt` / `params_beta.txt`.

---

## 1. Index — live or batch

```bash
# edit the CONFIG block at the top of run_laue.sh (WORK, PY, ALPHA_CONFIG, BETA_CONFIG, GPUs)
./run_laue.sh  /path/to/DATA_FOLDER  [/entry1/data/data]
```

- **Watch mode (default):** each new `.h5` frame is indexed as the detector writes it. Stop cleanly with
  `touch /path/to/DATA_FOLDER/STOP_LAUE`.
- **Batch an existing dataset:** set `WATCH=""` in the config block (or `export WATCH=""`).
- One GPU indexer runs per phase (α on GPU 0, β on GPU 1 by default). To index a single phase, set
  `BETA_CONFIG=""`.

**Output:** one result file per frame under `results/<phase>_<timestamp>/results/image_*.output.h5`,
containing the indexed orientations and their assigned spots.

---

## 2. Validate and map

No grain is reported without passing an independent statistical test. Run the analysis scripts in
[`analysis/`](analysis/) against the result folder:

| Script | What it does |
|---|---|
| `analysis/map_validate_cluster.py` | per-frame **spot test** (predicted pattern vs a random-orientation null, Poisson p<10⁻⁴) + map-wide clustering → the verified α grain map |
| `analysis/beta_map_validate.py`    | the same for the β (BCC) phase |
| `analysis/batch_peel_driver.py`    | **iterative peel** for dense frames: index → subtract each grain's full pattern → re-index the residual, until it stops finding grains |
| `analysis/grain_extent_backfill.py`| **cross-frame backfill**: project every confirmed grain into every frame, add present-but-missed detections → grain-extent (shape) map |

> **Set the paths first.** Each analysis script has a short config block at the very top
> (`WORK`, `DATA`, and the results sub-folder). Point them at your run before executing. Every
> reported number comes with its null.

---

## 3. Parent-β reconstruction (a two-phase hcp/bcc alloy)

Retained β and the α laths obey the Burgers orientation relationship, so each β predicts 12 α
variants. From the α orientations we infer the prior-β grain, checked against random **and**
Burgers-adjacent decoy nulls, and anchored to directly-indexed retained β.

```bash
python analysis/parentbeta_validate.py    alpha 32     # validate α on the scan  (repeat: beta)
python analysis/parentbeta_reconstruct.py 30           # infer parent(s); arg = min α-cluster size
```

Output: `parentbeta_reconstruction.png` (variant-ID map + occupancy) and a printed six-gate report
(synthetic control, 11/12-variant parent, retained-β anchor, rejected decoys).

---

## How it works (one paragraph)

Classical Laue indexing isolates one pattern at a time and stalls when a hundred grains overlap.
LaueMatching instead scores **every** orientation in a 0.4°-grid, 100-million-orientation database
against the whole image on the GPU, so overlap carries no penalty. Detection runs on an aggressively
thresholded image (only the brightest ~2% of pixels) to keep the search tractable, while all
verification uses the full background-subtracted frame — so no signal is discarded from the evidence.
The iterative peel then recovers fainter grains pass by pass.

---

## 4. Full analysis chain (one command per scan)

`analysis/run_analysis_chain.sh` runs the whole validation and interpretation sequence for
one indexed scan. Every path comes from the environment, so it runs on any host against any
scan:

```bash
env LAUE_WORK=/path/to/work \
    LAUE_SCAN_DATA=/path/to/raw/frames \
    LAUE_SCAN_ALPHA=/path/to/results/alpha_<ts> \
    LAUE_SCAN_BETA=/path/to/results/beta_<ts> \
    LAUE_OUT_PREFIX=myscan  NW=16 \
    bash analysis/run_analysis_chain.sh
```

Order matters — each step depends on the one before:

| step | script | what it does |
|---|---|---|
| 1 | `parentbeta_validate.py {phase} {nw} env` | per-frame Poisson spot test, p<1e-4 |
| 2 | `null_model.py` | **measures the random-orientation null on this scan** |
| 3 | `empirical_gate.py` | re-scores the validated set against that measured null |
| 4 | `beta_alpha_exclusion_census.py env {nw}` | β scored only on peaks α cannot explain |
| 5 | `exclusion_null.py` | measured null for that exclusion test |
| 6 | `parentbeta_reconstruct.py {minsize} {prefix}` | Burgers prior-β inference |
| 7 | `anchor_null.py` | tests whether the retained-β anchor beats chance |
| 8 | `variant_coherence.py` | spatial coherence of the variant map vs a shuffle null |
| 9 | `validated_figures.py` | report plates |

Supporting tools: `scan_map.py` (unvalidated catalog map), `regrain.py` (contiguity-aware
grain counts), `tolerance_sensitivity.py`, `big_grain_diagnostic.py` /
`big_grain_split_test.py` (is one large "grain" actually several?), `collect_scan_metrics.py`
(cross-scan summary JSON), `catalog_figures.py`.

### Three things this chain exists to enforce

1. **Measure the null on the scan in hand.** The built-in `p<1e-4` gate assumes peaks are
   scattered uniformly. Real Laue peak fields are clustered, so the true null has much
   heavier tails and the analytic gate under-rejects. Across nine scans the measured α null
   maximum ranged 14–17 and β 11–16 — a single inherited value misstates the others. On one
   dataset this changed the defensible β count by two orders of magnitude.
2. **A grain is a *contiguous* region of consistent orientation.** Clustering on orientation
   alone merges regions that are spatially disjoint. `regrain.py` splits clusters into
   connected components; on one scan that moved α from 325 to 614 and β from 40 to 27 — the
   two phases moving in opposite directions, which is what tells you it is a definitional
   fix and not a tuning knob.
3. **Check that a "corroborating" statistic beats chance.** The retained-β anchor is
   compelling with few β clusters and meaningless with many: with 2,537 candidates a *random*
   orientation lands within 1.74° of one 9% of the time. `anchor_null.py` measures this.

## Gotchas (each of these cost hours)

- **tcsh `noclobber` silently refuses `>` on an existing file.** Remote writes over ssh must
  go through `bash -s` (feed the script on stdin), not `ssh host "cmd > file"`. A refused
  write looks exactly like a successful one.
- **`ls dir/*.h5 | wc -l` returns 0 past ARG_MAX** (~40k files), and `ls -dt results/alpha_*`
  will happily return `alpha_<ts>.launch.log` because the log's mtime is newer than the run
  directory. Use `find`, and `-type d` when you mean a directory.
- **`CUDA_VISIBLE_DEVICES` without `CUDA_DEVICE_ORDER=PCI_BUS_ID`** selects by CUDA's
  FASTEST_FIRST ordering, which need not match nvidia-smi. On a mixed-GPU host this can put
  your job on someone else's card while the intended ones idle.
- **`pgrep -f <pattern>` matches its own command line.** Kill loops that name the target
  script will kill the ssh session running them. Use a bracketed pattern (`worke[r]`) or a
  script file.
- **Output files appear in one late batch after post-processing**, which is largely
  single-threaded — a 1.9 GB `solutions.txt` took ~45 min. "0 outputs after N frames" is not
  a stall.
- **Frames where the beam is off the specimen contain no diffraction and correctly produce no
  output.** Verification tolerances must allow for this: one 40,401-frame scan had a genuine
  1,947-frame blank band (peak counts 4–6 there against ~1,000 elsewhere).
