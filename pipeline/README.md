# LaueMatching many-grain pipeline

Index and validate **hundreds of overlapping grains per frame** from pink/white-beam Laue data,
live on the beamline's own GPUs, and (for a two-phase hcp/bcc alloy) reconstruct the prior-β grain from its α
variants. This directory is self-contained: a run script, annotated parameter templates, and the
analysis scripts.

Everything is driven by **one parameter file per crystallographic phase** plus a **data folder**.
The **indexer** is not specific to any experiment — change only the geometry, lattice, and energy
values in the parameter file to run on a different instrument or material. The **analysis**
scripts read the material (lattice, reflection list, geometry, energy window, space group) from
that same parameter file through `analysis/laue_material.py`, selected by `LAUE_PHASES` and
`LAUE_PARAMS_<PHASE>`; symmetry follows the space group, never the phase name.

**Starting on a dataset this pipeline has never seen?** Read the handbook,
[`../manuals/laue/README.md`](../manuals/laue/README.md) — survey the experiment folder, decide which analyses the
material system actually supports, index, analyse, report.

The handbook says *what to do*; the **lab notebooks** say *what was found*, including what
turned out to be wrong — one per geometry or station:
[`LAB_NOTEBOOK.md`](../manuals/laue/LAB_NOTEBOOK.md) (reflection, 34-ID-E),
[`LAB_NOTEBOOK_TPS21A.md`](../manuals/laue/LAB_NOTEBOOK_TPS21A.md) (reflection, TPS 21A, XMAS
calibration) and
[`LAB_NOTEBOOK_16BMD_Si.md`](../manuals/laue/LAB_NOTEBOOK_16BMD_Si.md) (transmission, 16-BM-D).

---

## Quick start

The launch is one line, but **it must carry your paths**: the CONFIG block at the top of
`run_laue.sh` ships with portable defaults (`WORK=$HOME/laue_run`, `PY=python`), not with any
site's values. Two kinds of environment work:

- a **pip environment** with `laue-index` installed (the C indexer is compiled in), plus a
  checkout of this repo for `pipeline/`; or
- an **editable checkout** (`pip install -e packages/laue_index`), for changing the code.

Either way, name the python by its full path -- a login shell over ssh does not have your conda
environment on `PATH`:

```bash
CHECKOUT=/path/to/LaueMatching
WORK=/path/to/work \
PY=/path/to/env/bin/python \
SCRIPTS=$CHECKOUT/scripts \
ALPHA_CONFIG=/path/to/work/params/params_alpha.txt \
BETA_CONFIG=/path/to/work/params/params_beta.txt \
$CHECKOUT/pipeline/run_laue.sh /path/to/DATA_FOLDER
touch /path/to/DATA_FOLDER/STOP_LAUE                  # stop a watch-mode run
```

Add `DRY_RUN=1` to print the resolved paths and stop without launching anything.

There is **one** single-host launcher, `pipeline/run_laue.sh`. Its **CONFIG block at the top** is
where these live; each can be set per run in the environment (as above) or made permanent by
editing the block:

| CONFIG value | what it is |
|---|---|
| `SCRIPTS` | directory holding `laue_orchestrator.py`: the checkout's `scripts/`. If unset, `run_laue.sh` uses `../scripts` next to itself, else the installed package's `laue_index/pipeline/` (asked of `PY`). Either way it **checks** the file is there before launching. |
| `PY` | python with `laue-index` installed |
| `WORK` | working dir: results land in `$WORK/results/` |
| `ALPHA_CONFIG` / `BETA_CONFIG` | the parameter files; `BETA_CONFIG=""` for a single phase |

`run_laue.sh` refuses to start if `SCRIPTS` has no orchestrator, a config is missing or still holds
`__SET_ME__`, and after each detached launch it checks the orchestrator is still alive a few
seconds later (printing the launch log's tail and exiting 1 if not). Alive at launch is not
success: count the per-frame outputs when the run ends.

If `WORK` is left unset the run lands in `$HOME/laue_run` and the parameter-file lookup fails
there -- a wrong path, not a missing one, so check the launcher's echoed paths before walking away.

**Many shards across several hosts/GPUs:** use [`dispatch/`](dispatch/README.md) -- it makes
row-aligned shards and a plan, checks the plan (unique ports and ResultDirs, a real background,
the indexer present on every host), sizes the preprocessing pool per shard, and watches the runs
to completion. `launch_shard.sh` is retired and only prints that pointer.

The analysis scripts in `analysis/` are run with the same Python, e.g.
`$PY analysis/parentbeta_reconstruct.py 30`.

---

## 0. Prerequisites (once)

- LaueMatching installed with a CUDA GPU (the indexer runs on the GPU; the refinement stage uses CPU cores).
- A **refined detector geometry** (the `geoN_*.xml` from your calibration) → the `P_Array` / `R_Array`
  values in the parameter file.
- The **crystal(s)**: space group + lattice parameters.
- Built **once per material** with the packaged tools -- `$CHECKOUT/scripts/<tool>.py` in a
  checkout (thin shims), or `python -c "from laue_index.pipeline import run_module; run_module('<tool>')" ...`
  from an installed package (the modules live in `laue_index/pipeline/`):
  - `GenerateOrientations.py` → the 100-million-orientation database (`100MilOrients.bin`), shared by all phases.
  - `GenerateHKLs.py` → the allowed-reflection list per phase (`valid_hkls_<phase>.csv`).
  - `GenerateSimulation.py` → the forward-spot cache per phase (`forward_<phase>.bin`).

Copy `params_alpha.template.txt` / `params_beta.template.txt`, replace **every** `__SET_ME__`, and
save as `params_alpha.txt` / `params_beta.txt`. The placeholders are deliberately not numbers.
From laue-index 0.7.2 the Python parsers refuse one in `SpaceGroup`, `Symmetry`,
`LatticeParameter`, `P_Array` and `R_Array`, the C binaries in `LatticeParameter`, `P_Array` and
`R_Array`, and `run_laue.sh` / `dispatch/mkrun.py` refuse any file that still holds one. (In
0.7.1 the parsers fell back to built-in defaults, another experiment's values.) `R_Array` is a rotation vector with the angle in **radians**;
`tol_LatC` / `tol_c_over_a` (optional lattice fit) are **fractions**, 0.001 = 0.1%.

---

## 1. Index — live or batch

```bash
# set WORK, PY, SCRIPTS, ALPHA_CONFIG, BETA_CONFIG, GPUs (environment or the CONFIG block)
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

> **Set the paths first.** The analysis scripts read their paths and settings from the
> environment. The one table of every variable, whether it is required, its default and which
> scripts read it, is at the end of
> [Phase 4](../manuals/laue/phase-4-analyse.md#environment-variables-of-the-analysis-scripts).
> Every reported number comes with its null.

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
env LAUE_PHASES=alpha,beta \
    LAUE_WORK=/path/to/work \
    LAUE_SCAN_DATA=/path/to/raw/frames \
    LAUE_SCAN_ALPHA=/path/to/results/alpha_<ts> \
    LAUE_SCAN_BETA=/path/to/results/beta_<ts> \
    LAUE_OUT_PREFIX=myscan  NW=16 \
    LAUE_PARAMS_ALPHA=/path/to/params_alpha.txt LAUE_PARAMS_BETA=/path/to/params_beta.txt \
    LAUE_MOUNT_DEG=45 \
    PY=/path/to/env/bin/python \
    bash analysis/run_analysis_chain.sh
```

Every variable: the table at the end of
[Phase 4](../manuals/laue/phase-4-analyse.md#environment-variables-of-the-analysis-scripts).

`LAUE_PHASES` is required. Validation, the null and the gate run for every listed phase;
steps 4-8 compare alpha with beta and run only when both are listed, and
`validated_figures.py` only when every phase is named alpha or beta -- otherwise they are
reported as SKIPPED. A single-phase scan: `LAUE_PHASES=zn LAUE_SCAN_ZN=... LAUE_PARAMS=...`
(no `LAUE_SCAN_BETA`, no `LAUE_MOUNT_DEG`). Every variable the selected steps need is
checked before the first step runs, and all missing ones are named together.

Every step must succeed or the chain stops with `STEP FAILED`. `null_model.py` writes the
measured null to `$LAUE_WORK/peel_map/<LAUE_OUT_PREFIX>_null.json`; the gating scripts
(`regrain.py`, `empirical_gate.py`, `validated_figures.py`) read that JSON directly, and the
chain also exports its maximum as `LAUE_NULLMAX_<PHASE>` (replacing any value inherited from
your environment), so the gate always uses the null of the scan in hand.
`LAUE_NULLMAX_<PHASE>` is only an override and must be the maximum of the statistic in force.
The gates default to `nhit` (`LAUE_GATE_STAT=nhit_distinct` is available, not yet validated as
a gate); the per-frame analytic Poisson gates use `nhit`.

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
   maximum ranged 14–17 and β 11–16 (source not in repo) — a single inherited value misstates
   the others. On one
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
- **Never combine `ssh -n` with `bash -s`.** `-n` points ssh's stdin at `/dev/null`, so the
  remote `bash -s` receives an **empty script**, runs nothing and exits 0 -- which reads as a
  failed or empty check ("no binary", "no process") rather than a broken one. Use `-n` only for
  a remote command that reads nothing; feed scripts on stdin without it, and have every remote
  check print a sentinel line whose absence you treat as an ssh failure.
- **`ssh` inside a `while read` loop eats the loop's stdin** and silently drops the remaining
  lines. Read the list into an array first and iterate with `for`.
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
