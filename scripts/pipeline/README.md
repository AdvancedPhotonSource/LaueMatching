# LaueMatching many-grain pipeline

Index and validate **hundreds of overlapping grains per frame** from pink/white-beam Laue data,
live on the beamline's own GPUs, and (for Ti-6Al-4V) reconstruct the prior-β grain from its α
variants. This directory is self-contained: a run script, annotated parameter templates, and the
analysis scripts.

Everything is driven by **one parameter file per crystallographic phase** plus a **data folder**.
Nothing here is specific to a single experiment — change only the geometry, lattice, and energy
values in the parameter file to run on a different instrument or material.

---

## Quick start — this beamline (34-ID-E)

Everything is already installed and configured; the command is one line:

```bash
cd /net/hpcs34/data34c/for_Hemant/lauematching_ti     # working dir (params, database, launcher)
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
| `WORK` — working dir | `/net/hpcs34/data34c/for_Hemant/lauematching_ti` (parameter files, database, results) |
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

## 3. Parent-β reconstruction (Ti-6Al-4V)

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
