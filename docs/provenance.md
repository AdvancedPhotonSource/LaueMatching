# Provenance in LaueMatching

Every artifact LaueMatching generates — from the HKL list through the
per-image indexing HDF5 — now carries a **provenance record** so a user
weeks later can tell:

- which build produced it (laue-index version, C-source hash, binary hashes; the git
  commit too when run from a checkout),
- what config was in effect,
- and which input files fed into it.

The single source of truth is `packages/laue_index/laue_index/pipeline/laue_provenance.py`.

## What gets stamped

| Producer                     | Where the record lives                          |
|------------------------------|--------------------------------------------------|
| `GenerateHKLs.py`            | `<hkl_file>.meta.json` artifact record (0.8.0; `.provenance.json` before, still read) |
| `GenerateSimulation.py`      | `/provenance` group inside the output HDF5       |
| `GenerateOrientations.py`    | `<output>.meta.json` artifact record              |
| `annotate_orientation_db.py`, `laue-index provenance stamp` | `<file>.meta.json` artifact record (retroactive) |
| `laue-index fetch-db`        | `<db>.meta.json` artifact record, full SHA-256 checked against the release |
| the C binaries (forward cache) | `<ForwardFile>.meta.json` artifact record      |
| `RunImage.py`, `laue_image_server.py` (backgrounds) | `<background>.meta.json` artifact record |
| `laue_image_server.py`       | `<mapping_file>.provenance.json` sidecar          |
| `laue_orchestrator.py`       | `<output_dir>/provenance.json` at start-of-run    |
| `laue_postprocess.py`        | `/entry/provenance` group inside each `image_XXXXX.output.h5` |
| `RunImage.py`                | `/entry/provenance` group inside the output HDF5 |

## Data artifacts: `<file>.meta.json` (laue-index 0.8.0)

The orientation database, HKL lists, forward caches and backgrounds are bare
binary (or plain-text) files. Each now carries an artifact record beside it,
written by `laue_index.artifacts` (Python) or `writeForwardCacheMeta` (C), in
one schema:

```
{"schema": "laue-artifact/1",
 "kind":   "orientation_db" | "hkl_list" | "forward_cache" | "background",
 "retroactive": false,            # true when stamped after the fact
 "artifact": {"path", "basename", "size", "quick", "sha256", "layout"},
 "config":   {...the FULL configuration that generated it...},
 "config_file": {"path", "sha256", "text"},   # when made from a params file
 "inputs":   [{"role", "path", "size", "quick" | "sha256", "record"}],
 "extra":    {...},
 "producer": {...host, user, time, build (as the run records above)...}}
```

| kind | `config` holds | written by |
|---|---|---|
| `orientation_db` | spacing, coverage, sampling method, crystal-system tag (or nothing, for a database whose generator is unknown) | GenerateOrientations, fetch-db, annotate / `provenance stamp` |
| `hkl_list` | SpaceGroup, Symmetry, LatticeParameter, P_Array, R_Array, NrPx*, Px*, Elo, Ehi (params-file key names) | GenerateHKLs |
| `forward_cache` | the key, the format, every key input under its params-file name, and the params file's full text in `config_file` | the C binaries |
| `background` | FilterRadius, NMeadianPasses; the source frame (file + index) in `inputs` | RunImage, laue_image_server |

`quick` is SHA-256 over the first and last MiB and the size; `sha256` is the
full hash. The released `100MilOrients.bin` is recognised by its full SHA-256,
`351dc8e0dec0db91aff67493bf749957663e9926dda55db145205a1e0915fd20`
(`artifacts.ORIENT_DB_SHA256`; `SHA256SUMS` on the v1.0-data release).

**Policy.** A missing record WARNS (the file is used unverified). A record that
disagrees with its file (size, hash, kind) or with the configuration REFUSES:
an HKL list recorded for another space group, symmetry or lattice; a forward
cache recorded for another key (the C prints `refusing to overwrite`; set
`DoFwd 1` to rebuild it deliberately). An HKL list made for a different
detector or energy window only warns. Runs check before launching the C
(`artifacts.check_run_inputs`) and write what they used under `artifacts` in
`provenance.json` / `/entry/provenance` (lineage).

**Tools.** `laue-index provenance show|verify [--full]|stamp --kind K [--params P]
FILE`; `stamp` writes a retroactive record and never invents what is unknown (a
`--params` file is recorded as declared, not verified). `laue-index doctor
--params FILE` reports every data artifact the params file names.

## Record shape (schema 2, laue-index 0.7.2 and later)

```
{
  "schema_version": "2",
  "timestamp_utc":  "2026-09-21T03:56:37.415468+00:00",
  "host":           "my-box",
  "user":           "<user>",
  "python":         "3.12.2",
  "laue_version":   "2.2.0",            # STALE: pipeline/_version.py, never bumped -- ignore
  "script":         {"argv0": "...", "argv": [...]},
  "git": {                              # "unknown" on a pip install (site-packages is no checkout)
    "commit":       "c7a111c...",
    "commit_short": "c7a111caa7d2",
    "dirty":        true,
    "branch":       "main",
    "remote":       "git@github.com:..."
  },
  "build": {                            # WHICH CODE RAN -- use this, not laue_version / git
    "laue_index_version": "0.7.2",
    "c_src_sha256":       "...",        # SHA-256 of the C sources, from the CMake manifest
    "manifest_version":   "0.7.2",
    "manifest":           {...laue_index.build_info()...},
    "bindir":             "/.../laue_index/bin",
    "binaries": {                       # full SHA-256 of each binary in bindir
      "LaueMatchingCPU":       {"path": "...", "size": ..., "mtime_utc": "...", "sha256": "..."},
      "LaueMatchingGPU":       {...},
      "LaueMatchingGPUStream": {...}    # or {"path": "...", "missing": true}
    },
    "executable": {                     # only when the caller passed the binary it ran
      "kind": "LaueMatchingGPUStream", "path": "...", "basename": "...", "realpath": "...",
      "size": ..., "mtime_utc": "...", "sha256": "..."     # or "missing": true
    }
    # if laue_index cannot be imported: "laue_index_version": "unknown" and "error": "...",
    # and no manifest / binaries (executable is still recorded, it is collected first)
  },
  "config":        {...LaueConfig snapshot...},
  "config_notes":  {"processing_type": "config label only ..."},   # when the snapshot has it
  "inputs":        [{"path": "...", "size": 7200000000, "sha256_head": "..."}, ...],
  "extra":         {caller-supplied fields}
}
```

What each identity answers:

- **`build.c_src_sha256`** -- was it built from the same *source*? Stable across rebuilds;
  use it to compare builds.
- **`build.binaries.<name>.sha256`** -- is it the same *executable file*? Not stable across
  rebuilds of identical source (nvcc embeds a PID-derived temporary filename), so use it only
  to prove two runs used the very same file. These hash whatever sits beside
  `indexer.binary_path()`, i.e. the CPU binary's directory.
- **`build.executable`** -- the binary that actually ran: `kind` (its file name), `path`,
  `realpath` and a full `sha256` (or `"missing": true`). Only two producers pass it: the
  streaming orchestrator (`provenance.json`), which resolves its daemon BEFORE stamping because
  it can run one from `<project_root>/build/` rather than from `bindir`, and also writes the
  path to `extra.daemon_bin` (`"NOT FOUND: ..."` if none; the run then fails after the stamp);
  and `RunImage.py` (`/entry/provenance`). Other producers leave it out.
- **`build.binaries`** -- a full-SHA-256 fingerprint of each of `LaueMatchingCPU`,
  `LaueMatchingGPU`, `LaueMatchingGPUStream` found in `bindir`, `{"path", "missing": true}`
  for one that is not there, or `{"error": ...}` if the lookup failed.
- **`config.processing_type`** is a config label (RunImage's `-g`; `"CPU"` by default, even
  on a streaming GPU run), not a record of the binary. Whenever the config snapshot carries it,
  the record adds `config_notes.processing_type` saying so and pointing at `build.executable`.
- **`config.robust_filter`** in the orchestrator's `provenance.json` is the
  `ConfigurationManager` value, which is RunImage's default (1) when the params file has no
  `RobustFilter` line. A streaming run applies the legacy filter in that case. From
  laue-index 0.7.3 the orchestrator also writes **`extra.streaming_postprocess`**
  (`robust_filter_key_present`, `robust_filter_effective`, `min_unique_effective`,
  `min_unique_source`: what the streaming post-processor actually applies) and
  `config_notes.robust_filter` pointing at it. In 0.7.2 records, read the params file instead.
  The per-image `/entry/provenance` and the image server's sidecar were always correct (they
  record the streaming parser's `null` for an absent key).
- The post-processor's own stdout/stderr are in `<output_dir>/postprocess.log` (0.7.3 and later).
- **`laue_version`** and, on a pip install, **`git`** identify nothing; they are kept so older
  records stay readable.

**Schema 1** (laue-index 0.7.1 and earlier) has no `build` or `config_notes` block. A schema-1
record cannot tell you which laue-index build produced it: `laue_version` read `2.2.0` across
many releases and `git.commit` is `"unknown"` on a pip install.

Text/CSV outputs carry the same record as commented header lines
(`laue_provenance.header_lines`): schema, timestamp, laue_index version, `c_src_sha256`, the
executable (kind, path, `(MISSING)` if absent) and its SHA-256 when known, one line per binary
(sha256 or missing), the stale `laue_version`, git commit/branch, host, user, argv, one line
per input, then the full record as one `provenance_json:` line.

## `sha256_head` — the weak fingerprint

The orientation database (7.2 GB, i.e. 6.7 GiB) would take ~40 s to full-SHA-256 on SSD.
To keep provenance writes near-instant, inputs are fingerprinted with:

    sha256( first_1_MiB  ||  last_1_MiB  ||  uint64_size )

This detects accidental replacement or silent corruption but is **not
cryptographically strong**. Producers that need a true digest can pass
`--strong-hash` (available on `GenerateOrientations.py` and
`annotate_orientation_db.py`).

## Reading a record

Python:

```python
import h5py
from laue_index.pipeline import laue_provenance as lp
with h5py.File("image_00001.output.h5") as hf:
    prov = lp.read_from_h5(hf, group="/entry/provenance")
print(prov["build"]["laue_index_version"], prov["build"]["c_src_sha256"])
```

Or from a sidecar:

```bash
jq '.build.laue_index_version, .build.c_src_sha256, .config' 100MilOrients.bin.meta.json
```
