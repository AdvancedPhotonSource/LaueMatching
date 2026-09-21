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
| `GenerateHKLs.py`            | `<hkl_file>.provenance.json` sidecar             |
| `GenerateSimulation.py`      | `/provenance` group inside the output HDF5       |
| `GenerateOrientations.py`    | `<output>.meta.json` sidecar                      |
| `annotate_orientation_db.py` | `<orient_file>.meta.json` sidecar                 |
| `laue_image_server.py`       | `<mapping_file>.provenance.json` sidecar          |
| `laue_orchestrator.py`       | `<output_dir>/provenance.json` at start-of-run    |
| `laue_postprocess.py`        | `/entry/provenance` group inside each `image_XXXXX.output.h5` |
| `RunImage.py`                | `/entry/provenance` group inside the output HDF5 |

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
