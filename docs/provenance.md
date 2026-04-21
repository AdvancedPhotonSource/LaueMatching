# Provenance in LaueMatching

Every artifact LaueMatching generates — from the HKL list through the
per-image indexing HDF5 — now carries a **provenance record** so a user
weeks later can tell:

- which commit produced it,
- what config was in effect,
- and which input files fed into it.

The single source of truth is `scripts/laue_provenance.py`.

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

## Record shape

```
{
  "schema_version": "1",
  "timestamp_utc":  "2026-04-21T03:56:37.415468+00:00",
  "host":           "my-box",
  "user":           "hsharma",
  "python":         "3.12.2",
  "laue_version":   "2.1.0",
  "script":         {"argv0": "...", "argv": [...]},
  "git": {
    "commit":       "c7a111c...",
    "commit_short": "c7a111caa7d2",
    "dirty":        true,
    "branch":       "main",
    "remote":       "git@github.com:..."
  },
  "config":  {...LaueConfig snapshot...},
  "inputs":  [{"path": "...", "size": 7200000000, "sha256_head": "..."}, ...],
  "extra":   {caller-supplied fields}
}
```

## `sha256_head` — the weak fingerprint

The 6.7 GB orientation database would take ~40 s to full-SHA-256 on SSD.
To keep provenance writes near-instant, inputs are fingerprinted with:

    sha256( first_1_MiB  ||  last_1_MiB  ||  uint64_size )

This detects accidental replacement or silent corruption but is **not
cryptographically strong**. Producers that need a true digest can pass
`--strong-hash` (available on `GenerateOrientations.py` and
`annotate_orientation_db.py`).

## Reading a record

Python:

```python
import h5py, scripts.laue_provenance as lp
with h5py.File("image_00001.output.h5") as hf:
    prov = lp.read_from_h5(hf, group="/entry/provenance")
print(prov["git"]["commit"], prov["config"]["space_group"])
```

Or from a sidecar:

```bash
jq '.git.commit, .config' 100MilOrients.bin.meta.json
```
