#!/usr/bin/env python
"""annotate_orientation_db.py — write a provenance sidecar next to an
orientation binary (``100MilOrients.bin``).

The 6.7 GB ``100MilOrients.bin`` that ships with LaueMatching was generated
before this repository kept generator provenance. This script writes a
``<orient_file>.meta.json`` sidecar so downstream runs can record at least
the file's fingerprint, size, record count, and what we *think* we know
about how it was produced. Use :file:`GenerateOrientations.py` for new
databases where full provenance is captured automatically.

Usage
-----
    python scripts/annotate_orientation_db.py [--file 100MilOrients.bin]

Invoked by ``build.sh`` after the binary is reassembled from its
GitHub-Release parts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import laue_provenance as lp  # noqa: E402

RECORD_BYTES = 9 * 8  # 3x3 float64, row-major

# Best-known facts about the ``100MilOrients.bin`` that ships with the
# repo. These are retroactive: the original generator script is lost.
DEFAULT_DB_METADATA = {
    "expected_n_orientations": 100_000_000,
    "record_bytes": RECORD_BYTES,
    "record_layout": "row-major 3x3 float64 rotation matrix",
    "crystal_system": "cubic",
    "step_deg": 0.4,
    "covers": "full SO(3), not the fundamental zone — oversampled on purpose",
    "origin": "GitHub Release tag v1.0-data, reassembled from 4 parts by build.sh",
    "generator_script": "unknown — pre-repo-history",
    "notes": (
        "Provenance added retroactively. For reproducible regeneration use "
        "scripts/GenerateOrientations.py, which writes a full provenance sidecar."
    ),
}


def annotate(orient_file: Path, extra_notes: str | None = None, *, strong_hash: bool = False) -> Path:
    if not orient_file.exists():
        raise FileNotFoundError(orient_file)
    size = orient_file.stat().st_size
    if size % RECORD_BYTES != 0:
        print(
            f"warning: file size {size} is not a multiple of {RECORD_BYTES} "
            f"bytes; record count may be wrong",
            file=sys.stderr,
        )
    n_records = size // RECORD_BYTES
    meta = dict(DEFAULT_DB_METADATA)
    meta["actual_n_orientations"] = n_records
    meta["size_bytes"] = size
    if extra_notes:
        meta["extra_notes"] = extra_notes

    prov = lp.collect(
        config=None,
        input_files=[orient_file],
        extra=meta,
        strong_hash=strong_hash,
    )
    out = orient_file.with_suffix(orient_file.suffix + ".meta.json")
    # Keep the classic ``.meta.json`` naming convention as a straight sibling.
    # ``with_suffix`` would give ``100MilOrients.bin.meta.json``; that's what
    # we want since the binary is supposed to be immutable.
    lp.write_sidecar_json(out, prov)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="100MilOrients.bin",
                   help="Orientation binary to annotate (default: 100MilOrients.bin)")
    p.add_argument("--notes", default=None, help="Extra notes to embed in the sidecar")
    p.add_argument("--strong-hash", action="store_true",
                   help="Compute a full SHA-256 (slow: ~40s for 6.7GB) instead of the weak head+tail hash")
    args = p.parse_args()

    try:
        out = annotate(Path(args.file), extra_notes=args.notes, strong_hash=args.strong_hash)
    except FileNotFoundError as exc:
        print(f"error: orientation file not found: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
