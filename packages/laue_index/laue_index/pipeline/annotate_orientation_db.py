#!/usr/bin/env python
"""annotate_orientation_db.py — write a provenance sidecar next to an
orientation binary (``100MilOrients.bin``).

The 7.2 GB (6.7 GiB) ``100MilOrients.bin`` that ships with LaueMatching was generated
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
#: What is known about the RELEASED database (GitHub release v1.0-data). It is
#: applied only when the file's full SHA-256 equals artifacts.ORIENT_DB_SHA256;
#: any other file is described as what it is: generator unknown.
RELEASE_CONFIG = {
    "source": "GitHub release v1.0-data (4 parts, reassembled)",
    "spacing_deg": 0.4,
    "covers": "full SO(3), not the fundamental zone (oversampled on purpose)",
    "crystal_system_tag": "cubic",
}
LAYOUT = {"record_bytes": RECORD_BYTES,
          "record_layout": "row-major 3x3 float64 rotation matrix"}


def annotate(orient_file: Path, extra_notes: str | None = None, *, strong_hash: bool = True) -> Path:
    """Stamp an EXISTING orientation database with an artifact record
    (``<file>.meta.json``, kind ``orientation_db``, ``retroactive: true``).

    The full SHA-256 is always computed (``strong_hash`` is kept for old
    callers): it is how the released database is recognised. Unknowns stay
    unknown; nothing is inferred from the file name or size.
    """
    from laue_index import artifacts as A
    orient_file = Path(orient_file)
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
    sha = lp._strong_hash(orient_file)
    is_release = sha == A.ORIENT_DB_SHA256
    extra = {"is_release": is_release, "actual_n_orientations": n_records,
             "crystal_system": "cubic",
             "covers": "full SO(3), not the fundamental zone",
             "generator": ("unknown (pre-repo history); the released v1.0-data database"
                           if is_release else
                           "unknown: not the released database and no record of how it "
                           "was made")}
    if extra_notes:
        extra["extra_notes"] = extra_notes
    A.write_record(orient_file, "orientation_db", layout=dict(LAYOUT, n_orientations=n_records),
                   config=dict(RELEASE_CONFIG) if is_release else {}, extra=extra,
                   retroactive=True)
    return A.sidecar_path(orient_file)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--file", default="100MilOrients.bin",
                   help="Orientation binary to annotate (default: 100MilOrients.bin)")
    p.add_argument("--notes", default=None, help="Extra notes to embed in the sidecar")
    p.add_argument("--strong-hash", action="store_true",
                   help="Compute a full SHA-256 (slow: ~40s for 7.2 GB) instead of the weak head+tail hash")
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
