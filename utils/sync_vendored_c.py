#!/usr/bin/env python3
"""Sync the vendored C in packages/laue_index/c_src/ from the canonical src/.

WHY THE DUPLICATION IS DELIBERATE. `pip install laue-index` builds from the
package's OWN sdist, which contains only what is under packages/laue_index/. A
CMakeLists that reached up to the repo-root src/ works perfectly in a checkout
and silently produces no binary for every pip user -- the sdist has no src/ to
find. So the C is copied in, and this script keeps the copy honest.

Same reasoning, and the same solution, as MIDAS utils/sync_vendored_c.py.

    python utils/sync_vendored_c.py           # canonical src/ -> package c_src/
    python utils/sync_vendored_c.py --check   # CI mode: exit 1 on drift

Run --check in CI. A one-copy edit leaves the pip-installed indexer computing
something different from the one a checkout builds, with every test green.
"""
from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "src"
VENDORED = ROOT / "packages" / "laue_index" / "c_src"

#: Everything the three binaries need to compile standalone. The CUDA sources
#: are vendored too: `LAUEMATCHING_CUDA=1 pip install laue-index` builds them
#: from the sdist, and the sdist has no repo-root src/ to reach for.
FILES = [
    "LaueMatchingCPU.c",
    "LaueMatchingHeaders.h",
    "nelder_mead.c",
    "nelder_mead.h",
    "LaueMatchingGPU.cu",
    "LaueMatchingGPUStream.cu",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report drift and exit 1; do not copy")
    args = ap.parse_args()

    VENDORED.mkdir(parents=True, exist_ok=True)
    drift, missing = [], []

    for name in FILES:
        src, dst = CANONICAL / name, VENDORED / name
        if not src.is_file():
            print(f"ERROR: canonical source missing: {src}", file=sys.stderr)
            return 2
        if not dst.is_file():
            missing.append(name)
        elif not filecmp.cmp(src, dst, shallow=False):
            drift.append(name)

    if args.check:
        if missing or drift:
            for n in missing:
                print(f"MISSING from c_src/: {n}", file=sys.stderr)
            for n in drift:
                print(f"DRIFTED from src/:   {n}", file=sys.stderr)
            print("\nRun: python utils/sync_vendored_c.py", file=sys.stderr)
            return 1
        print(f"vendored C in sync ({len(FILES)} files)")
        return 0

    for name in missing + drift:
        shutil.copy2(CANONICAL / name, VENDORED / name)
        print(f"  synced {name}")
    if not (missing or drift):
        print(f"already in sync ({len(FILES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
