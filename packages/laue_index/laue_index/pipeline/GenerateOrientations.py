#!/usr/bin/env python
"""GenerateOrientations.py — generate a LaueMatching orientation binary
from user-supplied spacing and crystal system, using :mod:`orix`.

Design note — **we sample the full SO(3), not the fundamental zone.**
This is load-bearing for LaueMatching: the over-sampling across symmetry
copies is what lets the indexer filter out spurious matches by voting
across symmetry-equivalent orientations. Reducing to the fundamental
zone would break that filter and silently degrade indexing quality.

Output format matches the existing ``100MilOrients.bin``:
    - row-major 3x3 ``float64`` rotation matrices
    - 9 * 8 = 72 bytes per record
    - no header, no padding

Usage
-----
    python scripts/GenerateOrientations.py \\
        --spacing-deg 0.4 \\
        --crystal-system cubic \\
        --output 100MilOrients.bin

Running with ``--spacing-deg 0.4`` produces ~100 million rotations on the
full SO(3), matching the size of the shipped binary within a few
percent (the exact count depends on discretization).

A ``<output>.meta.json`` sidecar is written alongside the binary with git
commit, CLI args, orix version, record count, and a weak fingerprint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import laue_provenance as lp  # noqa: E402

RECORD_BYTES = 9 * 8  # 3x3 float64
CHUNK_ROWS = 1 << 20  # 1M rotations per disk-write chunk (~72 MB)


# Map CLI crystal-system name → orix Symmetry class name. We import lazily
# so ``--help`` works without orix installed.
CRYSTAL_SYSTEMS = {
    "triclinic": "C1",
    "monoclinic": "C2h",
    "orthorhombic": "D2h",
    "tetragonal": "D4h",
    "trigonal": "D3d",
    "hexagonal": "D6h",
    "cubic": "Oh",
}


def _to_matrix_chunked(rot, chunk: int = CHUNK_ROWS):
    """Yield (N, 3, 3) float64 matrix chunks from an orix Rotation without
    materializing the full array in memory.
    """
    n = rot.size
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        yield rot[start:stop].to_matrix().astype(np.float64, copy=False)


def generate(
    spacing_deg: float,
    crystal_system: str,
    output_path: Path,
    *,
    sampling: str = "haar",
    strong_hash: bool = False,
) -> dict:
    from orix.sampling import uniform_SO3_sample
    import orix

    if crystal_system not in CRYSTAL_SYSTEMS:
        raise ValueError(
            f"unknown crystal system {crystal_system!r}; "
            f"choose from {sorted(CRYSTAL_SYSTEMS)}"
        )

    t0 = time.time()
    print(f"Sampling full SO(3) at spacing {spacing_deg} deg "
          f"(crystal system tag: {crystal_system})...")
    # ``uniform_SO3_sample`` generates rotations spanning the FULL sphere.
    # We deliberately do NOT call any fundamental-zone reduction on it.
    rot = uniform_SO3_sample(spacing_deg, method=sampling) if sampling != "haar" else uniform_SO3_sample(spacing_deg)
    n = rot.size
    print(f"Generated {n:,} rotations in {time.time() - t0:.1f}s")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    total_bytes = 0
    with open(output_path, "wb") as fh:
        for mats in _to_matrix_chunked(rot):
            # ``mats`` is row-major (N, 3, 3) float64. ``tobytes`` preserves
            # row-major ordering, which matches the C reader that does
            # ``for k { for l { read double } }``.
            contig = np.ascontiguousarray(mats, dtype=np.float64)
            fh.write(contig.tobytes())
            total_bytes += contig.nbytes
    write_time = time.time() - t0
    expected_bytes = n * RECORD_BYTES
    assert total_bytes == expected_bytes, (total_bytes, expected_bytes)
    print(f"Wrote {n:,} orientations ({total_bytes / 1e9:.2f} GB) in {write_time:.1f}s")

    prov = lp.collect(
        config={
            "spacing_deg": spacing_deg,
            "crystal_system": crystal_system,
            "crystal_system_symmetry_tag": CRYSTAL_SYSTEMS[crystal_system],
            "sampling_method": sampling,
            "covers": "full SO(3), not the fundamental zone",
        },
        input_files=[],
        extra={
            "orix_version": orix.__version__,
            "numpy_version": np.__version__,
            "n_orientations": int(n),
            "record_bytes": RECORD_BYTES,
            "record_layout": "row-major 3x3 float64 rotation matrix",
            "output_size_bytes": int(total_bytes),
        },
        strong_hash=strong_hash,
    )
    # file_fingerprint was skipped (input_files=[]) because the binary we
    # are describing *is* the output. Fingerprint it now and attach.
    prov["inputs"] = [lp.file_fingerprint(output_path, strong=strong_hash)]
    sidecar = output_path.with_suffix(output_path.suffix + ".meta.json")
    lp.write_sidecar_json(sidecar, prov)
    print(f"Wrote sidecar provenance → {sidecar}")

    return {"n_orientations": n, "output": str(output_path), "sidecar": str(sidecar)}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--spacing-deg", type=float, required=True,
                   help="Target angular spacing between adjacent rotations in degrees. "
                        "0.4 reproduces the shipped 100M-orientation database.")
    p.add_argument("--crystal-system", choices=sorted(CRYSTAL_SYSTEMS), default="cubic",
                   help="Crystal system tag recorded in the sidecar. Sampling itself covers the full SO(3); this is metadata only.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output binary path (e.g. my_orients.bin)")
    p.add_argument("--sampling", choices=["haar", "cubochoric"], default="haar",
                   help="Sampling method passed to orix.sampling.uniform_SO3_sample (default: haar)")
    p.add_argument("--strong-hash", action="store_true",
                   help="Compute full SHA-256 of the output binary (slow) instead of the weak head+tail hash")
    args = p.parse_args()

    try:
        generate(
            spacing_deg=args.spacing_deg,
            crystal_system=args.crystal_system,
            output_path=args.output,
            sampling=args.sampling,
            strong_hash=args.strong_hash,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
