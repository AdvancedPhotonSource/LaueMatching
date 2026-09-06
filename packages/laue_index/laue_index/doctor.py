"""`laue-index doctor` -- can this install actually do what it will be asked to?

The two states worth catching are both silent, and neither shows up in a
version string:

1. **A CUDA device is present but the install has no GPU binary.** The run falls
   back to CPU, or the streaming daemon refuses to start, long after the install
   that caused it. Measured 2026-09-06 on two beamline environments after a
   routine `pip install --upgrade`.

2. **The GPU binary cannot launch on this card.** ``cudaErrorNoKernelImageForDevice``
   is reported by the kernel *launch*, and a binary with no cubin for the device
   and no PTX at or below it prints ``Unique Orientations: 0`` and exits **0** --
   indistinguishable from a frame with no grains. PTX JIT rescues the
   older-toolkit-to-newer-card direction and never the reverse.

Everything here is a report, never a repair: it prints what it found, what that
means, and the one command that fixes it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import indexer
from .buildmeta import build_info

__all__ = ["detect_devices", "parse_architectures", "arch_supported",
           "diagnose", "main"]

_BINARIES = ("LaueMatchingCPU", "LaueMatchingGPU", "LaueMatchingGPUStream")


def detect_devices() -> list[dict[str, str]]:
    """CUDA devices visible right now, via nvidia-smi.

    nvidia-smi rather than a CUDA binding because it is what exists on a
    beamline host that has a driver but no toolkit and no pycuda -- exactly the
    machine where this question matters.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    devs = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1]:
            devs.append({"name": parts[0], "compute_cap": parts[1]})
    return devs


def parse_architectures(spec: str) -> tuple[set[int], set[int]]:
    """``"75-real;80-real;120-virtual"`` -> ({75, 80}, {120}) as (cubin, ptx).

    Bare numbers (CMake's shorthand) count as both, which is what CMake emits
    for them.
    """
    cubin: set[int] = set()
    ptx: set[int] = set()
    for tok in (spec or "").replace(",", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if tok.endswith("-real"):
            n = tok[:-5]
            if n.isdigit():
                cubin.add(int(n))
        elif tok.endswith("-virtual"):
            n = tok[:-8]
            if n.isdigit():
                ptx.add(int(n))
        elif tok.isdigit():
            cubin.add(int(tok)); ptx.add(int(tok))
    return cubin, ptx


def arch_supported(compute_cap: str, spec: str) -> tuple[bool, str]:
    """Can a binary built for *spec* launch on a device of *compute_cap*?

    Either a cubin for that exact architecture, or PTX at or BELOW it (JIT is
    forward-only). ``all-major`` and an unparseable spec are reported as
    unknown rather than guessed at.
    """
    try:
        cap = int(float(compute_cap) * 10)
    except (TypeError, ValueError):
        return True, f"device compute capability {compute_cap!r} unparseable"
    cubin, ptx = parse_architectures(spec)
    if not cubin and not ptx:
        return True, f"architecture list {spec!r} not parseable -- cannot check"
    if cap in cubin:
        return True, f"cubin for sm_{cap}"
    below = [p for p in ptx if p <= cap]
    if below:
        return True, f"no cubin for sm_{cap}, JIT from PTX sm_{max(below)}"
    return False, (f"no cubin for sm_{cap} and no PTX at or below it "
                   f"(cubins {sorted(cubin)}, ptx {sorted(ptx)}) -- the kernel "
                   f"cannot launch, and the run will report zero and exit 0")


def diagnose() -> dict[str, Any]:
    """Everything the report needs, as data, so it can be tested and scripted."""
    info = build_info()
    cuda = info.get("cuda", {}) if info.get("available") else {}
    bindir = Path(indexer.binary_path()).parent
    present = {b: (bindir / b).is_file() for b in _BINARIES}
    devices = detect_devices()

    problems: list[str] = []
    notes: list[str] = []

    if not present["LaueMatchingCPU"]:
        problems.append(
            "No CPU binary. Nothing can index. Reinstall on a machine with a C "
            "compiler and OpenMP, or set LAUEMATCHING_BIN to one you have.")

    have_gpu = present["LaueMatchingGPU"] and present["LaueMatchingGPUStream"]
    if devices and not have_gpu:
        why = cuda.get("reason", "unknown -- this install has no build manifest")
        problems.append(
            f"{len(devices)} CUDA device(s) visible but this install has no GPU "
            f"binary (build says: {why}). GPU and streaming runs are unavailable. "
            "Reinstall with nvcc on PATH: LAUEMATCHING_CUDA=1 pip install "
            "--force-reinstall --no-deps laue-index")
    elif not devices and not have_gpu:
        notes.append("No CUDA device visible and no GPU binary -- consistent.")

    if devices and have_gpu:
        spec = cuda.get("architectures", "")
        if not spec:
            notes.append(
                "GPU binaries present but the manifest records no architecture "
                "list, so they cannot be checked against this card. Likely built "
                "before 0.6.0 or restored by hand.")
        for d in devices:
            ok, why = arch_supported(d["compute_cap"], spec)
            line = f"{d['name']} (sm_{str(d['compute_cap']).replace('.', '')}): {why}"
            (notes if ok else problems).append(line)

    return {"build_info": info, "binaries": present, "bindir": str(bindir),
            "devices": devices, "problems": problems, "notes": notes,
            "healthy": not problems}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    d = diagnose()
    if as_json:
        print(json.dumps(d, indent=2))
        return 0 if d["healthy"] else 1

    info = d["build_info"]
    print("laue-index doctor")
    print(f"  package     {info.get('version', 'unknown')}")
    if not info.get("available"):
        print(f"  manifest    NONE -- {info.get('reason', '')}")
    else:
        print(f"  built       {info.get('built_at', '?')} on "
              f"{info.get('build_host', '?')}")
        c = info.get("cuda", {})
        print(f"  cuda build  mode={c.get('mode', '?')} built={c.get('built')} "
              f"({c.get('reason', '')})")
        if c.get("architectures"):
            print(f"  cuda archs  {c['architectures']}")
        if c.get("nvcc_version"):
            print(f"  nvcc        {c.get('nvcc_version')}  {c.get('nvcc', '')}")
    print(f"  bin dir     {d['bindir']}")
    for b, ok in d["binaries"].items():
        print(f"    {'yes' if ok else ' NO'}  {b}")
    if d["devices"]:
        for dev in d["devices"]:
            print(f"  device      {dev['name']}  compute {dev['compute_cap']}")
    else:
        print("  device      none visible (no nvidia-smi, or no GPU)")

    for n in d["notes"]:
        print(f"  note   {n}")
    for p in d["problems"]:
        print(f"  PROBLEM {p}")
    print("OK" if d["healthy"] else "PROBLEMS FOUND")
    return 0 if d["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
