"""What this install actually built, read from the manifest CMake wrote.

Module is ``buildmeta``, not ``build_info``, on purpose: the package exports a
FUNCTION called ``build_info``, and a module of the same name inside the same
package is shadowed by it -- ``from laue_index import build_info`` then hands
you the function and ``monkeypatch.setattr`` on it fails with AttributeError.

A version number says which sources were *meant* to be compiled; it says
nothing about which binaries came out. Those are different questions whenever
the build is conditional, and here it is: the CUDA binaries are attempted on
every install and skipped when there is no nvcc.

Measured 2026-09-06: `pip install --upgrade laue-index` on two beamline
environments removed ``LaueMatchingGPU`` and ``LaueMatchingGPUStream`` and left
``LaueMatchingCPU``, because the upgrade host had no CUDA toolkit. Nothing in
pip's output said so, both environments reported the new version correctly, and
the loss was found by listing ``bin/`` by hand. This manifest is what makes that
state answerable instead of invisible -- see :func:`laue_index.doctor.diagnose`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["MANIFEST_NAME", "build_info", "manifest_path", "c_src_sha256"]

MANIFEST_NAME = "_build_info.json"


def manifest_path() -> Path:
    """Where the manifest lives.

    Two layouts, and an EDITABLE install is the awkward one: the module is
    imported from the source checkout while the compiled artifacts and this
    manifest are installed into site-packages. Looking only beside ``__file__``
    therefore reports "no manifest" on a perfectly good dev install -- a false
    alarm, which costs the same investigation as a real one. So: beside the
    module first, then beside the binaries, which ``indexer`` already resolves
    correctly for both layouts.
    """
    here = Path(__file__).resolve().parent / MANIFEST_NAME
    if here.is_file():
        return here
    try:
        from . import indexer
        beside_bin = Path(indexer.binary_path()).resolve().parent.parent / MANIFEST_NAME
        if beside_bin.is_file():
            return beside_bin
    except Exception:
        pass
    return here


def build_info() -> dict[str, Any]:
    """The build manifest, or a clearly-marked placeholder if there is none.

    A missing manifest is itself information: it means this install predates
    the manifest (laue-index < 0.6.0) or was not produced by the CMake build at
    all, so nothing can be said about which binaries were attempted.
    """
    p = manifest_path()
    if not p.is_file():
        return {
            "schema": 0,
            "available": False,
            "reason": (f"no {MANIFEST_NAME} in {p.parent} -- this install "
                       "predates the build manifest (< 0.6.0) or was not built "
                       "by CMake"),
        }
    try:
        info = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return {"schema": 0, "available": False,
                "reason": f"{p} is unreadable: {e}"}
    info["available"] = True
    return info


def c_src_sha256() -> str | None:
    """Hash of the C sources this install was compiled from, if recorded.

    The point of recording it: a GPU binary salvaged from a previous install
    can be shown to be *equivalent* to what this install would have produced,
    rather than assumed to be. Restoring a binary whose source hash differs is
    how a stale executable ends up paired with new Python.
    """
    return build_info().get("c_src_sha256")
