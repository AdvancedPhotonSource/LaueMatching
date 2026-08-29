"""Indexer stage — thin wrapper around the C indexing executable.

REFACTOR_PLAN §3 / §6.5.  Builds the argv, picks CPU/GPU, sets the library
environment, runs the binary, captures stdout/stderr to logs, and returns a
typed result.  Input *preparation* (orientation-DB copy, HKL generation) stays
with the orchestrator; this stage just runs the binary on inputs that exist.

The 5-arg CLI contract is the integration boundary with the C side (REFACTOR_PLAN
§8) and is preserved exactly:
    <exe> <config.txt> <orientation_file> <hkls.csv> <blurred_image.bin> <ncpus>
"""
from __future__ import annotations

import importlib.resources
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("LaueMatching")

__all__ = ["IndexerResult", "BinaryUnavailableError", "available",
           "binary_path", "require_binary", "resolve_executable", "run_indexer"]

#: Override the binary location outright. Checked first.
BINARY_ENV = "LAUEMATCHING_BIN"


class BinaryUnavailableError(RuntimeError):
    """The indexing executable could not be found."""


@dataclass
class IndexerResult:
    success: bool
    returncode: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    error: str | None = None


def _executable_name(compute_type: str = "CPU", do_forward: bool = False) -> str:
    """Which of the three binaries this request needs.

    ``STREAM`` names the persistent GPU daemon. It is not a fallback candidate
    for anything: a caller that wants the daemon wants the daemon, so an
    unavailable one must say so rather than silently hand back the CPU
    single-image binary.
    """
    compute_type = compute_type.upper()
    if compute_type in ("STREAM", "GPUSTREAM"):
        return "LaueMatchingGPUStream"
    if compute_type == "GPU" and not do_forward:
        return "LaueMatchingGPU"
    if compute_type == "GPU" and do_forward:
        logger.warning("GPU requested but DoFwd is enabled. Using CPU implementation (LaueMatchingCPU).")
    elif compute_type != "CPU":
        logger.warning(f"Processing type '{compute_type}' not recognized or incompatible. Using CPU implementation.")
    return "LaueMatchingCPU"


def _candidates(name: str, repo_root: str | None = None) -> list[Path]:
    """Every place the binary might legitimately live, in priority order.

    The old implementation returned exactly one path,
    ``<repo_root>/bin/<name>``, and required the caller to know where a source
    checkout was. That made an installed package unusable: `pip install
    laue-index` gave a wrapper whose central function needed a repo, and whose
    error advised running `make` in a build directory that does not exist.
    """
    out: list[Path] = []
    env = os.environ.get(BINARY_ENV)
    if env:
        p = Path(env)
        # Accept either the executable itself or a directory holding it.
        out.append(p / name if p.is_dir() else p)
    # Built into the package by scikit-build-core at `pip install` time.
    try:
        out.append(Path(str(importlib.resources.files("laue_index") / "bin" / name)))
    except (ModuleNotFoundError, FileNotFoundError, TypeError):
        pass
    # Editable installs: importlib.resources resolves to the SOURCE tree, which
    # has no bin/, while the binary sits in site-packages.
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for prefix in {sys.prefix, sys.exec_prefix}:
        out.append(Path(prefix) / "lib" / pyver / "site-packages" / "laue_index" / "bin" / name)
    # On PATH (release tarball unpacked somewhere, conda, a manual install).
    which = shutil.which(name)
    if which:
        out.append(Path(which))
    # A source checkout, the historical location.
    if repo_root:
        out.append(Path(repo_root) / "bin" / name)
    return out


def binary_path(compute_type: str = "CPU", do_forward: bool = False,
                repo_root: str | None = None) -> Path:
    """Path to the indexing binary. May not exist -- test with :func:`available`.

    Returns the first candidate that exists; if none do, returns the first
    candidate so the caller can name a concrete path in a diagnostic.
    """
    name = _executable_name(compute_type, do_forward)
    cands = _candidates(name, repo_root)
    for c in cands:
        if c.is_file():
            return c
    return cands[0]


def available(compute_type: str = "CPU", do_forward: bool = False,
              repo_root: str | None = None) -> bool:
    """True if the binary this request needs is present and executable."""
    p = binary_path(compute_type, do_forward, repo_root)
    return p.is_file() and os.access(p, os.X_OK)


def _unavailable_message(compute_type: str, do_forward: bool,
                         repo_root: str | None) -> str:
    name = _executable_name(compute_type, do_forward)
    lines = [f"{name} not found. Looked in, in order:"]
    lines += [f"  - {c}" for c in _candidates(name, repo_root)]
    lines += [
        "",
        "The C indexer is compiled at `pip install` time. If your machine had no",
        "compiler then, reinstall after installing one, or point at a binary you",
        f"already have with {BINARY_ENV}=/path/to/{name} (or to its directory).",
        "From a source checkout, ./build.sh writes it to bin/.",
    ]
    if name.startswith("LaueMatchingGPU"):
        lines.append(
            "The CUDA binaries are opt-in: LAUEMATCHING_CUDA=1 pip install laue-index "
            "(needs nvcc), or download them from a GitHub release.")
    return "\n".join(lines)


def require_binary(compute_type: str = "CPU", do_forward: bool = False,
                   repo_root: str | None = None) -> Path:
    """The binary's path, or :class:`BinaryUnavailableError` explaining why not.

    For callers that cannot proceed without it -- the streaming daemon, above
    all -- so every one of them reports the same diagnosis (every path tried,
    plus the escape hatch) instead of inventing its own half of the story.
    """
    if not available(compute_type, do_forward, repo_root):
        raise BinaryUnavailableError(
            _unavailable_message(compute_type, do_forward, repo_root))
    return binary_path(compute_type, do_forward, repo_root)


def resolve_executable(repo_root: str | None = None, compute_type: str = "CPU",
                       do_forward: bool = False) -> str:
    """Back-compatible shim: the path as a string.

    Kept because `scripts/RunImage.py` calls it positionally with a repo root.
    Prefer :func:`binary_path` / :func:`available` in new code.
    """
    return str(binary_path(compute_type, do_forward, repo_root))


def _build_env(repo_root: str | None) -> dict:
    """The binary has no shared-library dependencies any more.

    It used to link NLopt from LIBS/NLOPT/{lib,lib64}, which had to be put on
    LD_LIBRARY_PATH here. NLopt was dropped (the simplex is vendored into the C),
    so there is nothing to add -- but the LIBS paths are still prepended when a
    repo root is given, harmlessly, in case an old tree's binary is being run.
    """
    env = dict(os.environ)
    if repo_root:
        lib = os.path.join(repo_root, "LIBS", "NLOPT", "lib")
        lib64 = os.path.join(repo_root, "LIBS", "NLOPT", "lib64")
        if os.path.isdir(lib) or os.path.isdir(lib64):
            env["LD_LIBRARY_PATH"] = f"{lib}:{lib64}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_indexer(*, repo_root: str, config_file: str, orient_db_file: str,
                hkl_file: str, image_bin: str, ncpus: int, output_path: str,
                compute_type: str = "CPU", do_forward: bool = False) -> IndexerResult:
    """Run the indexing binary on already-prepared inputs.

    stdout/stderr are written to ``<output_path>.LaueMatching_std{out,err}.txt``.
    """
    if not available(compute_type, do_forward, repo_root):
        msg = _unavailable_message(compute_type, do_forward, repo_root)
        for line in msg.splitlines():
            logger.error(line)
        return IndexerResult(success=False, error=msg)
    executable = str(binary_path(compute_type, do_forward, repo_root))

    cmd = [executable, config_file, orient_db_file, hkl_file, image_bin, str(ncpus)]
    logger.info(f'Running indexing command: {" ".join(cmd)}')
    stdout_log = f"{output_path}.LaueMatching_stdout.txt"
    stderr_log = f"{output_path}.LaueMatching_stderr.txt"

    try:
        process = subprocess.run(cmd, env=_build_env(repo_root),
                                 capture_output=True, text=True, check=False)
        with open(stdout_log, "w") as f:
            f.write(process.stdout)
        with open(stderr_log, "w") as f:
            f.write(process.stderr)

        if process.returncode == 0:
            logger.info(f"Indexing command completed successfully (exit code 0). Output saved to {stdout_log}")
            return IndexerResult(success=True, returncode=0,
                                 stdout_log=stdout_log, stderr_log=stderr_log)
        logger.error(f"Indexing command failed with exit code {process.returncode}.")
        logger.error(f"Check logs for details: {stdout_log} and {stderr_log}")
        logger.error(f"Stderr tail:\n{process.stderr[-500:]}")
        return IndexerResult(success=False, returncode=process.returncode,
                             stdout_log=stdout_log, stderr_log=stderr_log,
                             error=f"Indexing command failed with code {process.returncode}")
    except FileNotFoundError:
        logger.error(f"Executable not found at {executable} when trying to run.")
        return IndexerResult(success=False, error="Indexing executable not found during execution")
    except Exception as e:  # noqa: BLE001 — preserve legacy catch-all behaviour
        logger.error(f"An unexpected error occurred while running indexing: {str(e)}")
        return IndexerResult(success=False, error=f"Unexpected error during indexing execution: {e}")
