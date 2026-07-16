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

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger("LaueMatching")

__all__ = ["IndexerResult", "resolve_executable", "run_indexer"]


@dataclass
class IndexerResult:
    success: bool
    returncode: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    error: str | None = None


def resolve_executable(repo_root: str, compute_type: str = "CPU",
                       do_forward: bool = False) -> str:
    """Pick the indexing executable path (<repo_root>/bin/<name>)."""
    compute_type = compute_type.upper()
    if compute_type == "GPU" and not do_forward:
        name = "LaueMatchingGPU"
    else:
        name = "LaueMatchingCPU"
        if compute_type == "GPU" and do_forward:
            logger.warning("GPU requested but DoFwd is enabled. Using CPU implementation (LaueMatchingCPU).")
        elif compute_type != "CPU":
            logger.warning(f"Processing type '{compute_type}' not recognized or incompatible. Using CPU implementation.")
    return os.path.join(repo_root, "bin", name)


def _build_env(repo_root: str) -> dict:
    env = dict(os.environ)
    lib = os.path.join(repo_root, "LIBS", "NLOPT", "lib")
    lib64 = os.path.join(repo_root, "LIBS", "NLOPT", "lib64")
    env["LD_LIBRARY_PATH"] = f"{lib}:{lib64}:{env.get('LD_LIBRARY_PATH', '')}"
    return env


def run_indexer(*, repo_root: str, config_file: str, orient_db_file: str,
                hkl_file: str, image_bin: str, ncpus: int, output_path: str,
                compute_type: str = "CPU", do_forward: bool = False) -> IndexerResult:
    """Run the indexing binary on already-prepared inputs.

    stdout/stderr are written to ``<output_path>.LaueMatching_std{out,err}.txt``.
    """
    executable = resolve_executable(repo_root, compute_type, do_forward)
    if not os.path.exists(executable):
        logger.error(f"Indexing executable not found at: {executable}")
        logger.error("Please ensure the code is compiled (e.g., run 'make' in the build directory).")
        return IndexerResult(success=False, error="Indexing executable not found")

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
