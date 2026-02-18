#!/usr/bin/env python
"""
laue_orchestrator.py — Top-level orchestrator for LaueMatching streaming pipeline

Launches the LaueMatchingGPUStream daemon, starts the image server, monitors
progress, and runs post-processing.  Analogous to integrator_batch_process.py.

Workflow:
    1. Create timestamped output directory
    2. Start LaueMatchingGPUStream as subprocess (log captured)
    3. Wait for port 60517 to become ready
    4. Start laue_image_server.py as subprocess
    5. Monitor frame_mapping.json for progress
    6. Wait for server to finish
    7. Allow daemon flush time, then terminate daemon (SIGTERM)
    8. Run laue_postprocess.py
    9. Print summary

Usage:
    python laue_orchestrator.py \
        --config params.txt \
        --folder /path/to/h5s \
        [--h5-location /entry/data/data] \
        [--ncpus 8] \
        [--output-dir auto]
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import laue_stream_utils as lsu


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("laue_orchestrator")


def _setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_daemon_binary() -> str:
    """Locate the LaueMatchingGPUStream binary."""
    # scripts/ lives under the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    candidates = [
        os.path.join(project_root, "build", "LaueMatchingGPUStream"),
        os.path.join(project_root, "bin", "LaueMatchingGPUStream"),
        os.path.join(project_root, "LaueMatchingGPUStream"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    # Fallback: check PATH
    import shutil
    found = shutil.which("LaueMatchingGPUStream")
    if found:
        return found

    raise FileNotFoundError(
        "LaueMatchingGPUStream binary not found. "
        "Build it first (cmake --build build/) or add it to PATH."
    )


def _terminate_process(proc: subprocess.Popen, name: str, timeout: float = 10.0) -> None:
    """Send SIGTERM, wait, then SIGKILL if necessary."""
    if proc.poll() is not None:
        return  # Already exited

    logger.info(f"Sending SIGTERM to {name} (pid {proc.pid})...")
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=timeout)
        logger.info(f"{name} exited (code {proc.returncode})")
        return
    except subprocess.TimeoutExpired:
        logger.warning(f"{name} did not exit in {timeout}s after SIGTERM, sending SIGKILL...")
    except Exception as e:
        logger.error(f"Error sending SIGTERM to {name}: {e}")

    # SIGKILL fallback
    try:
        proc.kill()
        proc.wait(timeout=15)
        logger.info(f"{name} killed (code {proc.returncode}).")
    except subprocess.TimeoutExpired:
        logger.error(f"{name} (pid {proc.pid}) did not exit even after SIGKILL. "
                     "It may need to be killed manually.")
    except Exception as e:
        logger.error(f"Error killing {name}: {e}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    config_file: str,
    folder: str,
    orient_file: str = "",
    hkl_file: str = "",
    h5_location: str = "/entry/data/data",
    ncpus: int = 1,
    output_dir: str = "",
    port: int = lsu.LAUE_STREAM_PORT,
    port_timeout: float = 180.0,
    flush_time: float = 5.0,
    min_unique: int = 2,
) -> None:
    """
    Run the full LaueMatching streaming pipeline.

    Args:
        config_file:  Path to params.txt.
        folder:       Folder with H5 image files.
        orient_file:  Path to orientation database (.bin). Resolved from
                      CWD if relative.  Looked up in config if empty.
        hkl_file:     Path to HKL file (.csv/.bin). Resolved from CWD if
                      relative.  Looked up in config if empty.
        h5_location:  Internal H5 dataset path.
        ncpus:        Number of CPUs (passed to daemon).
        output_dir:   Output directory (auto-generated if empty).
        port:         Daemon TCP port.
        port_timeout: Max seconds to wait for daemon port.
        flush_time:   Seconds to wait after server finishes before killing daemon.
        min_unique:   Minimum unique spots for orientation filtering.
    """
    t_pipeline_start = time.time()

    # Resolve to absolute paths so they remain valid when the daemon
    # subprocess runs with cwd=output_dir.
    config_file = os.path.abspath(config_file)
    folder = os.path.abspath(folder)

    # Resolve orient / HKL files — fall back to defaults read from config.
    if not orient_file or not hkl_file:
        try:
            import laue_config
            cfg_mgr = laue_config.ConfigurationManager(config_file)
            if not orient_file:
                orient_file = cfg_mgr.get("orientation_file", "orientations.bin")
            if not hkl_file:
                hkl_file = cfg_mgr.get("hkl_file", "hkls.bin")
        except Exception as exc:
            logger.warning(f"Could not read config to resolve orient/hkl files: {exc}")
            if not orient_file:
                orient_file = "orientations.bin"
            if not hkl_file:
                hkl_file = "hkls.bin"
    orient_file = os.path.abspath(orient_file)
    hkl_file = os.path.abspath(hkl_file)
    logger.info(f"Orientation DB : {orient_file}")
    logger.info(f"HKL file       : {hkl_file}")

    # Read the daemon's ResultDir from config (the subdirectory where it
    # writes solutions.txt / spots.txt inside its CWD).
    daemon_result_dir = "results_stream"  # C-code default
    try:
        import laue_config
        cfg_mgr = laue_config.ConfigurationManager(config_file)
        daemon_result_dir = getattr(cfg_mgr.config, "result_dir", daemon_result_dir)
    except Exception:
        pass
    logger.info(f"Daemon ResultDir: {daemon_result_dir}")

    # --- 1. Create output directory ---
    if not output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"laue_stream_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # Paths inside output dir.
    # The daemon writes solutions/spots to <CWD>/<ResultDir>/.
    daemon_log = os.path.join(output_dir, "daemon.log")
    server_log = os.path.join(output_dir, "server.log")
    daemon_out_dir = os.path.join(output_dir, daemon_result_dir)
    solutions_file = os.path.join(daemon_out_dir, "solutions.txt")
    spots_file = os.path.join(daemon_out_dir, "spots.txt")
    mapping_file = os.path.join(output_dir, "frame_mapping.json")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Daemon output  : {daemon_out_dir}")

    # --- 2. Start GPU daemon ---
    daemon_bin = _find_daemon_binary()
    daemon_cmd = [
        daemon_bin,
        config_file,
        orient_file,
        hkl_file,
        str(ncpus),
    ]
    logger.info(f"Starting daemon: {' '.join(daemon_cmd)}")

    daemon_logf = open(daemon_log, "w")
    daemon_proc = subprocess.Popen(
        daemon_cmd,
        stdout=daemon_logf,
        stderr=subprocess.STDOUT,
        cwd=output_dir,
    )
    logger.info(f"Daemon started (pid {daemon_proc.pid}), log → {daemon_log}")

    # --- 3. Wait for daemon port ---
    logger.info(f"Waiting for port {port}...")
    if not lsu.wait_for_port("127.0.0.1", port, timeout=port_timeout):
        logger.error("Daemon did not open port in time. Check daemon log.")
        _terminate_process(daemon_proc, "daemon")
        daemon_logf.close()
        _print_log_tail(daemon_log)
        sys.exit(1)

    # --- 4. Start image server ---
    python = sys.executable
    server_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "laue_image_server.py",
    )
    server_cmd = [
        python, server_script,
        "--config", os.path.abspath(config_file),
        "--folder", os.path.abspath(folder),
        "--h5-location", h5_location,
        "--mapping-file", os.path.abspath(mapping_file),
        "--port", str(port),
        "--log-level", "INFO",
    ]
    logger.info(f"Starting image server...")

    server_logf = open(server_log, "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_logf,
        stderr=subprocess.STDOUT,
        cwd=output_dir,
    )
    logger.info(f"Image server started (pid {server_proc.pid}), log → {server_log}")

    # Count total frames for progress bar
    import glob
    total_frames = len(glob.glob(os.path.join(folder, "*.h5"))) + \
                   len(glob.glob(os.path.join(folder, "*.hdf5")))
    logger.info(f"Total frames to process: {total_frames}")

    # --- 5. Monitor progress ---
    try:
        _monitor(server_proc, daemon_proc, mapping_file, daemon_log,
                 total_frames=total_frames)
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        _terminate_process(server_proc, "image server")
        _terminate_process(daemon_proc, "daemon")
        daemon_logf.close()
        server_logf.close()
        sys.exit(130)

    # --- 6. Server finished — wait for daemon to flush output ---
    logger.info(f"Image server exited (code {server_proc.returncode}). "
                f"Waiting for daemon to write results...")

    # Poll for solutions.txt (mirrors integrator_batch_process.py pattern)
    flush_deadline = time.time() + flush_time + 60
    while time.time() < flush_deadline:
        if os.path.isfile(solutions_file) and os.path.getsize(solutions_file) > 0:
            logger.info(f"solutions.txt detected ({os.path.getsize(solutions_file)} bytes)")
            time.sleep(2.0)  # extra wait for file to be fully flushed
            break
        if daemon_proc.poll() is not None:
            logger.info("Daemon exited on its own.")
            break
        time.sleep(1.0)
    else:
        logger.warning("Timed out waiting for daemon to write solutions.txt")

    # --- 7. Terminate daemon ---
    _terminate_process(daemon_proc, "daemon")
    daemon_logf.close()
    server_logf.close()

    # --- 8. Post-processing ---
    logger.info("Starting post-processing...")
    postprocess_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "laue_postprocess.py",
    )

    if not os.path.isfile(solutions_file):
        logger.error(f"solutions.txt not found at {solutions_file}. Check daemon log.")
        _print_log_tail(daemon_log)
        sys.exit(1)
    if not os.path.isfile(spots_file):
        logger.error(f"spots.txt not found at {spots_file}. Check daemon log.")
        _print_log_tail(daemon_log)
        sys.exit(1)

    pp_cmd = [
        python, postprocess_script,
        "--solutions", solutions_file,
        "--spots", spots_file,
        "--config", os.path.abspath(config_file),
        "--output-dir", results_dir,
        "--mapping", os.path.abspath(mapping_file),
        "--min-unique", str(min_unique),
        "--nprocs", str(ncpus),
    ]
    logger.info(f"Running: {' '.join(os.path.basename(c) for c in pp_cmd)}")
    pp_result = subprocess.run(pp_cmd, capture_output=True, text=True)

    if pp_result.returncode != 0:
        logger.error(f"Post-processing failed (code {pp_result.returncode})")
        if pp_result.stderr:
            logger.error(pp_result.stderr[-2000:])
    else:
        logger.info("Post-processing complete.")

    # --- 9. Summary ---
    elapsed = time.time() - t_pipeline_start
    logger.info("=" * 60)
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Output directory:  {output_dir}")
    logger.info(f"  Daemon log:        {daemon_log}")
    logger.info(f"  Server log:        {server_log}")
    logger.info(f"  Frame mapping:     {mapping_file}")
    logger.info(f"  Results:           {results_dir}/")

    # List result files
    for f in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, f)
        sz = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        logger.info(f"    {f}  ({sz:,} bytes)")
    logger.info("=" * 60)


def _monitor(
    server_proc: subprocess.Popen,
    daemon_proc: subprocess.Popen,
    mapping_file: str,
    daemon_log: str,
    total_frames: int = 0,
    poll_interval: float = 1.0,
) -> None:
    """Monitor server progress and daemon health until server exits."""
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    last_count = 0

    if has_tqdm and total_frames > 0:
        pbar = tqdm(
            total=total_frames,
            desc="Streaming",
            unit="img",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
        )
    else:
        pbar = None

    try:
        while server_proc.poll() is None:
            # Check daemon is still running
            if daemon_proc.poll() is not None:
                logger.error(
                    f"Daemon exited unexpectedly (code {daemon_proc.returncode}). "
                    f"Aborting."
                )
                _print_log_tail(daemon_log)
                _terminate_process(server_proc, "image server")
                raise RuntimeError("Daemon died")

            # Read mapping for progress
            mapping = lsu.load_frame_mapping(mapping_file)
            count = len(mapping)
            if count > last_count:
                delta = count - last_count
                if pbar is not None:
                    pbar.update(delta)
                else:
                    sent = sum(1 for v in mapping.values()
                               if not v.get("skipped", False))
                    skipped = count - sent
                    logger.info(
                        f"Progress: {count}/{total_frames} frames "
                        f"({sent} sent, {skipped} skipped)"
                    )
                last_count = count

            time.sleep(poll_interval)

        # Final update — pick up any frames written after last poll
        mapping = lsu.load_frame_mapping(mapping_file)
        count = len(mapping)
        if count > last_count and pbar is not None:
            pbar.update(count - last_count)
    finally:
        if pbar is not None:
            pbar.close()


def _print_log_tail(log_path: str, n: int = 2000) -> None:
    """Print the last n characters of a log file."""
    if not os.path.exists(log_path):
        return
    try:
        with open(log_path) as f:
            content = f.read()
        tail = content[-n:] if len(content) > n else content
        if tail.strip():
            logger.info(f"--- Tail of {os.path.basename(log_path)} ---")
            for line in tail.strip().split("\n"):
                logger.info(f"  {line}")
            logger.info("--- End ---")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate LaueMatching streaming pipeline"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to params.txt configuration file"
    )
    parser.add_argument(
        "--folder", required=True,
        help="Folder containing H5 image files"
    )
    parser.add_argument(
        "--h5-location", default="/entry/data/data",
        help="HDF5 internal dataset path (default: /entry/data/data)"
    )
    parser.add_argument(
        "--ncpus", type=int, default=1,
        help="Number of CPUs for daemon (default: 1)"
    )
    parser.add_argument(
        "--output-dir", default="",
        help="Output directory (default: auto-timestamped)"
    )
    parser.add_argument(
        "--port", type=int, default=lsu.LAUE_STREAM_PORT,
        help=f"Daemon TCP port (default: {lsu.LAUE_STREAM_PORT})"
    )
    parser.add_argument(
        "--port-timeout", type=float, default=180.0,
        help="Max seconds to wait for daemon port (default: 180)"
    )
    parser.add_argument(
        "--flush-time", type=float, default=5.0,
        help="Seconds to wait after server finishes before killing daemon (default: 5)"
    )
    parser.add_argument(
        "--min-unique", type=int, default=2,
        help="Minimum unique spots for orientation filtering (default: 2)"
    )
    parser.add_argument(
        "--orient-file", default="",
        help="Path to orientation database file (default: from config or orientations.bin)"
    )
    parser.add_argument(
        "--hkl-file", default="",
        help="Path to HKL file (default: from config or hkls.bin)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    # Validate inputs
    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        sys.exit(1)

    run_pipeline(
        config_file=args.config,
        folder=args.folder,
        orient_file=args.orient_file,
        hkl_file=args.hkl_file,
        h5_location=args.h5_location,
        ncpus=args.ncpus,
        output_dir=args.output_dir,
        port=args.port,
        port_timeout=args.port_timeout,
        flush_time=args.flush_time,
        min_unique=args.min_unique,
    )


if __name__ == "__main__":
    main()
