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
    7. Wait until the daemon has reported every sent frame, then SIGTERM it
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
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional

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
    """Locate the LaueMatchingGPUStream binary.

    Discovery is delegated to laue_index.indexer, which is the single place
    that knows every location a binary can live -- including two this function
    used to miss entirely: LAUEMATCHING_BIN, and the site-packages bin/ that
    `LAUEMATCHING_CUDA=1 pip install laue-index` now populates. A pip user with
    the daemon installed was told to "build it first".

    The two checkout-only locations below are kept as a fallback for trees
    built in place with `cmake --build build/`, which the package cannot know
    about.
    """
    from laue_index.pipeline import repo_root
    root = repo_root()
    project_root = str(root) if root is not None else None

    primary_error = None
    try:
        from laue_index import indexer as _ix
    except ImportError:
        pass
    else:
        try:
            return str(_ix.require_binary("STREAM", repo_root=project_root))
        except _ix.BinaryUnavailableError as exc:
            primary_error = exc

    for c in ((os.path.join(project_root, "build", "LaueMatchingGPUStream"),
               os.path.join(project_root, "LaueMatchingGPUStream"))
              if project_root else ()):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        str(primary_error) if primary_error else
        "LaueMatchingGPUStream binary not found. Build it first "
        "(cmake --build build/) or add it to PATH."
    )


def _server_env() -> dict:
    """Environment for the image server: the caller's, unchanged.

    Deliberately does NOT set LAUE_DAEMON_NCPUS from --ncpus. Doing so (as an
    earlier 0.7.2 draft did) took the daemon's cores out of the preprocessing
    pool on every orchestrated run: run_laue.sh defaults NCPUS=32, which left 1
    worker on a 32-CPU host. The thread limit, not core contention, was the
    failure being fixed. A caller who wants the subtraction sets the variable
    itself, and it is passed through untouched.
    """
    return os.environ.copy()


def _wait_for_daemon_port(
    proc: subprocess.Popen,
    port: int,
    timeout: float,
    poll_interval: float = 2.0,
    progress_every: float = 30.0,
) -> bool:
    """Wait for the daemon to open *port*, watching the process while we wait.

    ``lsu.wait_for_port()`` is process-blind, which fails in both directions:

    * A daemon that dies immediately (bad params, missing file, GPU error) still
      holds the orchestrator for the whole timeout before it reports anything.
    * A daemon that is merely *slow* gets killed. Startup reads a multi-GB
      orientation database and then initialises a CUDA context; on a cold page
      cache, a busy GPU, or a loaded machine that legitimately exceeds a short
      fixed budget, and the run aborts with "Daemon did not open port in time"
      even though nothing is wrong.

    Polling ``proc`` as well as the port fixes both: we fail fast with the exit
    code when the daemon is genuinely dead, and we keep waiting (logging
    progress) as long as it is alive.

    Returns True once the port is open, False if the daemon died or the timeout
    elapsed while it was still running.
    """
    t0 = time.time()
    last_note = t0
    while True:
        if lsu.is_port_open("127.0.0.1", port):
            logger.info(f"Port {port} ready ({time.time() - t0:.1f}s)")
            return True

        rc = proc.poll()
        if rc is not None:
            logger.error(
                f"Daemon exited (code {rc}) after {time.time() - t0:.1f}s "
                f"without opening port {port}. Check daemon log."
            )
            return False

        now = time.time()
        if now - t0 >= timeout:
            logger.error(
                f"Daemon is still running but has not opened port {port} after "
                f"{timeout:.0f}s; giving up. If startup is legitimately this slow "
                f"(very large orientation database, contended GPU), raise "
                f"--port-timeout."
            )
            return False

        if now - last_note >= progress_every:
            logger.info(
                f"  still waiting for port {port} "
                f"({now - t0:.0f}s elapsed, daemon alive)..."
            )
            last_note = now

        time.sleep(poll_interval)


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


def _ensure_shm_files(orient_file: str) -> None:
    """Copy orientation database to /dev/shm if the path points there.

    When the resolved *orient_file* lives under ``/dev/shm`` and does not
    yet exist, this helper copies it from the LaueMatching checkout root
    (``<repo_root()>/<basename>``).
    """
    if not orient_file.startswith("/dev/shm/"):
        return  # not a shared-memory path, nothing to do

    if os.path.isfile(orient_file):
        sz = os.path.getsize(orient_file)
        logger.info(
            f"SHM file already present: {orient_file} ({sz / 1e9:.2f} GB) — skipping copy"
        )
        return

    # Source: <checkout root>/<basename>
    from laue_index.pipeline import repo_root
    basename = os.path.basename(orient_file)
    root = repo_root()
    if root is None:
        logger.error(
            f"Cannot copy to {orient_file}: no source checkout to copy it from. "
            "Put the file there yourself (`laue-index fetch-db` downloads the "
            "orientation database).")
        sys.exit(1)
    source = os.path.join(str(root), basename)

    if not os.path.isfile(source):
        logger.error(
            f"Cannot copy to {orient_file}: source file not found at {source}"
        )
        sys.exit(1)

    src_size = os.path.getsize(source)
    logger.info(
        f"Copying {source} → {orient_file} ({src_size / 1e9:.2f} GB) ..."
    )
    shutil.copy2(source, orient_file)
    logger.info("SHM copy complete.")


# The daemon's ResultDir when the params file has no ResultDir line
# (``char resultDir[1000] = "results_stream"`` in LaueMatchingGPUStream.cu).
DAEMON_RESULT_DIR_DEFAULT = "results_stream"


def _daemon_result_dir(config_file: str) -> str:
    """ResultDir exactly as LaueMatchingGPUStream reads it.

    The daemon matches the WHOLE first token of each raw line (``paramKeyCmp``,
    0.8.0; a prefix match before) and takes the second ``%s`` token; a later line overrides an earlier one, and a
    line with no value leaves the default. This used to come from
    ConfigurationManager, whose default is RunImage's ``results``: a params file
    without the line sent the orchestrator to a directory the daemon never
    wrote, where it waited out the flush timeout and exited 1.
    """
    result_dir = DAEMON_RESULT_DIR_DEFAULT
    with open(config_file, errors="replace") as f:
        for line in f:
            parts = line.split()
            if line.startswith("ResultDir") and parts and parts[0] == "ResultDir":
                if len(parts) >= 2:
                    result_dir = parts[1]
    return result_dir


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _streaming_postprocess_settings(config_file: str, min_unique=None) -> dict:
    """What the STREAMING post-processor will actually apply, for provenance.

    Read with the streaming parser, as the post-processor reads it. Up to 0.7.2 an
    absent ``RobustFilter`` meant 1 in the ``ConfigurationManager`` snapshot and 0
    on the streaming path, which is why this block exists; since 0.8 the key is
    required, so the two agree, and the block stays as the explicit record.
    """
    import laue_stream_utils as _lsu
    import laue_postprocess as _pp
    cfg = _lsu.parse_config(config_file)
    rf = cfg.get("robust_filter")
    floor = int(min_unique) if min_unique is not None else int(cfg.get("min_good_spots", 2))
    return {
        "robust_filter_key_present": rf is not None,
        "robust_filter_effective": bool(_pp._robust_in_force(cfg)),
        "min_unique_effective": floor,
        "min_unique_source": "--min-unique" if min_unique is not None else "MinGoodSpots",
    }


def _bg_for_lineage(config_file, output_dir):
    """The BackgroundFile a run will read, resolved where the image server looks
    (its cwd is output_dir), if it already exists; else None."""
    try:
        from laue_index import artifacts as _art
        bg = _art.read_params(config_file).get("BackgroundFile")
    except Exception:
        return None
    if not bg or not isinstance(bg, str):
        return None
    path = bg if os.path.isabs(bg) else os.path.join(output_dir, bg)
    return path if os.path.exists(path) else None


def run_pipeline(
    config_file: str,
    folder: str,
    orient_file: str = "",
    hkl_file: str = "",
    h5_location: str = "/entry/data/data",
    ncpus: int = 1,
    output_dir: str = "",
    port: int = lsu.LAUE_STREAM_PORT,
    port_timeout: float = 900.0,
    flush_time: float = 5.0,
    min_unique: Optional[int] = None,
    write_indexfile: bool = True,
    indexfile_dir: str = "",
    watch: bool = False,
    watch_poll: float = 2.0,
    watch_idle: float = 0.0,
    drain_stall: float = 600.0,
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
        port_timeout: Max seconds to wait for the daemon port while the
                      daemon is still alive (a dead daemon aborts immediately).
        flush_time:   Extra seconds allowed (on top of one hour) for the daemon
                      to finish the frames it was sent before it is stopped.
        drain_stall:  Give up waiting for the daemon's per-frame reports after
                      this many seconds without a new one (see
                      _wait_for_daemon_drain).
        min_unique:   Minimum EXCLUSIVE (winner-take-all) spots for orientation
                      filtering. None (default) lets postprocess use MinGoodSpots
                      from the config, as the non-streaming path does.
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

    # Copy orientation file to /dev/shm if the path points there.
    _ensure_shm_files(orient_file)

    # Read the daemon's ResultDir (as the daemon itself does) and ForwardFile.
    daemon_result_dir = _daemon_result_dir(config_file)
    forward_file = ""
    try:
        import laue_config
        cfg_mgr = laue_config.ConfigurationManager(config_file)
        forward_file = getattr(cfg_mgr.config, "forward_file", "")
    except Exception:
        pass
    logger.info(f"Daemon ResultDir: {daemon_result_dir}")

    # --- 1. Create output directory ---
    if not output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"laue_stream_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # The daemon and the image server both run with cwd=output_dir, so a
    # relative ForwardFile / BackgroundFile is looked up THERE. These checks
    # used to look in this process's cwd and so warned, or stayed silent, about
    # a different file from the one the run would use.
    def _as_run_sees_it(p):
        return p if os.path.isabs(p) else os.path.join(os.path.abspath(output_dir), p)

    # Warn if the forward simulation file does not exist yet.
    if forward_file and not os.path.isfile(_as_run_sees_it(forward_file)):
        logger.warning("=" * 60)
        logger.warning(
            f"Forward simulation file not found: {_as_run_sees_it(forward_file)}"
        )
        logger.warning(
            "The daemon will generate the forward simulation from scratch. "
            "This may take a considerable amount of time."
        )
        logger.warning("=" * 60)
    try:
        background_file = lsu.parse_config(config_file).get("background_file", "")
    except Exception:
        background_file = ""
    if background_file and not os.path.isfile(_as_run_sees_it(background_file)):
        logger.warning(
            f"BackgroundFile not found: {_as_run_sees_it(background_file)} "
            "(relative paths are taken from the output directory). The image "
            "server will compute a background from the first frame.")

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

    # Resolve the daemon binary BEFORE stamping provenance, so the record names
    # the executable that actually runs. _find_daemon_binary can fall back to
    # <project_root>/build/, which is not the directory laue_provenance hashes
    # by default. A missing binary is still fatal -- just after the stamp, so a
    # failed launch leaves a record too.
    daemon_bin_error = None
    try:
        daemon_bin = _find_daemon_binary()
    except FileNotFoundError as exc:
        daemon_bin, daemon_bin_error = None, exc

    # --- 1a. Data artifacts against their provenance records ---
    # A record that disagrees with its file, or an HKL list made for another
    # crystal, stops the run here, before the daemon loads 19 GB; a file with
    # no record only warns. The identities go into provenance.json (lineage).
    from laue_index import artifacts as _art
    _fwd = forward_file
    if _fwd and not os.path.isabs(_fwd):
        _fwd = os.path.join(output_dir, _fwd)
    try:
        artifact_lineage = _art.check_run_inputs(
            config_file, orient_db=orient_file if os.path.exists(orient_file) else None,
            hkl=hkl_file if os.path.exists(hkl_file) else None, forward=_fwd or None,
            background=_bg_for_lineage(config_file, output_dir), log=logger)
    except _art.ArtifactMismatch as exc:
        logger.error(f"Refusing to run: {exc}")
        sys.exit(1)

    # --- 1b. Stamp run-level provenance up-front ---
    # Written now (rather than at end-of-run) so a crashed/killed run still
    # leaves a record of which commit + config was in play.
    try:
        import laue_provenance as _lp
        run_prov = _lp.collect(
            config=getattr(cfg_mgr, "config", None),
            input_files=[f for f in (config_file, orient_file, hkl_file) if f],
            extra={
                "output_dir": output_dir,
                "folder": folder,
                "ncpus": ncpus,
                "port": port,
                "daemon_bin": daemon_bin or f"NOT FOUND: {daemon_bin_error}",
            },
            executable=daemon_bin,
        )
        try:
            spp = _streaming_postprocess_settings(config_file, min_unique)
            run_prov.setdefault("extra", {})["streaming_postprocess"] = spp
            run_prov.setdefault("config_notes", {})["robust_filter"] = (
                "the filter this streaming run applies is "
                "extra.streaming_postprocess.robust_filter_effective (RobustFilter is a "
                "required key since 0.8, so it equals config.robust_filter).")
        except Exception as spp_exc:          # never fail a run over a provenance note
            run_prov.setdefault("extra", {})["streaming_postprocess"] = f"ERROR: {spp_exc}"
        run_prov["artifacts"] = artifact_lineage
        _lp.write_sidecar_json(os.path.join(output_dir, "provenance.json"), run_prov)
        logger.info(f"Wrote run provenance: {os.path.join(output_dir, 'provenance.json')}")
    except Exception as prov_exc:
        logger.warning(f"Could not write run provenance: {prov_exc}")

    # --- 2. Start GPU daemon ---
    if daemon_bin is None:
        raise daemon_bin_error
    daemon_cmd = [
        daemon_bin,
        config_file,
        orient_file,
        hkl_file,
        str(ncpus),
    ]
    logger.info(f"Starting daemon: {' '.join(daemon_cmd)}")

    daemon_logf = open(daemon_log, "w")
    daemon_env = os.environ.copy()
    daemon_env["LAUE_STREAM_PORT"] = str(port)  # daemon reads this (default 60517)
    daemon_proc = subprocess.Popen(
        daemon_cmd,
        stdout=daemon_logf,
        stderr=subprocess.STDOUT,
        cwd=output_dir,
        env=daemon_env,
    )
    logger.info(f"Daemon started (pid {daemon_proc.pid}), log → {daemon_log}")

    # --- 3. Wait for daemon port ---
    logger.info(f"Waiting for port {port}...")
    if not _wait_for_daemon_port(daemon_proc, port, port_timeout):
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
    labels_file = os.path.join(output_dir, "labels.h5")
    server_cmd = [
        python, server_script,
        "--config", os.path.abspath(config_file),
        "--folder", os.path.abspath(folder),
        "--h5-location", h5_location,
        "--mapping-file", os.path.abspath(mapping_file),
        "--labels-file", os.path.abspath(labels_file),
        "--port", str(port),
        "--log-level", "INFO",
    ]
    if watch:
        server_cmd += ["--watch", "--watch-poll", str(watch_poll)]
        if watch_idle > 0:
            server_cmd += ["--watch-idle", str(watch_idle)]
    logger.info(f"Starting image server...")

    server_logf = open(server_log, "w")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=server_logf,
        stderr=subprocess.STDOUT,
        cwd=output_dir,
        env=_server_env(),
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

    # Wait until the daemon has REPORTED every frame the server sent.
    #
    # This used to wait for solutions.txt to stop growing (which itself replaced
    # "the file exists", after a 6561-frame batch lost its last 31 frames). But a
    # frame with no solution never grows that file, so a tail of such frames, or
    # one slow frame, read as "drained" and the SIGTERM discarded whatever was
    # still queued. The daemon prints one terminal line per frame it finishes
    # (see _DaemonProgress); the server's final frame_mapping.json says which
    # frames were sent. Done = every sent frame has its line.
    _wait_for_daemon_drain(daemon_proc, daemon_log, mapping_file,
                           deadline=flush_time + 3600, stall=drain_stall)

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
        "--labels", os.path.abspath(labels_file),
        "--folder", os.path.abspath(folder),
        "--nprocs", str(ncpus),
    ]
    if min_unique is not None:
        pp_cmd += ["--min-unique", str(min_unique)]
    if not write_indexfile:
        pp_cmd.append("--no-indexfile")
    elif indexfile_dir:
        pp_cmd.extend(["--indexfile-out", indexfile_dir])
    logger.info(f"Running: {' '.join(os.path.basename(c) for c in pp_cmd)}")
    pp_result = subprocess.run(pp_cmd, capture_output=True, text=True)
    # Keep the post-processor's own output: it was captured and dropped on success,
    # so its startup warnings reached no log.
    pp_log = os.path.join(output_dir, "postprocess.log")
    try:
        with open(pp_log, "w") as _f:
            _f.write(f"# {' '.join(pp_cmd)}\n# exit {pp_result.returncode}\n")
            _f.write("## stdout\n" + (pp_result.stdout or "") + "\n## stderr\n" + (pp_result.stderr or ""))
        logger.info(f"Post-processing output: {pp_log}")
    except OSError as exc:
        logger.warning(f"Could not write {pp_log}: {exc}")

    if pp_result.returncode != 0:
        # Fail the run. This used to log the error and then "Pipeline complete"
        # and exit 0, and pipeline/dispatch/wait_static.sh keys on that line --
        # so a run with no per-image results read as finished.
        logger.error(f"Post-processing failed (code {pp_result.returncode})")
        if pp_result.stderr:
            logger.error(pp_result.stderr[-2000:])
        logger.error("Pipeline FAILED at post-processing after "
                     f"{time.time() - t_pipeline_start:.1f}s; daemon output is in "
                     f"{daemon_out_dir}, results (incomplete) in {results_dir}")
        sys.exit(pp_result.returncode if pp_result.returncode > 0 else 1)
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

    # Summarise result files
    result_files = sorted(os.listdir(results_dir))
    total_sz = sum(
        os.path.getsize(os.path.join(results_dir, f))
        for f in result_files
        if os.path.isfile(os.path.join(results_dir, f))
    )
    logger.info(f"  Result files:      {len(result_files)} files, {total_sz / 1e6:.1f} MB total")
    logger.info("=" * 60)


class _DaemonProgress:
    """Frames LaueMatchingGPUStream has finished, read from its log.

    finalize_stream() in LaueMatchingGPUStream.cu ends every frame with exactly
    one of::

        [Image %u] No matches, skipping fitting.
        [Image %u] Total: %.3f s (GPU: ...)

    The log is read incrementally; a partial last line is kept for the next read.
    """
    _TERMINAL = re.compile(r"^\[Image (\d+)\] (?:Total:|No matches, skipping fitting\.)")

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.done: set = set()
        self._offset = 0
        self._partial = ""

    def update(self) -> set:
        try:
            with open(self.log_path, errors="replace") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except OSError:
            return self.done
        lines = (self._partial + chunk).split("\n")
        self._partial = lines.pop()
        for line in lines:
            m = self._TERMINAL.match(line)
            if m:
                self.done.add(int(m.group(1)))
        return self.done


def _wait_for_daemon_drain(daemon_proc, daemon_log: str, mapping_file: str,
                           deadline: float = 3600.0, stall: float = 600.0,
                           poll: float = 1.0) -> bool:
    """Block until the daemon has reported every non-skipped frame in the mapping.

    Returns True when it has (or the daemon exited on its own). Returns False,
    with a warning naming the frames not yet reported, after *deadline* seconds,
    or after *stall* seconds in which no new frame was reported: the daemon
    prints its "No matches" line without flushing stdout, so a trailing run of
    such frames can sit in its buffer until it exits.
    """
    sent = {int(k) for k, v in lsu.load_frame_mapping(mapping_file).items()
            if not v.get("skipped", False)}
    prog = _DaemonProgress(daemon_log)
    t0 = last_new = time.time()
    n_done = -1
    while True:
        if daemon_proc.poll() is not None:
            logger.info("Daemon exited on its own.")
            return True
        done = prog.update()
        missing = sent - done
        if not missing:
            logger.info(f"Daemon reported all {len(sent)} sent frame(s); drained")
            return True
        now = time.time()
        if len(done) != n_done:
            n_done, last_new = len(done), now
        if now - t0 >= deadline or now - last_new >= stall:
            head = sorted(missing)[:20]
            logger.warning(
                f"Daemon has not reported {len(missing)} of {len(sent)} sent "
                f"frame(s) (e.g. {head}) after {now - t0:.0f}s, {now - last_new:.0f}s "
                "since its last report; stopping it anyway. Results "
                "may be truncated (unless those frames had no matches, whose "
                "log line the daemon may not have flushed yet).")
            return False
        time.sleep(poll)


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
        "--port-timeout", type=float, default=900.0,
        help="Max seconds to wait for daemon port (default: 900)"
    )
    parser.add_argument(
        "--flush-time", type=float, default=5.0,
        help="Extra seconds (on top of one hour) the daemon may take to finish "
             "the frames it was sent before it is stopped (default: 5)"
    )
    parser.add_argument(
        "--drain-stall", type=float, default=600.0,
        help="After the image server finishes, stop waiting for the daemon when "
             "it has reported no new frame for this many seconds (default: 600)"
    )
    parser.add_argument(
        "--min-unique", type=int, default=None,
        help="Minimum exclusive (winner-take-all) spots for orientation "
             "filtering. Default: MinGoodSpots from --config"
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
    parser.add_argument(
        "--no-indexfile", action="store_true",
        help="Disable the default Tischler-format .indexing.txt per-image output"
    )
    parser.add_argument(
        "--indexfile-out", default="",
        help="Directory for .indexing.txt files (default: alongside HDF5 outputs)"
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Real-time mode: image server keeps watching the folder for new "
             "files. Stop with a STOP_LAUE file in the folder or --watch-idle."
    )
    parser.add_argument(
        "--watch-poll", type=float, default=2.0,
        help="Seconds between folder rescans in watch mode (default: 2)"
    )
    parser.add_argument(
        "--watch-idle", type=float, default=0.0,
        help="Watch mode exits after N seconds with no new files (default: 0 = never)"
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
        write_indexfile=not args.no_indexfile,
        indexfile_dir=args.indexfile_out,
        watch=args.watch,
        watch_poll=args.watch_poll,
        watch_idle=args.watch_idle,
        drain_stall=args.drain_stall,
    )


if __name__ == "__main__":
    main()
