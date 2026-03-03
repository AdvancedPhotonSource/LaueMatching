#!/usr/bin/env python
"""
laue_image_server.py — TCP image-sending client for LaueMatchingGPUStream

Reads HDF5 images from a folder, pre-processes each frame (background
subtraction → enhancement → thresholding → component filtering → Gaussian
blur), and sends processed images to the GPU daemon over TCP port 60517.

Wire protocol (must match LaueMatchingGPUStream.cu handle_client):
    | uint16_t image_num (2 bytes, LE) | float[NrPxX*NrPxY] pixel data |

Usage:
    python laue_image_server.py \
        --config params.txt \
        --folder /path/to/h5s \
        [--h5-location /entry/data/data] \
        [--mapping-file frame_mapping.json] \
        [--save-interval 50] \
        [--host 127.0.0.1] \
        [--port 60517]
"""

import argparse
import glob
import logging
import os
import queue
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, Future

import numpy as np

import laue_stream_utils as lsu

# Optional imports
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("laue_image_server")


def _setup_logging(level: str = "INFO") -> None:
    """Configure module-level logging to stderr."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Top-level worker for multiprocessing (must be picklable)
# ---------------------------------------------------------------------------

def _preprocess_one(h5_path, h5loc, frame_idx, nr_px_y, nr_px_x, cfg, background):
    """Load and preprocess a single frame. Runs in a worker process."""
    try:
        raw = lsu.load_h5_image(h5_path, h5loc, frame_index=frame_idx)
    except Exception as e:
        return {"error": f"load: {e}"}
    if raw.shape != (nr_px_y, nr_px_x):
        return {"error": f"shape {raw.shape} != ({nr_px_y},{nr_px_x})"}
    blurred, _filt_img, filt_labels, centers = lsu.preprocess_image(
        raw, cfg, background=background
    )
    if not centers:
        return {"error": "no_spots"}
    pixels_bytes = np.ascontiguousarray(blurred, dtype=np.float32).tobytes()
    return {"pixels_bytes": pixels_bytes, "filt_labels": filt_labels,
            "n_spots": len(centers)}


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def serve_images(
    config_file: str,
    folder: str,
    h5_location: str = "/entry/data/data",
    mapping_file: str = "frame_mapping.json",
    labels_file: str = "labels.h5",
    save_interval: int = 50,
    host: str = "127.0.0.1",
    port: int = lsu.LAUE_STREAM_PORT,
) -> dict:
    """
    Iterate over all H5 files in *folder*, preprocess each frame, and send
    it to the GPU daemon listening on *host:port*.

    Returns the final frame-mapping dict.
    """
    # --- 1. Load configuration ---
    cfg = lsu.parse_config(config_file)
    # CLI h5_location overrides config
    if h5_location:
        cfg["h5_location"] = h5_location
    h5loc = cfg["h5_location"]

    nr_px_x = cfg["nr_px_x"]
    nr_px_y = cfg["nr_px_y"]

    # --- 2. Discover H5 files ---
    h5_files = sorted(glob.glob(os.path.join(folder, "*.h5")))
    if not h5_files:
        # Also try .hdf5
        h5_files = sorted(glob.glob(os.path.join(folder, "*.hdf5")))
    if not h5_files:
        logger.error(f"No H5 files found in {folder}")
        return {}

    # Count total frames for progress
    total_frames = 0
    file_frame_counts = []
    for h5f in h5_files:
        n = lsu.count_h5_frames(h5f, h5loc)
        file_frame_counts.append(n)
        total_frames += n
    logger.info(f"Found {len(h5_files)} H5 file(s), {total_frames} total frame(s)")

    # Validate image number fits in uint16 (max 65535)
    if total_frames > 65535:
        logger.warning(
            f"Total frames ({total_frames}) exceeds uint16 max (65535). "
            f"Only first 65535 frames will be processed."
        )
        total_frames = min(total_frames, 65535)

    # --- 3. Pre-compute / load background ---
    # Use first frame to compute background if not already available
    background = lsu.load_background(
        cfg.get("background_file", ""), nr_px_x, nr_px_y
    )
    if np.count_nonzero(background) == 0:
        logger.info("Computing background from first frame...")
        first_img = lsu.load_h5_image(h5_files[0], h5loc, frame_index=0)
        background = lsu.compute_background(
            first_img,
            filter_radius=cfg["filter_radius"],
            median_passes=cfg["median_passes"],
        )
        # Save for re-use
        bg_path = cfg.get("background_file", "")
        if bg_path:
            try:
                background.tofile(bg_path)
                logger.info(f"Background saved to {bg_path}")
            except Exception as e:
                logger.warning(f"Could not save background: {e}")

    # --- 4. Connect to daemon ---
    logger.info(f"Connecting to daemon at {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(30.0)
    try:
        sock.connect((host, port))
    except socket.error as e:
        logger.error(f"Failed to connect: {e}")
        return {}
    logger.info("Connected to daemon.")

    # --- 5. Parallel preprocess + pipelined send ---
    #
    # ProcessPoolExecutor preprocesses multiple frames in parallel.
    # Futures are consumed in order to maintain sequential image_num.
    # A sender thread handles TCP I/O concurrently.
    #
    SEND_QUEUE_SIZE = 4  # buffer up to 4 frames
    PREPROCESS_WORKERS = min(os.cpu_count() or 4, 8)  # cap at 8 workers
    send_q: queue.Queue = queue.Queue(maxsize=SEND_QUEUE_SIZE)
    send_error: list = []  # shared error flag

    frame_mapping: dict = {}
    sent_count_lock = threading.Lock()
    counters = {"sent": 0, "skip": 0}

    def _sender_thread(sock, send_q, send_error, frame_mapping, counters,
                        labels_h5f):
        """Drain the queue and send frames over TCP."""
        while True:
            item = send_q.get()
            if item is None:  # Sentinel: done
                send_q.task_done()
                break
            image_num, pix_bytes, mapping_entry, filt_labels = item
            try:
                header = struct.pack("<H", image_num)
                sock.sendall(header)
                sock.sendall(pix_bytes)
                mapping_entry["skipped"] = False
                with sent_count_lock:
                    counters["sent"] += 1
                # Save label array to HDF5
                if labels_h5f is not None and filt_labels is not None:
                    ds_name = f"labels/{image_num}"
                    labels_h5f.create_dataset(
                        ds_name, data=filt_labels, dtype=np.int32,
                        compression="gzip", compression_opts=1,
                    )
            except Exception as e:
                logger.error(f"  Send/save error for image_num={image_num}: {e}")
                mapping_entry["skipped"] = True
                mapping_entry["reason"] = f"send_error: {e}"
                with sent_count_lock:
                    counters["skip"] += 1
                send_error.append(e)
            frame_mapping[str(image_num)] = mapping_entry
            send_q.task_done()

    # Open labels HDF5 for writing
    labels_h5f = None
    if HAS_H5PY:
        labels_h5f = h5py.File(labels_file, "w")
        logger.info(f"Labels will be saved to {labels_file}")
    else:
        logger.warning("h5py not available — labels will not be saved.")

    sender = threading.Thread(
        target=_sender_thread,
        args=(sock, send_q, send_error, frame_mapping, counters, labels_h5f),
        daemon=True,
    )
    sender.start()

    image_num = 1    # 1-based, uint16
    skip_count = 0
    t_start = time.time()

    logger.info(f"Using {PREPROCESS_WORKERS} parallel preprocessing workers")

    try:
        with ProcessPoolExecutor(max_workers=PREPROCESS_WORKERS) as pool:
            # Stage 1: Submit ALL frames to the pool upfront
            all_futures = []  # (future, image_num, h5_basename, frame_idx)
            img_num = 1
            for file_idx, (h5_path, n_frames) in enumerate(
                zip(h5_files, file_frame_counts)
            ):
                h5_basename = os.path.basename(h5_path)
                logger.info(
                    f"[{file_idx+1}/{len(h5_files)}] Queuing {h5_basename} "
                    f"({n_frames} frame{'s' if n_frames > 1 else ''})"
                )
                for frame_idx in range(n_frames):
                    if img_num > 65535:
                        break
                    fut = pool.submit(
                        _preprocess_one,
                        h5_path, h5loc, frame_idx,
                        nr_px_y, nr_px_x, cfg, background,
                    )
                    all_futures.append((fut, img_num, h5_basename, frame_idx))
                    img_num += 1
                if img_num > 65535:
                    break

            logger.info(f"Submitted {len(all_futures)} frames for preprocessing")

            # Stage 2: Consumer thread — drains futures in order → send queue
            def _consumer_thread():
                nonlocal skip_count
                for fut, img_num, h5bn, fr_idx in all_futures:
                    if send_error:
                        break
                    try:
                        result = fut.result()
                    except Exception as e:
                        result = {"error": f"worker_exception: {e}"}

                    if "error" in result:
                        reason = result["error"]
                        if reason != "no_spots":
                            logger.error(f"  Frame {fr_idx} of {h5bn}: {reason}")
                        frame_mapping[str(img_num)] = {
                            "file": h5bn, "frame": fr_idx,
                            "skipped": True, "reason": reason,
                        }
                        skip_count += 1
                    else:
                        mapping_entry = {
                            "file": h5bn, "frame": fr_idx,
                            "n_spots": result["n_spots"],
                        }
                        send_q.put((
                            img_num, result["pixels_bytes"],
                            mapping_entry, result["filt_labels"],
                        ))

                    # Progress
                    with sent_count_lock:
                        sc = counters["sent"]
                    processed = sc + skip_count
                    if processed > 0 and (processed % 10 == 0 or processed == 1):
                        elapsed = time.time() - t_start
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = (total_frames - processed) / rate if rate > 0 else 0
                        logger.info(
                            f"  Progress: {processed}/{total_frames} "
                            f"({rate:.1f} img/s, ETA {remaining:.0f}s)"
                        )
                    if processed > 0 and processed % save_interval == 0:
                        lsu.save_frame_mapping(frame_mapping, mapping_file)

                # Signal sender thread: no more frames
                send_q.put(None)

            consumer = threading.Thread(target=_consumer_thread, daemon=True)
            consumer.start()
            consumer.join()  # wait for all futures to be consumed

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except BrokenPipeError:
        logger.error("Daemon closed connection (broken pipe).")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Signal sender thread to finish and wait
        send_q.put(None)  # sentinel
        sender.join(timeout=60)

        # Final mapping save
        lsu.save_frame_mapping(frame_mapping, mapping_file)
        logger.info(f"Final mapping saved to {mapping_file}")

        # Close socket
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()
        logger.info("Socket closed.")

        # Close labels HDF5
        if labels_h5f is not None:
            labels_h5f.close()
            logger.info(f"Labels saved to {labels_file}")

    # Summary
    elapsed = time.time() - t_start
    with sent_count_lock:
        sc = counters["sent"]
    logger.info(
        f"Done: {sc} sent, {skip_count + counters['skip']} skipped, "
        f"{elapsed:.1f}s total ({sc/elapsed:.1f} img/s)"
    )
    return frame_mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Send preprocessed H5 images to LaueMatchingGPUStream daemon"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to params.txt configuration file"
    )
    parser.add_argument(
        "--folder", required=True,
        help="Folder containing .h5 image files"
    )
    parser.add_argument(
        "--h5-location", default="/entry/data/data",
        help="HDF5 internal dataset path (default: /entry/data/data)"
    )
    parser.add_argument(
        "--mapping-file", default="frame_mapping.json",
        help="Output JSON file for frame mapping (default: frame_mapping.json)"
    )
    parser.add_argument(
        "--save-interval", type=int, default=50,
        help="Save mapping to disk every N frames (default: 50)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Daemon host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=lsu.LAUE_STREAM_PORT,
        help=f"Daemon port (default: {lsu.LAUE_STREAM_PORT})"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )
    parser.add_argument(
        "--labels-file", default="labels.h5",
        help="Output HDF5 file for image segmentation labels (default: labels.h5)"
    )
    args = parser.parse_args()

    _setup_logging(args.log_level)

    if not os.path.isfile(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    if not os.path.isdir(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        sys.exit(1)

    serve_images(
        config_file=args.config,
        folder=args.folder,
        h5_location=args.h5_location,
        mapping_file=args.mapping_file,
        labels_file=args.labels_file,
        save_interval=args.save_interval,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
