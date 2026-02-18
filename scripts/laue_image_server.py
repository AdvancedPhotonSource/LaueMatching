#!/usr/bin/env python
"""
laue_image_server.py — TCP image-sending client for LaueMatchingGPUStream

Reads HDF5 images from a folder, pre-processes each frame (background
subtraction → enhancement → thresholding → component filtering → Gaussian
blur), and sends processed images to the GPU daemon over TCP port 60517.

Wire protocol (must match LaueMatchingGPUStream.cu handle_client):
    | uint16_t image_num (2 bytes, LE) | double[NrPxX*NrPxY] pixel data |

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
import socket
import sys
import time

import numpy as np

import laue_stream_utils as lsu

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
# Core loop
# ---------------------------------------------------------------------------

def serve_images(
    config_file: str,
    folder: str,
    h5_location: str = "/entry/data/data",
    mapping_file: str = "frame_mapping.json",
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
    sock.settimeout(30.0)
    try:
        sock.connect((host, port))
    except socket.error as e:
        logger.error(f"Failed to connect: {e}")
        return {}
    logger.info("Connected to daemon.")

    # --- 5. Process & send loop ---
    frame_mapping: dict = {}
    image_num = 1    # 1-based, uint16
    sent_count = 0
    skip_count = 0
    t_start = time.time()

    try:
        for file_idx, (h5_path, n_frames) in enumerate(
            zip(h5_files, file_frame_counts)
        ):
            h5_basename = os.path.basename(h5_path)
            logger.info(
                f"[{file_idx+1}/{len(h5_files)}] Processing {h5_basename} "
                f"({n_frames} frame{'s' if n_frames > 1 else ''})"
            )

            for frame_idx in range(n_frames):
                if image_num > 65535:
                    logger.warning("Reached uint16 image_num limit, stopping.")
                    break

                # Load raw frame
                try:
                    raw = lsu.load_h5_image(h5_path, h5loc, frame_index=frame_idx)
                except Exception as e:
                    logger.error(
                        f"  Frame {frame_idx} of {h5_basename}: load error: {e}"
                    )
                    skip_count += 1
                    continue

                # Validate dimensions
                if raw.shape != (nr_px_y, nr_px_x):
                    logger.warning(
                        f"  Frame {frame_idx}: shape {raw.shape} != "
                        f"expected ({nr_px_y},{nr_px_x}), skipping."
                    )
                    skip_count += 1
                    continue

                # Preprocess (steps 1-6)
                blurred, _filt_img, _filt_labels, centers = lsu.preprocess_image(
                    raw, cfg, background=background
                )

                # Skip if no spots detected
                if not centers:
                    logger.debug(
                        f"  Frame {frame_idx} of {h5_basename}: "
                        f"no spots after filtering, skipping."
                    )
                    skip_count += 1
                    # Still record in mapping but mark as skipped
                    frame_mapping[str(image_num)] = {
                        "file": h5_basename,
                        "frame": frame_idx,
                        "skipped": True,
                        "reason": "no_spots",
                    }
                    image_num += 1
                    continue

                # Send to daemon
                try:
                    lsu.send_image(sock, image_num, blurred)
                    sent_count += 1
                except Exception as e:
                    logger.error(f"  Send error for image_num={image_num}: {e}")
                    skip_count += 1
                    frame_mapping[str(image_num)] = {
                        "file": h5_basename,
                        "frame": frame_idx,
                        "skipped": True,
                        "reason": f"send_error: {e}",
                    }
                    image_num += 1
                    continue

                # Record mapping
                frame_mapping[str(image_num)] = {
                    "file": h5_basename,
                    "frame": frame_idx,
                    "n_spots": len(centers),
                    "skipped": False,
                }

                # Progress
                elapsed = time.time() - t_start
                rate = sent_count / elapsed if elapsed > 0 else 0
                remaining = (total_frames - (sent_count + skip_count)) / rate if rate > 0 else 0
                if sent_count % 10 == 0 or sent_count == 1:
                    logger.info(
                        f"  Sent image_num={image_num} "
                        f"({sent_count}/{total_frames} done, "
                        f"{rate:.1f} img/s, ETA {remaining:.0f}s)"
                    )

                # Periodic save
                if sent_count % save_interval == 0:
                    lsu.save_frame_mapping(frame_mapping, mapping_file)
                    logger.debug(f"  Saved mapping ({sent_count} entries)")

                image_num += 1

            # Break outer loop too if limit reached
            if image_num > 65535:
                break

    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except BrokenPipeError:
        logger.error("Daemon closed connection (broken pipe).")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
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

    # Summary
    elapsed = time.time() - t_start
    logger.info(
        f"Done: {sent_count} sent, {skip_count} skipped, "
        f"{elapsed:.1f}s total ({sent_count/elapsed:.1f} img/s)"
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
        save_interval=args.save_interval,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
