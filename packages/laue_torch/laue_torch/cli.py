"""Command-line entry point: forward-simulate a Laue pattern with PyTorch.

Mirrors :mod:`scripts.GenerateSimulation` but uses the differentiable
:class:`laue_torch.forward.LaueForwardModel` under the hood. Useful as a
sanity-check that the torch path matches the reference NumPy/C path.

Example
-------
    python -m laue_torch.cli \\
        -configFile simulation/params_sim.txt \\
        -orientationFile simulation/fourOrientations.csv \\
        -outputFile out.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from .forward import LaueForwardModel
from .io import generate_hkls, load_orientations, parse_params

logger = logging.getLogger("laue_torch.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward-simulate a Laue pattern using laue_torch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-configFile", required=True,
                        help="Path to params_sim.txt-style config.")
    parser.add_argument("-orientationFile", required=True,
                        help="CSV of orientation matrices, one per row (9 floats).")
    parser.add_argument("-outputFile", required=True,
                        help="Output HDF5 path; .tif written alongside.")
    parser.add_argument("-strainMode", default="none",
                        choices=("none", "voigt", "deviatoric", "F"),
                        help="Strain parameterization. With 'none' a single B0 is used.")
    parser.add_argument("-strainFile", default=None,
                        help="Optional CSV of per-grain strains. Voigt-6, deviatoric-5, "
                             "or 9 floats (F deformation gradient).")
    parser.add_argument("-energyImage", action="store_true",
                        help="Also write an intensity-weighted energy image.")
    parser.add_argument("-dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("-noStretch", action="store_true",
                        help="Disable contrast stretch on the saved TIFF.")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("Parsing config: %s", args.configFile)
    p = parse_params(args.configFile)
    logger.info("Generating HKLs via midas-hkls (sg=%d, E_hi=%g keV)",
                p.sg_num, p.E_hi)
    hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
    logger.info("HKL count: %d", hkls.shape[0])

    logger.info("Loading orientations: %s", args.orientationFile)
    U = load_orientations(args.orientationFile)
    logger.info("Grains: %d", U.shape[0])

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    t = p.to_tensors(dtype=dtype)

    strain = None
    if args.strainMode != "none":
        if args.strainFile is None:
            logger.warning("strainMode=%s but no -strainFile given; using zero strain.",
                           args.strainMode)
            strain = _zero_strain(args.strainMode, U.shape[0], dtype)
        else:
            strain = _load_strain(args.strainFile, args.strainMode, dtype)
            if strain.shape[0] != U.shape[0]:
                raise ValueError(f"strain rows ({strain.shape[0]}) != grains ({U.shape[0]})")

    model = LaueForwardModel(
        hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
        psf_sigma=t["psf_sigma"],
        rotation="matrix", detector_rotation="rodrigues",
        strain_mode=args.strainMode,
        hard=True,
        energy_image=args.energyImage,
    )

    with torch.no_grad():
        if args.energyImage:
            img, aux = model(U.to(dtype), t["lattice"], t["P"], t["R"],
                             strain=strain, E_range=t["E_range"], return_aux=True)
        else:
            img = model(U.to(dtype), t["lattice"], t["P"], t["R"],
                        strain=strain, E_range=t["E_range"])
            aux = None

    img_np = img.detach().cpu().numpy()
    out = Path(args.outputFile)
    tif_path = out.with_suffix(out.suffix + ".tif")
    _save_tiff(img_np, tif_path, stretch=not args.noStretch)
    logger.info("Wrote TIFF: %s", tif_path)

    with h5py.File(out, "w") as hf:
        hf.create_dataset("/entry1/data/data", data=img_np)
        hf.create_dataset("/entry1/orientation_matrices", data=U.cpu().numpy())
        if aux is not None and aux.energy_image is not None:
            hf.create_dataset("/entry1/energy_image",
                              data=aux.energy_image.detach().cpu().numpy())
            # Average energy per pixel where intensity > 0.
            with np.errstate(invalid="ignore", divide="ignore"):
                avg = np.where(img_np > 1e-6,
                               aux.energy_image.detach().cpu().numpy() / np.maximum(img_np, 1e-30),
                               0.0)
            hf.create_dataset("/entry1/average_energy", data=avg)
    logger.info("Wrote HDF5: %s", out)
    return 0


# ── Helpers ────────────────────────────────────────────────────────────────

def _zero_strain(mode: str, G: int, dtype: torch.dtype) -> torch.Tensor:
    if mode == "voigt":
        return torch.zeros(G, 6, dtype=dtype)
    if mode == "deviatoric":
        return torch.zeros(G, 5, dtype=dtype)
    if mode == "F":
        return torch.eye(3, dtype=dtype).unsqueeze(0).expand(G, 3, 3).contiguous()
    raise ValueError(mode)


def _load_strain(path: str, mode: str, dtype: torch.dtype) -> torch.Tensor:
    arr = np.atleast_2d(np.genfromtxt(path))
    if mode == "voigt":
        if arr.shape[1] != 6:
            raise ValueError(f"voigt strain expects 6 columns, got {arr.shape[1]}")
        return torch.tensor(arr, dtype=dtype)
    if mode == "deviatoric":
        if arr.shape[1] != 5:
            raise ValueError(f"deviatoric strain expects 5 columns, got {arr.shape[1]}")
        return torch.tensor(arr, dtype=dtype)
    if mode == "F":
        if arr.shape[1] != 9:
            raise ValueError(f"F strain expects 9 columns (row-major 3x3), got {arr.shape[1]}")
        return torch.tensor(arr.reshape(-1, 3, 3), dtype=dtype)
    raise ValueError(mode)


def _save_tiff(img: np.ndarray, path: Path, stretch: bool) -> None:
    if stretch and img.max() > 0:
        scaled = (img / img.max() * 65535).astype(np.uint16)
    else:
        scaled = np.clip(img, 0, 65535).astype(np.uint16)
    Image.fromarray(scaled).save(path)


if __name__ == "__main__":
    sys.exit(main())
