# `laue_torch`

A fully differentiable PyTorch forward model for white-beam Laue
micro-diffraction, plus a per-voxel orientation- and strain-distribution
function (ODF / SDF) recovery pipeline that runs on top of any
conventional Laue indexer.

The package is built as a drop-in **refinement** stage downstream of the
existing [LaueMatching](https://github.com/AdvancedPhotonSource/LaueMatching)
pipeline: LaueMatching's coarse-grid indexer (or any equivalent) gives
an approximate mean orientation per voxel; `laue_torch` refines the
*distribution* parameters around that mean from peak shape, with
analytic gradients through every step of the forward map.

## Highlights

- **Differentiable forward**: parity-tested against the reference
  NumPy/C simulator (94/94 spot match on the canonical test case);
  passes a strict `torch.autograd.gradcheck` on the geometry path for
  every parameter group (orientation, strain, lattice, detector pose).
- **Distribution-level recovery**: tangent-Gaussian on SO(3) for
  unimodal mosaic; mixture for twins / sub-grain modes; multivariate
  Gaussian on Voigt-6 strain.
- **Tensor GND density**: Nye's dislocation density tensor follows
  analytically from the recovered per-voxel ODF gradient; FCC slip-
  system projection helper included.
- **Posterior uncertainty**: Laplace approximation at convergence
  gives per-parameter marginal credible intervals.
- **Hessian eigenanalysis**: explicit identification of the physical
  degeneracies of polychromatic Laue (hydrostatic-strain null, lattice
  scale null, $P_z\!\leftrightarrow\!$lattice trade-off, etc.).
- **Real-data adapter**: reads LaueMatching post-processed H5 output
  and runs the per-voxel ODF refinement with one function call.

## Quick start

```python
import torch
from laue_torch import LaueForwardModel, generate_hkls, parse_params

p = parse_params("simulation/params_sim.txt")
hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
t = p.to_tensors()

model = LaueForwardModel(
    hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
    psf_sigma=t["psf_sigma"],
    rotation="quat", strain_mode="voigt",
)

U = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)   # (G, 4) quaternion
img = model(U, t["lattice"], t["P"], t["R"])                    # (Nx, Ny)
```

## Per-voxel ODF refinement on real data

```python
from laue_torch import parse_params
from laue_torch.realdata import LaueScanLoader, VoxelODFRefiner, plot_sigma_map

params = parse_params("simulation/params_sim.txt")
refiner = VoxelODFRefiner(params, sigma_init_deg=1.0,
                          M_render=128, n_steps=500)

results = []
for voxel in LaueScanLoader("/path/to/scan/"):
    results.append(refiner.refine(voxel))

plot_sigma_map(results, grid_shape=(20, 20),
               out_path="sigma_map.png", show_posterior=True)
```

Each `result` is a `VoxelODFResult` carrying:

- `U_mean`: refined mean orientation (3×3)
- `sigma_U_deg`: recovered isotropic mosaic spread
- `posterior_sigma_U_deg`: 1-σ Laplace posterior on `sigma_U_deg`
- `final_loss`, `n_steps`, `dt_s`, ...

For multi-voxel scans, the `plots` module supplies `plot_sigma_map`,
`plot_orientation_map`, `plot_gnd_map` (which computes Nye's tensor on
the recovered orientation field by central differences).

## Tutorial

The single-script tutorial `examples/tutorial_per_voxel_odf.py` walks
through the full synthetic pipeline end-to-end (forward → render →
Adam → Nye tensor → Laplace posterior) in $\sim$3 min on CPU.

## Synthetic experiments

The numerical experiments that characterised this package (mosaic-spread
recovery, twin-variant identification, intragranular gradients and the Nye
tensor, Laplace-posterior calibration, basin-of-convergence and Hessian
eigenanalysis studies) are research scripts and are **not** distributed with the
package.

What ships instead is the part that transfers: the two runnable tutorials in
`examples/`, and the test suite, which exercises every one of those code paths
against synthetic ground truth.

## Tests

```bash
cd packages/laue_torch
pip install -e '.[dev]'
KMP_DUPLICATE_LIB_OK=TRUE pytest
```

219 tests cover parity against the NumPy/C reference, gradient flow /
`gradcheck` on every parameter group, calibration recovery, distribution
moments, mixture / Nye correctness, coded-aperture and joint-fit paths, and a
contract test pinning the `midas-stress` misorientation convention.

The four parity tests need a LaueMatching **checkout** (they read
`simulation/params_sim.txt` and the reference simulator). Run from an installed
wheel instead and they skip cleanly rather than failing.

## Documentation

- [`docs/torch-forward-model.md`](../../docs/torch-forward-model.md) — math + API
  reference: the forward map, every differentiable parameter group, and the
  parameterisation choices.
- `examples/` — two runnable tutorials: per-voxel ODF recovery, and the
  coded-aperture voxel workflow.
- The test suite is the other reference. `tests/test_parity.py` pins the forward
  model against the repository's NumPy/C simulator, and
  `tests/test_grad.py` gradchecks every parameter group.

## Dependencies

Runtime (installed automatically):

- `torch >= 2.0`, `numpy >= 1.20`, `scipy >= 1.7`, `h5py >= 3.0`, `Pillow >= 9.0`
- `midas-stress >= 0.11.0` — symmetry-reduced misorientation. Note it returns
  **radians**; `laue_torch.symmetry` converts to degrees in one place.
- `midas-hkls >= 0.9.0` — lattice, space group, hkl generation, absorption
- `midas-invert >= 0.1.1` — Laplace uncertainty, fit / loss primitives

Optional: `matplotlib >= 3.5` (`pip install 'laue-torch[viz]'`) for the plotting
helpers in `laue_torch.realdata.plots`.

## Citation

If you use this package, please cite the LaueMatching pipeline that produces the
indexer seed orientations:

> H. Sharma, D. Sheyfer, R. Harder, J.Z. Tischler.
> *LaueMatching: an approach for rapid and robust indexing of Laue
> diffraction patterns.*
> J. Appl. Cryst. **59**, 552–563 (2026).

## License

BSD-3-Clause (UChicago Argonne, LLC) — see [`LICENSE`](LICENSE).
