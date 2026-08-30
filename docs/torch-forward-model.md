# `laue_torch` — Differentiable Laue forward projection

`laue_torch` is a PyTorch implementation of the LaueMatching forward model
(the math behind [scripts/GenerateSimulation.py](../scripts/GenerateSimulation.py)
and [LaueMatchingCPU.c](../packages/laue_index/c_src/LaueMatchingCPU.c)). Every output pixel
is a smooth function of the inputs, so PyTorch's autograd can backprop
through the rendered image into:

- per-grain orientation `U` (quaternion / Rodrigues / 6D / matrix),
- per-grain strain `ε` (Voigt-6 / deviatoric-5 / full deformation gradient `F`),
- shared lattice parameters `(a, b, c, α, β, γ)`,
- detector pose `P_Array`, `R_Array`,
- pixel size, energy bounds.

This unlocks gradient-based detector calibration, orientation refinement,
strain fitting, and orientation-distribution-function (ODF) inference —
all on the same forward kernel that produces parity with the existing
NumPy / C implementation.

## Install

`torch>=2.1` and [`midas-hkls`](https://github.com/marinerhemant/MIDAS) are
required. Both are listed in [requirements.txt](../requirements.txt). Until
`midas-hkls` propagates to PyPI, install it from the MIDAS source tree:

```bash
pip install -e ~/opt/MIDAS/packages/midas_hkls/
```

## Forward model

For each `(grain g, reflection h)`:

```text
B0  = recip(a, b, c, α, β, γ)            # 1/nm, lattice → reciprocal
B   = (I − ε) · B0    or   F⁻ᵀ · B0       # apply per-grain strain
q   = U · B · h                            # 1/nm
q̂   = q / ‖q‖
k_f = k_i − 2 (q̂·k_i) q̂                  # Bragg, k_i = (0, 0, 1)
xyz = R_detᵀ · k_f                         # rotate into detector frame
xy  = xy · P[2] / z                        # project onto detector plane
px  = (x − P[0]) / dx + (Nx − 1)/2         # to pixel coords
py  = (y − P[1]) / dy + (Ny − 1)/2
sinθ = −q̂_z;   E = h·c · ‖q‖ / (4π · sinθ)
```

Each spot is splatted onto the `(Nx, Ny)` image with a sub-pixel-accurate
2D Gaussian PSF; rejection masks (`z>0`, detector bounds, energy window)
are sigmoids by default for differentiability, and become hard booleans
with `hard=True`.

## Quickstart

```python
from laue_torch import LaueForwardModel, generate_hkls, parse_params

p = parse_params("simulation/params_sim.txt")
hkls = generate_hkls(p.sg_num, p.lattice, p.E_hi)
t = p.to_tensors()

model = LaueForwardModel(
    hkls=hkls, n_pix=t["n_pix"], px_size=t["px_size"],
    psf_sigma=t["psf_sigma"],
    rotation="quat", strain_mode="voigt",
)

import torch
U = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)   # (G, 4)
img = model(U, t["lattice"], t["P"], t["R"])                    # (Nx, Ny)
```

Set `reduce="stack"` to get per-grain images of shape `(G, Nx, Ny)`,
useful for per-grain losses or compositing.

## Gradient-based refinement examples

Each example follows the same pattern: render an observed image with known
parameters, then refine an initial guess via Adam.

### Detector translation

```python
P = (P_true + perturbation).requires_grad_(True)
opt = torch.optim.Adam([P], lr=5e-4)
for _ in range(400):
    opt.zero_grad()
    I_pred = model(U, lat, P, R_true)
    ((I_pred - I_obs) ** 2).mean().backward()
    opt.step()
```

### Orientation

For sub-pixel matching, image-MSE alone has a narrow basin (perturbations
beyond a few PSF widths produce zero overlap). Use the `aux.px / aux.py`
position outputs and a per-spot MSE for a wider basin:

```python
_, aux_obs = model(U_true, lat, P, R, return_aux=True)
def loss_fn():
    _, aux = model(U, lat, P, R, return_aux=True)
    w = aux.mask * aux_obs.mask
    return (((aux.px - aux_obs.px) ** 2 + (aux.py - aux_obs.py) ** 2) * w).sum() \
            / w.sum().clamp_min(1.0)
```

### Strain

Three parameterizations, selectable via `strain_mode=` at model
construction:

| `strain_mode` | params | shape | notes |
|---------------|--------|-------|-------|
| `"voigt"`      | 6     | `(G, 6)` | `(ε11, ε22, ε33, ε23, ε13, ε12)`; full symmetric strain |
| `"deviatoric"` | 5     | `(G, 5)` | `(ε11, ε22, ε23, ε13, ε12)`; `ε33 = −(ε11+ε22)` (no hydrostatic null) |
| `"F"`          | 9     | `(G, 3, 3)` | full deformation gradient; `B = F⁻ᵀ · B0` |

In white-beam Laue **the hydrostatic component of `ε` is invisible to spot
positions** (each spot picks its own energy from the white beam to satisfy
Bragg). Use `strain_mode="deviatoric"` for position-only fits to drop the
null direction; use `strain_mode="voigt"` plus an energy-resolved loss
(see below) to recover all six components.

### Energy-resolved fitting

Setting `energy_image=True` makes the forward also splat per-spot energies
into a `(Nx, Ny)` image. Per-spot energies are always available in
`aux.energy`. Build a hydrostatic-sensitive loss:

```python
_, aux_obs = model(U, lat, P, R, strain=eps_true, return_aux=True)
E_obs = aux_obs.energy.detach()
m_obs = aux_obs.mask.detach()

eps = torch.zeros_like(eps_true).requires_grad_(True)
def loss_fn():
    _, aux = model(U, lat, P, R, strain=eps, return_aux=True)
    w = aux.mask * m_obs
    return ((aux.energy - E_obs) ** 2 * w).sum() / w.sum().clamp_min(1.0)
```

The position-only loss has zero gradient on the trace direction of `ε`;
the energy loss does not. The test suite contains a direct contrast of
the two in `test_energy_loss_sees_hydrostatic_strain`.

## ODF / orientation mixture

Use the multi-grain forward kernel as the building block:

```python
from laue_torch import DiscreteOrientationMixture, fibonacci_so3

# 200 candidate orientations on a quasi-uniform SO(3) grid.
odf = DiscreteOrientationMixture(
    U_init=fibonacci_so3(200, dtype=torch.float64),
    weight_param="softmax",
)
def loss_fn():
    U = odf.orientations()
    w = odf.weights()
    img = model(U, lat, P, R, weights=w)
    data_term = ((img - I_obs) ** 2).mean()
    return data_term + 1e-3 * odf.entropy()    # sparsity prior
```

This is sketch-level — a serious ODF refinement adds:

- a robust loss (Huber, masked MSE),
- multi-scale annealing of `psf_sigma`,
- L1 / entropy / spherical-harmonic priors on the weights,
- coarse-to-fine candidate refinement.

## Parity with the reference NumPy / C implementation

Every detector pixel that the existing simulator lights up is also lit
by `laue_torch` (94/94 spots, exact set match) for the golden case
[simulation/params_sim.txt](../simulation/params_sim.txt) +
[simulation/fourOrientations.csv](../simulation/fourOrientations.csv).
This is exercised by `tests/test_parity.py`. The differentiable forward
*also* passes a strict `torch.autograd.gradcheck` on the geometry-only
output (orientation, strain, lattice, detector pose) — see
`tests/test_grad.py::test_gradcheck_on_aux_positions`.

## CLI

```bash
python -m laue_torch.cli \
    -configFile simulation/params_sim.txt \
    -orientationFile simulation/fourOrientations.csv \
    -outputFile out.h5
```

Mirrors `scripts/GenerateSimulation.py` arguments. Add `-strainMode voigt`
+ `-strainFile strains.csv` for per-grain strain. Add `-energyImage` to
also write the intensity-weighted energy image.
