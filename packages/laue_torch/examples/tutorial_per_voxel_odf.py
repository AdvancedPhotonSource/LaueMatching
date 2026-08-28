"""Tutorial: per-voxel ODF + tensor-GND recovery from a single Laue exposure.

Run this as a script or pasted into a Jupyter notebook one cell at a
time.  Demonstrates the full pipeline that the npj CM paper covers:

  1. Build a synthetic ground-truth voxel with a known mosaic spread.
  2. Render its Laue pattern (the "observation").
  3. Adam refinement: recover the per-voxel ODF parameters from peak
     shape, given an indexer-supplied mean orientation seed.
  4. Compute Nye's dislocation density tensor across a 1-D scan.
  5. Compute Laplace posterior σ on the recovered σ_U.

Total runtime: ~3 minutes on CPU.
"""

# ── 1. Setup ──────────────────────────────────────────────────────────────
import math
import os
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from laue_torch import LaueForwardModel, generate_hkls, parse_params
from laue_torch.distributions import (
    GaussianStrain, IndependentVoxelDistribution, TangentGaussianSO3,
)
from laue_torch.synthetic import default_truth, make_model
from laue_torch.nye import nye_tensor, gnd_density, fcc_slip_systems
from laue_torch.uncertainty import laplace_posterior

print("Step 1: building forward model and ground-truth voxel ...")
truth = default_truth(n_grains=1, strain=False)
model = make_model(strain_mode="voigt", rotation="matrix",
                   n_pix=384, px_size=0.0008, h_max=8,
                   psf_sigma=2.5, hard=False, energy_image=False)

# ── 2. Truth voxel: tangent-Gaussian mosaic spread of σ_U = 0.5° ──────────
sigma_U_truth = 0.5
truth_orient = TangentGaussianSO3(
    U_init=truth.U[0], sigma_init=math.radians(sigma_U_truth))
truth_strain = GaussianStrain(sigma_init=1e-6)
truth_voxel = IndependentVoxelDistribution(truth_orient, truth_strain)
for p in truth_voxel.parameters():
    p.requires_grad_(False)

# ── 3. Render the observed Laue pattern (peak shape encodes the ODF) ─────
print("Step 2: rendering observation ...")
gen_obs = torch.Generator().manual_seed(2026)
with torch.no_grad():
    I_obs = truth_voxel.render(model, truth.lat, truth.P, truth.R,
                               M=128, generator=gen_obs)
print(f"  observation: shape={tuple(I_obs.shape)}, peak={I_obs.max().item():.3f}")

# ── 4. Refinement: indexer gives U_mean; we refine Σ_orient from peak ─────
print("Step 3: ODF refinement via Adam ...")
init_orient = TangentGaussianSO3(
    U_init=truth.U[0],                              # indexer-supplied seed (= truth here)
    sigma_init=math.radians(sigma_U_truth * 3.0))   # initial guess 3× truth
init_strain = GaussianStrain(sigma_init=1e-6)
init_voxel = IndependentVoxelDistribution(init_orient, init_strain)

# Freeze the strain and the mean orientation (we only refine Σ_orient here).
init_voxel.orient.mean_d6.requires_grad_(False)
init_voxel.strain.mean.requires_grad_(False)
init_voxel.strain.cov.log_diag.requires_grad_(False)
init_voxel.strain.cov.off_diag.requires_grad_(False)

opt = torch.optim.Adam([
    {"params": [init_voxel.orient.cov.log_diag], "lr": 5e-3},
    {"params": [init_voxel.orient.cov.off_diag], "lr": 5e-3},
])

n_steps = 250
for step in range(n_steps):
    opt.zero_grad()
    g_truth = torch.Generator().manual_seed(step + 1)
    g_pred = torch.Generator().manual_seed(step + 1)
    with torch.no_grad():
        I_obs_step = truth_voxel.render(model, truth.lat, truth.P, truth.R,
                                        M=64, generator=g_truth)
    I_pred = init_voxel.render(model, truth.lat, truth.P, truth.R,
                               M=64, generator=g_pred)
    loss = ((I_pred - I_obs_step) ** 2).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    sigma_U_pred = math.degrees(math.sqrt(
        init_voxel.orient.covariance().diag().mean().item()))
print(f"  truth σ_U     = {sigma_U_truth:.3f}°")
print(f"  recovered σ_U = {sigma_U_pred:.3f}°  "
      f"(loss = {loss.item():.3g})")

# ── 5. Tensor GND from a 1-D scan with a linear orientation gradient ─────
print("\nStep 4: tensor GND from a synthetic 1-D scan ...")
from laue_torch.nye import synthetic_linear_gradient_field
n_voxels = 7
R_field, alpha_truth = synthetic_linear_gradient_field(
    n_voxels=n_voxels, axis_index=2,
    rate_per_voxel_deg=0.2, spacing=1.0,
    U_base=truth.U[0],
)
# In a real experiment R_field would come from per-voxel U_mean recoveries.
alpha_recovered = nye_tensor(R_field, spacing=1.0)
alpha_central = alpha_recovered[n_voxels // 2]
relerr = (alpha_central - alpha_truth).abs().max().item() / \
         alpha_truth.abs().max().item()
print(f"  Nye tensor at central voxel:")
print(f"    truth     : {alpha_truth.numpy()}")
print(f"    recovered : {alpha_central.numpy()}")
print(f"    rel error = {relerr:.4f}")

# Total GND density and slip-system breakdown.
b_burgers = 2.5e-10                               # ~0.25 nm for Cu/Al
rho_total = gnd_density(alpha_central.unsqueeze(0).unsqueeze(0),
                        burgers_m=b_burgers)
print(f"  Total GND density (central voxel): {rho_total.item():.3g} m^-2")

# ── 6. Laplace posterior on σ_U ──────────────────────────────────────────
print("\nStep 5: Laplace posterior on the recovered σ_U ...")
fixed_seed = 0xABCDEF
g_obs_fixed = torch.Generator().manual_seed(fixed_seed)
with torch.no_grad():
    I_obs_fixed = truth_voxel.render(model, truth.lat, truth.P, truth.R,
                                     M=64, generator=g_obs_fixed)

map_log_diag = init_voxel.orient.cov.log_diag.detach().clone()
map_off_diag = init_voxel.orient.cov.off_diag.detach().clone()
map_theta = torch.cat([map_log_diag, map_off_diag])

from laue_torch.geometry import rodrigues_to_matrix, sixd_to_matrix


def loss_fn_flat(theta):
    cov_log = theta[:3]
    cov_off = theta[3:6]
    L = torch.diag(cov_log.exp())
    tril_i, tril_j = init_voxel.orient.cov.tril_idx
    L = L.clone()
    L[tril_i, tril_j] = cov_off
    g_pred = torch.Generator().manual_seed(fixed_seed)
    z = torch.randn(64, 3, dtype=theta.dtype, generator=g_pred)
    delta = z @ L.T
    U_pert = rodrigues_to_matrix(delta)
    U_mean = sixd_to_matrix(init_voxel.orient.mean_d6.detach())
    U_samples = U_mean.unsqueeze(0) @ U_pert
    eps = torch.zeros(64, 6, dtype=theta.dtype)
    weights = torch.full((64,), 1.0 / 64, dtype=theta.dtype)
    I_pred = model(U_samples, truth.lat, truth.P, truth.R,
                   strain=eps, weights=weights)
    return ((I_pred - I_obs_fixed) ** 2).mean()


map_loss = loss_fn_flat(map_theta).item()
posterior = laplace_posterior(loss_fn_flat, map_theta,
                              noise_variance=max(map_loss, 1e-12))
sigma_orient_rad = float(map_log_diag.exp().mean().item())
posterior_sigma_log_diag = posterior.sigma[:3].mean().item()
posterior_sigma_orient_deg = math.degrees(
    sigma_orient_rad * posterior_sigma_log_diag)
print(f"  σ_U^MAP = {sigma_U_pred:.3f}°  ± {posterior_sigma_orient_deg:.3f}° "
      f"(1σ posterior; cond # {posterior.cond_number:.3g})")
print(f"  truth lies inside 1σ?  "
      f"{abs(sigma_U_pred - sigma_U_truth) <= posterior_sigma_orient_deg}")

print("\nDone. See the package README for the method and references.")
