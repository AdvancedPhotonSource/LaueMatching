# `laue_jax`

A differentiable **JAX** forward projection for white-beam Laue
microdiffraction — a single-framework port of
[`laue_torch`](../laue_torch/README.md).

## Why a second implementation

`laue_torch` is the reference: it is the one that is parity-tested against the
NumPy/C simulator, and it carries the full inverse pipeline (ODF/SDF recovery,
coded aperture, joint fitting, uncertainty).

`laue_jax` exists for one reason: so the Laue forward map composes with
**JAX-CPFEM** inside a *single* autodiff graph. Bridging two frameworks across
an optimisation loop means either detaching gradients or paying for a custom
VJP at the boundary; porting the forward is cheaper and keeps the graph intact.

It is deliberately a **forward model only** — geometry, rasterisation and
projection. Everything downstream stays in `laue_torch`.

## Enable float64

The parity target is `laue_torch` in double precision. JAX defaults to float32,
which will not reproduce it:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Install

```bash
pip install laue-jax
```

JAX itself is installed CPU-only by default. For GPU, follow the
[JAX install guide](https://docs.jax.dev/en/latest/installation.html) for your
CUDA version — this package does not pin an accelerator build.

## Use

```python
import jax
jax.config.update("jax_enable_x64", True)

from laue_jax import laue_forward, quat_to_matrix

# U: orientation, lat: lattice parameters, P/R: detector pose
image = laue_forward(...)
```

The public surface mirrors `laue_torch`'s geometry and rasterize modules:
`quat_to_matrix`, `rodrigues_to_matrix`, `sixd_to_matrix`, `to_rotation_matrix`,
`reciprocal_matrix`, `voigt_to_symmetric`, `deviatoric5_to_symmetric`,
`strain_to_B`, `gaussian_splat`, `hard_window`, `soft_window`,
`pseudo_voigt_splat`, and `laue_forward`.

## Tests

The suite is a **parity test against `laue_torch`**, so it needs both:

```bash
pip install -e ".[dev]"
pytest
```

`laue-torch` is a dev-only dependency and never a runtime one — keeping this
package's install footprint JAX-only is the point.

## License

BSD-3-Clause (UChicago Argonne, LLC) — see [`LICENSE`](LICENSE), as for the rest
of [LaueMatching](https://github.com/AdvancedPhotonSource/LaueMatching).
