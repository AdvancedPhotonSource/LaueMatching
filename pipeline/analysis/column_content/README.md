# column_content: the Laue reference implementation

Which crystal orientations are present in one beam column, each with its intensity share and orientation spread, and
how completely they explain the frame. The method, the interface, the validated envelope and the traps are in MIDAS
`manuals/column-content/`. The monochromatic (DAC) implementation of the same method is `midas_defect.column_content`.

## Use

```python
import sys; sys.path.insert(0, "<LaueMatching>/pipeline/analysis")
from column_content.geom import Geom, EmpKernel
from column_content.fit import ColumnFit
G = Geom(params_path, "zn")                       # LaueMatching params file + phase name
kern = EmpKernel("si_kernel_lsq.npz")             # MEASURED on a thin single-crystal calibrant at the same beam settings
fit = ColumnFit(G, raw, bg, orientation_matrices, all_hkls, kern, K=24)
fit.fit(n_iter=300, lr=3e-4, inits=(0.05, 0.4))
report = fit.report()                             # per orientation: share, extent_perp_deg, extent_blind_deg, mean_om
residual = fit.residual_frame()                   # feed to the indexer for discovery rounds
```

**Multi-frame runs:** `pipeline.run(...)`, with your indexer injected as `index_fn` and `solutions`. Measure the
discovery gate on scrambled residual frames (`gate_from_scrambled`); do not reuse the round-0 gate.

**Validate first:** `synthetic.render_columns(...)` builds known-content frames with YOUR geometry, calibrant kernel,
background and spread classes. Then `evaluate.evaluate(...)` applies the frozen gates.

**Unexplained intensity:** `arcs.extract_arcs` (the bead classification) and `arcs.common_axis_test` (pooled
detection, with the axis reported as a band).

## Provenance
Ported from the sampleH (Zn on Zn, 34-ID-E) development code, 2026-09. The port is checked against the original:
`ColumnFit` vs the original on 4 validation frames, and `extract_arcs` vs the original on 5 real frames. See the
LAB_NOTEBOOK in the manual.
