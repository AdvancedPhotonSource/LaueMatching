# Laue Handbook — moved

Split into a **doc set** on 2026-08-11, matching the FF-HEDM and NF-HEDM sets in MIDAS and
the contract in `beamreport/DOCS_SPEC.md`.

**→ [`laue/README.md`](laue/README.md)** — the spine: scope gate, install gate, halt
conditions, the invariants, the worked example and done-means. The phases open as you
reach them.

| Was | Is now |
|---|---|
| Phases 0–6 | [`laue/phase-0-survey.md`](laue/phase-0-survey.md) … [`laue/phase-6-material.md`](laue/phase-6-material.md) |
| Invariants, Worked example, Done means | [`laue/README.md`](laue/README.md) |
| `Laue_Lab_Notebook_bt_34ide_jul26.md` | [`laue/LAB_NOTEBOOK.md`](laue/LAB_NOTEBOOK.md) |
| `Laue_Lab_Notebook_bt_34ide_jul26b.md` | [`laue/LAB_NOTEBOOK_ZnZn.md`](laue/LAB_NOTEBOOK_ZnZn.md) |
| — (new) | [`laue/DIAGNOSIS.md`](laue/DIAGNOSIS.md), [`laue/RUNBOOK.md`](laue/RUNBOOK.md) |

**The phase text was moved, not rewritten.** The spine gained a scope gate, an install gate
and a halt list, which the handbook did not have — `beamreport-doc-lint` refused it for all
three, and it was right.
