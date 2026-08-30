# Laue Handbook — moved

Split into a **doc set** on 2026-08-11, matching the FF-HEDM and NF-HEDM sets in MIDAS and
the contract in `beamreport/DOCS_SPEC.md`.

**→ [`../manuals/laue/README.md`](../manuals/laue/README.md)** — the spine: scope gate, install gate, halt
conditions, the invariants, the worked example and done-means. The phases open as you
reach them.

| Was | Is now |
|---|---|
| Phases 0–6 | [`../manuals/laue/phase-0-survey.md`](../manuals/laue/phase-0-survey.md) … [`../manuals/laue/phase-6-material.md`](../manuals/laue/phase-6-material.md) |
| Invariants, Worked example, Done means | [`../manuals/laue/README.md`](../manuals/laue/README.md) |
| `Laue_Lab_Notebook_bt_34ide_jul26.md` | [`../manuals/laue/LAB_NOTEBOOK.md`](../manuals/laue/LAB_NOTEBOOK.md) |
| `Laue_Lab_Notebook_bt_34ide_jul26b.md` | merged into the same file — the public record is one notebook per *geometry*, not per campaign |
| — (new) | [`../manuals/laue/DIAGNOSIS.md`](../manuals/laue/DIAGNOSIS.md), [`../manuals/laue/RUNBOOK.md`](../manuals/laue/RUNBOOK.md) |

**The phase text was moved, not rewritten.** The spine gained a scope gate, an install gate
and a halt list, which the handbook did not have — `beamreport-doc-lint` refused it for all
three, and it was right.
