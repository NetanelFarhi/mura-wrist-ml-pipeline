# Notebook 09 integration note

## Source preserved

Hagit's supplied draft is preserved unchanged at
`notebooks/legacy/09_MURA_Agentic_AI_Hagit_contributor_draft.ipynb` and is
not executed directly as the active runtime. This organization does not
rewrite the notebook; the safety-oriented components listed below were adapted
into the canonical deterministic path.

## What was retained

| Contribution from the incoming notebook | Canonical project location |
| --- | --- |
| Technical input QC for empty, corrupt, undersized, and near-uniform images | `src/runtime/deterministic_safety.py::technical_qc` and Notebook 09 §1 |
| Anatomical scope blocking before inference | `validate_declared_anatomy` and Notebook 09 §2 |
| Study-level view-evidence/disagreement summary | `view_evidence` and Notebook 09 §2 |
| A safe route that preserves human review | `guarded_study_route` and Notebook 09 §2 |
| Safety checks for unsupported anatomy and malformed images | `tests/test_deterministic_safety.py` |

## Deliberate adaptations

The incoming notebook referenced a Drive-local checkpoint named
`RESNET18_FINE_TUNING/resnet18_best.pt`, a separate threshold, and official
MURA `valid` records. Those dependencies are not the verified canonical Model
B bundle and would re-open the already-observed official-valid population.
They were therefore **not** made part of the runnable project path.

The canonical Notebook 09 instead uses only synthetic image bytes. It runs
without MURA images or a checkpoint and demonstrates the deterministic safety
layer. A future runtime may connect the verified Model B bundle only after the
artifact identity check passes; it must not use official-valid labels for
selection or agent evaluation.

## Scope boundary

The current implementation is a deterministic safety baseline, not an
adaptive LLM agent. TTA, Grad-CAM, adaptive planning, and their independent
evaluation remain planned work. See `agents/AGENT_DESIGN.md` and
`evaluation/AGENT_EVAL_PLAN.md`.
