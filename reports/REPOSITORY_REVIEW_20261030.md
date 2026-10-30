# Repository review — 2026-10-30

## Purpose and scope

This is a documentation and governance review before the lecturer's project
review. It does not retrain a model, alter a threshold, recompute a historical
metric, or open official MURA `valid`.

## What was checked

| Review item | Evidence inspected | Result |
|---|---|---|
| Current Checkpoint 1 claims | `docs/results/checkpoint1/model_comparison.csv`, `README.md`, `docs/REVIEW_GUIDE.md` | The displayed values match the persisted internal-validation table. |
| Baseline comparison | `docs/results/checkpoint1/model_a_grouped_cv_candidates.csv`, `model_comparison.csv`, Notebook 03 | The frozen ResNet-18 embedding + Logistic Regression baseline is explicit and compared on the same 691-study internal validation population. |
| Model B result and policy | `model_comparison.csv`, `frozen_operating_policies.json`, Notebook 06 | Mean probability pooling is the primary Model B policy; its threshold is validation-only and targets sensitivity >=95%. |
| Historical official-valid use | `reports/TEST_SET_GOVERNANCE.md`, Phase 1 reports | The once-opened official MURA `valid` result is labelled historical reference only, not an untouched future test. |
| Phase 2 conclusions | `reports/phase2a/`, `reports/phase2b/`, `reports/phase2c/PHASE2C_SUMMARY.md` | Calibration gains were limited/inconsistent; attention and partial fine-tuning did not justify further architecture tuning. |
| Agent claims | `agents/AGENT_DESIGN.md`, `evaluation/AGENT_EVAL_PLAN.md`, Notebook 09 documents | Deterministic safety tools exist; adaptive Agentic AI is designed but not yet implemented or evaluated. |
| Data governance | `.gitignore`, `data/README.md`, `scripts/check_repo_data_governance.py` | Raw MURA images, checkpoints, local manifests, predictions, and credentials are excluded from Git. |
| Product/app clarity | `apps/wrist_review/`, `docs/app/Wrist_Review_App.md` | A local deterministic research prototype is documented separately from the planned adaptive agent. |

## Reviewer navigation

Start at [docs/REVIEW_GUIDE.md](../docs/REVIEW_GUIDE.md). It gives the
notebook order, the measured-result table, the evidence hierarchy, and the
important limits in one place.

## Verification performed

From the repository root on 2026-10-30:

```powershell
python -m pytest tests -q
python scripts/check_repo_data_governance.py
```

Result: **74 tests passed** (14 third-party matplotlib deprecation warnings)
and the data-governance check passed with no unsafe tracked paths. The app was
also smoke-tested locally: its home route returned HTTP 200 and, without a local model bundle, its status
endpoint returned `ready: false` with `model_bundle_missing`. This is the
intended fail-closed behavior; it does not run inference without the verified,
non-versioned artifact bundle.

## Remaining, clearly marked work

- The adaptive Agentic-AI layer and its independent evaluation suite remain
  planned rather than executed.
- A genuinely untouched advanced-model evaluation protocol remains a team and
  lecturer decision; do not reuse observed official MURA `valid` for selection.
- The app is a local workflow prototype, not clinically validated or ready for
  deployment.
- The all-anatomy notebooks are experimental and must not be reported as the
  validated XR_WRIST MVP result.
