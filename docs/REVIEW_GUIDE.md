# Reviewer guide — current project state

**Last repository review:** 2026-10-30
**Project scope:** Stanford MURA v1.1, **XR_WRIST only**, study-level
`NORMAL` / `ABNORMAL` classification. This is a research prototype for
workflow support before specialist review. It is not a fracture detector,
localization system, autonomous diagnosis, or clinical deployment.

## Start here

1. Read the root [README](../README.md) for the notebook sequence.
2. Read [result lineage](RESULT_LINEAGE.md) before comparing any historical
   2,768/692 result with the current 2,769/691 development protocol.
3. Read [Checkpoint 1 results](results/checkpoint1/README.md) for the current
   internal-validation comparison and bootstrap intervals.
4. Read [test-set governance](../reports/TEST_SET_GOVERNANCE.md) before
   interpreting any historical official-MURA-`valid` result.
5. Read the [charter](../CHARTER.md), [agent design](../agents/AGENT_DESIGN.md),
   and [agent-evaluation plan](../evaluation/AGENT_EVAL_PLAN.md) for the
   intended final workflow and its safety boundaries.
6. See the dated [repository review](../reports/REPOSITORY_REVIEW_20261030.md)
   for the evidence and checks performed for this update.
7. See the concise [Checkpoint 2 status](../reports/CHECKPOINT2_STATUS.md)
   and [presentation/demo preparation](../presentation/README.md) for what
   is ready now and what must still be completed.

## What is currently supported by measured evidence

The current Checkpoint 1 comparison uses **691 internal-validation studies
from 653 patients**, held out at patient level from official MURA `train`.
The operating threshold for each model is the highest threshold achieving
internal-validation sensitivity >=95% at study level.

| Model | AUROC | AUPRC | Sensitivity | Specificity | FN | FP | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen ImageNet ResNet-18 embeddings + Logistic Regression | 0.880 | 0.852 | 95.2% | 39.3% | 13 | 256 | Required baseline; development/validation |
| Fine-tuned ResNet-18, mean probability pooling | 0.913 | 0.878 | 95.2% | 58.3% | 13 | 176 | Current Model B operating-policy result; development/validation |
| Fine-tuned ResNet-18, max probability pooling | 0.883 | 0.806 | 95.2% | 62.1% | 13 | 160 | Secondary diagnostic only |

For Model B mean pooling, the frozen development threshold is **0.263568**;
62.5% of validation studies are flagged for human review under the >=95%
sensitivity policy. Full values and patient-cluster bootstrap intervals are in
the [result table](results/checkpoint1/model_comparison.csv).

## Historical official-valid reference — not an untouched future test

Official MURA `valid` was opened once for Phase 1 after validation-only
selection. It contains 237 wrist studies. Its historical results are preserved
for transparency: Model A AUROC 0.876 / AUPRC 0.860 and Model B AUROC 0.897 /
AUPRC 0.894. At their historical thresholds, Model B had 26 false negatives
versus 17 for Model A.

These values are **historical reference evidence only**. Because they were
observed, official `valid` may not be used for later model, threshold,
calibration, architecture, or agent selection. See
[TEST_SET_GOVERNANCE.md](../reports/TEST_SET_GOVERNANCE.md).

## What Phase 2 established

- **Phase 2A:** changing the operating threshold can increase sensitivity, but
  requires a substantial specificity/workload trade-off; calibration gains
  from temperature scaling were limited/inconsistent.
- **Phase 2B — PARTIAL:** gated attention did not show a consistent advantage
  over simpler embedding pooling on the internal-validation development set.
  Completed: mean/max/attention comparison and a 2,000-replicate paired
  bootstrap, with all requested paired confidence intervals crossing zero.
  Remaining: the attention-specific regression assertions and comprehensive
  view-count reporting requested by the protocol. It must not be presented as
  a completed superior multi-view result.
- **Phase 2C:** partial study-level fine-tuning did not justify its extra
  complexity. Frozen mean embedding remained slightly stronger at the >=95%
  policy (specificity 0.585 vs 0.571); paired bootstrap intervals included
  zero. The ML core is therefore frozen rather than further tuned. This is a
  separate mean-embedding research comparison; it does not replace the
  Checkpoint 1 Model B mean-probability runtime or borrow its threshold.

The detailed conclusion is in
[PHASE2C_SUMMARY.md](../reports/phase2c/PHASE2C_SUMMARY.md).

## Agent and product status

The **deterministic safety layer is implemented and tested**: input QC,
anatomy support checks, view-evidence routing, and a policy that can raise
review priority but never remove standard human review. The adaptive Agentic
AI layer is **designed, not yet implemented or evaluated**. Its independent
evaluation is pre-specified rather than inferred from model thresholds.

The local [Wrist Review app overview](app/Wrist_Review_App.md) documents the
implemented deterministic research interface. It is not a clinical product and
does not implement the planned adaptive agent.

## Notebook map and status

| Location | Meaning | Use for new selection decisions? |
|---|---|---|
| `notebooks/01`–`06` | Canonical wrist data/model/evaluation workflow | Yes, internal train/validation only |
| `notebooks/07` | Geometry/shortcut diagnostic | Diagnostic only |
| `notebooks/08` | Local study-view utility | No model selection |
| `notebooks/09_MURA_Deterministic_Safety_Tools.ipynb` | Supporting synthetic safety smoke tests, retained unchanged | No model selection |
| `notebooks/09_MURA_Deterministic_Agent_Policy.ipynb` | Active deterministic policy demonstration using synthetic inputs and policy v1.1 | No model selection; not an adaptive agent |
| `notebooks/legacy/09_MURA_Agentic_AI_Hagit_contributor_draft.ipynb` | Hagit's preserved contributor draft; safety-oriented components were adapted into the deterministic path, while the notebook itself is not executed directly | No |
| `notebooks/experimental_all_anatomies/` | Seven-anatomy development extension | Experimental only; not the XR_WRIST MVP claim |
| `notebooks/legacy/` | Traceability archive | No |

## Reproducibility and data governance checks

From the repository root, the reviewed local software checks were:

```powershell
python -m pytest tests -q
python scripts/check_repo_data_governance.py
```

On 2026-10-30, these completed with **74 tests passing** and with the
data-governance check reporting no unsafe tracked paths. Raw MURA images,
checkpoints, local manifests, per-study predictions, and credentials remain
outside Git by design.

## Important limits for review

- MURA `ABNORMAL` is not synonymous with acute fracture.
- Model probabilities are not automatically calibrated clinical confidence.
- Shortcut/geometry evidence is diagnostic; it cannot prove lesion reasoning
  or localize pathology.
- The app and deterministic routing are workflow prototypes; they do not
  establish clinical utility.
- A genuinely untouched evaluation protocol for advanced claims remains an
  open methodological decision with the team and lecturer.
