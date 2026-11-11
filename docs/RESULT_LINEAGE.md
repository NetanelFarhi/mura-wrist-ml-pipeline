# Result lineage and split versions

This note prevents two different internal development splits from being
mistakenly treated as one experiment. It does not replace the original reports
or change any historical result.

## Terms used throughout the repository

- **Official MURA `train`:** the only source population used to create the
  project's internal development splits.
- **Internal train / validation:** patient-grouped partitions derived from
  official MURA `train`. They are used for fitting, model selection, and
  threshold selection during development.
- **Official MURA `valid`:** the source split that was opened once for the
  historical Phase 1 frozen-reference evaluation. It must not be used for any
  later model, threshold, calibration, architecture, or agent decision.

## Split versions

| Version | Internal train | Internal validation | Evidence and status |
|---|---:|---:|---|
| **Historical Phase 1 split** | 2,768 studies / 2,614 patients / 7,805 images | 692 studies / 653 patients / 1,947 images | Used for the original Phase 1, Phase 1.5, Phase 2A, and Phase 2B reports. It is retained unchanged for traceability. |
| **Current canonical Checkpoint 1 split** | 2,769 studies / 2,614 patients / 7,810 images | 691 studies / 653 patients / 1,942 images | Used by the current numbered workflow and the current Checkpoint 1 result table. This is the split to use for any continuation of the current development protocol. |

Both versions contain 3,460 XR_WRIST studies from the official training
population. The one-study shift is a change in the internal patient-grouped
partition, not evidence of a new test set. Historical reports used the
Phase-1 seed-23 split; the current Notebook 01 documents the current
StratifiedGroupKFold, seed-42 split. The repository does not retroactively
rewrite the historical artifacts or compare metric point estimates across the
two split versions as if they came from the same validation cohort.

## What is current versus historical

### Current Checkpoint 1 protocol

The canonical model workflow is [Notebook 01](../notebooks/01_MURA_Data_Preparation_and_EDA.ipynb)
through [Notebook 06](../notebooks/06_MURA_ModelB_Evaluation_and_Comparison.ipynb),
using the 2,769/691 split above. [Notebook 07](../notebooks/07_MURA_Shortcut_and_Geometry_Audit.ipynb)
is a diagnostic that reads the frozen artifacts and does not select a model or threshold. The current results are in
[Checkpoint 1 results](results/checkpoint1/README.md) and the machine-readable
[comparison table](results/checkpoint1/model_comparison.csv).

The 691-study validation cohort has 422 NORMAL and 269 ABNORMAL studies. Its
published primary policy selects, separately for each model, the highest
study-level threshold reaching validation sensitivity >=95%. These are
development/validation results only.

[Phase 2C](../reports/phase2c/PHASE2C_SUMMARY.md) also uses the current
2,769/691 partition: it regenerated a compatible frozen mean-embedding
reference before its single controlled partial-fine-tuning comparison.

### Model and threshold boundaries within the current split

The current Checkpoint 1 reporting and local deterministic-app bundle are
the fine-tuned **Model B with mean probability pooling** and its own frozen
95%-sensitivity threshold (`0.263568`). Phase 2C's **frozen mean-embedding**
reference is a separate study-level research comparison with its own threshold
(`0.138672`). It was retained as the simpler winner of that Phase 2C
comparison, but it does not silently replace the Checkpoint 1 Model B runtime.

Do not transfer a threshold between these models, combine their rows in one
fair-comparison table, or describe the Phase 2C reference as an independently
tested clinical replacement for Model B. Each result must retain its model,
aggregation method, split version, and operating policy.

### Current split, archived operating policy

[The pre-95% checkpoint archive](results/archive/checkpoint1_pre_95pct/)
contains a previous >=90% operating-policy presentation on the **current
691-study validation cohort**. It is archived because its operating target is
not the current >=95% project target; it is not the historical 2,768/692 split.

### Historical Phase 1 / Phase 1.5 / Phase 2A / Phase 2B evidence

The following are preserved historical development artifacts based on the
2,768/692 split:

- [Phase 1 summary](../reports/PHASE1_SUMMARY.md) and
  [technical handoff](../reports/PHASE1_TECHNICAL_HANDOFF.md);
- [Phase 1.5 decision note](../reports/PHASE1_5_DECISION.md);
- [Phase 2A summary](../reports/phase2a/PHASE2A_SUMMARY.md); and
- [Phase 2B summary](../reports/phase2b/PHASE2B_SUMMARY.md).

They remain useful for transparent negative findings and design history. In
particular, Phase 2B is **PARTIAL**, not a completed superior multi-view
result. Its gated-attention comparison and paired bootstrap were completed,
but the remaining attention-specific regression assertions and comprehensive
view-count reporting are not complete. See its summary for the exact scope.

## Official MURA `valid` status

The historical Phase 1 report evaluated the official MURA `valid` split once:
237 XR_WRIST studies (140 NORMAL, 97 ABNORMAL). Those observed results are
historical reference evidence, not a fresh final test for the current workflow.
The governing rule and options for a future advanced-model evaluation are in
[TEST_SET_GOVERNANCE.md](../reports/TEST_SET_GOVERNANCE.md).

## Reporting rule

Every future table or slide must state all three of the following:

1. the cohort: current internal validation, historical internal validation, or
   historical official MURA `valid`;
2. the operating policy used; and
3. whether the result is current development evidence, historical reference,
   or exploratory work.

Do not combine rows from different split versions in a single fair-comparison
table. Do not call any result from official MURA `valid` an untouched test
result for new model selection.
