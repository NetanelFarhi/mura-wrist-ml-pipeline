# Current results

The canonical Model B has two separately reported evaluations:

- **Internal validation (development):** 691 studies, AUROC 0.913, sensitivity 95.2%, specificity 58.3%. This cohort also supports the paired Model A/B comparison and threshold selection: [full table and intervals](../../docs/results/checkpoint1/README.md).
- **Official MURA `valid` (14 November 2026):** 237 studies, AUROC 0.903, sensitivity 89.7% (95% CI 82.8–95.3%), specificity 64.3%, 10 missed abnormal studies and 50 false positives: [final run report](../official_valid_final/report.md).

The official run used the frozen model and threshold from [pre-registration](../../PREREGISTRATION.md). Its sensitivity point estimate is below 95%; the interval includes the target. This population had prior exposure, so it is not an untouched test. The deterministic layer routed 159 studies to priority review and left 7 abnormal studies in routine workflow. The LLM explanation layer was not run.

The separate Phase 2C result is a controlled development experiment, not a replacement for the Model B runtime: [Phase 2C summary](../phase2c/PHASE2C_SUMMARY.md). Use [result lineage](../../docs/RESULT_LINEAGE.md) to keep the models, thresholds, and split versions distinct.
