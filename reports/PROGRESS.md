# Phase 1 progress log

> **Current governance note:** entries using “sealed test” are preserved historical logs. Official MURA `valid` was opened once for the frozen Phase-1 reference and must not be used for any later selection; see [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md).

## 2026-09-23 - Pre-run requirements and environment audit (superseded by actual run)

- **Executed:** Read the BIU DS23 guidelines, the supplied project charter, and the Phase 1 implementation brief. Inspected the workspace, Python runtimes, GPU state, and plausible local MURA locations.
- **Actual outputs:** A Phase 1 repository scaffold, pinned requirements, Phase 1 charter, data-access instructions, and configuration files.
- **Measured results at this pre-run snapshot:** NVIDIA GTX 1660 Ti (6 GB) was present, but the inspected PyTorch runtime was CPU-only. No MURA dataset or archive was then found, and no model metrics existed.
- **Decision:** Preserve the official MURA validation split as sealed test by default. Build all independent pipeline code and CPU-safe smoke tests; do not fabricate training or evaluation results.
- **Blocker:** Authorized local MURA access is required before indexing, training, EDA, and final evaluation.
- **Next step:** Run the data audit after `MURA_DATA_ROOT` is set, then generate the patient-safe split manifests.

## 2026-09-23 - Independent Phase 1 implementation and verification (pre-run)

- **Executed:** Implemented the XR_WRIST index, patient-safe split builder, audit, study datasets, preprocessing, frozen-embedding Logistic Regression baseline, ResNet-18 fine-tuning path, study-level evaluation, bootstrap CI support, error-analysis reporting, notebooks, and Phase 1 documentation.
- **Actual outputs at this pre-run stage:** `scripts/audit_data.py --help`, all three model CLI help commands, notebook JSON validation, and `python -m compileall src scripts tests` completed successfully. The then-installed CPU-only runtime also completed a synthetic frozen-embedding Logistic Regression train/validation/test-prediction flow.
- **Measured software results:** 16 synthetic tests passed. They cover audit findings, manifest and split contracts, patient/study isolation, label consistency, study grouping, validation-only threshold selection, test-leakage rejection, and report generation.
- **Measured ML results:** No MURA metrics. The synthetic Logistic Regression smoke test verifies plumbing only and is not a dataset result.
- **Decision:** Final test prediction generation is separated from final test metric computation. Both training CLIs can write sealed test prediction CSVs only after validation choices freeze; `scripts/evaluate_phase1.py` is the sole final-metric owner.
- **Blocker at that time:** MURA was absent. The then-installed Anaconda runtime was PyTorch 2.2.0 CPU-only, and its torchvision import was blocked by a Pillow `_imaging` DLL error, so actual ResNet-18 construction/training could not run.
- **Next step:** Obtain authorized MURA access, install a compatible CUDA-enabled PyTorch/torchvision/Pillow environment, then execute the documented ordered commands once.

## 2026-09-23 - Real XR_WRIST execution and sealed final evaluation

- **Executed:** Indexed and audited the authorized local MURA XR_WRIST data; created the deterministic patient-safe manifests; extracted frozen ResNet-18 study embeddings; trained the frozen-embedding Logistic Regression baseline; fine-tuned ResNet-18; froze validation-only selections; generated sealed test prediction files; and ran the sole study-level final evaluator with metrics, bootstrap intervals, comparison, and descriptive error-analysis reports.
- **Actual outputs:** [wrist_manifest.csv](../data/manifests/wrist_manifest.csv), [split_manifest.csv](../data/manifests/split_manifest.csv), [wrist_data_audit.json](wrist_data_audit.json), both model directories under `models/`, [model_comparison.csv](metrics/model_comparison.csv), [baseline_results.md](baseline_results.md), and [error_analysis.md](error_analysis.md).
- **Measured data results:** 3,474 patients, 3,697 studies, and 10,411 images; 2,274 NORMAL and 1,423 ABNORMAL studies. The sealed official-valid test set contains 237 studies (140 NORMAL, 97 ABNORMAL). The data audit and final evaluator found no duplicate paths, missing/corrupt images, inconsistent study labels, patient overlap, study overlap, or final-manifest integrity violations.
- **Measured software results:** 17 synthetic tests passed. They remain separate from the ML-performance claims.
- **Model A result:** Frozen ResNet-18 embeddings plus Logistic Regression selected `C=0.1` and threshold 0.3542854529699628 on validation only. On the sealed test set: AUROC 0.8756 (95% CI 0.8249–0.9201), AUPRC 0.8596, recall 0.8247, specificity 0.8071, F1 0.7843, and false-negative rate 0.1753 (17 false negatives).
- **Model B result:** Fine-tuned ResNet-18 selected mean aggregation and threshold 0.3860229253768921 on validation only. The CUDA run stopped early after 8 of 15 permitted epochs; its validation-best checkpoint was epoch 4 (AUROC 0.9107). On the sealed test set: AUROC 0.8970 (95% CI 0.8518–0.9353), AUPRC 0.8937, recall 0.7320, specificity 0.9500, F1 0.8114, and false-negative rate 0.2680 (26 false negatives).
- **Decision:** Freeze the reported Phase 1 outcome. Model B has higher discrimination/specificity/F1 point estimates but worse recall and nine additional false negatives at the selected operating points. Per-model AUROC intervals overlap and no paired difference test was run, so no statistical-significance claim is made.
- **Blocker / limitation:** No Phase 1 execution blocker remains. Limitations are the single 237-study sealed test set, one controlled Model B configuration, no paired difference interval, no external validation, and no clinically justified sensitivity/FNR operating target. MURA labels abnormality, not fracture.
- **Next step:** Do not retune on the sealed test set. If a subsequent research phase is authorized, pre-register the comparison/operating-point analysis and address false-negative performance before any advanced modelling or agentic workflow.
