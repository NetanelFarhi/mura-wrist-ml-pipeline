# Phase 1 scope and course alignment

> **Current governance note:** “sealed final test” below describes the historical Phase-1 protocol. The official MURA `valid` results were observed once for that frozen reference evaluation and are not untouched for subsequent selection; see [TEST_SET_GOVERNANCE.md](TEST_SET_GOVERNANCE.md).

## Authority applied

The implementation brief supplied by the project owner defines the requested work: **Phase 1 only**. It explicitly requires real baseline machine-learning and deep-learning training when MURA and compute are available, and explicitly prohibits Agentic AI, Grad-CAM, uncertainty modelling, RAG, Streamlit, attention, ensembles, and additional CNN families in this phase.

The BIU DS23 guidelines are treated as authoritative for course-level requirements. The supplied project charter provides project context and constraints where compatible with the Phase 1 request. Neither source is used to fabricate missing experiment results, team members, data access, or a clinical claim.

## Requirements implemented in Phase 1

| Requirement source | Requirement | Phase 1 response |
|---|---|---|
| User brief and charter | XR_WRIST NORMAL/ABNORMAL at study level | Patient-study-view hierarchy, study aggregation, and study-level metrics. |
| User brief and course guidelines | Cheap baseline compared fairly with primary model | Frozen ResNet-18 embeddings plus Logistic Regression versus fine-tuned ResNet-18 on identical split definitions. |
| User brief and course guidelines | Reproducible data and evaluation workflow | Pinned dependencies, configs, exact manifests, ordered notebooks, audit script, tests, environment inventory, and progress log. |
| User brief and course guidelines | Evaluation and error analysis | AUROC, AUPRC, recall, specificity, F1, false-negative rate, confusion matrix, bootstrap CI support, leakage checks, and error reports. |
| Charter | Scientific boundary | No fracture diagnosis, validated localization, autonomous diagnosis, or claimed clinical outcome improvement. |
| Course guidelines | Clean repository and data dictionary | Raw data, caches, checkpoints, and secrets are ignored; the data dictionary and access instructions are included. |

## Deferred requirements

The course's final-capstone guidelines require a justified, guarded, working agent and final team role ownership. The charter also sketches an agentic decision workflow, QC, Grad-CAM, and human escalation. These are **not** Phase 1 deliverables because the owner explicitly deferred them. Therefore, Phase 1 may be reported as a baseline/checkpoint deliverable, but not as the entire final course capstone.

The charter's FracAtlas contingency is documented but inactive. Changing from MURA abnormality classification to fracture classification requires an explicit project-owner decision.

## Current evidence boundary

The authorized MURA XR_WRIST run is complete. The executed audit measured 3,474 patients, 3,697 studies, and 10,411 images; the official `valid` source was sealed as the 237-study final test set. Both required models were evaluated once at study level after their validation-only selections were frozen. The exact results, confidence intervals, thresholds, comparison, and descriptive error indices are recorded in [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md), [baseline_results.md](baseline_results.md), and [error_analysis.md](error_analysis.md).

This does not change the scientific boundary: the findings are a single research-baseline result for MURA NORMAL/ABNORMAL labels. They do not establish fracture diagnosis, localization, clinical benefit, statistical significance of the Model B versus Model A difference, or readiness for clinical deployment. Agentic functionality remains deferred beyond Phase 1.
