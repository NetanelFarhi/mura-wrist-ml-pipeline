# MURA X Phase 1 Charter

## Problem and user

The primary user is an Emergency Department clinician reviewing musculoskeletal radiographs before specialist review. Phase 1 evaluates whether a research decision-support classifier can identify MURA wrist studies labeled ABNORMAL while preserving a human clinician as the decision maker. It does not diagnose fractures, localize pathology, improve clinical outcomes, or make autonomous clinical decisions.

## Success metric

Before training, the primary comparison is study-level AUROC for a fine-tuned ImageNet ResNet-18 versus frozen ResNet-18 embeddings with Logistic Regression on the same patient-safe sealed test split. Secondary metrics are AUPRC, recall, specificity, F1, false-negative rate, confusion matrix, and a bootstrap AUROC confidence interval where feasible. The operating threshold and study aggregation rule are selected on validation data only.

## Data and contingency

Plan A is Stanford MURA, restricted to `XR_WRIST`, subject to Stanford's non-commercial research terms. The audit records measured local counts, labels, dimensions, views, and integrity findings; no global MURA counts are presented as local results. Plan B is documented only: FracAtlas would change the prediction task to fracture classification and requires an explicit project decision. It is not activated by this repository.

## Phase 1 architecture

Patient-safe manifests feed reproducible grayscale-to-three-channel preprocessing. Model A freezes ImageNet ResNet-18, mean-pools image embeddings to a study embedding, then fits Logistic Regression. Model B fine-tunes ImageNet ResNet-18 per view, aggregates study probabilities by validation-selected mean or max, then evaluates once on the sealed test set. Leakage checks, metrics, and error analysis are separate from model training.

## Agent boundary

The course charter describes a later agentic decision-support workflow. Phase 1 deliberately contains no agent, Grad-CAM, uncertainty engine, RAG, or interface. The Phase 1 pipeline is usable without an agent; a future phase must separately justify any agent, define its tools and guardrails, and evaluate it.

## Risks and cut line

The main risks are unavailable MURA access, CPU-only training, and accidental leakage across studies or patients. Mitigations are configurable external data access, CPU smoke tests, conservative training defaults, exact manifests, and automated integrity checks. If MURA is unavailable, independent code and tests are completed, actual training results remain unreported, and the next required action is obtaining authorized MURA access.

## Milestones and ownership

Phase 1 ownership is data indexing and audit, ML baseline and fine-tuning, and evaluation/reporting by the project lead. Before final-course submission, the project team must assign named Data, Model, Agent, and Product responsibilities as required by the BIU guidelines. Phase 1 completes when both model paths, test safeguards, reports, and measured results are available; agentic work is deferred.

