# Pre-registration: single final run on official MURA `valid` (XR_WRIST)

Registered before the run. The run notebook is committed to `main` immediately before this file; both enter `main` before the run and are not changed afterwards. The registration commit is the commit that adds this file.

## 1. Canonical model
Bundle `checkpoint1-model-b-mean-probability-95-sensitivity` (`configs/model_b_artifact_bundle.json`): fine-tuned ImageNet ResNet-18, one logit per view. Checkpoint `best_resnet18_study_level.pt`, SHA-256 `63AEDC6BEC1534BDED7C3CA4D0DAF03AFC17BF92166CB7848506A6F3A82DE5B6`. The run stops if the hash differs.

## 2. Study aggregation
Study probability = mean of the view probabilities.

## 3. Frozen threshold
`0.263567715883255`: the highest threshold reaching sensitivity ≥ 0.95 on internal validation (691 studies, 653 patients; sensitivity 0.952, specificity 0.583, FN 13). Official MURA `valid` was not used to choose it. A study is flagged if p ≥ threshold.

## 4. Agent version and deterministic rules
Repository code: the repository at the registration commit (the commit that adds this file); the run uses `CODE_COMMIT` = its full hash. The internal check of the run notebook is repeated on that commit before the official run.
- `tools/agent_policy.py`, `POLICY_VERSION = "1.1.0"` (`view_consistency`, `evaluate_agent_policy`), SHA-256 `10AC53D6529DBCC42F0B504EF138E42D8D84CAD3FD14A2E3B114F9BD04F643F4`;
- `src/runtime/deterministic_safety.py` (`technical_qc`, `validate_declared_anatomy`, `view_evidence`), SHA-256 `2DA0AE3F43432C7EB3A87B8A5AEC87D7D6F7CD9B14E4C4208B1DCB32B9A73A45`.

Rule: `priority_human_review` if p ≥ threshold or the view probabilities lie on both sides of the threshold; otherwise `routine_workflow`. The policy re-checks every study record; if it does not return `VALID` (QC not PASS, unsupported anatomy or an inconsistent record), the route is `SAFETY_BLOCK`. Every study stays in human review. This routing rule is the same as in policy 1.1 at commit `3ecbcb9`: on internal validation both give 490 priority / 200 routine / 1 SAFETY_BLOCK.

Other agent implementations in `main` are not part of this run: notebook `09_MURA_Agentic_AI-final.ipynb`, `src/runtime/bounded_agent.py`, `src/runtime/agent_tool_policy.py` (application tool policy with a ±0.05 borderline zone) and `src/runtime/crewai_research.py`. The LLM explanation layer is not run and cannot change the class or the route.

Run code: `notebooks/10_MURA_Official_Valid_Final_Evaluation.ipynb`, committed immediately before this file; only `MODE` and `CODE_COMMIT` are set at run time.

## 5. Reported metrics
- Model B alone (study probability vs. the frozen threshold), all 237 studies: sensitivity, specificity, AUROC, misses (FN), share of studies flagged (p ≥ threshold).
- Also reported: FP, AUPRC; decision layer (routes): share of studies routed to priority review, studies raised to priority only by view disagreement, ABNORMAL studies routed to routine or kept outside priority review, SAFETY_BLOCK count.
- 95% CIs: patient-cluster bootstrap as in notebook 06 (`bootstrap_interval`): whole patients resampled with replacement, 1,000 resamples, seed 42, threshold fixed, percentile intervals.
- Reference values on internal validation (691 studies; development data, not a test): sensitivity 0.952, specificity 0.583, AUROC 0.913, FN 13, FP 176, flagged 62.5% (`docs/results/checkpoint1/README.md`).
- Reading of sensitivity, fixed in advance: upper 95% CI bound < 0.95 → target not reproduced; otherwise, point estimate ≥ 0.95 → target reproduced in the point estimate; otherwise → compatible with the target but below it in the point estimate.

## 6. Single run
This is a single run. After it, no model, threshold or rule is changed, and the result is reported as is, alongside the Phase 1 results.

Phase 1 was the project's earlier single evaluation on this set, with the models of that time (another training split, other thresholds): the Phase 1 Model A missed 17 and the Phase 1 Model B missed 26 of the 97 ABNORMAL studies (`reports/PHASE1_5_DECISION.md`). This run is therefore a second look at an already-observed set, not an untouched test. The run's `report.md` shows its result next to these numbers; the Phase 1 numbers are a historical reference, not a fair comparison.

Other contact with this set before registration: notebook `09_MURA_Agentic_AI-final.ipynb` and its working versions (08–11.11.2026) ran the canonical Model B on a few studies without metrics (7 studies, 20 images in the final version), choosing them with cached predictions of an earlier, non-canonical model on all 659 images; notebook `09A_MURA_Ground_True_Independent_Model_B_Validation.ipynb` (12.11.2026) compared 3 of these studies with their labels (1 FP, 1 TP, 1 TN). The model, aggregation, threshold and routing rule were all in `main` by commit `3ecbcb9` (07.11.2026), before these contacts, and none of them was changed because of them.

Only aggregate results go to Git (`report.md`, `summary.json`).
