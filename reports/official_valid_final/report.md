# Official MURA valid (XR_WRIST) — run report

- Mode: `official`; run folder: `official_20261114T203242Z`
- Started: 2026-11-14T20:30:19+00:00; summary saved: 2026-11-14T20:32:50+00:00 (UTC)
- Code commit (repository ZIP): `7d860849d643a96c0efdf2869923818bfe55ecec`
- Model B checkpoint SHA-256: `63AEDC6BEC1534BDED7C3CA4D0DAF03AFC17BF92166CB7848506A6F3A82DE5B6`; threshold 0.263567715883255; study aggregation: mean of view probabilities; agent policy 1.1.0
- Device: cpu; python 3.13.18, torch 2.14.1+cpu, torchvision 0.29.1+cpu, pillow 11.3.0, numpy 2.4.6, pandas 2.2.3, sklearn 1.6.1
- Checks: 38 of 38 PASS

## Population

237 studies (97 abnormal, 140 normal), 207 patients; studies within 0.001 of the threshold: 0.

## Model B (study probability vs. the frozen threshold)

| Measure | Value [95 % CI] |
|---|---|
| Sensitivity (primary) | 0.897 [0.828, 0.953] |
| Missed abnormal studies (FN) | 10 |
| Specificity | 0.643 [0.562, 0.727] |
| False positives (FP) | 50 |
| Share flagged (p ≥ threshold) | 0.578 [0.512, 0.639] |
| AUROC | 0.903 [0.859, 0.943] |
| AUPRC | 0.902 [0.850, 0.940] |

**Primary reading (rule fixed before the run):** compatible with the target, but below it in the point estimate.

## Agent decision layer (routes)

| Measure | Value [95 % CI] |
|---|---|
| Share routed to priority review | 0.671 [0.606, 0.730] |
| Raised to priority only by view disagreement | 22 |
| Abnormal studies in routine workflow | 7 |
| Abnormal studies outside priority review | 7 |
| SAFETY_BLOCK | 0 |

Routes: {'priority_human_review': 159, 'routine_workflow': 78}. Image QC (worst per study): {'PASS': 237}.

## Method

95 % CIs: patient-cluster bootstrap, 1000 resamples, seed 42, threshold fixed, percentile 2.5/97.5; one-class resamples skipped: 0.

## Phase 1 (historical reference, not a fair comparison)

Phase 1 evaluated earlier models on this set (another training split, other thresholds): Model A missed 17 and Model B missed 26 of the 97 abnormal studies (`reports/PHASE1_5_DECISION.md`). This run: Model B missed 10 of 97. This run is a second look at an already-observed set, not an untouched test.

## Files in the run folder (SHA-256)

- `view_predictions.csv`: `2C685D421B264CB61B02A43623295D85A996697B38F88BCF3C78AE94F04B6520`
- `study_predictions.csv`: `DA5662B5BAAC9650F86A06D1DDCE4BD16CDB8F33826DBFED6ECF1FAE9D0AE229`
- `study_routes.csv`: `F06E708CBE36DC57393A4B3BC963AE67E9161A7D4641CB855C74804F6C5DE263`
- `summary.json`: `4E155479A491F7126EC80367EC039D0C71741A9734901DCDCD8BE280CC044BD2`

Per-study files stay on Drive; only this report and summary.json are meant for Git.
