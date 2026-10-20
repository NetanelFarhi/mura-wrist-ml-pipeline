# MURA XR_WRIST Research Project

Course project for study-level **NORMAL / ABNORMAL** classification of wrist radiograph studies from Stanford MURA v1.1.

The project is a research prototype for decision support before specialist review. It predicts the MURA study label, which is an abnormality label — **not** a fracture diagnosis, fracture location, or autonomous clinical decision.

## Current checkpoint status (Checkpoint 1, 18.10.2026)

The Checkpoint 1 path runs end to end in Google Colab:

```text
01 data preparation: patient-level split, leakage checks, EDA
        ↓
02 preprocessing and image-decode checks
        ↓
03 Model A: frozen ResNet-18 embeddings + Logistic Regression  →  04 evaluation of Model A (sensitivity ≥ 90%)
05 Model B: ResNet-18 fine-tuned at study level (three training seeds)
        ↓
06a operating policy frozen on internal validation (sensitivity ≥ 95%)
        ↓                                         ↓
07 end-to-end run with the agent stub       06b locked test on the official MURA valid split
                                                 (ready, not run: only after the team freezes the model)
```

Until 18.10.2026 the agent layer is a **stub** (`src/pipeline/agent_stub.py`): it fixes the contract
`ModelOutput → AgentReport` (contract version 1.0) and has no LLM, no guardrails and no tools. Its only rule is
the frozen threshold: score ≥ threshold → `ABOVE_THRESHOLD` (flag for specialist review), otherwise
`BELOW_THRESHOLD`. The agents (guardrails, QC, tools, clinician report, comparison with a rule-based policy) are
Checkpoint 2 work. They receive the same `ModelOutput`; the additional report fields required by
[agents/AGENT_DESIGN.md](agents/AGENT_DESIGN.md) §5 will be added as a new contract version.

| Current run | Train | Internal validation |
|---|---:|---:|
| Images | 7,810 | 1,942 |
| Studies | 2,769 | 691 |
| Patients | 2,614 | 653 |
| NORMAL studies | 1,712 | 422 |
| ABNORMAL studies | 1,057 | 269 |

## Results — internal validation (development only)

All numbers below come from the executed notebooks in this branch and use the same 691 internal-validation
studies that were used for model, aggregation and threshold selection. They are development results, not a final
untouched-test result.

| Model | AUROC | AUPRC | Notebook |
|---|---:|---:|---|
| A: frozen ResNet-18 + Logistic Regression (`C = 0.1`) | 0.8804 | 0.8515 | 03 |
| B: fine-tuned ResNet-18, seed 23 (best epoch 6) | 0.9020 | 0.8690 | 05 |
| B: fine-tuned ResNet-18, seed 42 (best epoch 7) | 0.9018 | 0.8731 | `seed_runs/` |
| B: fine-tuned ResNet-18, seed 7 (best epoch 5) — **pipeline model** | 0.9043 | 0.8721 | `seed_runs/` |

Model B scores a study as the mean of its view probabilities (mean was selected over max on validation in every run).
Paired patient-cluster bootstrap of AUROC(B) − AUROC(A), 1,000 resamples: seed 23 +0.0216 (95% CI 0.0006 to 0.0429),
seed 42 +0.0214 (−0.0006 to 0.0430), seed 7 +0.0239 (0.0038 to 0.0461).

### Frozen operating policy (Notebook 06a, run `run_20261017T105209Z`)

Rules fixed before the candidates were compared: (1) per model, the highest internal-validation threshold with
sensitivity ≥ 95%; (2) the Model B run with the highest specificity at that sensitivity; (3) Model B becomes the
pipeline model only if its specificity exceeds Model A's by at least 0.02, otherwise the simpler Model A is kept.

| Model | Threshold | Sensitivity | Specificity | False negatives | False positives | Studies flagged |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.0638 | 0.952 | 0.393 | 13 | 256 | 512 / 691 (74.1%) |
| B, seed 23 | 0.0306 | 0.952 | 0.400 | 13 | 253 | 509 / 691 (73.7%) |
| B, seed 42 | 0.0435 | 0.952 | 0.355 | 13 | 272 | 528 / 691 (76.4%) |
| B, seed 7 | 0.0704 | 0.952 | 0.469 | 13 | 224 | 480 / 691 (69.5%) |

Result: Model B seed 7 (specificity gain +0.076 over Model A) is the pipeline model, `model_b@cde5a48a7b0f`,
threshold 0.070447. **Caveat:** over the three seeds the mean specificity is 0.408 (SD 0.057), a mean gain of only
+0.015 over Model A — below the 0.02 margin. Seed 7 is the best of three runs on the same validation split, so its
advantage is optimistic; the locked test (06b) is the unbiased check.

For comparison, Notebook 04's reference policy for Model A (sensitivity ≥ 90%): threshold 0.1277, sensitivity
0.903, specificity 0.531, 26 false negatives, 198 false positives, 441 / 691 studies flagged (63.8%).

## Locked test — official MURA `valid`, XR_WRIST (Notebook 06b)

**Not run yet.** By team decision the official MURA `valid` split is evaluated **once**, with Notebook 06b, only
after the model is frozen. Notebook 06b reads the frozen `operating_policy.json` from 06a, checks the hashes of the
frozen models, and refuses to run if a locked-test result already exists.

## End-to-end run with the agent stub (Notebook 07)

Notebook 07 reads the frozen policy and (1) sends every internal-validation study through the stub
(`case_reports.jsonl`, one `AgentReport` per study), (2) recomputes the scores from the radiographs with the frozen
model and checks them against the saved scores (tolerance 0.002), and (3) shows one study end to end. It needs a
**T4 GPU**: the saved Model B scores were computed on GPU with mixed precision, and a CPU recomputation exceeds the
tolerance.

Last executed run, `run_20261017T113214Z` (previous notebook version, live check on 20 studies): 691 reports,
480 `ABOVE_THRESHOLD` / 211 `BELOW_THRESHOLD`, 13 ABNORMAL studies below the threshold, maximum score difference
7.6 × 10⁻⁵, flags identical. The current version (all 691 studies in the live check, demonstration cell) awaits its
acceptance run. Contract tests: `python -m pytest -q tests/test_agent_stub.py`.

## Notebook workflow

| Notebook | Status | Purpose |
|---|---|---|
| `00_data_access_audit.ipynb` | Legacy scaffold | Local-folder data audit; not the current Colab ZIP workflow. |
| `01_eda.ipynb` | Legacy scaffold | Earlier local EDA template; superseded by `01_MURA_Data_Preparation.ipynb`. |
| [01_MURA_Data_Preparation.ipynb](notebooks/01_MURA_Data_Preparation.ipynb) | Executed | Verifies the source data, parses patient/study/label, creates (or verifies) the patient-level split, leakage tests, exports `WRIST_SPLITS`, EDA tables and charts. |
| [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb) | Executed | Cleans the internal manifests, checks decode validity, preserves study grouping, and defines deterministic evaluation preprocessing. |
| [03_MURA_Baseline_ResNet18_LogReg.ipynb](notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb) | Executed | Model A: frozen ResNet-18 embeddings pooled by study + Logistic Regression; internal-validation predictions. |
| [04_MURA_Model_Evaluation.ipynb](notebooks/04_MURA_Model_Evaluation.ipynb) | Executed | Model A: validation-only operating point (sensitivity ≥ 90%), ROC/PR, confusion matrices, bootstrap intervals, error tables. |
| [05_MURA_ResNet18_FineTune.ipynb](notebooks/05_MURA_ResNet18_FineTune.ipynb) | Executed (seed 23) | Model B: ImageNet ResNet-18 fine-tuned with a study-level loss (`configs/resnet18_finetune.yaml`); paired comparison with Model A. |
| [notebooks/seed_runs/](notebooks/seed_runs/) | Executed (seeds 42, 7) | The same Notebook 05 with other training seeds; the seed-7 run is the frozen pipeline model. |
| [06a_MURA_Operating_Policy.ipynb](notebooks/06a_MURA_Operating_Policy.ipynb) | Executed | Internal validation only: thresholds at sensitivity ≥ 95% (90% for reference), Model B seed-to-seed spread, pre-specified choice of the Model B run and the pipeline model; writes `operating_policy.json`. |
| [06b_MURA_Locked_Test.ipynb](notebooks/06b_MURA_Locked_Test.ipynb) | Ready, not run | One evaluation of the frozen models on the official MURA `valid` split; run only after the model is frozen. |
| [07_MURA_Agent_Stub_Pipeline.ipynb](notebooks/07_MURA_Agent_Stub_Pipeline.ipynb) | Updated; acceptance run pending | End-to-end run: frozen model output → `agent_stub` → `AgentReport` for every validation study, live image-to-report check, demonstration of one study. |
| `notebooks/archive/` | Archive | Earlier data-preparation versions (v1, v2). |

## Data and Git policy

Raw MURA images, ZIP archives, per-study predictions, embeddings, checkpoints, credentials, and local Drive paths must never be committed. The repository contains code, notebook logic, reviewed notebook outputs, documentation, schemas, and aggregate results only.

For local data access and expected layout, see:

- [data/README.md](data/README.md)
- [data/data_dictionary.md](data/data_dictionary.md)
- [.gitignore](.gitignore)

The Colab notebooks use one shared Google Drive folder. Members reach it as `MyDrive/DATA` (a shortcut) or as
`MyDrive/Final Project DS23/Agentic AI-Assisted Detection of Musculoskeletal Abnormalities in Emergency X-rays/DATA`;
Notebooks 01, 05, 06a, 06b and 07 accept either path, Notebooks 02–04 expect `MyDrive/DATA`.

```text
DATA/
├── MURA-v1.1_files.zip
├── WRIST_SPLITS/            # Notebook 01 (written once, then only verified)
├── PREPROCESSED_WRIST/      # Notebook 02
├── BASELINE_MODEL_A/        # Notebooks 03 and 04 (frozen Model A, hashed in operating_policy.json)
└── RUNS_ARTEM/              # time-stamped run folders: 05, 06a, 06b, 07 and re-runs of 02–04
```

Shared files are never replaced. Notebook 01 verifies existing split files instead of rewriting them; Notebooks
02–04 write to `RUNS_ARTEM/<stage>/run_<UTC time>/` when their outputs already exist in the shared folder;
Notebooks 05–07 always write to a new run folder and refuse to overwrite a file; 06b and 07 check the SHA-256 of
every frozen file before using it.

## Evaluation governance

The project uses patient-safe internal train/validation manifests. A patient and a study may not appear in both internal splits, and all views from one study stay together.

The official MURA `valid` split was used once for the historical Phase 1 frozen-reference evaluation, so it cannot honestly be described as untouched. Checkpoint 1 reports it as a *held-out test previously observed in Phase 1*: thresholds, aggregation and models are frozen on internal validation (Notebook 06a) before Notebook 06b scores it once, and Notebook 06b refuses a second evaluation. One internal-train image is byte-identical to one test image (`patient11359/study1_negative`); Notebook 06b reports a pre-specified analysis without that study.

See [reports/TEST_SET_GOVERNANCE.md](reports/TEST_SET_GOVERNANCE.md) for the full historical record and constraints.

## Scope and next work

Implemented for Checkpoint 1:

- data preparation with a saved patient-level split, leakage checks and EDA (Notebook 01);
- deterministic preprocessing and integrity checks;
- Model A (frozen ResNet-18 + Logistic Regression) and Model B (fine-tuned ResNet-18, three training seeds), same study-level protocol;
- validation-only operating policy at sensitivity ≥ 95%, Model B seed-to-seed spread, bootstrap confidence intervals;
- locked-test notebook (06b) with a one-evaluation guard, to be run once after the model is frozen;
- agent stub with a fixed contract, contract tests (`tests/test_agent_stub.py`) and an end-to-end run (Notebook 07).

After 18.10.2026 (Checkpoint 2):

- agent layer: guardrails, QC, tools, clinician-facing report, comparison with a rule-based policy;
- calibration, Grad-CAM diagnostic analysis, shortcut/error analysis;
- external validation or another approved evaluation protocol.

The agent design and evaluation plan are in [agents/AGENT_DESIGN.md](agents/AGENT_DESIGN.md) and [evaluation/AGENT_EVAL_PLAN.md](evaluation/AGENT_EVAL_PLAN.md).

## Important limitations

- MURA `ABNORMAL` does not mean acute fracture.
- High-sensitivity operating points substantially increase review workload (about 70% of studies flagged at sensitivity ≥ 95%); they are research tradeoffs, not clinical policies.
- Probability values are model scores, not calibrated confidence.
- Model B scores depend on the compute setting (GPU with mixed precision); CPU inference would need re-freezing or a documented tolerance.
- No model in this repository is ready for clinical use.

## Project documentation

- [CHARTER.md](CHARTER.md)
- [Requirements traceability](REQUIREMENTS_TRACEABILITY.md)
- [Product and cost plan](reports/PRODUCT_AND_COST_PLAN.md)
- [Shortcut-analysis plan](evaluation/SHORTCUT_ANALYSIS_PLAN.md)
- [Historical Phase 1 summary](reports/PHASE1_SUMMARY.md)
