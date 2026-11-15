# MURA XR_WRIST — Study-Level Abnormality Classification

A research project using **Stanford MURA v1.1 wrist radiographs** to classify an imaging study as **NORMAL / ABNORMAL** and support human review. A study can contain several X-ray views; the pipeline combines their evidence into one study-level result.

The project brings together data preparation, a frozen-feature baseline, a fine-tuned ResNet-18, reproducible evaluation, deterministic safety checks, a local **Wrist Review** application, and a fully implemented **Agentic AI research workflow**.

The scope is MURA-defined abnormality: the labels do not establish an acute-fracture diagnosis or lesion location.

**This is a research prototype, not a clinically validated system.**

---

## Team

| Team member | Current responsibilities |
| --- | --- |
| **Netanel** | Model-bundle and runtime integration, study-level results, deterministic safety, local application/demo, bounded-agent workstream, Demo App and coordination of the experimental all-anatomy extension. |
| **Hagit** | Data access and governance, patient/study/view organization, preprocessing, and data-scope review. Full development and implementation of the MURA Agentic AI workflow (Notebook 09), including deterministic agent policy, frozen Model B integration, TTA and Grad-CAM tools, CrewAI research explanations, structured reporting, safety controls, and end-to-end testing. |
| **Hodaya** | Core fine-tuned model workflow, model-training evidence, model review, and the project presentation. |
| **Artem** | Product framing, reproducibility review, and final-presentation readiness. |

These workstreams follow the [project charter](CHARTER.md); ownership does not imply sole authorship of every notebook.

Notebook 09 implements a separate Agentic AI research workflow developed by Hagit. It is distinct from the deterministic safety controls used by the local Wrist Review application.

---

## How the Project Works

1. **Prepare the data:** Organize patients, studies, and views; create patient-grouped internal train/validation splits from official MURA `train`.

2. **Build the baseline — Model A:** Extract frozen ImageNet ResNet-18 embeddings, average them within each study, and fit Logistic Regression. Select regularization with patient-grouped cross-validation inside internal train.

3. **Evaluate the existing fine-tuned model — Model B:** Use the saved ResNet-18 checkpoint and average per-view probabilities into one study probability.

4. **Apply the operating policy:** Select each model's highest internal-validation threshold that reaches sensitivity ≥95%, then freeze that model-specific threshold.

5. **Support human review:** Apply technical QC, declared-anatomy checks, and deterministic routing before presenting evidence in the local app. Routing can raise priority; human review remains required.

6. **Generate agent-assisted research explanations — Notebook 09:** Integrate frozen Model B predictions with deterministic policy checks, authorized optional TTA/Grad-CAM tools, CrewAI-based explanations, and structured research artifacts.

The Agentic AI workflow is a separate research component and does not replace the local application or the frozen model operating policy.

---

## Main Result and Model Comparison

**Model B reached AUROC 0.913 on internal validation and 0.903 on official MURA `valid`. Sensitivity was 95.2% internally and 89.7% on official validation at the same frozen threshold.**

### Model A versus Model B on internal validation

Both models were evaluated on the same **691 studies from 653 patients**, comprising 422 NORMAL and 269 ABNORMAL studies.

This cohort comes from official MURA `train`; these are development/validation results, not an untouched final test.

| Metric | Model A — Frozen ResNet-18 + Logistic Regression | Model B — Fine-tuned ResNet-18 |
| --- | ---: | ---: |
| AUROC | 0.880 | **0.913** |
| AUPRC | 0.852 | **0.878** |
| Sensitivity at the ≥95% target | 95.2% | 95.2% |
| Specificity at that operating point | 39.3% | 58.3% |
| False negatives | 13 | 13 |
| False positives | 256 | 176 |
| Studies flagged at that operating point | 74.1% | 62.5% |
| Model-specific threshold, rounded | 0.063774 | 0.263568 |

Values come from the [current comparison CSV](docs/results/checkpoint1/model_comparison.csv).

The paired patient-cluster bootstrap estimates a Model B minus Model A AUROC difference of **+0.032 (95% CI: +0.012 to +0.052)**.

The observed specificity improvement is not statistically supported on this cohort; threshold selection used the same development population.

See the [full Checkpoint 1 results](docs/results/checkpoint1/README.md) for uncertainty intervals, paired comparisons, figures, training history, and the secondary max-pooling diagnostic.

Mean probability pooling remains the primary Model B policy.

### Model B: internal versus official validation (pre-registered final run)

One pre-registered run of the frozen Model B and the deterministic decision layer on official MURA `valid`, XR_WRIST (237 studies: 97 ABNORMAL, 140 NORMAL; 207 patients). The plan is in [PREREGISTRATION.md](PREREGISTRATION.md), committed before the run (commit `7d86084`). This set was evaluated in Phase 1 and selected cases were inspected in Notebooks 09/09A before registration. The result is a repeat evaluation of an already-observed set, not an untouched test. The current model, aggregation, threshold and routing rule were frozen before the November case checks and were not changed because of them; see [pre-registration, section 6](PREREGISTRATION.md#6-single-run).

| Measure | Official `valid` [95% CI] | Internal validation (development) |
|---|---|---|
| Studies / patients | 237 / 207 | 691 / 653 |
| NORMAL / ABNORMAL studies | 140 / 97 | 422 / 269 |
| Sensitivity | 89.7% [82.8%, 95.3%] | 95.2% |
| Missed ABNORMAL studies (FN) | 10 of 97 | 13 of 269 |
| Specificity | 64.3% [56.2%, 72.7%] | 58.3% |
| False positives (FP) | 50 of 140 | 176 of 422 |
| Flagged | 57.8% | 62.5% |
| AUROC | 0.903 [0.859, 0.943] | 0.913 |
| AUPRC | 0.902 [0.850, 0.940] | 0.878 |
| Frozen threshold | 0.263567715883255 | 0.263567715883255 |

**The official sensitivity point estimate is below the 95% target.** Its 95% confidence interval includes 95%, so the pre-registered reading is compatible with the target, but below it in the point estimate; this does not establish that the target was achieved. Official intervals use 1,000 patient-cluster bootstrap resamples with the threshold fixed.

The threshold was selected on internal validation, so internal operating-point performance can be optimistic. Different case difficulty and sampling variation may also contribute to the difference; the official cohort is smaller. These are possible explanations, not established causes or a statistical test of the difference between cohorts.

The **deterministic decision layer** routed 159/237 studies (67.1%) to priority review and 78 to routine, including 7 ABNORMAL studies. View disagreement added 22 priority routes beyond the model's threshold flags; `SAFETY_BLOCK` was 0. All 38 execution checks passed. These checks and routes do not establish diagnostic correctness. The LLM/CrewAI explanation layer was **not run** in this evaluation.

Phase 1 on the same set is a historical reference, not a fair comparison (other models, split and thresholds): Model A missed 17 and Model B missed 26 of the 97 ABNORMAL studies.

Deviation from the pre-registration: the first attempt stopped while parsing the manifest, before any image was read by the model; four lines of manifest parsing in notebook 10 were fixed and committed before the run. Model, threshold, rule and metrics were not changed.

Full report: [reports/official_valid_final/report.md](reports/official_valid_final/report.md). Run code: [notebooks/10_MURA_Official_Valid_Final_Evaluation.ipynb](notebooks/10_MURA_Official_Valid_Final_Evaluation.ipynb).

---

## Notebook Map

| Notebook | Purpose | Status |
| --- | --- | --- |
| [01 — Data Preparation and EDA](notebooks/01_MURA_Data_Preparation_and_EDA.ipynb) | Patient/study/view organization, EDA and grouped split | Canonical workflow |
| [02 — Preprocessing](notebooks/02_MURA_Preprocessing.ipynb) | Preprocessing and manifests | Canonical workflow |
| [03 — Model A Baseline](notebooks/03_MURA_Baseline_ResNet18_LogReg.ipynb) | Frozen ResNet-18 embeddings and Logistic Regression | Canonical workflow |
| [04 — Model A Evaluation](notebooks/04_MURA_Baseline_Evaluation.ipynb) | Baseline evaluation | Canonical workflow |
| [05 — Model B](notebooks/05_MURA_FineTuned_ResNet18.ipynb) | Verify the existing fine-tuned checkpoint | Canonical workflow; keep `verify_existing` mode |
| [06 — Model B Evaluation and Comparison](notebooks/06_MURA_ModelB_Evaluation_and_Comparison.ipynb) | Evaluate and compare Models A and B | Canonical workflow |
| [06 — Results Viewer](notebooks/06_results_viewer.ipynb) | Read-only presentation of saved results | Supporting utility |
| [07 — Shortcut and Geometry Audit](notebooks/07_MURA_Shortcut_and_Geometry_Audit.ipynb) | Diagnostic audit | Diagnostic |
| [08 — Runtime Study Viewer](notebooks/08_MURA_Runtime_Study_Viewer.ipynb) | Study preview | Optional utility |
| [09 — MURA Agentic AI](notebooks/09_MURA_Agentic_AI-final.ipynb) | Frozen Model B, deterministic policy, TTA, Grad-CAM, CrewAI, and structured reporting | Implemented and tested; research only |
| [10 — Official Valid Final Evaluation](notebooks/10_MURA_Official_Valid_Final_Evaluation.ipynb) | Single pre-registered evaluation of frozen Model B and the deterministic decision layer on official MURA valid | Completed; see PREREGISTRATION.md |
| [Legacy Notebooks](notebooks/legacy/README.md) | Historical notebooks and earlier policy implementations | Archived |
| [Experimental All-Anatomy Notebooks](notebooks/experimental_all_anatomies/README.md) | Experimental extension beyond XR_WRIST | Outside wrist MVP |

The [notebook guide](notebooks/README.md) documents the workflow and dependencies.

The [per-notebook reviews](notebooks/reviews/README.md) are static reviews and do not themselves establish sequential execution.

Notebook 09 was previously executed end-to-end, separately from these static reviews.

---

## Use the Existing Model B Artifacts

The canonical external bundle is:

`DATA/MODEL_B_FINE_TUNING/`

Its checkpoint is:

`best_resnet18_study_level.pt`

The frozen operating configuration uses:

| Parameter | Value |
| --- | --- |
| Architecture | Fine-tuned ResNet-18 |
| Study aggregation | `mean_probability` |
| Frozen threshold | `0.263567715883255` |
| Threshold selection | Internal validation |
| Human review | Mandatory |

The [bundle specification](configs/model_b_artifact_bundle.json) records required files and the checkpoint checksum.

For the main ML environment, use Python 3.10 or 3.11 and install the [repository dependencies](requirements.txt):

```powershell
python -m pip install -r requirements.txt
```

Obtain the existing Model B bundle from authorized storage and verify it using the [artifact setup instructions](models/README.md).

Notebook 05 defaults to `verify_existing`.

**Do not rerun fine-tuning merely to set up or review this project.**

The following files must remain available and unchanged for current policy and bundle checks:

- [`tools/agent_policy.py`](tools/agent_policy.py) — deterministic agent policy, packaged version `1.1.0`.
- [`src/runtime/deterministic_safety.py`](src/runtime/deterministic_safety.py) — deterministic safety implementation.
- [`configs/model_b_artifact_bundle.json`](configs/model_b_artifact_bundle.json) — canonical bundle metadata.

Raw radiographs, model weights, private manifests, and sensitive study data must remain outside Git.

---

## Wrist Review Application

The runnable browser application is deliberately located in
[`apps/wrist_review/`](apps/wrist_review/), with an
[application-folder guide](apps/README.md) that distinguishes it from reusable
runtime code in `src/`, the agent assets in `agents/`, evaluation materials in
`evaluation/`, and final presentation assets in `presentation/`.

The [local application guide](apps/wrist_review/README.md) explains how to point `MURA_APP_MODEL_DIR` to an authorized local copy of the canonical bundle and launch the browser interface.

The application uses the verified Model B bundle and deterministic safety policy.

It checks technical image quality and declared anatomy. The anatomy selector is not an automatic body-part detector.

Missing or invalid artifacts prevent a model result. The deterministic route
always runs independently of an LLM. When explicitly enabled by a local operator,
the app can request the bounded, de-identified CrewAI research explanation that
implements the same fixed Model B and safety constraints described in Notebook 09;
it cannot change the score, threshold, priority route, or human-review requirement.

The app's optional LLM explanation is a runnable product integration of the
Notebook 09 design, not a new model-training or evaluation workflow.

---

## Notebook 09 — Agentic AI Research Workflow

**Developed and implemented by Hagit.**

[Notebook 09](notebooks/09_MURA_Agentic_AI-final.ipynb) implements an Agentic AI research workflow integrating frozen Model B predictions, deterministic policy decisions, optional explainability tools, CrewAI-based research explanations, and structured reports.

### Main Components

- **Model B integration:** Uses the existing frozen ResNet-18 checkpoint and operating threshold.
- **Agent adapter:** Converts model outputs into structured inputs for the agent workflow.
- **Deterministic policy:** Evaluates consistency, operating threshold compliance, and authorized tool selection.
- **TTA tool:** Provides test-time augmentation analysis when authorized.
- **Grad-CAM tool:** Produces visual explanations for model predictions.
- **CrewAI workflow:** Generates bounded research explanations based on available evidence.
- **Report generation:** Produces structured JSON research artifacts and PDF reports.
- **Safety and execution controls:** Include deterministic checks and bounded execution behavior.

The agent workflow does not retrain Model B or modify the frozen operating threshold.

Its outputs support research review and are not independent clinical diagnoses.

### Implementation and Testing

The Agentic AI workflow was developed and tested in Google Colab.

Technical testing included the deterministic policy, adapter, TTA, Grad-CAM, and report-generation components.

Notebook 09 was previously executed end-to-end.

The research workflow was also tested using selected wrist studies with and without optional explainability tools.

These technical tests do not establish clinical validation or generalization to an untouched official test population.

---

## Agentic AI — Python Modules

The six Python modules are stored in the [`tools/`](tools/) directory.

| File | Description |
| --- | --- |
| [`agent_policy.py`](tools/agent_policy.py) | Deterministic policy, consistency checks, and tool authorization |
| [`agent_policy_integration.py`](tools/agent_policy_integration.py) | Policy integration into the agent workflow |
| [`mura_agent_adapter.py`](tools/mura_agent_adapter.py) | Adapter between Model B outputs and the agent |
| [`tta_tool.py`](tools/tta_tool.py) | Test-time augmentation |
| [`gradcam_tool.py`](tools/gradcam_tool.py) | Grad-CAM explainability |
| [`report_generator.py`](tools/report_generator.py) | Structured research report generation |

---

## Agentic AI — Configuration

Configuration files are stored in:

[`configs/agentic_ai/`](configs/agentic_ai/)

| File | Description |
| --- | --- |
| [`mura_agent_specification.json`](configs/agentic_ai/mura_agent_specification.json) | Agent specification and workflow configuration |
| [`frozen_primary_operating_policy.json`](configs/agentic_ai/frozen_primary_operating_policy.json) | Frozen Model B operating policy |

The frozen policy uses `mean_probability` aggregation and threshold `0.263567715883255`.

These settings were selected using internal validation and must not be modified based on subsequent test results.

---

## Agentic AI — Research Outputs

Research outputs are organized under:

[`reports/agentic_ai/`](reports/agentic_ai/)

### Integrated Reports

Directory: [`reports/agentic_ai/integrated/`](reports/agentic_ai/integrated/)

| File |
| --- |
| `valid_XR_WRIST_patient11185_study1_positive_final_integrated_research_report.json` |
| `valid_XR_WRIST_patient11185_study1_positive_complete_agent_research_report.json` |
| `valid_XR_WRIST_patient11185_study1_positive_frozen_agent_result.json` |

These JSON files contain integrated research outputs for a selected wrist study.

### Research Artifacts

Directory: [`reports/agentic_ai/research_artifacts/`](reports/agentic_ai/research_artifacts/)

| File | Study |
| --- | --- |
| [`MURA_WRIST_RANDOM_001_artifact_v1.json`](reports/agentic_ai/research_artifacts/MURA_WRIST_RANDOM_001_artifact_v1.json) | RANDOM_001 |
| [`MURA_WRIST_RANDOM_002_artifact_v1.json`](reports/agentic_ai/research_artifacts/MURA_WRIST_RANDOM_002_artifact_v1.json) | RANDOM_002 |
| [`MURA_WRIST_NO_TOOLS_001_artifact_v1.json`](reports/agentic_ai/research_artifacts/MURA_WRIST_NO_TOOLS_001_artifact_v1.json) | NO_TOOLS_001 |

### PDF Research Reports

Directory: [`reports/agentic_ai/research_reports/`](reports/agentic_ai/research_reports/)

| File | Status |
| --- | --- |
| [`MURA_WRIST_RANDOM_001_research_report_with_metrics.pdf`](reports/agentic_ai/research_reports/MURA_WRIST_RANDOM_001_research_report_with_metrics.pdf) | Included — final metrics version |
| [`MURA_WRIST_RANDOM_002_research_report.pdf`](reports/agentic_ai/research_reports/MURA_WRIST_RANDOM_002_research_report.pdf) | Included |
| [`MURA_WRIST_NO_TOOLS_001_research_report.pdf`](MURA_WRIST_NO_TOOLS_001_research_report.pdf) |Included |

The research artifacts and the two available PDF reports have been uploaded to GitHub.

The older RANDOM_001 PDF without metrics is excluded from the final inventory.

---

## Agentic AI Dependencies

The Agentic AI components use a separate dependency file:

[`requirements_agent.txt`](requirements_agent.txt)

This file is maintained separately from the main [`requirements.txt`](requirements.txt) used for the ML pipeline.

The agent dependency installation was tested in a separate Google Colab environment.

| Verification | Result |
| --- | --- |
| Dependency installation | PASS |
| Eight selected Python library imports | 8/8 PASS |
| Environment-wide `pip check` | Six dependency issues reported |
| Full end-to-end execution in the fresh environment | Not verified |

The successful installation and import tests do not establish a completely conflict-free environment.

---

## Research Scope and Limitations

This project is intended for research and educational purposes.

Important limitations include:

- MURA abnormality labels do not establish a specific fracture diagnosis.
- The models do not provide validated lesion localization.
- Model performance reported here comes from internal validation, except the pre-registered final run on official MURA valid reported above.
- Official MURA `valid` was previously observed for historical reference and must not be described as an untouched final test for subsequent model selection. The pre-registered final run is a second look at that already-observed set.
- Agent-generated explanations may contain errors and require human review.
- Grad-CAM visualizations are explanatory aids, not validated lesion-localization outputs.
- The Agentic AI workflow does not replace professional clinical judgment.
- Human review is mandatory.

---

## Results, Governance, and Further Reading

- [Pre-registration](PREREGISTRATION.md) — plan of the single final run on official MURA valid, committed before the run.
- [Official-valid final run report](reports/official_valid_final/report.md) — pre-registered result with confidence intervals.
- [Full Current Results](docs/results/checkpoint1/README.md) — metrics, uncertainty, paired comparison, and diagnostic figures.
- [Test-Set Governance](reports/TEST_SET_GOVERNANCE.md) — official MURA validation-set history and limitations.
- [Result Lineage](docs/RESULT_LINEAGE.md) — current versus historical splits, operating policies, and model boundaries.
- [Documentation Index](docs/README.md) — reviewer materials and supporting documentation.
- [Reports Index](reports/README.md) — current results, historical evidence, experiments, governance, and reviews.
- [Project Charter](CHARTER.md) — project scope, success criteria, and team workstreams.

---

##
