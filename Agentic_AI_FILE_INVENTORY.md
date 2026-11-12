# MURA Agentic AI — File Inventory

## Overview

This inventory documents the Notebook 09 Agentic AI components, the independent Model B verification notebook (09A), and associated research outputs prepared for the GitHub upload package.

The workflow integrates frozen Model B predictions, deterministic policy evaluation, authorized TTA and Grad-CAM tools, CrewAI-based research explanations, and structured research reporting.

**Scope:** Research use only. Human review is mandatory.

## 1. Notebooks — `notebooks/`

| File | Description |
|---|---|
| `09_MURA_Agentic_AI-final.ipynb` | Final Notebook 09 implementing the Agentic AI workflow and documenting policy-guided tool execution. |
| `09A_MURA_Ground_True_Independent_Model_B_Validation.ipynb` | Independent verification of frozen Model B on three XR_WRIST studies: eight image-level predictions, original-image identity checks, ground-truth outcomes, false-positive analysis, and Grad-CAM visualization. Confirm the exact exported notebook filename before GitHub upload. |

### Notebook 09A — Verified results

The notebook was rerun after disconnecting and reconnecting the Colab runtime. Its reported checks reproduced all eight image-level probabilities within numerical tolerance (maximum absolute difference approximately `4.81316e-07`).

| Study | Ground truth | Model B prediction | Outcome | Identity |
|---|---|---|---|---|
| `MURA_WRIST_RANDOM_001` | NORMAL | ABNORMAL | False Positive | `VERIFIED_MAPPING` |
| `MURA_WRIST_RANDOM_002` | ABNORMAL | ABNORMAL | True Positive | `VERIFIED_MAPPING` |
| `MURA_WRIST_NO_TOOLS_001` | NORMAL | NORMAL | True Negative | `VERIFIED_MAPPING` |

The two NO_TOOLS images matched the decoded grayscale pixels of their originals in `DATA/MURA-v1.1_files.zip` exactly; their file SHA-256 hashes differed. STEP 7 creates `df_ground_truth`, and STEP 7A confirms NO_TOOLS image identity. STEP 7B shows all three mappings verified. STEP 8 reproduced the RANDOM_001 false-positive classification and image-level threshold disagreement. STEP 9A found all six original/Grad-CAM image files; STEP 9B images were visually checked and reported as displaying correctly.

For RANDOM_001, the frozen study probability was approximately `0.405498`, above the threshold `0.263567715883255`, although the original ground truth was NORMAL. Notebook 09 documents `priority_human_review` and successful execution of Grad-CAM and TTA without changing Model B's classification. The exact deterministic policy trigger still requires inspection of its implementation.

**Scope of verification:** Three selected studies and eight images; this is not a full official MURA validation-set performance evaluation. The independent notebook reproduces Model B inference; evidence of CrewAI tool execution comes from Notebook 09's documented run, not a new agent execution in 09A.

## 2. Python Modules — `tools/`

| File | Description |
|---|---|
| `agent_policy.py` | Deterministic agent policy, tool selection and consistency checks. |
| `agent_policy_integration.py` | Integration of policy evaluation into the agent workflow. |
| `mura_agent_adapter.py` | Adapter connecting Model B outputs with the agent workflow. |
| `tta_tool.py` | Test-time augmentation tool for authorized analysis. |
| `gradcam_tool.py` | Grad-CAM explainability tool. |
| `report_generator.py` | Structured research report generation. |

## 3. Configuration — `configs/agentic_ai/`

| File | Description |
|---|---|
| `mura_agent_specification.json` | Agent specification and configuration. |
| `frozen_primary_operating_policy.json` | Frozen Model B operating policy and decision threshold. |

The frozen Model B threshold is `0.263567715883255`, with `mean_probability` study-level aggregation. The operating threshold was selected using internal validation.

## 4. Integrated Reports — `reports/agentic_ai/integrated/`

| File |
|---|
| `valid_XR_WRIST_patient11185_study1_positive_final_integrated_research_report.json` |
| `valid_XR_WRIST_patient11185_study1_positive_complete_agent_research_report.json` |
| `valid_XR_WRIST_patient11185_study1_positive_frozen_agent_result.json` |

These files document integrated agent research outputs for a selected wrist study.

## 5. Research Artifacts — `reports/agentic_ai/research_artifacts/`

| File | Study |
|---|---|
| `MURA_WRIST_RANDOM_001_artifact_v1.json` | RANDOM_001 |
| `MURA_WRIST_RANDOM_002_artifact_v1.json` | RANDOM_002 |
| `MURA_WRIST_NO_TOOLS_001_artifact_v1.json` | NO_TOOLS_001 |

## 6. PDF Research Reports — `reports/agentic_ai/research_reports/`

| File | Inventory status |
|---|---|
| `MURA_WRIST_RANDOM_001_research_report_with_metrics.pdf` | Listed in prepared package; metrics version retained |
| `MURA_WRIST_RANDOM_002_research_report.pdf` | Listed in prepared package |
| `MURA_WRIST_NO_TOOLS_001_research_report.pdf` | Listed in prepared package; verify actual repository upload |

## 7. Dependencies

| File | Description |
|---|---|
| `requirements_agent.txt` | Python dependencies for the Agentic AI components, maintained separately from the main ML requirements. |

Installation and eight core library imports passed in a separate Colab environment. The environment-wide `pip check` reported six dependency issues; a fully conflict-free environment has not been verified.

## 8. Additional Package Documentation

| File | Description |
|---|---|
| `GITHUB_UPLOAD_GUIDE.md` | Manual upload and repository organization guide. |
| `UPLOAD_MANIFEST.json` | Package manifest for tracking included files. |

These files were listed in the prepared archive; their presence in GitHub has not been independently confirmed.

## Validation and Limitations

- Notebook 09 was previously executed successfully end-to-end, according to its documented run.
- The six Python modules underwent technical testing.
- Notebook 09A's reported independent verification was rerun in a fresh Colab runtime, with STEP 7 initialization present and successful; the earlier missing-STEP-7 concern is resolved for the tested Colab notebook.
- Frozen Model B configuration and predictions must remain unchanged by the agentic workflow.
- Grad-CAM and TTA are supplementary research tools, not proof of diagnostic correctness.
- Outputs are research artifacts, not clinical diagnostic reports.
- Official validation performance must not be inferred from these three selected studies or from internal validation results.
- Verify filenames and actual repository presence after uploading to GitHub.

**Inventory scope:** Notebook 09 Agentic AI package and Notebook 09A independent verification; not the entire MURA project repository.
