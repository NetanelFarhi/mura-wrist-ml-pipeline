# MURA Agentic AI — File Inventory

## Overview

This inventory documents the Notebook 09 Agentic AI components and research outputs included in the GitHub upload package.

The system integrates frozen Model B predictions, deterministic policy evaluation, optional TTA and Grad-CAM tools, CrewAI-based research explanations, and structured research reporting.

**Scope:** Research use only. Human review is mandatory.

## 1. Notebook

| File | Description |
|---|---|
| `notebooks/09_MURA_Agentic_AI-final.ipynb` | Final Notebook 09 implementing the Agentic AI workflow. |

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

| File | Status |
|---|---|
| `MURA_WRIST_RANDOM_001_research_report_with_metrics.pdf` | Included; metrics version retained |
| `MURA_WRIST_RANDOM_002_research_report.pdf` | Included |
| `MURA_WRIST_NO_TOOLS_001_research_report.pdf` | Included |

## 7. Dependencies

| File | Description |
|---|---|
| `requirements_agent.txt` | Python dependencies for the Agentic AI components, maintained separately from the main ML requirements. |

Installation and eight core library imports passed in a separate Colab environment. The environment-wide `pip check` reported six dependency issues; a fully conflict-free environment has not been verified.

## 8. Additional Package Documentation

The prepared upload archive also contains:

| File | Description |
|---|---|
| `GITHUB_UPLOAD_GUIDE.md` | Manual upload and repository organization guide. |
| `UPLOAD_MANIFEST.json` | Package manifest for tracking included files. |

These two files are part of the prepared archive and are not yet confirmed as uploaded to GitHub.

## Validation and Limitations

- Notebook 09 was previously executed successfully end-to-end.
- The six Python modules underwent technical testing.
- Frozen Model B configuration must remain unchanged.
- Outputs are research artifacts, not clinical diagnostic reports.
- Official validation performance must not be inferred from internal validation results.
- The missing NO_TOOLS_001 PDF will be addressed separately.

**Inventory scope:** Notebook 09 Agentic AI upload package only; not the entire MURA project repository.
