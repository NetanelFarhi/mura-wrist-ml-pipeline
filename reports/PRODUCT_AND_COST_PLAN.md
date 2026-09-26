# Product and future-cost plan — design only

## User, workflow, and value under test

**Primary user:** an Emergency Department physician reviewing a wrist radiograph study before expert radiology review is available.

**Intended future workflow:** intake validates a supported wrist study and image quality; the study-level model and deterministic safety policy may raise priority for human review; a bounded agent may only investigate genuine ambiguity and produce a grounded evidence summary; the clinician remains responsible for every clinical decision. The system never replaces standard review or provides autonomous diagnosis/treatment.

**Value hypothesis:** a high-sensitivity, well-governed study-level abnormality flag and concise evidence trail could help a clinician identify studies that deserve prompt human attention. This project tests workflow safety and ML discrimination on MURA-defined abnormality; it does not measure diagnosis time, ED revisits, treatment benefit, or deployment readiness.

## What real deployment would require

Clinical validation on an appropriate independently governed population, legal/privacy/security review, DICOM/PACS and identity-integration design, cybersecurity and hostile-input testing, monitoring/drift/incident response, clinician usability validation, human-oversight protocol, model/agent version governance, and local regulatory/institutional approvals.

## Transparent monthly-cost framework

No production price is claimed because study volume, storage retention, runtime, vendor, region, and model/API choices are unknown. Before a later cost estimate, record these assumptions: studies/month (`N`), images/study (`V`), retained GB/study (`S`), inference seconds/study (`I`), optional-agent fraction (`A`), and LLM tokens/case (`T`). Then report these categories separately:

| Category | Estimation basis to record later |
|---|---|
| Compute | CPU/GPU inference, QC, optional TTA, monitoring; `N × I` and chosen infrastructure price |
| Storage and retention | images/logs/outputs; `N × S × retention period` plus backup |
| Integration and security | PACS/DICOM interface, authentication, audit logs, vulnerability management |
| LLM/API, only if enabled | `N × A × T` under the selected vendor/model price and privacy agreement |
| Operations | monitoring, human review, support, model/agent evaluation and incident response |

The future cost report must disclose its assumptions, distinguish one-time implementation from recurring cost, and compare agent-enabled versus deterministic-only workflows.
