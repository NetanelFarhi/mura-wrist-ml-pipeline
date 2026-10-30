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

## Illustrative future research-pilot scenario

This is a transparent **illustrative monthly scenario in USD**, not a vendor
quotation, clinical-deployment estimate, or current project bill. It assumes
40 GPU inference/QC hours per month, 100 GB of object storage, and (only if a
future bounded text-summary agent is enabled) 1,000 text-only summaries per
month. It assumes that no radiograph or patient identifier is sent to an LLM.

| Component | Assumption and arithmetic | Illustrative monthly amount |
|---|---|---:|
| GPU compute | 40 hours × $0.7364/hour | $29.46 |
| Object storage | 100 GB × $0.023/GB-month | $2.30 |
| Optional text-only LLM summaries | 1,000 calls × (3,000 input + 750 output tokens), using an example $1/M-input and $5/M-output token rate | $6.75 |
| **Illustrative subtotal with optional LLM** | compute + storage + optional summaries | **$38.51/month** |
| **Illustrative subtotal without LLM** | compute + storage only | **$31.76/month** |

The compute example uses the published AWS SageMaker `ml.g4dn.xlarge`
illustration, and the storage example uses the published S3 Standard rate;
the optional LLM line is a token-based example that must be refreshed against
the selected provider before procurement. Prices vary by region, usage,
discounts, taxes, service requests, and date. This scenario excludes data
transfer, disks, logging, backups, monitoring, security/compliance,
integration, and human-review costs. The current repository has no deployed
LLM, cloud service, or actual cloud bill.

Sources checked 2026-10-30:

- [AWS SageMaker pricing](https://aws.amazon.com/sagemaker/ai/pricing/)
- [AWS S3 Standard pricing example](https://docs.aws.amazon.com/pdfs/solutions/latest/live-streaming-on-aws-with-amazon-s3/live-streaming-on-aws-with-amazon-s3.pdf)
- [AWS pricing-calculator assumptions](https://aws.amazon.com/calculator/calculator-assumptions/)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
