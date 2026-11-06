# MURA Agentic AI — Steps 81–88

**Project:** MURA XR_WRIST abnormality classification and policy-controlled research explanation  
**Environment:** Google Colab / Python / CrewAI / Gemini  
**Documentation status:** Reconstructed from observed notebook outputs and troubleshooting discussions through 2026-11-06. This is **not** a line-by-line audit of the complete notebook source.

## 1. Purpose and scope

These steps integrate a frozen ResNet18 study-level classification with a deterministic safety/workflow policy and a CrewAI research explanation workflow. The language model may summarize supplementary evidence but **must not alter the classifier's predicted class, probability, threshold, or workflow action**. Outputs are for research only, are not clinical diagnoses, and require human review.

The work in this interval covered blind-study testing, official-validation error-case replays, tool registration, report validation, evidence restoration, archiving, and an inventory of wrist studies with four images.

## 2. End-to-end conceptual workflow

```text
MURA XR_WRIST study
    -> QC / anatomy checks and frozen ResNet18 study result
    -> deterministic policy (workflow action + authorized tools)
    -> protected TTA / Grad-CAM evidence tools, if authorized
    -> CrewAI research explanation agent using Gemini
    -> deterministic trusted report envelope + validation
    -> archived Markdown / JSON + audit qualifications
    -> mandatory human review
```

**Important:** This is the intended workflow. A successful restored-report validation is not proof that a fresh CrewAI run occurred or that all tool calls were independently audited.

## 3. Key system components

| Component | Current evidence / status |
|---|---|
| Trigger | Study classification and policy decision are available; full automatic trigger wiring has not been audited here. |
| Frozen classifier | ResNet18 study classification; threshold `0.0600893441587686` was used in the documented blind-study validation. |
| Policy engine | Deterministic workflow and tool authorization; official FP case uses `priority_human_review`. |
| LLM | Gemini via CrewAI; used in earlier runs. No new Gemini calls were needed during restoration. |
| CrewAI tools | `mura_tta_summary` and `mura_gradcam_summary`. |
| Policy tool IDs | `tta_uncertainty` and `gradcam_explainer`; explicitly map to the CrewAI names. |
| Memory | Not yet audited; do not assume persistent memory is enabled. |
| Agent loop / limits | CrewAI objects exist; `max_iter`, token caps, retries, and actual loop behavior still require a configuration audit. |
| Reports | Trusted metadata envelope, lexical/numerical checks, JSON/Markdown archive. |
| Human oversight | Mandatory; research-only restrictions retained. |

## 4. Step-by-step record

### Step 81 — Frozen blind-study safety baseline

A four-image synthetic/controlled blind case, `BLIND_STUDY_001`, was used in the earlier workflow. The preserved classification was compared against a frozen snapshot (`blind_result == frozen_before_81k`). This case is **not** interchangeable with an official MURA validation study.

The exact full set of Step 81 cells and their definitions was not reviewed for this README.

### Steps 82–83 — CrewAI execution, protected tools, and validation

For `BLIND_STUDY_001`, the observed trusted result included:

- Predicted class: `1`
- Study probability: `0.545678123831749`
- Frozen threshold: `0.0600893441587686`
- Workflow action: `priority_human_review`
- Four image-level Grad-CAM entries

The Gemini report used rounded TTA values and did not follow the exact labels required by the numerical validator. A revised **Step 83X.3** appended deterministic, high-precision evidence from trusted tool results while preserving the original Gemini text. Its nine numerical checks passed (`9/9`). **Step 83W** was then updated to validate this combined report and recorded `VALIDATED` with ten checks passing (`10/10`). **Step 83Y** archived the post-run audit with a unique timestamp to avoid overwriting an existing file.

These results validate the combined report against the specified checks. They do **not** establish an independent audit of every tool call.

### Step 84 — Official-validation policy replays

Two saved-prediction cases were compared: a false negative and a false positive. The policy replay comparison (Step 84T) produces CSV and JSON artifacts without model inference or Gemini calls. The output filenames were changed to include a timestamp so repeat notebook runs do not fail when files already exist.

Observed official cases from the earlier work:

| Case | Study | Ground truth | Predicted class | Study probability | Policy outcome |
|---|---|---:|---:|---:|---|
| False positive | `OFFICIAL_STUDY_00137` | 0 | 1 | `0.09145868569612503` | Priority human review; supplementary TTA and Grad-CAM authorized |
| False negative | `OFFICIAL_STUDY_00005` | 1 | 0 | `0.027500247582793236` | Routine workflow; no supplementary tools authorized |

The false-negative CrewAI run recorded no tool calls; its historical audit status remained `NOT_INDEPENDENTLY_VERIFIED`.

### Steps 85F–85H — Official false-positive evidence restoration

The restored study is `OFFICIAL_STUDY_00137`.

- `trusted_result_85f`: frozen classification.
- `policy_85f`: successful policy decision with `selected_tools = ['tta_uncertainty', 'gradcam_explainer']`.
- CrewAI tool mapping: `tta_uncertainty -> mura_tta_summary`, `gradcam_explainer -> mura_gradcam_summary`.

During a full notebook rerun, the earlier 85G/85H cells referenced missing variables (`trusted_result_85g`, `report_85g`, `agent_85h`, `crew_85h`, `inputs_85h`). Rather than regenerate a Gemini report, the workflow was revised to restore saved evidence and reconstruct the two tool wrappers. **Step 85G** subsequently completed successfully; **Step 85H** verified the restored evidence and registered tool names without executing CrewAI.

**Critical notebook order:** Step **85I.1** and **85I.2** must execute **before** the revised 85G and 85H, because 85G depends on the saved report and `validation_85i2`. Step numbers alone do not define executable dependency order.

### Steps 85I.1–85I.2 — Saved-report verification

An existing report was located:

`/content/drive/MyDrive/DATA/MURA_AGENTIC_AI/reports/OFFICIAL_STUDY_00137_crewai_20261105T205030Z.json`

A matching Markdown file exists with the same basename. The JSON includes `trusted_classification`, `policy`, `tta_evidence`, `gradcam_metadata`, `gemini_original_report`, `combined_report`, and audit qualifications.

Step 85I.2 checked study identity, class, probability, threshold, workflow action, policy tools/version/status, original TTA probability, Grad-CAM metadata presence, and research/human-review restrictions. **Result: 12/12 PASS; `VALIDATED`.** The saved Grad-CAM metadata has one entry. The historical tool-execution audit remains **`NOT_INDEPENDENTLY_VERIFIED`**.

### Step 85K — Deterministic report envelope

The original Gemini text was loaded from the saved JSON rather than from a new CrewAI execution. The original text alone was `REJECTED` by the report validator. A Python-generated trusted header was prepended, preserving the original Gemini text unchanged. The combined report passed **10/10 content checks**, including identity, class, probability, workflow action, TTA, Grad-CAM, limitations, human review, and research-only status. `report_accepted_85k = True`.

The header is trusted metadata; it does not independently verify the original model-generated explanation.

### Step 85L — Archive verification

The revised archive cell defaults to `SAVE_NEW_COPY_85L = False`. It verified the existing JSON and Markdown files agree, confirmed `VALIDATED` report status and frozen-classification preservation, and made **no new Gemini calls or file modifications**. A new copy is created only on explicit opt-in.

### Steps 86–87 — Execution and usage observations

Earlier runs included a successful false-positive research report with supplementary tools and a false-negative report without authorized tool calls. Step 87 measured reported CrewAI usage for the false-negative case:

| Metric | Reported value |
|---|---:|
| Prompt tokens | 4,534 |
| Completion tokens | 1,386 |
| Total tokens | 5,920 |
| Cached tokens | 0 |
| Successful requests | 6 |

CrewAI's Gemini token accounting may be approximate. Six requests do **not** necessarily mean six agent reasoning iterations. These figures should not be treated as a four-image-study benchmark.

### Step 88 / 88A — Original ZIP study inventory

A reduced/derived table did not identify suitable four-image studies. Step 88A inspected the original MURA ZIP and found **237 XR_WRIST studies**, including **32 studies with exactly four images**.

| Images per study | Number of studies |
|---:|---:|
| 1 | 24 |
| 2 | 44 |
| 3 | 133 |
| 4 | 32 |
| 5 | 4 |

One four-image candidate is `patient11222/study2_negative` (`true_label = 0`). Before any new Gemini run, check whether a saved frozen prediction and policy result exist for the chosen study. A four-image study from the original ZIP is distinct from `BLIND_STUDY_001`.

## 5. Notebook reproducibility and safety rules

1. **Do not overwrite** historical audit artifacts. Use timestamped filenames or explicit reuse of existing archives.
2. **Avoid accidental Gemini reruns.** Restoration and validation cells must not call `kickoff`, `kickoff_async`, or another live LLM execution method.
3. **Enforce dependencies.** The revised Step 85I.1 and 85I.2 run before 85G, 85H, 85K, and 85L.
4. **Never mix studies.** Frozen classification, policy, and evidence must share the same study ID and pass the applicable numerical checks.
5. **Keep policy authorization separate from tool registration.** A registered tool is not automatically authorized; runtime enforcement requires separate verification.
6. **Preserve audit qualifications.** `NOT_INDEPENDENTLY_VERIFIED` must not be upgraded based solely on a validated report.
7. **Preserve human review and research-only limitations.** Neither TTA nor Grad-CAM is a validated clinical diagnostic or lesion-localization claim.

## 6. Open architecture questions — proposed Step 89

The following are **not yet established** by the completed Step 81–88 checks:

- Whether CrewAI memory is enabled, and whether memory is isolated per study.
- The actual agent loop configuration (`max_iter`, execution time, retry behavior, and delegation).
- Whether there is an enforced LLM token/cost budget; the previously measured usage is descriptive, not a cap.
- Whether all tool calls are independently audited and policy-checked at execution time.
- Whether the trigger-to-report pipeline is automatically orchestrated end-to-end, rather than invoked by notebook cells.

**Recommended next action:** perform a read-only architecture audit of the actual Agent, Task, Crew, tools, and policy objects; then benchmark a selected four-image study only if its existing prediction and authorization data are verified. Avoid enabling persistent memory by default in a medical research workflow without explicit data-isolation rules.

## 7. Scope of verification

This README summarizes outputs and code fragments observed during troubleshooting. It does not claim to have inspected every cell from Steps 81–88, reproduced all experiments, or independently audited Gemini requests, tool events, or source images. The complete notebook and archived artifacts remain the primary reproducibility records.
