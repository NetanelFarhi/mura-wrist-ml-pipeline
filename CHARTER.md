# MURA XR_WRIST Capstone Charter

**Submission:** 4.10.2026 | **Team:** Artium, Hagit, Hodaya, Netanel

## 1. Problem + user
User: Emergency Department physician reviewing wrist radiographs before expert review. The MVP flags **MURA-defined musculoskeletal abnormality** per study (NORMAL/ABNORMAL), not acute fracture, lesion location, treatment, or autonomous diagnosis. In the historical Phase-1 reference, selected operating points missed 17 or 26 of 97 abnormal studies; this makes the sensitivity/specificity trade-off measurable. Delayed review is clinical motivation only, because this dataset cannot measure ED revisits/outcomes.

## 2. Success metric
**Pre-specified target:** study-level sensitivity >=90%. During development, choose the highest threshold meeting that floor on internal patient-level validation, freeze it before the defined evaluation, then report specificity, precision, F1, AUROC, AUPRC, FN rate (=1-sensitivity among abnormal studies), and 95% patient/study bootstrap CIs. NPV is not primary because prevalence-dependent. This is a target, not an untouched advanced-model claim; the future advanced-model evaluation protocol remains an open methodological decision.

## 3. Data + Plan B
Plan A: Stanford MURA v1.1, research/non-commercial, `XR_WRIST`: local access was verified and indexed as 3,474 patients, 3,697 studies and 10,411 images ([audit](reports/wrist_data_audit.json)); patient -> study -> 1-8 views, one NORMAL/ABNORMAL study label. Official train -> patient-level internal train/validation; official valid was used once as frozen Phase-1 reference after validation-only choices, and is now observed/not untouched for later selection. No raw images in Git.

Plan B: FracAtlas only if access fails; it changes the task to fracture detection plus anatomy/QC/evaluation assumptions and requires documented adaptation.

## 4. Architecture sketch
`Data -> Patient/Study/View -> Model -> declared study aggregation -> probability/evidence -> deterministic safety/routing -> optional ambiguity investigation -> bounded adaptive agent -> workflow escalation -> human clinician.`

The deterministic system works without agent: supported anatomy, corruption/QC rejection, mandatory review rules, safety thresholds. AI may raise priority; never lower a case below standard review. Historical aggregation: Model A mean embeddings; Model B mean probability; learned aggregation remains experimental.

## 5. Agent seat
Known rules remain deterministic. The bounded agent only investigates ambiguity: conflicting view/model evidence, borderline QC, whether permitted uncertainty/TTA is worthwhile, missing evidence, and grounded summary. It cannot diagnose/treat, override safety, or alter thresholds; it may return `NEEDS_HUMAN_REVIEW` or `UNSUPPORTED`.

Tools: untrusted metadata, QC/support checker, primary model, per-view disagreement, approved uncertainty/TTA, Grad-CAM diagnostic, model card, calculator. It is evaluated against rule and fixed-workflow baselines on the same curated cases for escalation, groundedness, evidence, tool use/cost, unsupported handling and violations. Metadata instructions never alter policy.

## 6. Risks + cut line
1. MURA abnormality != acute fracture: honest scope; exploratory errors only. 2. Shortcuts (markers/text/borders/hardware): Grad-CAM diagnostic + targeted review; no localization claims. 3. Agent wrapper/hallucination: non-overridable rules, hostile-input eval, bounded tools/failure-to-review. 4. Small evaluation: bootstrap CIs/no over-precision.

**Cut line:** reproducible wrist-abnormality classifier, cheap baseline comparison, pre-specified operating point, error/shortcut analysis, deterministic safety policy, separately evaluated bounded agent. Optional advanced ML/UI cut first.

## 7. Milestones + hats
| Checkpoint | Deliverable | Ownership |
|---|---|---|
| 18.10.2026 | Thin end-to-end baseline without agent | Data/Model/Product: [NAME REQUIRED] |
| 30.10.2026 | Guarded agent integrated + evaluated | Agent: [NAME REQUIRED]; presenters: [NAME REQUIRED] |
| 12.11.2026 | ML/error analysis, rule-vs-agent eval, demo/docs/cost | Data/Model/Agent/Product: [NAME REQUIRED] |

Mandatory cheap baseline: frozen ImageNet ResNet-18 embeddings + study aggregation + Logistic Regression, fairly compared with fine-tuned ResNet-18 on same population/metrics. Preliminary results are historical, not future untouched-test evidence.
