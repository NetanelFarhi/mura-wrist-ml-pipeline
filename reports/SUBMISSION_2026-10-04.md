# Charter-submission alignment checklist — 4.10.2026

This checklist separates material that is ready to submit as **design/protocol** from work that remains an implementation or evaluation task. It must not be used to imply that planned agent work or future advanced-model testing has occurred.

## Charter packet

- [x] Seven explicit charter fields in `CHARTER.md`.
- [x] Specific user, scoped MURA-abnormality task, measurable >=95% sensitivity target, and bootstrap-CI requirement.
- [x] Patient-level internal train/validation design; Logistic Regression C selected inside train by grouped CV; model-specific study-level cutoff chosen only on internal validation.
- [x] Honest governance note: official MURA `valid` was historically opened for frozen Phase-1 reference evaluation and cannot be called untouched for later model selection.
- [x] Mandatory frozen-embedding Logistic Regression baseline named alongside fine-tuned ResNet-18.
- [x] Deterministic safety policy, bounded adaptive-agent seat, agent-evaluation comparator design, Plan B distinction, risks/cut line, and named hats.
- [x] Research/non-commercial data governance statement: raw MURA images, credentials, and protected manifests are not committed.

## Verification completed for this branch

1. The current Charter was rendered as `docs/submissions/MURA_Agentic_AI_Capstone_Charter_final.docx` and `.pdf`; the PDF is one page.
2. The rendered packet contains no raw radiograph or protected-data image.
3. Unverified clinical/background claims were removed or rephrased; outstanding source checks remain tracked in `reports/SOURCE_VERIFICATION.md` rather than presented as success claims.
4. The repository data-governance check must pass immediately before every push, including this branch.
5. The branch does not state that the bounded agent is implemented/evaluated, clinically validated, or that advanced models have an untouched final test result.

## Checkpoint distinction

The thin end-to-end baseline is the early deliverable. The working agent, agent evaluation, advanced model claims, UI/demo, and deployment cost estimate are later work. Historical Phase-1 results stay preserved, but later model selection must not use the already observed official MURA `valid` labels.
