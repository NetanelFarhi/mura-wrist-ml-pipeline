# Source-verification register

No external web research was performed for this documentation update. A claim marked **NEEDS VERIFICATION** must be checked against a primary or official source before it is used as factual background in a final submission. If it cannot be verified, remove it or rephrase it as a project-design statement.

| Claim or intended wording | Status | Required action |
|---|---|---|
| This project uses local MURA-derived XR_WRIST manifests organised patient -> study -> views, with study-level NORMAL/ABNORMAL labels. | VERIFIED in local project artifacts | Keep the scoped wording; do not change it to fracture detection. |
| Official MURA `train` was split internally at patient level for development. | VERIFIED in local split/audit artifacts | Retain with the exact split protocol. |
| Official MURA `valid` was opened once for frozen Phase-1 reference evaluation. | VERIFIED in `reports/TEST_SET_GOVERNANCE.md` | Retain the reuse limitation in all later reports. |
| MURA access/use is subject to research or non-commercial terms. | NEEDS VERIFICATION | Cite the current official Stanford MURA terms before final submission. |
| MURA abnormality can include non-acute-fracture findings such as hardware, casts, degenerative change, markers, or artifacts. | NEEDS VERIFICATION | Verify against the MURA paper/documentation before using as a factual distribution claim; treat shortcut review as exploratory unless annotated. |
| Missing radiographic abnormalities or delayed specialist review are clinically important. | NEEDS VERIFICATION | Use a reliable clinical source or keep this only as non-quantified motivation. |
| A 3% miss target is clinically established or appropriate for this dataset. | REMOVE OR REPHRASE | No project source establishes this. Use the pre-specified >=95% sensitivity protocol instead. |
| “<1% 24-hour ED revisit” is a measurable project outcome on MURA. | REMOVE OR REPHRASE | MURA does not contain this outcome; it is not a success metric. |
| Graif/revisit citation supports the project's success metric. | REMOVE OR REPHRASE | Verify the exact citation before any background use; never use it as model-performance evidence. |
| Published original-MURA wrist results can anchor context for this project. | NEEDS VERIFICATION | Cite the primary MURA paper and explicitly state that dataset/protocol/model comparisons are not identical. |
| FracAtlas is a compatible Plan B with stated licence/labels. | NEEDS VERIFICATION | Verify the current source, licence, anatomy, labels, and task compatibility before activation. |
| The AI may raise priority but must not lower a case below standard review. | VERIFIED AS PROJECT POLICY | Describe as a safety rule, not proof of clinical effectiveness. |
| The system can replace radiology review, diagnose a fracture, localize a lesion, or improve patient outcomes. | REMOVE OR REPHRASE | Outside this dataset and evidence base. |
