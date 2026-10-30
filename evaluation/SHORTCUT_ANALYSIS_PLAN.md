# Shortcut and error-analysis plan — geometry diagnostic completed; manual/Grad-CAM work planned

## Scope and boundary

MURA supplies a study-level NORMAL/ABNORMAL label, not reliable localization or clean subtype labels for all causes of abnormality. Therefore this plan is a diagnostic/error-analysis protocol, not proof that a model recognized a lesion or a specific pathology. Any manual review is exploratory, small-sample, and clearly labelled as such.

## Current evidence status

A non-selection geometry/shortcut diagnostic has already been completed on the development workflow. It tested whether simple geometry features carried label-associated information and whether saved-model scores or decisions changed under controlled padding, border, and resize perturbations. Its saved results are in [the checkpoint shortcut-audit report](../docs/results/checkpoint1/shortcut_audit.md). This is a warning/sensitivity check only: it does **not** prove that a model used a shortcut, identify a particular marker or border as the cause, establish lesion localization, or show clinical relevance.

No manual visual annotation study and no Grad-CAM review has been executed as part of this evidence. The remaining items below are therefore prospective analysis steps, not completed findings.

## Remaining candidate shortcuts to inspect

| Candidate signal | Planned evidence | Permitted conclusion |
|---|---|---|
| L/R markers and text overlays | blinded visual audit; optional counterfactual masking only if pre-specified | possible shortcut sensitivity, not clinical relevance |
| Borders, collimation and acquisition artifacts | stratify errors/QC flags; inspect representative cases | association only |
| Metal/hardware and casts | record visible presence during manual review when unambiguous | possible reliance or dataset confounding, not abnormality subtype ground truth |
| Old/degenerative-looking patterns | do not infer labels; include only if a qualified reviewer can defensibly annotate the visible pattern | exploratory descriptive finding only |
| View count and view disagreement | predefined study-level subgroup analysis | operational association, not causation |
| Cropping, rotation, exposure/other QC artifacts | QC statistics and paired error table | input-quality limitation, not diagnosis |

## Grad-CAM boundary and status

Grad-CAM has not been executed yet. If implemented later, it is used primarily as a model-diagnostic aid to inspect whether coarse model evidence appears to fall on anatomy versus markers/text/borders. ResNet-18's final feature map is coarse, and MURA has no localization ground truth. Reports must never say that a heatmap proves the model detected a lesion in a named bone.

## Procedure and reporting

Predefine a small, balanced sample across correct cases, false negatives, false positives, low/high view disagreement, and QC-flagged cases. Keep reviewers blinded to model label/probability when feasible; record a simple codebook, reviewer/date, image IDs, ambiguity, and whether the signal is visibly present. Report sample size, missingness, agreement, and all exclusions. Do not fabricate subtype ground truth or generalize exploratory manual observations to the whole dataset.
