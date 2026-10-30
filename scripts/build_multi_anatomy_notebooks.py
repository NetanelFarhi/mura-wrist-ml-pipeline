#!/usr/bin/env python
"""Create executed, aggregate-only notebooks for the all-anatomy baseline.

The notebooks are presentation layers over local development artifacts.  They
do not embed MURA pixels, patient identifiers, raw predictions, model weights,
or the official MURA valid split.  The underlying training implementation is
``run_multi_anatomy_baseline.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


SETUP = """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
RUNTIME = ROOT.parent / 'runtime' / 'multianatomy_v1'
MANIFESTS = RUNTIME / 'manifests'
QC = RUNTIME / 'qc'
BASELINE = RUNTIME / 'baseline_all_v1'
assert ROOT.name == 'mura-wrist-ml-pipeline', ROOT
print('Repository: local project checkout')
print('Runtime artifacts: local only (not committed to Git)')
"""


def data_preparation_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md(
            "# 10 — MURA all-anatomy data preparation\n\n"
            "This local development extension covers all seven MURA anatomy categories. "
            "It uses **official MURA train only** and creates a single internal patient-level train/validation split. "
            "Official MURA `valid` is deliberately excluded."
        ),
        code(SETUP),
        code(
            "summary = pd.read_csv(MANIFESTS / 'development_split_summary.csv')\n"
            "provenance = json.loads((MANIFESTS / 'development_provenance.json').read_text(encoding='utf-8'))\n"
            "assert provenance['official_valid_read'] is False\n"
            "assert provenance['test_used_for_selection'] is False\n"
            "display(Markdown('**Data-governance check passed:** official MURA `valid` was not read.'))\n"
            "display(summary)"
        ),
        code(
            "pivot = summary.pivot(index='anatomy', columns='final_split', values='studies').fillna(0)\n"
            "ax = pivot.plot(kind='bar', figsize=(10, 4.5), color=['#2563eb', '#0f766e'])\n"
            "ax.set(title='Study counts by anatomy and internal split', xlabel='Anatomy', ylabel='Studies')\n"
            "ax.grid(axis='y', alpha=0.25)\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Important design choice\n\n"
            "The same patient can appear in more than one body part. Therefore the split is global by patient — "
            "not a separate split per anatomy. Each study ID includes the anatomy (`XR_*`) so images from different body parts cannot be grouped into one study."
        ),
        code(
            "combined = pd.read_csv(MANIFESTS / 'all_anatomies_train_manifest.csv')\n"
            "study_audit = combined.groupby('study_id').agg(\n"
            "    anatomy=('anatomy', 'nunique'), patient=('patient_id', 'nunique'),\n"
            "    label=('label', 'nunique'), declared_views=('num_views_in_study', 'first'),\n"
            "    indexed_views=('image_path', 'nunique')\n"
            ").reset_index()\n"
            "assert study_audit['anatomy'].eq(1).all()\n"
            "assert study_audit['patient'].eq(1).all()\n"
            "assert study_audit['label'].eq(1).all()\n"
            "assert study_audit['declared_views'].eq(study_audit['indexed_views']).all()\n"
            "print({'studies': len(study_audit), 'images': len(combined), 'unique_patients': combined['patient_id'].nunique()})\n"
            "print('Study hierarchy and view-count checks passed.')"
        ),
    ]
    return nb


def preprocessing_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md(
            "# 11 — MURA all-anatomy preprocessing and QC\n\n"
            "This notebook reports the local image-integrity and geometry audit that was run before feature extraction. "
            "It stores aggregate metadata only; no radiographs are displayed or saved here."
        ),
        code(SETUP),
        code(
            "qc_report = json.loads((QC / 'qc_report.json').read_text(encoding='utf-8'))\n"
            "assert qc_report['official_valid_read'] is False\n"
            "display(Markdown('**QC provenance:** official train only; official `valid` was not read.'))\n"
            "display(pd.DataFrame([qc_report]))"
        ),
        code(
            "qc_summary = pd.read_csv(QC / 'qc_summary_by_anatomy.csv')\n"
            "geometry = pd.read_csv(QC / 'geometry_summary_by_anatomy.csv')\n"
            "display(qc_summary)\n"
            "display(geometry)"
        ),
        code(
            "plot = qc_summary.pivot(index='anatomy', columns='final_split', values='decode_fail_images').fillna(0)\n"
            "ax = plot.plot(kind='bar', figsize=(10, 4.2), color=['#2563eb', '#0f766e'])\n"
            "ax.set(title='Unreadable image files found by anatomy', xlabel='Anatomy', ylabel='Decode failures')\n"
            "ax.grid(axis='y', alpha=0.25)\n"
            "plt.xticks(rotation=30, ha='right')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Preprocessing contract\n\n"
            "Images are decoded as grayscale, replicated to three channels for ImageNet ResNet-18, padded to preserve aspect ratio, resized to 224×224, and normalized with fixed ImageNet values. "
            "The evaluation transform is deterministic. Narrow but valid source images are recorded as warnings, not silently removed."
        ),
    ]
    return nb


def baseline_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md(
            "# 12 — MURA all-anatomy frozen ResNet-18 + Logistic Regression baseline\n\n"
            "This is the required cheap baseline, trained **separately for each anatomy**. Each image passes through a frozen ImageNet ResNet-18; embeddings are mean-pooled within a study and Logistic Regression produces the study-level abnormality score."
        ),
        code(SETUP),
        code(
            "baseline_provenance = json.loads((BASELINE / 'baseline_provenance.json').read_text(encoding='utf-8'))\n"
            "assert baseline_provenance['official_valid_read'] is False\n"
            "assert baseline_provenance['test_used_for_selection'] is False\n"
            "display(Markdown('**Run check passed:** the baseline used only official train → internal train/validation.'))\n"
            "display(pd.DataFrame([{\n"
            "    'model': baseline_provenance['model'],\n"
            "    'C selection': baseline_provenance['C_selection'],\n"
            "    'operating policy': baseline_provenance['operating_policy'],\n"
            "    'target sensitivity': baseline_provenance['target_sensitivity'],\n"
            "    'device': baseline_provenance['device'],\n"
            "    'runtime (seconds)': baseline_provenance['runtime_seconds'],\n"
            "}]))"
        ),
        code(
            "metrics = pd.read_csv(BASELINE / 'baseline_metrics_by_anatomy.csv')\n"
            "display(metrics[[\n"
            "    'anatomy', 'n_studies', 'selected_C', 'threshold', 'auroc', 'auprc',\n"
            "    'sensitivity', 'specificity', 'precision', 'f1',\n"
            "    'false_negative', 'false_positive', 'flagged_percent'\n"
            "]].round(4))"
        ),
        code("display(Image(filename=str(BASELINE / 'baseline_roc_pr_by_anatomy.png')))"),
        code("display(Image(filename=str(BASELINE / 'baseline_operating_point_workload.png')))"),
        md(
            "### How the threshold was chosen\n\n"
            "The 95% sensitivity target was fixed before reporting the results. For each anatomy, the code considers all internal-validation study probabilities and chooses the **highest** threshold that still reaches at least 95% sensitivity. This maximizes specificity among thresholds meeting that target. The threshold is not transferred from one anatomy to another."
        ),
    ]
    return nb


def evaluation_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    nb.cells = [
        md(
            "# 13 — MURA all-anatomy development evaluation\n\n"
            "This notebook evaluates the all-anatomy baseline on the internal validation set. It is not an official-test result and must not be described as clinical performance."
        ),
        code(SETUP),
        code(
            "metrics = pd.read_csv(BASELINE / 'baseline_metrics_by_anatomy.csv')\n"
            "thresholds = pd.read_csv(BASELINE / 'baseline_thresholds_by_anatomy.csv')\n"
            "bootstrap = pd.read_csv(BASELINE / 'baseline_patient_bootstrap_ci.csv')\n"
            "assert metrics['official_valid_read'].eq(False).all()\n"
            "assert thresholds['official_valid_read'].eq(False).all()\n"
            "display(Markdown('**Evaluation scope:** internal validation only; no official MURA `valid` label was read.'))\n"
            "display(thresholds.round(4))"
        ),
        code(
            "summary = metrics[[\n"
            "    'anatomy', 'auroc', 'auprc', 'sensitivity', 'specificity', 'precision',\n"
            "    'f1', 'false_negative_rate', 'false_negative', 'false_positive', 'flagged_percent'\n"
            "]].copy()\n"
            "display(summary.round(4))"
        ),
        code(
            "ci_table = bootstrap.pivot(index='anatomy', columns='metric', values=['estimate', 'lower_95_ci', 'upper_95_ci'])\n"
            "display(ci_table.round(4))"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))\n"
            "plot = metrics.set_index('anatomy')[['auroc', 'auprc']].sort_index()\n"
            "plot.plot(kind='bar', ax=axes[0], color=['#2563eb', '#0f766e'])\n"
            "axes[0].set(title='Ranking metrics by anatomy', xlabel='Anatomy', ylabel='Score', ylim=(0, 1.05))\n"
            "axes[0].grid(axis='y', alpha=0.25)\n"
            "metrics.sort_values('anatomy').plot.bar(x='anatomy', y=['false_negative', 'false_positive'], ax=axes[1], color=['#b91c1c', '#d97706'])\n"
            "axes[1].set(title='Validation errors at the fixed 95% sensitivity policy', xlabel='Anatomy', ylabel='Studies')\n"
            "axes[1].grid(axis='y', alpha=0.25)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        md(
            "### Interpretation boundary\n\n"
            "Differences between body parts can reflect sample size, prevalence, acquisition differences, and task difficulty. These results establish a reproducible development baseline only. A future fine-tuned model must be compared against this baseline on the same internal validation studies, with its own validation-selected threshold."
        ),
    ]
    return nb


def write_and_execute(notebook, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    # Ensure a plain "python" kernel resolves to the interpreter used for this command.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=str(ROOT))
    nbf.write(executed, path)
    print(path)


def main() -> None:
    expected = [
        ROOT.parent / "runtime" / "multianatomy_v1" / "manifests" / "development_split_summary.csv",
        ROOT.parent / "runtime" / "multianatomy_v1" / "qc" / "qc_summary_by_anatomy.csv",
        ROOT.parent / "runtime" / "multianatomy_v1" / "baseline_all_v1" / "baseline_metrics_by_anatomy.csv",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise SystemExit("Baseline artifacts are required before notebook generation:\n" + "\n".join(missing))
    builders = [
        ("10_MURA_All_Anatomies_Data_Preparation.ipynb", data_preparation_notebook),
        ("11_MURA_All_Anatomies_Preprocessing.ipynb", preprocessing_notebook),
        ("12_MURA_All_Anatomies_Baseline_ResNet18_LogReg.ipynb", baseline_notebook),
        ("13_MURA_All_Anatomies_Development_Evaluation.ipynb", evaluation_notebook),
    ]
    for name, builder in builders:
        write_and_execute(builder(), NOTEBOOKS / name)


if __name__ == "__main__":
    main()
