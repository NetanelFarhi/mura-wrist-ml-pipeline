#!/usr/bin/env python
"""Build executed aggregate-only notebooks for all-anatomy fine-tuning results."""

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
FINE = RUNTIME / 'finetuned_all_v1'
COMPARISON = RUNTIME / 'model_comparison_v1'
SHORTCUT = RUNTIME / 'shortcut_audit_v1'
assert ROOT.name == 'mura-wrist-ml-pipeline', ROOT
print('Notebook environment configured; results are loaded from saved development artifacts.')
"""


def fine_tuning_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    notebook.cells = [
        md(
            "# 14 — MURA all-anatomy study-level fine-tuned ResNet-18\n\n"
            "This notebook reports one controlled Model B training run per anatomy. It is a development/validation experiment only: the historical official MURA `valid` split is not read."
        ),
        code(SETUP),
        code(
            "provenance = json.loads((FINE / 'finetuned_provenance.json').read_text(encoding='utf-8'))\n"
            "assert provenance['official_valid_read'] is False\n"
            "assert provenance['test_used_for_selection'] is False\n"
            "display(Markdown('**Safety check passed:** no official MURA `valid`/test prediction was used.'))\n"
            "display(pd.DataFrame([{\n"
            "    'initialization': provenance['initialization'],\n"
            "    'architecture': provenance['architecture'],\n"
            "    'maximum epochs': provenance['max_epochs'],\n"
            "    'early stopping': provenance['early_stopping'],\n"
            "    'operating policy': provenance['operating_policy'],\n"
            "}]))"
        ),
        code(
            "metrics = pd.read_csv(FINE / 'finetuned_metrics_by_anatomy.csv')\n"
            "display(metrics[[\n"
            "    'anatomy', 'n_studies', 'threshold', 'auroc', 'auprc', 'sensitivity',\n"
            "    'specificity', 'precision', 'f1', 'false_negative', 'false_positive', 'flagged_percent', 'runtime_seconds'\n"
            "]].round(4))"
        ),
        code(
            "ci_rows = []\n"
            "for anatomy in sorted(metrics['anatomy'].unique()):\n"
            "    ci = pd.read_csv(FINE / anatomy.lower() / 'patient_bootstrap_ci.csv')\n"
            "    ci['anatomy'] = anatomy\n"
            "    ci_rows.append(ci)\n"
            "bootstrap_ci = pd.concat(ci_rows, ignore_index=True)\n"
            "display(bootstrap_ci[bootstrap_ci['metric'].isin(['auroc', 'auprc', 'sensitivity', 'specificity'])][['anatomy', 'metric', 'estimate', 'lower_95_ci', 'upper_95_ci', 'valid_resamples']].round(4))"
        ),
        code(
            "history = pd.read_csv(COMPARISON / 'training_history_by_anatomy.csv')\n"
            "best = history.sort_values('validation_auroc').groupby('anatomy', as_index=False).tail(1)\n"
            "display(best[['anatomy', 'best_epoch', 'epoch', 'train_loss', 'validation_auroc', 'learning_rate']].sort_values('anatomy').round(4))\n"
            "display(Image(filename=str(COMPARISON / 'finetuned_training_history.png')))"
        ),
        md(
            "### What the model learns\n\n"
            "A view is passed through ResNet-18 to produce an abnormality probability. The real-view probabilities within the same study are mean-pooled, and the loss is computed once per study. Padding views are masked and cannot influence the study score. This is not attention pooling and it is not lesion localization."
        ),
    ]
    return notebook


def comparison_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    notebook.cells = [
        md(
            "# 15 — MURA all-anatomy baseline vs fine-tuned model comparison\n\n"
            "The comparison is paired within each anatomy: both models predict the same internal-validation studies. Each model uses its own threshold selected on internal validation to meet the fixed 95% sensitivity target."
        ),
        code(SETUP),
        code(
            "provenance = json.loads((COMPARISON / 'comparison_provenance.json').read_text(encoding='utf-8'))\n"
            "assert provenance['official_valid_read'] is False\n"
            "assert provenance['test_used_for_selection'] is False\n"
            "display(Markdown('**Comparison scope:** development/internal validation only.'))\n"
            "display(pd.DataFrame([provenance]))"
        ),
        code(
            "comparison = pd.read_csv(COMPARISON / 'model_comparison_by_anatomy.csv')\n"
            "display(comparison[[\n"
            "    'anatomy', 'model', 'auroc', 'auprc', 'sensitivity', 'specificity',\n"
            "    'precision', 'f1', 'false_negative', 'false_positive', 'flagged_percent'\n"
            "]].round(4))\n"
            "display(Image(filename=str(COMPARISON / 'baseline_vs_finetuned.png')))"
        ),
        code(
            "paired = pd.read_csv(COMPARISON / 'paired_patient_bootstrap.csv')\n"
            "key_metrics = paired[paired['metric'].isin(['auroc', 'auprc', 'specificity', 'sensitivity'])]\n"
            "display(key_metrics[['anatomy', 'metric', 'candidate_minus_baseline', 'lower_95_ci', 'upper_95_ci', 'valid_resamples']].round(4))"
        ),
        code(
            "errors = pd.read_csv(COMPARISON / 'paired_error_summary.csv')\n"
            "display(errors)"
        ),
        md(
            "### Interpretation boundary\n\n"
            "A point estimate alone is not enough to say that fine-tuning wins. The paired bootstrap interval must be considered, especially for the smaller Humerus and Forearm validation samples. All intervals are conditional on validation-selected thresholds, so they are not a replacement for an untouched final test."
        ),
    ]
    return notebook


def shortcut_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    notebook.cells = [
        md(
            "# 16 — MURA all-anatomy peripheral-pixel sensitivity audit\n\n"
            "This is a bounded diagnostic of the saved fine-tuned models. It does not retrain a model, select a threshold, or make a lesion-localization claim."
        ),
        code(SETUP),
        code(
            "provenance = json.loads((SHORTCUT / 'shortcut_audit_provenance.json').read_text(encoding='utf-8'))\n"
            "assert provenance['official_valid_read'] is False\n"
            "assert provenance['test_used_for_selection'] is False\n"
            "assert provenance['threshold_reselected'] is False\n"
            "display(Markdown('**Audit scope:** saved internal-validation checkpoints and frozen validation thresholds only.'))\n"
            "display(pd.DataFrame([provenance]))"
        ),
        code(
            "summary = pd.read_csv(SHORTCUT / 'MULTI_ANATOMY_SHORTCUT_AUDIT_SUMMARY.csv')\n"
            "shown = summary[summary['variant'] != 'canonical']\n"
            "display(shown[[\n"
            "    'anatomy', 'variant', 'mean_abs_probability_change', 'median_abs_probability_change',\n"
            "    'p95_abs_probability_change', 'decision_flip_count', 'decision_flip_percent',\n"
            "    'new_false_negatives_after_perturbation', 'new_false_positives_after_perturbation'\n"
            "]].round(4))\n"
            "display(Image(filename=str(SHORTCUT / 'shortcut_audit_summary.png')))"
        ),
        code(
            "intervals = pd.read_csv(SHORTCUT / 'shortcut_audit_bootstrap_ci.csv')\n"
            "display(intervals[['anatomy', 'variant', 'metric', 'point_estimate', 'lower_95_ci', 'upper_95_ci', 'valid_resamples']].round(4))"
        ),
        md(
            "### Interpretation boundary\n\n"
            "The perturbations measure sensitivity to synthetic padding, borders, and corners. They do **not** identify a specific L/R marker, prove shortcut learning, prove lesion localization, or prove that a model is free of shortcuts. A border or corner can include anatomy, collimation, or acquisition content. The detailed per-study rows remain local and are not part of this notebook."
        ),
    ]
    return notebook


def execute(notebook, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, target)
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(cwd=str(ROOT))
    nbf.write(executed, target)
    print(target)


def main() -> None:
    required = [
        ROOT.parent / "runtime" / "multianatomy_v1" / "finetuned_all_v1" / "finetuned_metrics_by_anatomy.csv",
        ROOT.parent / "runtime" / "multianatomy_v1" / "model_comparison_v1" / "model_comparison_by_anatomy.csv",
        ROOT.parent / "runtime" / "multianatomy_v1" / "shortcut_audit_v1" / "MULTI_ANATOMY_SHORTCUT_AUDIT_SUMMARY.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Fine-tuning/comparison artifacts are required before notebook generation:\n" + "\n".join(missing))
    execute(fine_tuning_notebook(), NOTEBOOKS / "14_MURA_All_Anatomies_FineTuned_ResNet18.ipynb")
    execute(comparison_notebook(), NOTEBOOKS / "15_MURA_All_Anatomies_Model_Comparison.ipynb")
    execute(shortcut_notebook(), NOTEBOOKS / "16_MURA_All_Anatomies_Shortcut_Audit.ipynb")


if __name__ == "__main__":
    main()
