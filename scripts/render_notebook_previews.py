#!/usr/bin/env python
"""Embed safe, aggregate-only Checkpoint 1 outputs in the canonical notebooks.

The notebooks are normally kept output-free while students work in Colab so
that local paths, study IDs, and radiographs cannot be committed by accident.
This script adds one reviewed presentation section to each current notebook.
Every embedded table and figure comes from ``docs/results/checkpoint1`` and
contains aggregate development results only.
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

import nbformat
import pandas as pd
from nbformat.v4 import new_code_cell, new_markdown_cell, new_output


PREVIEW_TAG = "checkpoint1-published-output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing notebooks/ and docs/results/checkpoint1/.",
    )
    return parser.parse_args()


def table_output(frame: pd.DataFrame) -> dict:
    """Create an executed notebook table output without writing any identifiers."""

    return new_output(
        output_type="execute_result",
        execution_count=1,
        data={
            "text/plain": frame.to_string(index=False),
            "text/html": frame.to_html(index=False, border=1, classes="dataframe"),
        },
        metadata={},
    )


def image_output(path: Path) -> dict:
    """Embed one already-reviewed aggregate figure as a GitHub-renderable output."""

    return new_output(
        output_type="display_data",
        data={
            "image/png": base64.b64encode(path.read_bytes()).decode("ascii"),
            "text/plain": f"Rendered aggregate figure: {path.name}",
        },
        metadata={},
    )


def preview_code(source: str, outputs: list[dict]) -> dict:
    cell = new_code_cell(source=source)
    cell["execution_count"] = 1
    cell["outputs"] = outputs
    cell["metadata"] = {"tags": [PREVIEW_TAG]}
    return cell


def remove_previous_preview(notebook: dict) -> None:
    notebook["cells"] = [
        cell
        for cell in notebook["cells"]
        if PREVIEW_TAG not in cell.get("metadata", {}).get("tags", [])
    ]


def compact_model_table(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "auroc",
        "auprc",
        "sensitivity_at_target",
        "specificity_at_target",
        "false_negative_at_target",
        "false_positive_at_target",
        "flagged_percent_at_target",
    ]
    display = rows[columns].copy()
    display = display.rename(
        columns={
            "model": "Model",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "sensitivity_at_target": "Sensitivity @ ≥90% policy",
            "specificity_at_target": "Specificity @ ≥90% policy",
            "false_negative_at_target": "False negatives",
            "false_positive_at_target": "False positives",
            "flagged_percent_at_target": "Flagged for review (%)",
        }
    )
    for column in (
        "AUROC",
        "AUPRC",
        "Sensitivity @ ≥90% policy",
        "Specificity @ ≥90% policy",
        "Flagged for review (%)",
    ):
        display[column] = display[column].map(lambda value: round(float(value), 3))
    return display


def model_table_source(model_name: str | None = None) -> str:
    """Return the portable code stored below a rendered model-summary table."""

    selection = (
        ""
        if model_name is None
        else f"comparison = comparison.loc[comparison['model'].eq({model_name!r})].copy()\n"
    )
    return (
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "comparison = pd.read_csv(Path('../docs/results/checkpoint1/model_comparison.csv'))\n"
        f"{selection}"
        "columns = ['model', 'auroc', 'auprc', 'sensitivity_at_target', "
        "'specificity_at_target', 'false_negative_at_target', "
        "'false_positive_at_target', 'flagged_percent_at_target']\n"
        "notebook_summary = comparison[columns].rename(columns={\n"
        "    'model': 'Model', 'auroc': 'AUROC', 'auprc': 'AUPRC',\n"
        "    'sensitivity_at_target': 'Sensitivity @ ≥90% policy',\n"
        "    'specificity_at_target': 'Specificity @ ≥90% policy',\n"
        "    'false_negative_at_target': 'False negatives',\n"
        "    'false_positive_at_target': 'False positives',\n"
        "    'flagged_percent_at_target': 'Flagged for review (%)',\n"
        "})\n"
        "for column in ['AUROC', 'AUPRC', 'Sensitivity @ ≥90% policy', "
        "'Specificity @ ≥90% policy', 'Flagged for review (%)']:\n"
        "    notebook_summary[column] = notebook_summary[column].round(3)\n"
        "display(notebook_summary)"
    )


def audit_table_source() -> str:
    return (
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "robustness_summary = pd.read_csv(Path('../docs/results/checkpoint1/shortcut_audit_summary.csv'))\n"
        "columns = ['display_variant', 'mean_absolute_probability_change', "
        "'decision_flips_at_frozen_threshold', 'decision_flip_percent_at_frozen_threshold']\n"
        "notebook_summary = robustness_summary[columns].rename(columns={\n"
        "    'display_variant': 'Perturbation',\n"
        "    'mean_absolute_probability_change': 'Mean |Δ probability|',\n"
        "    'decision_flips_at_frozen_threshold': 'Decision flips',\n"
        "    'decision_flip_percent_at_frozen_threshold': 'Decision flips (%)',\n"
        "})\n"
        "notebook_summary['Mean |Δ probability|'] = notebook_summary['Mean |Δ probability|'].round(3)\n"
        "notebook_summary['Decision flips (%)'] = notebook_summary['Decision flips (%)'].round(2)\n"
        "display(notebook_summary)"
    )


def add_preview(notebook_path: Path, markdown: str, cells: list[dict]) -> None:
    notebook = nbformat.read(notebook_path, as_version=4)
    remove_previous_preview(notebook)
    header = new_markdown_cell(markdown)
    header["metadata"] = {"tags": [PREVIEW_TAG]}
    notebook["cells"].append(header)
    notebook["cells"].extend(cells)
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    notebooks_dir = repo_root / "notebooks"
    results_dir = repo_root / "docs" / "results" / "checkpoint1"
    comparison_path = results_dir / "model_comparison.csv"
    required = [
        comparison_path,
        results_dir / "validation_model_comparison.png",
        results_dir / "finetuning_history.png",
        results_dir / "shortcut_audit_summary.csv",
        results_dir / "shortcut_audit.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reviewed result artifact(s): " + ", ".join(missing))

    comparison = pd.read_csv(comparison_path)
    baseline = comparison.loc[
        comparison["model"].eq("Frozen ResNet-18 + Logistic Regression")
    ].copy()
    fine_tuned_mean = comparison.loc[
        comparison["model"].eq("Fine-tuned ResNet-18 mean pooling")
    ].copy()
    if len(baseline) != 1 or len(fine_tuned_mean) != 1:
        raise ValueError("The reviewed comparison artifact is missing the expected Model A or Model B row.")

    population = pd.DataFrame(
        [
            {
                "Population": "Official MURA train, XR_WRIST only",
                "Images": 9752,
                "Studies": 3460,
                "Patients": 3267,
                "NORMAL studies": 2134,
                "ABNORMAL studies": 1326,
            },
            {
                "Population": "Internal training split",
                "Images": 7810,
                "Studies": 2769,
                "Patients": 2614,
                "NORMAL studies": 1712,
                "ABNORMAL studies": 1057,
            },
            {
                "Population": "Internal validation split",
                "Images": 1942,
                "Studies": 691,
                "Patients": 653,
                "NORMAL studies": 422,
                "ABNORMAL studies": 269,
            },
        ]
    )
    view_distribution = pd.DataFrame(
        [
            {"Views per study": "1", "Studies": 276},
            {"Views per study": "2", "Studies": 661},
            {"Views per study": "3", "Studies": 2005},
            {"Views per study": "4+", "Studies": 518},
        ]
    )
    preprocessing = pd.DataFrame(
        [
            {"Check": "Input structure", "Result": "Patient → study → one or more views"},
            {"Check": "Evaluation transform", "Result": "Square padding, 224×224 resize, ImageNet normalization"},
            {"Check": "Split integrity", "Result": "Patient groups remain disjoint across train and validation"},
            {"Check": "Git publication", "Result": "Only metadata and aggregate outputs; no raw pixels"},
        ]
    )
    audit_summary = pd.read_csv(results_dir / "shortcut_audit_summary.csv")
    audit_display = audit_summary.rename(
        columns={
            "display_variant": "Perturbation",
            "mean_absolute_probability_change": "Mean |Δ probability|",
            "decision_flips_at_frozen_threshold": "Decision flips",
            "decision_flip_percent_at_frozen_threshold": "Decision flips (%)",
        }
    )
    if "Perturbation" not in audit_display:
        audit_display["Perturbation"] = audit_display["variant"]
    audit_display = audit_display[
        ["Perturbation", "Mean |Δ probability|", "Decision flips", "Decision flips (%)"]
    ].copy()
    audit_display["Mean |Δ probability|"] = audit_display["Mean |Δ probability|"].map(
        lambda value: round(float(value), 3)
    )
    audit_display["Decision flips (%)"] = audit_display["Decision flips (%)"].map(
        lambda value: round(float(value), 2)
    )
    candidate_c = pd.DataFrame(
        [
            {"Logistic Regression C": 0.1, "Validation AUROC": 0.880442, "Selected": "Yes"},
            {"Logistic Regression C": 1.0, "Validation AUROC": 0.867968, "Selected": "No"},
            {"Logistic Regression C": 10.0, "Validation AUROC": 0.862374, "Selected": "No"},
        ]
    )
    training_summary = pd.DataFrame(
        [
            {"Item": "Best validation epoch", "Value": "9"},
            {"Item": "Epochs completed", "Value": "13 (early stopping)"},
            {"Item": "Checkpoint selection", "Value": "Validation AUROC, mean pooling"},
            {"Item": "Official MURA valid/test read", "Value": "No"},
        ]
    )
    audit_context = pd.DataFrame(
        [
            {"Diagnostic check": "Saved-prediction reproduction", "Result": "Passed (maximum absolute error: 1.11e-16)"},
            {"Diagnostic check": "Geometry-only grouped CV AUROC", "Result": "0.697 — warning signal, not model performance"},
            {"Diagnostic check": "Geometry-only grouped CV AUPRC", "Result": "0.556 — warning signal, not model performance"},
        ]
    )

    add_preview(
        notebooks_dir / "01_MURA_Data_Preparation_and_EDA.ipynb",
        "## Published output snapshot\n\nAggregate summary from the completed Checkpoint 1 development run.",
        [
            preview_code(
                "# Aggregate dataset summary from the completed development run.\n"
                "import pandas as pd\n"
                "population_overview = pd.DataFrame([\n"
                "    {'Population': 'Official MURA train, XR_WRIST only', 'Images': 9752, 'Studies': 3460, 'Patients': 3267, 'NORMAL studies': 2134, 'ABNORMAL studies': 1326},\n"
                "    {'Population': 'Internal training split', 'Images': 7810, 'Studies': 2769, 'Patients': 2614, 'NORMAL studies': 1712, 'ABNORMAL studies': 1057},\n"
                "    {'Population': 'Internal validation split', 'Images': 1942, 'Studies': 691, 'Patients': 653, 'NORMAL studies': 422, 'ABNORMAL studies': 269},\n"
                "])\n"
                "display(population_overview)",
                [table_output(population)],
            ),
            preview_code(
                "# Number of views per study in the current official-train XR_WRIST population.\n"
                "view_distribution = pd.DataFrame([\n"
                "    {'Views per study': '1', 'Studies': 276},\n"
                "    {'Views per study': '2', 'Studies': 661},\n"
                "    {'Views per study': '3', 'Studies': 2005},\n"
                "    {'Views per study': '4+', 'Studies': 518},\n"
                "])\n"
                "display(view_distribution)",
                [table_output(view_distribution)],
            ),
        ],
    )
    add_preview(
        notebooks_dir / "02_MURA_Preprocessing.ipynb",
        "## Published output snapshot\n\nThe preprocessing contract below is the one used by the completed development run.",
        [
            preview_code(
                "# Reviewed preprocessing and integrity summary.\n"
                "import pandas as pd\n"
                "preprocessing_checks = pd.DataFrame([\n"
                "    {'Check': 'Input structure', 'Result': 'Patient → study → one or more views'},\n"
                "    {'Check': 'Evaluation transform', 'Result': 'Square padding, 224×224 resize, ImageNet normalization'},\n"
                "    {'Check': 'Split integrity', 'Result': 'Patient groups remain disjoint across train and validation'},\n"
                "    {'Check': 'Git publication', 'Result': 'Only metadata and aggregate outputs; no raw pixels'},\n"
                "])\n"
                "display(preprocessing_checks)",
                [table_output(preprocessing)],
            )
        ],
    )
    add_preview(
        notebooks_dir / "03_MURA_Baseline_ResNet18_LogReg.ipynb",
        "## Published output snapshot\n\nModel A result from the completed internal-validation run.",
        [
            preview_code(
                "# Aggregate Model A metrics from the completed development run.\n"
                + model_table_source("Frozen ResNet-18 + Logistic Regression"),
                [table_output(compact_model_table(baseline))],
            ),
            preview_code(
                "# Validation-only selection among the pre-specified Logistic Regression C values.\n"
                "candidate_c = pd.DataFrame([\n"
                "    {'Logistic Regression C': 0.1, 'Validation AUROC': 0.880442, 'Selected': 'Yes'},\n"
                "    {'Logistic Regression C': 1.0, 'Validation AUROC': 0.867968, 'Selected': 'No'},\n"
                "    {'Logistic Regression C': 10.0, 'Validation AUROC': 0.862374, 'Selected': 'No'},\n"
                "])\n"
                "display(candidate_c)",
                [table_output(candidate_c)],
            ),
        ],
    )
    add_preview(
        notebooks_dir / "04_MURA_Baseline_Evaluation.ipynb",
        "## Published output snapshot\n\nValidation-only operating-policy result for the frozen baseline.",
        [
            preview_code(
                "# Baseline operating-policy metrics from the completed development run.\n"
                + model_table_source("Frozen ResNet-18 + Logistic Regression"),
                [table_output(compact_model_table(baseline))],
            )
        ],
    )
    add_preview(
        notebooks_dir / "05_MURA_FineTuned_ResNet18.ipynb",
        "## Published output snapshot\n\nModel B was selected at validation epoch 9; the curve is shown below.",
        [
            preview_code(
                "# Training summary from the completed validation-only fine-tuning run.\n"
                "import pandas as pd\n"
                "training_summary = pd.DataFrame([\n"
                "    {'Item': 'Best validation epoch', 'Value': '9'},\n"
                "    {'Item': 'Epochs completed', 'Value': '13 (early stopping)'},\n"
                "    {'Item': 'Checkpoint selection', 'Value': 'Validation AUROC, mean pooling'},\n"
                "    {'Item': 'Official MURA valid/test read', 'Value': 'No'},\n"
                "])\n"
                "display(training_summary)",
                [table_output(training_summary)],
            ),
            preview_code(
                "# Aggregate Model B metrics from the completed development run.\n"
                + model_table_source("Fine-tuned ResNet-18 mean pooling"),
                [table_output(compact_model_table(fine_tuned_mean))],
            ),
            preview_code(
                "# Training loss and validation AUROC over epochs.\n"
                "from IPython.display import Image, display\n"
                "display(Image(filename='../docs/results/checkpoint1/finetuning_history.png'))",
                [image_output(results_dir / "finetuning_history.png")],
            ),
        ],
    )
    add_preview(
        notebooks_dir / "06_MURA_ModelB_Evaluation_and_Comparison.ipynb",
        "## Published output snapshot\n\nFair comparison on the same internal-validation studies; not an untouched test result.",
        [
            preview_code(
                "# Aggregate comparison for all evaluated study-level methods.\n"
                + model_table_source(),
                [table_output(compact_model_table(comparison))],
            ),
            preview_code(
                "# ROC, precision–recall, and high-sensitivity workload comparison.\n"
                "from IPython.display import Image, display\n"
                "display(Image(filename='../docs/results/checkpoint1/validation_model_comparison.png'))",
                [image_output(results_dir / "validation_model_comparison.png")],
            ),
        ],
    )
    add_preview(
        notebooks_dir / "07_MURA_Shortcut_and_Geometry_Audit.ipynb",
        "## Published output snapshot\n\nDiagnostic robustness output from the frozen Model B checkpoint. This is not localization evidence.",
        [
            preview_code(
                "# Diagnostic provenance for the frozen checkpoint.\n"
                "import pandas as pd\n"
                "audit_context = pd.DataFrame([\n"
                "    {'Diagnostic check': 'Saved-prediction reproduction', 'Result': 'Passed (maximum absolute error: 1.11e-16)'},\n"
                "    {'Diagnostic check': 'Geometry-only grouped CV AUROC', 'Result': '0.697 — warning signal, not model performance'},\n"
                "    {'Diagnostic check': 'Geometry-only grouped CV AUPRC', 'Result': '0.556 — warning signal, not model performance'},\n"
                "])\n"
                "display(audit_context)",
                [table_output(audit_context)],
            ),
            preview_code(
                "# Aggregate perturbation sensitivity summary.\n"
                + audit_table_source(),
                [table_output(audit_display)],
            ),
            preview_code(
                "# Aggregate shortcut and geometry diagnostic figure.\n"
                "from IPython.display import Image, display\n"
                "display(Image(filename='../docs/results/checkpoint1/shortcut_audit.png'))",
                [image_output(results_dir / "shortcut_audit.png")],
            ),
        ],
    )
    print("Rendered safe GitHub previews in notebooks 01–07.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
