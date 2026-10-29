"""Execute the canonical MURA notebooks locally and retain safe rendered outputs.

This is deliberately a *development-only reproduction* runner.  It reads the
official MURA train population and the local MURA ZIP only; it never points a
notebook at official MURA ``valid``.  Generated manifests, models, predictions
and logs are written outside Git.  Before each notebook is saved, the runner
rejects raw-image output, local paths and study/patient identifiers.

Example (from the repository root)::

    python scripts/execute_canonical_notebooks.py --kernel mura-gpu \
      --jupyter-path C:/.../mura-gpu-jupyter/share/jupyter

The command is intended for an authorised local MURA copy.  It does not make a
new model-selection or test-set decision.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_TAG = "checkpoint1-published-output"
NOTEBOOK_NAMES = [
    "01_MURA_Data_Preparation_and_EDA.ipynb",
    "02_MURA_Preprocessing.ipynb",
    "03_MURA_Baseline_ResNet18_LogReg.ipynb",
    "04_MURA_Baseline_Evaluation.ipynb",
    "05_MURA_FineTuned_ResNet18.ipynb",
    "06_MURA_ModelB_Evaluation_and_Comparison.ipynb",
    "07_MURA_Shortcut_and_Geometry_Audit.ipynb",
    "09_MURA_Deterministic_Safety_Tools.ipynb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="mura-gpu")
    parser.add_argument("--jupyter-path", type=Path, required=True)
    parser.add_argument("--source-csv-root", type=Path, required=True)
    parser.add_argument("--mura-data-root", type=Path, required=True)
    parser.add_argument("--raw-zip", type=Path, required=True)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--start-at", choices=[name[:2] for name in NOTEBOOK_NAMES], default="01")
    parser.add_argument("--stop-after", choices=[name[:2] for name in NOTEBOOK_NAMES], default="09")
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    return parser.parse_args()


def replace_once(source: str, old: str, new: str, context: str) -> str:
    if old not in source:
        raise ValueError(f"Expected source fragment was not found for {context}.")
    return source.replace(old, new, 1)


def replace_unless_present(source: str, old: str, new: str, context: str) -> str:
    """Apply a source edit once, while allowing a previously prepared notebook."""

    if new in source:
        return source
    return replace_once(source, old, new, context)


def cell_containing(notebook: nbformat.NotebookNode, text: str) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if cell.cell_type == "code" and text in cell.source:
            return cell
    raise ValueError(f"Could not find code cell containing {text!r}.")


def cell_with_prefix(notebook: nbformat.NotebookNode, prefix: str) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.lstrip().startswith(prefix):
            return cell
    raise ValueError(f"Could not find code cell starting with {prefix!r}.")


def remove_published_snapshots(notebook: nbformat.NotebookNode) -> None:
    notebook.cells = [
        cell
        for cell in notebook.cells
        if PREVIEW_TAG not in cell.get("metadata", {}).get("tags", [])
    ]


def prepare_sources(notebook_path: Path) -> nbformat.NotebookNode:
    """Make the Colab notebooks portable and prevent unsafe text output."""

    notebook = nbformat.read(notebook_path, as_version=4)
    remove_published_snapshots(notebook)
    name = notebook_path.name

    if name == "01_MURA_Data_Preparation_and_EDA.ipynb":
        cell = cell_with_prefix(notebook, "try:")
        cell.source = replace_unless_present(
            cell.source,
            "DATA_ROOT = Path(os.environ.get(\n    \"MURA_DATA_ROOT\",\n    \"/content/drive/MyDrive/DATA/MURA-v1.1\",\n))",
            "DATA_ROOT = Path(os.environ.get(\n    \"MURA_SOURCE_CSV_DIR\",\n    os.environ.get(\"MURA_DATA_ROOT\", \"/content/drive/MyDrive/DATA/MURA-v1.1\"),\n))",
            "Notebook 01 source CSV root",
        )
        cell.source = replace_unless_present(
            cell.source,
            'print({"data_root": str(DATA_ROOT), "splits_dir": str(SPLITS_DIR), "seed": SEED})',
            'print({"source_csv_files_found": True, "split_directory_ready": True, "seed": SEED})',
            "Notebook 01 safe setup output",
        )
        cell = cell_with_prefix(notebook, "studies = pd.read_csv")
        cell.source = replace_unless_present(
            cell.source,
            "display(manifest.head())",
            "display(pd.DataFrame([{\n"
            "    'manifest_rows': len(manifest),\n"
            "    'unique_images': manifest['file_name'].nunique(),\n"
            "    'studies': manifest['study_uid'].nunique(),\n"
            "    'patients': manifest['patient_id'].nunique(),\n"
            "}]))",
            "Notebook 01 safe manifest output",
        )
        cell = cell_with_prefix(notebook, "train_path = SPLITS_DIR")
        cell.source = replace_once(
            cell.source,
            'print({"train_manifest": str(train_path), "validation_manifest": str(validation_path)})',
            'print({"train_manifest_written": True, "validation_manifest_written": True})',
            "Notebook 01 safe save output",
        )

    if name == "02_MURA_Preprocessing.ipynb":
        cell = cell_with_prefix(notebook, "import io")
        cell.source = replace_unless_present(
            cell.source, "import io\n", "import io\nimport os\n", "Notebook 02 os import"
        )
        cell = cell_containing(notebook, "google.colab import drive")
        if not cell.source.lstrip().startswith("try:"):
            cell.source = (
                "try:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
                "except ModuleNotFoundError:\n"
                "    pass"
            )
        cell = next(
            cell for cell in notebook.cells
            if cell.cell_type == "code" and (
                cell.source.lstrip().startswith("# EDIT THESE PATHS ONLY")
                or cell.source.lstrip().startswith("# Colab defaults may be overridden")
            )
        )
        if not cell.source.lstrip().startswith("# Colab defaults may be overridden"):
            cell.source = """# Colab defaults may be overridden through local environment variables.
# Do NOT substitute the official MURA valid manifest here.
PROJECT_DIR = Path(os.environ.get('MURA_PROJECT_DATA_DIR', '/content/drive/MyDrive/DATA'))
SPLITS_DIR = Path(os.environ.get('MURA_SPLITS_DIR', str(PROJECT_DIR / 'WRIST_SPLITS')))
RAW_ZIP = Path(os.environ.get('MURA_RAW_ZIP', str(PROJECT_DIR / 'MURA-v1.1_files.zip')))
OUTPUT_DIR = Path(os.environ.get('MURA_PREPROCESSING_DIR', str(PROJECT_DIR / 'PREPROCESSED_WRIST')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_MANIFEST = SPLITS_DIR / 'wrist_internal_train.csv'
VALIDATION_MANIFEST = SPLITS_DIR / 'wrist_internal_val.csv'
for required_path in (RAW_ZIP, TRAIN_MANIFEST, VALIDATION_MANIFEST):
    assert required_path.exists(), f'Not found: {required_path}'
print({
    'raw_zip_available': RAW_ZIP.exists(),
    'train_manifest_available': TRAIN_MANIFEST.exists(),
    'validation_manifest_available': VALIDATION_MANIFEST.exists(),
})
"""
        cell = cell_with_prefix(notebook, "REJECTION_COLUMNS")
        cell.source = replace_unless_present(
            cell.source,
            "display(rejected)",
            "display(rejected['reason'].value_counts().rename_axis('reason').reset_index(name='rejected_rows'))",
            "Notebook 02 safe rejected output",
        )
        cell = cell_with_prefix(notebook, "# Deterministic evaluation-transform smoke test")
        old = """print({
    \"study_uid\": sample.study_uid,
    \"label_binary\": int(sample.label_binary),
    \"raw_size\": raw.size,
    \"tensor_shape\": tuple(tensor_a.shape),
})"""
        new = """print({
    \"preprocessing_smoke_test\": \"passed\",
    \"deterministic_transform\": True,
    \"tensor_shape\": tuple(tensor_a.shape),
})"""
        cell.source = replace_unless_present(cell.source, old, new, "Notebook 02 safe smoke-test output")

    if name == "03_MURA_Baseline_ResNet18_LogReg.ipynb":
        cell = cell_with_prefix(notebook, "import io")
        cell.source = replace_unless_present(
            cell.source, "import io\n", "import io\nimport os\n", "Notebook 03 os import"
        )
        cell = cell_containing(notebook, "google.colab import drive")
        if not cell.source.lstrip().startswith("try:"):
            cell.source = (
                "try:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
                "except ModuleNotFoundError:\n"
                "    pass"
            )
        cell = next(
            cell for cell in notebook.cells
            if cell.cell_type == "code" and (
                cell.source.lstrip().startswith("# Change only this path")
                or cell.source.lstrip().startswith("# Colab defaults may be overridden")
            )
        )
        if not cell.source.lstrip().startswith("# Colab defaults may be overridden"):
            cell.source = """# Colab defaults may be overridden through local environment variables.
DATA_DIR = Path(os.environ.get('MURA_PROJECT_DATA_DIR', '/content/drive/MyDrive/DATA'))
RAW_ZIP = Path(os.environ.get('MURA_RAW_ZIP', str(DATA_DIR / 'MURA-v1.1_files.zip')))
PREPROCESSING_DIR = Path(os.environ.get('MURA_PREPROCESSING_DIR', str(DATA_DIR / 'PREPROCESSED_WRIST')))
OUTPUT_DIR = Path(os.environ.get('MURA_MODEL_A_DIR', str(DATA_DIR / 'BASELINE_MODEL_A')))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = PREPROCESSING_DIR / 'wrist_internal_train_clean.csv'
VALIDATION_MANIFEST = PREPROCESSING_DIR / 'wrist_internal_val_clean.csv'
PREPROCESSING_SPEC = PREPROCESSING_DIR / 'preprocessing_spec.json'

for required_path in (RAW_ZIP, TRAIN_MANIFEST, VALIDATION_MANIFEST):
    assert required_path.exists(), 'Not found: {}'.format(required_path)

print({
    'raw_zip_available': RAW_ZIP.exists(),
    'train_manifest_available': TRAIN_MANIFEST.exists(),
    'validation_manifest_available': VALIDATION_MANIFEST.exists(),
    'output_directory_ready': OUTPUT_DIR.exists(),
})
"""
        cell = cell_with_prefix(notebook, "def save_embedding_artifact")
        cell.source = replace_unless_present(
            cell.source,
            "print(\"Saved:\", save_embedding_artifact(\"train\", train_studies, train_study_embeddings))\n"
            "print(\"Saved:\", save_embedding_artifact(\"validation\", validation_studies, validation_study_embeddings))",
            "save_embedding_artifact(\"train\", train_studies, train_study_embeddings)\n"
            "save_embedding_artifact(\"validation\", validation_studies, validation_study_embeddings)\n"
            "print({'train_embedding_artifact_saved': True, 'validation_embedding_artifact_saved': True})",
            "Notebook 03 safe embedding-save output",
        )
        for cell in notebook.cells:
            if cell.cell_type == "code" and "validation_predictions.to_csv" in cell.source:
                cell.source = replace_unless_present(
                    cell.source,
                    "print(\"Saved model:\", model_path)\n"
                    "print(\"Saved validation predictions:\", predictions_path)\n"
                    "print(\"Saved metrics:\", metrics_path)",
                    "print({'model_saved': True, 'validation_predictions_saved': True, 'metrics_saved': True})",
                    "Notebook 03 safe final-save output",
                )
                break
        else:
            raise ValueError("Could not find Notebook 03 final-save cell.")

    if name == "04_MURA_Baseline_Evaluation.ipynb":
        cell = cell_with_prefix(notebook, "import json")
        cell.source = replace_unless_present(
            cell.source, "import json\n", "import json\nimport os\n", "Notebook 04 os import"
        )
        cell = cell_containing(notebook, "google.colab import drive")
        if not cell.source.lstrip().startswith("try:"):
            cell.source = (
                "try:\n"
                "    from google.colab import drive\n"
                "    drive.mount('/content/drive')\n"
                "except ModuleNotFoundError:\n"
                "    pass"
            )
        cell = next(
            cell for cell in notebook.cells
            if cell.cell_type == "code" and (
                cell.source.lstrip().startswith("# Change only this path")
                or cell.source.lstrip().startswith("# Colab defaults may be overridden")
            )
        )
        if not cell.source.lstrip().startswith("# Colab defaults may be overridden"):
            cell.source = """# Colab defaults may be overridden through local environment variables.
DATA_DIR = Path(os.environ.get('MURA_PROJECT_DATA_DIR', '/content/drive/MyDrive/DATA'))
MODEL_DIR = Path(os.environ.get('MURA_MODEL_A_DIR', str(DATA_DIR / 'BASELINE_MODEL_A')))
OUTPUT_DIR = MODEL_DIR / 'internal_validation_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = MODEL_DIR / 'validation_predictions.csv'
RANKING_METRICS_PATH = MODEL_DIR / 'validation_ranking_metrics.json'
SELECTION_PATH = MODEL_DIR / 'model_selection.json'
RUN_METADATA_PATH = MODEL_DIR / 'run_metadata.json'
CANDIDATES_PATH = MODEL_DIR / 'logistic_regression_candidates.csv'

for required_path in (
    PREDICTIONS_PATH, RANKING_METRICS_PATH, SELECTION_PATH,
    RUN_METADATA_PATH, CANDIDATES_PATH,
):
    assert required_path.exists(), f'Notebook 03 output not found: {required_path}'

print({'model_artifacts_available': True, 'evaluation_directory_ready': OUTPUT_DIR.exists()})
"""
        for old, new, context in [
            ("print('Saved:', roc_pr_path)", "print('ROC/PR figure saved.')", "Notebook 04 ROC save output"),
            ("print('Saved:', tradeoff_path)", "print('Threshold trade-off figure saved.')", "Notebook 04 trade-off save output"),
            ("print('Saved:', confusion_path)", "print('Confusion-matrix figure saved.')", "Notebook 04 confusion save output"),
        ]:
            for cell in notebook.cells:
                if cell.cell_type == "code" and old in cell.source:
                    cell.source = replace_unless_present(cell.source, old, new, context)
        for cell in notebook.cells:
            if cell.cell_type == "code" and "for path in saved_paths:" in cell.source:
                cell.source = replace_unless_present(
                    cell.source,
                    "for path in saved_paths:\n    print('Saved:', path)",
                    "print({'evaluation_artifacts_saved': len(saved_paths)})",
                    "Notebook 04 safe bulk-save output",
                )

    if name == "05_MURA_FineTuned_ResNet18.ipynb":
        cell = cell_with_prefix(notebook, "try:")
        legacy_setup_output = (
            'print({"project_root": str(PROJECT_ROOT), "data_root": str(DATA_ROOT), '
            '"artifact_dir": str(ARTIFACT_DIR)})'
        )
        if legacy_setup_output in cell.source:
            cell.source = replace_once(
                cell.source,
                legacy_setup_output,
                'print({"project_root_available": PROJECT_ROOT.exists(), '
                '"data_root_available": DATA_ROOT.exists(), '
                '"artifact_directory_ready": ARTIFACT_DIR.exists()})',
                "Notebook 05 legacy setup output",
            )
        for cell in notebook.cells:
            if cell.cell_type == "code" and 'print(" ".join(command))' in cell.source:
                cell.source = replace_unless_present(
                    cell.source,
                    'print(" ".join(command))',
                    'print("Starting controlled fine-tuning run on internal train/validation only.")',
                    "Notebook 05 safe command output",
                )
            if cell.cell_type == "code" and "subprocess.run(command, cwd=PROJECT_ROOT, check=True)" in cell.source:
                cell.source = replace_unless_present(
                    cell.source,
                    "subprocess.run(command, cwd=PROJECT_ROOT, check=True)",
                    "required_outputs = [\n"
                    "    ARTIFACT_DIR / 'best_resnet18_study_level.pt',\n"
                    "    ARTIFACT_DIR / 'training_history.json',\n"
                    "    ARTIFACT_DIR / 'validation_metrics.json',\n"
                    "    ARTIFACT_DIR / 'validation_predictions.csv',\n"
                    "]\n"
                    "if all(path.exists() for path in required_outputs):\n"
                    "    print({'fine_tuning_completed': True, 'existing_run_artifacts_verified': True})\n"
                    "else:\n"
                    "    completed = subprocess.run(\n"
                    "        command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True\n"
                    "    )\n"
                    "    print({'fine_tuning_completed': completed.returncode == 0})",
                    "Notebook 05 safe training-process output",
                )
            if cell.cell_type == "code" and "validation_predictions = pd.read_csv(ARTIFACT_DIR / \"validation_predictions.csv\")" in cell.source:
                cell.source = replace_unless_present(
                    cell.source,
                    "validation_predictions = pd.read_csv(ARTIFACT_DIR / \"validation_predictions.csv\")",
                    "validation_predictions = pd.read_csv(ARTIFACT_DIR / \"validation_predictions.csv\")\n"
                    "if 'split' not in validation_predictions.columns:\n"
                    "    validation_predictions['split'] = 'validation'",
                    "Notebook 05 validation split metadata",
                )

    if name == "07_MURA_Shortcut_and_Geometry_Audit.ipynb":
        cell = cell_containing(notebook, "subprocess.run([")
        cell.source = replace_unless_present(
            cell.source,
            "subprocess.run([",
            "completed = subprocess.run([",
            "Notebook 07 audit-process assignment",
        )
        cell.source = replace_unless_present(
            cell.source,
            "], check=True)",
            "], check=True, capture_output=True, text=True)\n"
            "print({'shortcut_audit_completed': completed.returncode == 0})",
            "Notebook 07 safe audit-process output",
        )

    nbformat.validate(notebook)
    return notebook


def output_text(output: dict) -> Iterable[str]:
    if output.get("output_type") == "stream":
        yield str(output.get("text", ""))
    for value in output.get("data", {}).values():
        if isinstance(value, str):
            yield value
    for value in output.get("traceback", []):
        yield str(value)


def validate_safe_outputs(notebook: nbformat.NotebookNode, notebook_name: str) -> None:
    """Reject outputs that would disclose a local path or individual study."""

    unsafe = [
        re.compile(r"[A-Za-z]:\\\\"),
        re.compile(r"/content/drive", re.IGNORECASE),
        re.compile(r"(?:patient|study)\d+", re.IGNORECASE),
        re.compile(r"data:image", re.IGNORECASE),
    ]
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            for text in output_text(output):
                if any(pattern.search(text) for pattern in unsafe):
                    raise ValueError(f"Unsafe rendered output in {notebook_name}; notebook was not written.")


def execute_notebook(
    path: Path,
    kernel: str,
    timeout_seconds: int,
    environment: dict[str, str],
) -> None:
    notebook = prepare_sources(path)
    client = NotebookClient(
        notebook,
        timeout=timeout_seconds,
        kernel_name=kernel,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    executed = client.execute(cwd=str(path.parent), env=environment)
    validate_safe_outputs(executed, path.name)
    nbformat.write(executed, path)
    print(f"Executed and saved safe outputs: {path.name}")


def main() -> int:
    args = parse_args()
    source_csv_root = args.source_csv_root.resolve()
    mura_data_root = args.mura_data_root.resolve()
    raw_zip = args.raw_zip.resolve()
    artifacts_root = args.artifacts_root.resolve()
    for path in (source_csv_root, mura_data_root, raw_zip):
        if not path.exists():
            raise FileNotFoundError(path)
    if "valid" in str(source_csv_root).lower() or "valid" in str(mura_data_root).lower():
        raise ValueError("This runner accepts the official MURA train population only.")

    artifacts_root.mkdir(parents=True, exist_ok=True)
    # NotebookClient resolves the kernel before it passes the execution
    # environment to that kernel, so set the parent process path as well.
    os.environ["JUPYTER_PATH"] = str(args.jupyter_path.resolve())
    environment = os.environ.copy()
    environment["JUPYTER_PATH"] = str(args.jupyter_path.resolve())
    environment["IPYTHONDIR"] = str(artifacts_root / "ipython")
    environment.update(
        {
            "MURA_SOURCE_CSV_DIR": str(source_csv_root),
            "MURA_DATA_ROOT": str(mura_data_root),
            "MURA_RAW_ZIP": str(raw_zip),
            "MURA_PROJECT_DATA_DIR": str(artifacts_root),
            "MURA_DATA_DIR": str(artifacts_root),
            "MURA_SPLITS_DIR": str(artifacts_root / "WRIST_SPLITS"),
            "MURA_PREPROCESSING_DIR": str(artifacts_root / "PREPROCESSED_WRIST"),
            "MURA_MODEL_A_DIR": str(artifacts_root / "BASELINE_MODEL_A"),
            "MURA_MODEL_B_DIR": str(artifacts_root / "MODEL_B_FINE_TUNING"),
            "MURA_SHORTCUT_OUTPUT": str(artifacts_root / "SHORTCUT_AUDIT"),
            "MURA_PROJECT_ROOT": str(ROOT),
        }
    )
    start = NOTEBOOK_NAMES.index(next(name for name in NOTEBOOK_NAMES if name.startswith(args.start_at)))
    stop = NOTEBOOK_NAMES.index(next(name for name in NOTEBOOK_NAMES if name.startswith(args.stop_after)))
    if stop < start:
        raise ValueError("--stop-after must not come before --start-at.")
    for name in NOTEBOOK_NAMES[start : stop + 1]:
        execute_notebook(ROOT / "notebooks" / name, args.kernel, args.timeout_seconds, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
