"""Build and execute a read-only notebook for GitHub-visible project results.

The notebook is intentionally restricted to already committed report artifacts.
It never reads raw MURA images, model checkpoints, prediction caches, or the
official-valid labels; it does not train, evaluate, calibrate, or select a
threshold.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "06_results_viewer.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(source)


def code_cell(source: str):
    return nbf.v4.new_code_cell(source)


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (project results viewer)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"},
    }
    notebook.cells = [
        markdown_cell(
            "# MURA XR_WRIST - saved results viewer\n\n"
            "This notebook is a **read-only presentation of committed project artifacts** for GitHub. "
            "It does not access raw MURA images, retrain a model, recompute metrics, select a threshold, "
            "or use official MURA `valid` for any new decision.\n\n"
            "The historical Phase-1 official-valid reference has already been observed. All later Phase-2 "
            "material shown here is explicitly development/validation evidence, not an untouched final-test claim."
        ),
        code_cell(
            "from pathlib import Path\n"
            "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "SAFE_ARTIFACTS = [\n"
            "    ROOT / 'reports' / 'results.md',\n"
            "    ROOT / 'reports' / 'team_share' / 'PHASE1_5_KEY_FIGURE.png',\n"
            "    ROOT / 'reports' / 'phase2a' / 'operating_points.csv',\n"
            "    ROOT / 'reports' / 'phase2b' / 'model_comparison.csv',\n"
            "    ROOT / 'reports' / 'TEST_SET_GOVERNANCE.md',\n"
            "]\n"
            "missing = [str(path.relative_to(ROOT)) for path in SAFE_ARTIFACTS if not path.exists()]\n"
            "if missing:\n"
            "    raise FileNotFoundError(f'Missing committed presentation artifacts: {missing}')\n"
            "print('Read-only artifact check passed.')\n"
            "print('Project root:', ROOT)"
        ),
        markdown_cell("## Historical Phase-1 baseline reference"),
        code_cell(
            "import pandas as pd\n\n"
            "# Values transcribed from the committed reports/results.md reference table.\n"
            "# This cell displays preserved history; it does not recalculate anything.\n"
            "phase1_reference = pd.DataFrame([\n"
            "    {\n"
            "        'Model': 'Frozen ResNet-18 embeddings + Logistic Regression',\n"
            "        'AUROC': 0.876, 'AUPRC': 0.860, 'Sensitivity': 0.825,\n"
            "        'Specificity': 0.807, 'F1': 0.784, 'False negatives': '17 / 97',\n"
            "    },\n"
            "    {\n"
            "        'Model': 'Fine-tuned ResNet-18',\n"
            "        'AUROC': 0.897, 'AUPRC': 0.894, 'Sensitivity': 0.732,\n"
            "        'Specificity': 0.950, 'F1': 0.811, 'False negatives': '26 / 97',\n"
            "    },\n"
            "])\n"
            "phase1_reference"
        ),
        code_cell(
            "from IPython.display import Markdown, display\n\n"
            "display(Markdown((ROOT / 'reports' / 'results.md').read_text(encoding='utf-8')))"
        ),
        markdown_cell("## Validation threshold trade-off (saved Phase 1.5 figure)"),
        code_cell(
            "from IPython.display import Image, display\n\n"
            "display(Image(filename=str(ROOT / 'reports' / 'team_share' / 'PHASE1_5_KEY_FIGURE.png')))"
        ),
        markdown_cell("## Development-only multi-view results"),
        code_cell(
            "phase2b = pd.read_csv(ROOT / 'reports' / 'phase2b' / 'model_comparison.csv')\n"
            "phase2b = phase2b.rename(columns={\n"
            "    'auroc': 'AUROC', 'auprc': 'AUPRC',\n"
            "    'specificity_at_0.85': 'Specificity @ >=85% sensitivity',\n"
            "    'specificity_at_0.9': 'Specificity @ >=90% sensitivity',\n"
            "})\n"
            "phase2b[[\n"
            "    'model', 'AUROC', 'AUPRC',\n"
            "    'Specificity @ >=85% sensitivity',\n"
            "    'Specificity @ >=90% sensitivity',\n"
            "]]"
        ),
        markdown_cell("## Test-set governance"),
        code_cell(
            "display(Markdown((ROOT / 'reports' / 'TEST_SET_GOVERNANCE.md').read_text(encoding='utf-8')))"
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)

    # The registered python3 kernel calls `python`; prepend this interpreter's
    # directory so execution uses the same known Anaconda environment.
    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
    executed = NotebookClient(notebook, timeout=90, kernel_name="python3").execute(cwd=str(ROOT))
    nbf.write(executed, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
