#!/usr/bin/env python
"""Generate the code-only local/Colab MURA study viewer notebook.

The notebook deliberately ships with no embedded radiographs or cell outputs.
It is for an authorised local MURA copy only; GitHub receives the reproducible
viewer code, not MURA pixels.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08_MURA_Runtime_Study_Viewer.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def build() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    notebook.cells = [
        markdown(
            "# MURA XR_WRIST — local study viewer\n\n"
            "This optional viewer displays separate radiograph panels from one study **only while running locally or in authorised Colab storage**. "
            "It reads internal-train data only and makes no prediction, threshold decision, or model change. "
            "Keep `SHOW_RUNTIME_PREVIEW = False` and clear outputs before committing: MURA image pixels must not be stored in Git, including a private repository."
        ),
        code(
            "from pathlib import Path\n"
            "import math\n"
            "import matplotlib.pyplot as plt\n"
            "import pandas as pd\n"
            "from PIL import Image\n\n"
            "# Set these local/Colab paths for your authorised MURA copy.\n"
            "DATA_ROOT = Path('/content/drive/MyDrive/DATA/MURA-v1.1')\n"
            "MANIFEST_PATH = Path('/content/drive/MyDrive/DATA/WRIST_SPLITS/wrist_manifest_train_only.csv')\n\n"
            "# Privacy/governance default: the public notebook never embeds an X-ray output.\n"
            "SHOW_RUNTIME_PREVIEW = False\n"
            "STUDY_ID = None  # Optional local study ID. None selects one deterministic internal-train example.\n"
            "MAX_VIEWS_TO_SHOW = 6\n"
            "RANDOM_SEED = 42"
        ),
        markdown(
            "## Runtime-only preview\n\n"
            "When enabled, each real view is shown in its own panel rather than merged into one image. "
            "The code does not print patient IDs, study IDs, paths, labels, predictions, or pixel files. It is a visual data-quality check only."
        ),
        code(
            "if SHOW_RUNTIME_PREVIEW:\n"
            "    if not DATA_ROOT.exists() or not MANIFEST_PATH.exists():\n"
            "        raise FileNotFoundError('Set DATA_ROOT and MANIFEST_PATH to your authorised local MURA copy.')\n\n"
            "    rows = pd.read_csv(MANIFEST_PATH)\n"
            "    required = {'study_id', 'image_path', 'source_split'}\n"
            "    missing = required - set(rows.columns)\n"
            "    if missing:\n"
            "        raise ValueError(f'Manifest is missing: {sorted(missing)}')\n"
            "    rows = rows.loc[rows['source_split'].eq('train')].copy()\n"
            "    rows['image_path'] = rows['image_path'].astype(str).str.replace('\\\\', '/', regex=False)\n"
            "    if not rows['image_path'].str.startswith('train/').all():\n"
            "        raise ValueError('Viewer accepts official-train paths only.')\n\n"
            "    grouped = list(rows.sort_values(['study_id', 'image_path']).groupby('study_id', sort=True))\n"
            "    if STUDY_ID is None:\n"
            "        candidates = [(key, frame) for key, frame in grouped if 2 <= len(frame) <= MAX_VIEWS_TO_SHOW]\n"
            "        if not candidates:\n"
            "            candidates = [(key, frame) for key, frame in grouped if len(frame) >= 1]\n"
            "        if not candidates:\n"
            "            raise RuntimeError('No train studies were found in this manifest.')\n"
            "        selected_key, selected = candidates[RANDOM_SEED % len(candidates)]\n"
            "    else:\n"
            "        selected = rows.loc[rows['study_id'].eq(STUDY_ID)].sort_values('image_path')\n"
            "        if selected.empty:\n"
            "            raise ValueError('The requested local STUDY_ID is not in the internal-train manifest.')\n"
            "        selected_key = STUDY_ID\n\n"
            "    selected = selected.head(MAX_VIEWS_TO_SHOW).reset_index(drop=True)\n"
            "    columns = min(3, len(selected))\n"
            "    rows_count = math.ceil(len(selected) / columns)\n"
            "    figure, axes = plt.subplots(rows_count, columns, figsize=(4 * columns, 5 * rows_count))\n"
            "    axes = list(getattr(axes, 'flat', [axes]))\n"
            "    for view_number, (_, record) in enumerate(selected.iterrows(), start=1):\n"
            "        with Image.open(DATA_ROOT / record['image_path']) as image:\n"
            "            axes[view_number - 1].imshow(image.convert('L'), cmap='gray')\n"
            "        axes[view_number - 1].set_title(f'View {view_number} of {len(selected)}')\n"
            "        axes[view_number - 1].axis('off')\n"
            "    for axis in axes[len(selected):]:\n"
            "        axis.axis('off')\n"
            "    figure.suptitle('Runtime-only study preview — internal train', y=0.99)\n"
            "    figure.tight_layout()\n"
            "    plt.show()"
        ),
        markdown(
            "## Before a Git commit\n\n"
            "1. Set `SHOW_RUNTIME_PREVIEW = False`.\n"
            "2. Use **Edit → Clear all outputs** in Colab/Jupyter.\n"
            "3. Commit the notebook code and the aggregate project charts/tables, never the displayed radiographs."
        ),
    ]
    return notebook


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build(), NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
