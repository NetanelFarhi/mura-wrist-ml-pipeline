#!/usr/bin/env python
"""Fail closed unless all seven all-anatomy notebooks have safe executed output.

This is a publishing-quality audit for the local report notebooks. It checks
that every notebook is executed, has no cell error, contains aggregate output,
does not embed a raw radiograph display in code, and does not expose an
absolute local Windows path in its output. It does not inspect or modify model
results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPECTED = (
    "10_MURA_All_Anatomies_Data_Preparation.ipynb",
    "11_MURA_All_Anatomies_Preprocessing.ipynb",
    "12_MURA_All_Anatomies_Baseline_ResNet18_LogReg.ipynb",
    "13_MURA_All_Anatomies_Development_Evaluation.ipynb",
    "14_MURA_All_Anatomies_FineTuned_ResNet18.ipynb",
    "15_MURA_All_Anatomies_Model_Comparison.ipynb",
    "16_MURA_All_Anatomies_Shortcut_Audit.ipynb",
)
FORBIDDEN_CODE_FRAGMENTS = ("Image.open(", ".imshow(", "plt.imshow(")
FORBIDDEN_OUTPUT_FRAGMENTS = ("C:\\Users\\", "Traceback (most recent call last)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _output_text(output: dict[str, object]) -> str:
    value = output.get("text", "")
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    return str(value)


def audit_notebook(path: Path) -> dict[str, object]:
    try:
        import nbformat
    except ModuleNotFoundError as exc:  # pragma: no cover - installation dependent.
        raise RuntimeError("nbformat is required for notebook output audit.") from exc
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors: list[str] = []
    output_count = 0
    for index, cell in enumerate(code_cells, start=1):
        source = str(cell.get("source", ""))
        for fragment in FORBIDDEN_CODE_FRAGMENTS:
            if fragment in source:
                errors.append(f"code cell {index} contains forbidden raw-image display fragment {fragment!r}")
        outputs = list(cell.get("outputs", []))
        output_count += len(outputs)
        if not outputs:
            errors.append(f"code cell {index} has no saved output")
        for output in outputs:
            if output.get("output_type") == "error":
                errors.append(f"code cell {index} contains an execution error")
            text = _output_text(output)
            for fragment in FORBIDDEN_OUTPUT_FRAGMENTS:
                if fragment in text:
                    errors.append(f"code cell {index} exposes forbidden output fragment {fragment!r}")
    if not code_cells:
        errors.append("notebook has no code cells")
    return {
        "notebook": path.name,
        "code_cells": len(code_cells),
        "saved_outputs": output_count,
        "passed": not errors,
        "issues": errors,
    }


def run(args: argparse.Namespace) -> Path:
    notebooks_dir = args.notebooks_dir.resolve()
    output = args.output.resolve()
    missing = [name for name in EXPECTED if not (notebooks_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing expected all-anatomy notebooks: " + ", ".join(missing))
    results = [audit_notebook(notebooks_dir / name) for name in EXPECTED]
    report = {
        "expected_notebooks": list(EXPECTED),
        "all_passed": all(bool(item["passed"]) for item in results),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["all_passed"]:
        failures = [f"{item['notebook']}: {item['issues']}" for item in results if not item["passed"]]
        raise RuntimeError("Notebook output audit failed. " + " | ".join(failures))
    print(f"Notebook output audit passed: {len(results)} notebooks, {sum(int(item['saved_outputs']) for item in results)} saved outputs.")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
