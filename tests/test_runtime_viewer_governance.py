"""Ensure the optional local radiograph viewer cannot publish pixel outputs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08_MURA_Runtime_Study_Viewer.ipynb"


class RuntimeViewerGovernanceTests(unittest.TestCase):
    def test_viewer_is_code_only_with_preview_disabled(self) -> None:
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        source = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
        self.assertIn("SHOW_RUNTIME_PREVIEW = False", source)
        self.assertIn("source_split", source)
        self.assertIn("eq('train')", source)
        for cell in notebook.cells:
            self.assertEqual(cell.get("outputs", []), [], "Runtime viewer must not commit any output.")


if __name__ == "__main__":
    unittest.main()
