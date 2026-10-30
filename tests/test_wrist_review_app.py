"""Safety checks for the local Wrist Review app configuration.

These tests do not load a checkpoint or any radiograph.  They verify that the
local app resolves its repository paths predictably and refuses inference when
the required, non-versioned Model B bundle is absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPOSITORY_ROOT / "apps" / "wrist_review" / "core.py"


def load_app_core():
    """Load the standalone app adapter without requiring an apps package."""

    module_name = "wrist_review_app_core_for_test"
    specification = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_local_app_default_paths_resolve_inside_repository(monkeypatch):
    monkeypatch.delenv("MURA_APP_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("MURA_APP_MODEL_DIR", raising=False)

    app_core = load_app_core()
    configuration = app_core.resolve_configuration()

    assert configuration.project_root == REPOSITORY_ROOT.resolve()
    assert configuration.model_dir == (
        REPOSITORY_ROOT / "local_artifacts" / "MODEL_B_FINE_TUNING"
    ).resolve()


def test_local_app_fails_closed_without_model_bundle(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    missing_bundle = tmp_path / "missing_bundle"
    monkeypatch.setenv("MURA_APP_PROJECT_ROOT", str(project_root))
    monkeypatch.setenv("MURA_APP_MODEL_DIR", str(missing_bundle))

    app_core = load_app_core()
    status = app_core.artifact_status(app_core.resolve_configuration())

    assert status["ready"] is False
    assert status["issues"] == ["model_bundle_missing"]
