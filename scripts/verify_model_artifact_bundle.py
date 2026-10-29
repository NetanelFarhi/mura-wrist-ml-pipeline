#!/usr/bin/env python
"""Fail closed when a local Model B artifact directory is not the canonical bundle.

This checker reads no radiographs and performs no evaluation. It verifies that
the exact local/Drive artifacts expected by Notebooks 05 and 06 are present,
that the checkpoint hash is known, and that the saved operating policy was
selected from internal validation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = REPOSITORY_ROOT / "configs" / "model_b_artifact_bundle.json"


class BundleVerificationError(ValueError):
    """Raised when a Model B artifact bundle cannot be verified safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleVerificationError(f"Could not read JSON artifact: {path.name}") from error
    if not isinstance(data, dict):
        raise BundleVerificationError(f"Expected a JSON object in {path.name}")
    return data


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return an uppercase SHA-256 digest without loading a checkpoint into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_bundle(artifact_dir: Path, bundle_spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Verify the expected Model B artifact bundle and return a safe summary."""

    spec = _load_json(bundle_spec_path)
    required_files = spec.get("required_files")
    if not isinstance(required_files, dict) or not required_files:
        raise BundleVerificationError("Bundle specification has no required_files mapping")

    missing = [name for name in required_files if not (artifact_dir / name).is_file()]
    if missing:
        raise BundleVerificationError(
            "Canonical Model B bundle is incomplete. Missing: " + ", ".join(sorted(missing))
        )

    checkpoint = artifact_dir / "best_resnet18_study_level.pt"
    expected_hash = str(required_files[checkpoint.name].get("sha256", "")).upper()
    actual_hash = sha256_file(checkpoint)
    if expected_hash and actual_hash != expected_hash:
        raise BundleVerificationError(
            "Unexpected Model B checkpoint hash. Do not silently substitute another experiment."
        )

    selection = _load_json(artifact_dir / "selection.json")
    policy = _load_json(artifact_dir / "frozen_primary_operating_policy.json")
    model = spec.get("model", {})
    expected_threshold = float(model["threshold"])
    target = float(model["operating_policy"].rsplit(">=", maxsplit=1)[1].strip())

    if selection.get("selection_data") != "validation only":
        raise BundleVerificationError("Model B selection.json does not declare validation-only selection")
    if selection.get("test_used_for_selection") is not False:
        raise BundleVerificationError("Model B selection.json does not rule out test use")

    selection_policy = selection.get("threshold_selection", {})
    if selection_policy.get("selection_rule") != "highest_threshold_with_sensitivity_at_least_target":
        raise BundleVerificationError("Unexpected Model B threshold-selection rule")
    if not math.isclose(float(selection_policy.get("target_sensitivity")), target, abs_tol=1e-12):
        raise BundleVerificationError("Unexpected Model B sensitivity target")
    if not math.isclose(float(selection_policy.get("threshold")), expected_threshold, abs_tol=1e-12):
        raise BundleVerificationError("Unexpected Model B selected threshold")

    if policy.get("source_split") != "internal_validation":
        raise BundleVerificationError("Frozen policy does not name internal validation as its source")
    if policy.get("study_aggregation") != "mean_probability":
        raise BundleVerificationError("Frozen policy does not use mean probability pooling")
    if policy.get("test_used_for_selection") is not False:
        raise BundleVerificationError("Frozen policy does not rule out test use")
    if not math.isclose(float(policy.get("threshold")), expected_threshold, abs_tol=1e-12):
        raise BundleVerificationError("Frozen policy threshold does not match bundle specification")

    return {
        "bundle_id": spec.get("bundle_id"),
        "artifact_dir": str(artifact_dir),
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": actual_hash,
        "selection_population": selection.get("selection_data"),
        "threshold": expected_threshold,
        "target_sensitivity": target,
        "study_aggregation": policy.get("study_aggregation"),
        "official_mura_valid_read_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle-spec", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.artifact_dir, args.bundle_spec), indent=2))


if __name__ == "__main__":
    main()
