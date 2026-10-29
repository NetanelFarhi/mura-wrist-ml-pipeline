"""Synthetic tests for verified reconstruction of a split Model B checkpoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.reassemble_model_b_checkpoint import BundleVerificationError, reassemble_checkpoint


def write_parts(parts_dir: Path, payload: bytes) -> tuple[Path, str]:
    parts_dir.mkdir()
    pieces = [payload[:7], payload[7:]]
    manifest_parts = []
    for index, piece in enumerate(pieces):
        name = f"best_resnet18_study_level.pt.part-{index:03d}"
        path = parts_dir / name
        path.write_bytes(piece)
        manifest_parts.append(
            {"file": name, "bytes": len(piece), "sha256": hashlib.sha256(piece).hexdigest().upper()}
        )
    target_hash = hashlib.sha256(payload).hexdigest().upper()
    (parts_dir / "checkpoint_parts.json").write_text(
        json.dumps(
            {
                "target_file": "best_resnet18_study_level.pt",
                "target_sha256": target_hash,
                "parts": manifest_parts,
            }
        ),
        encoding="utf-8",
    )
    return parts_dir, target_hash


def test_reassembles_only_when_every_part_and_final_hash_match(tmp_path: Path) -> None:
    parts_dir, expected_hash = write_parts(tmp_path / "parts", b"synthetic checkpoint payload")
    result = reassemble_checkpoint(parts_dir, tmp_path / "bundle")
    restored = tmp_path / "bundle" / "best_resnet18_study_level.pt"
    assert result["status"] == "reassembled_and_verified"
    assert result["checkpoint_sha256"] == expected_hash
    assert restored.read_bytes() == b"synthetic checkpoint payload"


def test_reassembly_rejects_a_modified_part(tmp_path: Path) -> None:
    parts_dir, _ = write_parts(tmp_path / "parts", b"synthetic checkpoint payload")
    (parts_dir / "best_resnet18_study_level.pt.part-001").write_bytes(b"modified")
    with pytest.raises(BundleVerificationError, match="byte count|SHA-256"):
        reassemble_checkpoint(parts_dir, tmp_path / "bundle")


def test_reassembly_rejects_a_manifest_path_outside_the_output_directory(tmp_path: Path) -> None:
    parts_dir, _ = write_parts(tmp_path / "parts", b"synthetic checkpoint payload")
    manifest_path = parts_dir / "checkpoint_parts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_file"] = "../not_a_checkpoint.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleVerificationError, match="invalid target_file"):
        reassemble_checkpoint(parts_dir, tmp_path / "bundle")
