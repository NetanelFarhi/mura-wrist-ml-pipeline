#!/usr/bin/env python
"""Reassemble the canonical Model B checkpoint from verified Drive parts.

The shared Drive connector cannot upload a single 134 MB checkpoint. This
utility joins the explicitly listed checkpoint parts into the expected .pt
file, verifies the SHA-256 of every part and of the final checkpoint, and
refuses to overwrite a file with a different identity. It reads no MURA image
data and does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class BundleVerificationError(ValueError):
    """Raised when the supplied checkpoint parts cannot prove their identity."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return an uppercase SHA-256 digest without loading a file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleVerificationError(f"Could not read checkpoint-parts manifest: {path}") from error
    if not isinstance(data, dict):
        raise BundleVerificationError("Checkpoint-parts manifest must be a JSON object")
    return data


def reassemble_checkpoint(parts_dir: Path, output_dir: Path) -> dict[str, object]:
    """Write one verified checkpoint from a verified, ordered parts manifest."""

    manifest_path = parts_dir / "checkpoint_parts.json"
    manifest = _load_manifest(manifest_path)
    target_name = manifest.get("target_file")
    target_sha256 = str(manifest.get("target_sha256", "")).upper()
    parts = manifest.get("parts")
    if target_name != "best_resnet18_study_level.pt":
        raise BundleVerificationError("Checkpoint-parts manifest has an invalid target_file")
    if len(target_sha256) != 64:
        raise BundleVerificationError("Checkpoint-parts manifest has an invalid target_sha256")
    if not isinstance(parts, list) or not parts:
        raise BundleVerificationError("Checkpoint-parts manifest has no parts")

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / target_name
    if target.exists():
        existing_hash = sha256_file(target)
        if existing_hash == target_sha256:
            return {
                "status": "already_verified",
                "checkpoint": str(target),
                "checkpoint_sha256": existing_hash,
            }
        raise BundleVerificationError(
            f"Refusing to overwrite a different checkpoint: {target.name}. "
            "Move it aside only after confirming its provenance."
        )

    temp_target = output_dir / f".{target_name}.reassembling"
    if temp_target.exists():
        temp_target.unlink()

    total_bytes = 0
    seen_part_names: set[str] = set()
    try:
        with temp_target.open("wb") as target_handle:
            for index, part in enumerate(parts):
                if not isinstance(part, dict):
                    raise BundleVerificationError(f"Invalid part entry at index {index}")
                part_name = part.get("file")
                expected_hash = str(part.get("sha256", "")).upper()
                expected_bytes = part.get("bytes")
                if (
                    not isinstance(part_name, str)
                    or Path(part_name).name != part_name
                    or part_name in seen_part_names
                    or len(expected_hash) != 64
                    or not isinstance(expected_bytes, int)
                    or expected_bytes <= 0
                ):
                    raise BundleVerificationError(f"Invalid manifest fields for part {index}")
                seen_part_names.add(part_name)
                source = parts_dir / part_name
                if not source.is_file():
                    raise BundleVerificationError(f"Missing checkpoint part: {part_name}")
                if source.stat().st_size != expected_bytes:
                    raise BundleVerificationError(f"Unexpected byte count for checkpoint part: {part_name}")
                if sha256_file(source) != expected_hash:
                    raise BundleVerificationError(f"Unexpected SHA-256 for checkpoint part: {part_name}")
                with source.open("rb") as source_handle:
                    while chunk := source_handle.read(1024 * 1024):
                        target_handle.write(chunk)
                        total_bytes += len(chunk)

        final_hash = sha256_file(temp_target)
        if final_hash != target_sha256:
            raise BundleVerificationError("Reassembled checkpoint SHA-256 does not match the manifest")
        os.replace(temp_target, target)
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise

    return {
        "status": "reassembled_and_verified",
        "checkpoint": str(target),
        "checkpoint_sha256": target_sha256,
        "bytes_written": total_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(reassemble_checkpoint(args.parts_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
