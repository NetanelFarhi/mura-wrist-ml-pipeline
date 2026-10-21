"""Fail if staged/tracked content looks like protected data or credentials.

This is a lightweight pre-commit guard, not a replacement for manual review.
Run ``python scripts/check_repo_data_governance.py --staged`` before committing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


RAW_DATA_PATH = re.compile(
    r"(?:^|/)(?:data/raw|MURA-v\d+(?:\.\d+)?|XR_WRIST)(?:/|$)", re.IGNORECASE
)
RAW_IMAGE = re.compile(
    r"\.(?:dcm|png|jpe?g|bmp|tiff?)$", re.IGNORECASE
)
SECRET_PATH = re.compile(r"(?:^|/)(?:\.env(?:\.|$)|[^/]+\.(?:pem|key))", re.IGNORECASE)
IDENTIFIER_LIKE = re.compile(r"\b(?:patient|study)[_-]?\d{3,}\b", re.IGNORECASE)
WINDOWS_ABSOLUTE_PATH = re.compile(r"\b[A-Za-z]:[\\/]", re.IGNORECASE)
PUBLIC_TEXT_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".yaml", ".yml"}


def listed_paths(staged: bool) -> list[str]:
    # This script reads only path names.  The temporary safe-directory setting
    # keeps the check usable in sandboxed/CI environments where Git sees a
    # different filesystem owner than the repository owner.
    prefix = ["git", "-c", "safe.directory=*"]
    command = prefix + (["diff", "--cached", "--name-only", "-z"] if staged else ["ls-files", "-z"])
    result = subprocess.run(command, check=True, capture_output=True)
    return [item.decode("utf-8", errors="replace") for item in result.stdout.split(b"\0") if item]


def unsafe_paths(paths: list[str]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if SECRET_PATH.search(normalized):
            violations.append(f"credential-like path: {path}")
        elif RAW_DATA_PATH.search(normalized) and RAW_IMAGE.search(normalized):
            violations.append(f"raw-medical-image-like path: {path}")
    return violations


def _is_public_text_artifact(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if Path(normalized).suffix.lower() not in PUBLIC_TEXT_EXTENSIONS:
        return False
    return (
        normalized.startswith("reports/")
        or normalized.startswith("docs/results/")
        or normalized in {"README.md", "CHARTER.md", "REQUIREMENTS_TRACEABILITY.md"}
    )


def unsafe_public_content(paths: list[str]) -> list[str]:
    """Reject identifiers and personal paths in GitHub-facing text artifacts.

    The check intentionally excludes code and notebooks: those may contain safe
    local setup examples. It protects published reports/results, where row-level
    data and machine-specific paths have no place.
    """

    violations: list[str] = []
    for path in paths:
        if not _is_public_text_artifact(path):
            continue
        artifact = Path(path)
        if not artifact.is_file():
            # A staged deletion has no working-tree file to inspect.
            continue
        try:
            text = artifact.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"non-text public artifact: {path}")
            continue
        if IDENTIFIER_LIKE.search(text):
            violations.append(f"identifier-like value in public artifact: {path}")
        if WINDOWS_ABSOLUTE_PATH.search(text):
            violations.append(f"Windows absolute path in public artifact: {path}")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="inspect the current Git index instead of all tracked files",
    )
    args = parser.parse_args()
    try:
        paths = listed_paths(args.staged)
        violations = unsafe_paths(paths) + unsafe_public_content(paths)
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Could not read Git paths: {error}", file=sys.stderr)
        return 2

    if violations:
        print("Refusing to proceed: possible protected data or credentials found:")
        print("\n".join(f"- {item}" for item in violations))
        return 1

    scope = "staged" if args.staged else "tracked"
    print(f"Data-governance check passed: no unsafe {scope} paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
