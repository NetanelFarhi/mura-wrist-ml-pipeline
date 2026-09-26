"""Fail if staged/tracked paths look like raw medical data or credentials.

This is a lightweight pre-commit guard, not a replacement for manual review.
Run ``python scripts/check_repo_data_governance.py --staged`` before committing.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


RAW_DATA_PATH = re.compile(
    r"(?:^|/)(?:data/raw|MURA[^/]*|XR_WRIST)(?:/|$)", re.IGNORECASE
)
RAW_IMAGE = re.compile(
    r"\.(?:dcm|png|jpe?g|bmp|tiff?)$", re.IGNORECASE
)
SECRET_PATH = re.compile(r"(?:^|/)(?:\.env(?:\.|$)|[^/]+\.(?:pem|key))", re.IGNORECASE)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="inspect the current Git index instead of all tracked files",
    )
    args = parser.parse_args()
    try:
        violations = unsafe_paths(listed_paths(args.staged))
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
