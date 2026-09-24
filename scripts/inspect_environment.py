"""Write a privacy-conscious Phase 1 runtime inventory without requiring ML packages."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _torch_inventory() -> dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    payload: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        payload["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return payload


def _nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    return completed.stdout.strip() or None


def collect_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch": _torch_inventory(),
        "nvidia_smi": _nvidia_smi(),
        "note": (
            "This inventory records runtime capabilities only. "
            "The training scripts select CPU unless torch reports CUDA availability."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the Phase 1 runtime inventory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/environment.json"),
        help="JSON output path relative to the project root.",
    )
    args = parser.parse_args()

    payload = collect_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

