"""Local HTTP shell for the Wrist Review mobile interface.

This server deliberately reuses the checked Model B runtime in ``core.py``.
It keeps uploaded images in process memory only and never exposes a route for
training, threshold editing, or result persistence.
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import sys
import threading
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core
from PIL import Image, UnidentifiedImageError
from src.runtime.deterministic_safety import StudyImage


STATIC_ROOT = APP_ROOT / "static"
MAX_REQUEST_BYTES = 40 * 1024 * 1024
MAX_VIEWS = 8
MAX_WORKLIST_ZIP_BYTES = 100 * 1024 * 1024
MAX_WORKLIST_ZIP_FILES = 400
MAX_WORKLIST_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"})

_runtime: core.ModelRuntime | None = None
_runtime_lock = threading.Lock()
_status: dict[str, Any] | None = None
_status_lock = threading.Lock()


def verified_status() -> dict[str, Any]:
    """Return a cached, checksum-verified artifact status."""

    global _status
    with _status_lock:
        if _status is None:
            _status = core.artifact_status(core.resolve_configuration())
        return _status


def verified_runtime() -> core.ModelRuntime:
    """Load the immutable inference runtime once, on the first actual review."""

    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = core.load_verified_runtime(core.resolve_configuration())
        return _runtime


def decode_view(item: Any, index: int) -> StudyImage:
    """Decode a browser data URL into a guarded, in-memory study view."""

    if not isinstance(item, dict):
        raise ValueError("Each uploaded view must be an object.")
    name = str(item.get("name") or f"view_{index + 1}").strip()
    encoded = item.get("data_url")
    if not isinstance(encoded, str) or "," not in encoded:
        raise ValueError(f"{name}: image data was missing.")
    _prefix, payload = encoded.split(",", 1)
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError(f"{name}: image data could not be read.") from exc
    if not image_bytes:
        raise ValueError(f"{name}: image is empty.")
    if len(image_bytes) > 15 * 1024 * 1024:
        raise ValueError(f"{name}: image exceeds the 15 MB local limit.")
    return StudyImage(image_id=name, image_bytes=image_bytes)


def extract_worklist_archive(archive_bytes: bytes) -> dict[str, Any]:
    """Read an image-only patient worklist ZIP entirely in process memory.

    The archive is never extracted to disk.  Strict compressed, uncompressed,
    file-count, path, and per-image limits protect this local prototype from
    malformed or unexpectedly large archives.
    """

    if not archive_bytes:
        raise ValueError("The ZIP archive is empty.")
    if len(archive_bytes) > MAX_WORKLIST_ZIP_BYTES:
        raise ValueError("The ZIP archive exceeds the 100 MB local limit.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("The selected file is not a readable ZIP archive.") from exc

    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) > MAX_WORKLIST_ZIP_FILES:
            raise ValueError("The ZIP archive contains more than 400 files.")
        uncompressed_size = sum(info.file_size for info in infos)
        if uncompressed_size > MAX_WORKLIST_UNCOMPRESSED_BYTES:
            raise ValueError("The ZIP archive expands beyond the 50 MB local limit.")

        files: list[dict[str, str]] = []
        for info in infos:
            raw_name = info.filename.replace("\\", "/")
            path = PurePosixPath(raw_name)
            if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
                raise ValueError("The ZIP archive contains an unsafe file path.")
            if path.parts[0] == "__MACOSX" or path.name.startswith("."):
                continue
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if info.flag_bits & 0x1:
                raise ValueError("Password-protected ZIP archives are not supported.")
            if info.file_size > MAX_IMAGE_BYTES:
                raise ValueError(f"{path.name}: image exceeds the 15 MB local limit.")
            image_bytes = archive.read(info)
            try:
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
                raise ValueError(f"{path.name}: archive entry is not a readable image.") from exc
            mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            files.append(
                {
                    "name": path.name,
                    "relative_path": path.as_posix(),
                    "mime_type": mime_type,
                    "data_url": f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}",
                }
            )

    if not files:
        raise ValueError("No supported image files were found in the ZIP archive.")
    return {"files": files, "image_count": len(files), "storage": "in_memory_only"}


class WristReviewHandler(BaseHTTPRequestHandler):
    """Serve static UI assets and the narrowly-scoped local review API."""

    server_version = "WristReviewLocal/1.0"

    def log_message(self, _format: str, *args: object) -> None:
        """Keep routine browser logging quiet; no identifiers are logged."""

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/status":
            status = verified_status()
            from src.runtime.crewai_research import research_agent_status

            research = research_agent_status(core.resolve_configuration().project_root)
            safe = {
                "ready": bool(status.get("ready")),
                "model": status.get("model"),
                "threshold": status.get("threshold"),
                "target_sensitivity": status.get("target_sensitivity"),
                "validation_sensitivity": status.get("validation_sensitivity"),
                "validation_specificity": status.get("validation_specificity"),
                "study_aggregation": status.get("study_aggregation"),
                "selection_split": status.get("selection_split"),
                "optional_tool_policy": status.get("optional_tool_policy", {}),
                "issues": status.get("issues", []),
                "research_agent": {
                    "enabled": bool(research.get("enabled")),
                    "ready": bool(research.get("ready")),
                    "agent_version": research.get("agent_version"),
                    "mode": research.get("mode"),
                    "max_iterations": research.get("max_iterations"),
                    "max_llm_calls": research.get("max_llm_calls"),
                    "raw_images_shared": bool(research.get("raw_images_shared")),
                    "reason": research.get("reason"),
                },
            }
            return self.send_json(HTTPStatus.OK, safe)
        return self.serve_static(route)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route not in {"/api/review", "/api/worklist-zip"}:
            return self.send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint."})
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        max_size = MAX_WORKLIST_ZIP_BYTES if route == "/api/worklist-zip" else MAX_REQUEST_BYTES
        if not 0 < content_length <= max_size:
            label = "ZIP archive" if route == "/api/worklist-zip" else "Study payload"
            return self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"{label} is too large."})
        if route == "/api/worklist-zip":
            try:
                archive_result = extract_worklist_archive(self.rfile.read(content_length))
                return self.send_json(HTTPStatus.OK, archive_result)
            except (RuntimeError, ValueError, OSError, zipfile.BadZipFile) as exc:
                return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Could not read the review request."})
        try:
            raw_views = payload.get("views") if isinstance(payload, dict) else None
            if not isinstance(raw_views, list) or not raw_views:
                raise ValueError("Add at least one image before reviewing a study.")
            if len(raw_views) > MAX_VIEWS:
                raise ValueError(f"A study can contain up to {MAX_VIEWS} views.")
            images = [decode_view(item, index) for index, item in enumerate(raw_views)]
            duplicate_ids = [image.image_id for image in images]
            if len(duplicate_ids) != len(set(duplicate_ids)):
                raise ValueError("Each selected file needs a unique filename.")
            result = core.run_guarded_review(
                verified_runtime(),
                case_alias=str(payload.get("case_alias") or "LOCAL_CASE"),
                declared_anatomy=str(payload.get("declared_anatomy") or ""),
                images=images,
            )
            return self.send_json(HTTPStatus.OK, result)
        except (RuntimeError, ValueError, OSError) as exc:
            return self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def serve_static(self, route: str) -> None:
        relative = "index.html" if route in {"", "/"} else route.lstrip("/")
        candidate = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents and candidate != STATIC_ROOT.resolve():
            return self.send_error(HTTPStatus.NOT_FOUND)
        if not candidate.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        content = candidate.read_bytes()
        media_type, _encoding = mimetypes.guess_type(str(candidate))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", (media_type or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("MURA_APP_PORT", "8601"))
    server = ThreadingHTTPServer((host, port), WristReviewHandler)
    print(f"Wrist Review mobile app: http://{host}:{port}")
    print("Local-only. Uploaded images remain in memory for the current request.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
