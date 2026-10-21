"""Backward-compatible entry point for the current Charter PDF.

The older standalone renderer contained a second, stale Charter draft. Keep
this command name for existing team instructions, but delegate to the single
current submission builder so the 95% protocol and governance text cannot
drift between files.
"""

from __future__ import annotations

from build_charter_submission_pdf import build


if __name__ == "__main__":
    build()
