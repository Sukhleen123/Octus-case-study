"""
Compute a fingerprint (SHA-256) that captures:
  - raw Octus file mtimes + sizes
  - settings fields that affect processed artifacts

If the fingerprint matches the stored manifest, bootstrap skips rebuilding.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.app.settings import Settings


def compute_fingerprint(settings: Settings) -> str:
    """Return a SHA-256 hex digest representing the current build inputs."""
    raw = Path(settings.octus_raw_dir)
    tracked_files = [
        raw / "company_metadata.json",
        raw / "transcripts.json",
        raw / "sec_filings_metadata.json",
    ]

    parts: list[str] = []
    for f in tracked_files:
        if f.exists():
            stat = f.stat()
            parts.append(f"{f.name}:{stat.st_mtime_ns}:{stat.st_size}")
        else:
            parts.append(f"{f.name}:MISSING")

    # Include settings fields that affect artifact content
    parts.append(json.dumps(settings.fingerprint_subset(), sort_keys=True))

    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()
