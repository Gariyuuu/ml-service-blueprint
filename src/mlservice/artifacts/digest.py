"""Content digests used to pin datasets and detect tampered artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file, so large datasets do not need to fit in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
