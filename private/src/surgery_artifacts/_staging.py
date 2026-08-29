"""Atomic staging paths that retain the destination parent's access policy."""

from __future__ import annotations

import secrets
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def unique_staging_path(
    parent: str | Path,
    *,
    prefix: str,
    suffix: str = "",
) -> Path:
    """Return a high-entropy unused child path without creating it."""

    root = Path(parent)
    for _attempt in range(100):
        candidate = root / f"{prefix}{secrets.token_hex(12)}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"could not allocate a unique staging path under {root}")


def create_staging_directory(parent: str | Path, *, prefix: str) -> Path:
    """Create a private-name directory using the parent's normal ACL inheritance."""

    root = Path(parent)
    for _attempt in range(100):
        candidate = unique_staging_path(root, prefix=prefix)
        try:
            # Deliberately use Path.mkdir's normal mode. tempfile.mkdtemp forces
            # mode 0700; on Windows that produces an owner-only DACL which then
            # survives an atomic rename into a shared workspace.
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique staging directory under {root}")


@contextmanager
def staging_directory(parent: str | Path, *, prefix: str) -> Iterator[Path]:
    """Yield an inheritance-friendly staging directory and remove it afterward."""

    temporary = create_staging_directory(parent, prefix=prefix)
    try:
        yield temporary
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def write_staging_text(
    parent: str | Path,
    *,
    prefix: str,
    suffix: str,
    text: str,
) -> Path:
    """Create an exclusive staging file using the parent's normal ACL inheritance."""

    root = Path(parent)
    for _attempt in range(100):
        candidate = unique_staging_path(root, prefix=prefix, suffix=suffix)
        try:
            with candidate.open("x", encoding="utf-8") as stream:
                stream.write(text)
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError(f"could not create a unique staging file under {root}")
