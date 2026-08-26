"""Content-addressed local registry for verified surgery capsules."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .canonical_json import pretty_text
from .errors import CapsuleFormatError
from .validation import validate_capsule

_IDENTIFIER = re.compile(r"[0-9a-f]{64}")
_REF_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


@dataclass(frozen=True, slots=True)
class RegistryEntry:
    surgery_id: str
    path: Path
    ref: str | None
    created: bool


class ArtifactRegistry:
    """Store immutable objects and atomically update small human-readable refs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def add(self, capsule: str | Path, *, ref: str | None = None) -> RegistryEntry:
        validation = validate_capsule(capsule)
        objects = self.root / "objects"
        refs = self.root / "refs"
        objects.mkdir(parents=True, exist_ok=True)
        refs.mkdir(parents=True, exist_ok=True)
        destination = objects / validation.surgery_id
        created = False
        if destination.exists():
            existing = validate_capsule(destination)
            if existing.surgery_id != validation.surgery_id:
                raise CapsuleFormatError("registry object identity collision")
        else:
            staging = Path(tempfile.mkdtemp(prefix=".incoming-", dir=objects))
            shutil.rmtree(staging)
            try:
                shutil.copytree(validation.path, staging)
                copied = validate_capsule(staging)
                if copied.surgery_id != validation.surgery_id:
                    raise CapsuleFormatError("registry copy changed capsule identity")
                try:
                    os.replace(staging, destination)
                    created = True
                except OSError:
                    if not destination.exists():
                        raise
                    validate_capsule(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if ref is not None:
            self.set_ref(ref, validation.surgery_id)
        return RegistryEntry(validation.surgery_id, destination, ref, created)

    def set_ref(self, ref: str, surgery_id: str) -> Path:
        if _IDENTIFIER.fullmatch(surgery_id) is None:
            raise ValueError("surgery id must be a lowercase SHA-256 digest")
        object_path = self.root / "objects" / surgery_id
        validation = validate_capsule(object_path)
        if validation.surgery_id != surgery_id:
            raise CapsuleFormatError("registry object does not match requested id")
        relative = _ref_path(ref)
        destination = self.root / "refs" / Path(f"{relative.as_posix()}.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "ref": relative.as_posix(),
            "surgery_id": surgery_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            temporary.write_text(pretty_text(payload), encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def resolve(self, value: str) -> Path:
        if _IDENTIFIER.fullmatch(value):
            path = self.root / "objects" / value
        else:
            relative = _ref_path(value)
            ref_path = self.root / "refs" / Path(f"{relative.as_posix()}.json")
            try:
                payload = json.loads(ref_path.read_text(encoding="utf-8"))
                surgery_id = payload["surgery_id"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
                raise CapsuleFormatError(f"registry ref is unreadable: {value}") from error
            if not isinstance(surgery_id, str) or _IDENTIFIER.fullmatch(surgery_id) is None:
                raise CapsuleFormatError(f"registry ref has an invalid surgery id: {value}")
            path = self.root / "objects" / surgery_id
        validate_capsule(path)
        return path


def _ref_path(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("registry ref must be a non-empty name")
    normalized = value.replace("\\", "/").strip("/")
    path = Path(normalized)
    if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
        raise ValueError("registry ref must be a safe relative name")
    if any(_REF_SEGMENT.fullmatch(part) is None for part in path.parts):
        raise ValueError("registry ref contains unsupported characters")
    return path
