"""Env-only credential resolution. Installed by private/tools/shadow/sync.py.

Upstream resolves credentials from an explicit value, the environment, ``NAME_FILE``,
secret directories, and finally an executable broker (a subprocess). The lab keeps the
public contract — :func:`resolve_secret`, :func:`resolve_first`,
:func:`secret_available`, :class:`SecretResolutionError` — and only the first two
sources: explicit value, then environment variable. No files, no directories, no
subprocess. Nothing in the airgapped lab needs a credential; this exists so callers
import cleanly and resolve to ``None``.
"""

from __future__ import annotations

import os
import re

__all__ = ["SecretResolutionError", "resolve_secret", "resolve_first", "secret_available"]

_SECRET_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SecretResolutionError(RuntimeError):
    """A configured secret source could not be resolved safely."""


def _validate_name(name: str) -> str:
    if not isinstance(name, str) or not _SECRET_NAME_RE.match(name):
        raise SecretResolutionError(f"invalid credential name: {name!r}")
    return name


def _normalize_value(raw: str) -> str:
    value = raw.rstrip("\r\n")
    if not value:
        raise SecretResolutionError("credential value is empty")
    return value


def resolve_secret(name: str, *, explicit: str | None = None) -> str | None:
    """Return the explicit value, else the environment value, else ``None``."""
    name = _validate_name(name)
    if explicit is not None and explicit.rstrip("\r\n"):
        return _normalize_value(explicit)
    environment_value = os.environ.get(name)
    if environment_value:
        return _normalize_value(environment_value)
    return None


def resolve_first(*names: str) -> str | None:
    """Return the first available credential in caller-defined priority order."""
    for name in names:
        value = resolve_secret(name)
        if value is not None:
            return value
    return None


def secret_available(*names: str) -> bool:
    """Return whether any requested credential resolves successfully."""
    return resolve_first(*names) is not None
