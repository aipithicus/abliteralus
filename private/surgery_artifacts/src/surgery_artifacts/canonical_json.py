"""Canonical JSON used by content identities and manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def normalize(value: Any) -> Any:
    """Return a JSON-safe value with deterministic mapping and sequence semantics."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple):
        return [normalize(item) for item in value]
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [normalize(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonical-JSON compatible: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Serialize a value using the capsule's stable JSON representation."""

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_text(value: Any) -> str:
    """Serialize human-inspectable JSON without changing canonical identity rules."""

    return json.dumps(
        normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
