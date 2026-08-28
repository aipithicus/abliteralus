"""Canonical UTF-8 JSON and LF-delimited record framing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .errors import JsonlFormatError


def canonical_bytes(value: Any) -> bytes:
    """Encode one JSON value under the engine's deterministic byte policy."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def encode_record(record: Mapping[str, Any]) -> bytes:
    """Encode exactly one canonical LF-terminated JSON object."""

    if not isinstance(record, Mapping):
        raise JsonlFormatError("JSONL records must be JSON objects")
    try:
        return canonical_bytes(dict(record)) + b"\n"
    except (RecursionError, TypeError, ValueError) as error:
        raise JsonlFormatError("record contains a non-portable JSON value") from error


def decode_record(line: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    """Decode and verify one canonical LF-terminated JSON object."""

    if len(line) > maximum_bytes:
        raise JsonlFormatError(f"{label} exceeds {maximum_bytes} bytes")
    if not line.endswith(b"\n"):
        raise JsonlFormatError(f"{label} is not LF-terminated")
    if b"\r" in line:
        raise JsonlFormatError(f"{label} contains a raw carriage return")
    try:
        text = line[:-1].decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise JsonlFormatError(f"{label} is not valid UTF-8") from error
    if not text:
        raise JsonlFormatError(f"{label} is blank")
    try:
        record = json.loads(text)
    except (RecursionError, ValueError) as error:
        raise JsonlFormatError(f"{label} is not valid bounded JSON") from error
    if not isinstance(record, dict):
        raise JsonlFormatError(f"{label} is not a JSON object")
    if encode_record(record) != line:
        raise JsonlFormatError(f"{label} is not in canonical JSONL form")
    return record
