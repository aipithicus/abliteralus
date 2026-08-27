"""Deterministic JSON encoding and record hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

HASH_PREFIX = "sha256:"


def canonical_bytes(value: Any) -> bytes:
    """Encode one JSON value under the journal's portable byte policy."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def calculate_record_hash(record: Mapping[str, Any]) -> str:
    """Hash a record without its self-describing ``record_hash`` member."""

    unsigned = dict(record)
    unsigned.pop("record_hash", None)
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def encode_record(record: Mapping[str, Any]) -> bytes:
    """Encode exactly one LF-terminated JSONL record."""

    return canonical_bytes(dict(record)) + b"\n"
