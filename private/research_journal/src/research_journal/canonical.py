"""Research-journal record hashing over the shared JSONL byte policy."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from jsonl_engine import canonical_bytes, encode_record

HASH_PREFIX = "sha256:"


def calculate_record_hash(record: Mapping[str, Any]) -> str:
    """Hash a record without its self-describing ``record_hash`` member."""

    unsigned = dict(record)
    unsigned.pop("record_hash", None)
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


__all__ = ["calculate_record_hash", "canonical_bytes", "encode_record"]
