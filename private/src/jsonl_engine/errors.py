"""Errors raised by the JSONL engine."""

from __future__ import annotations


class JsonlStoreError(RuntimeError):
    """Base class for physical JSONL store failures."""


class JsonlFormatError(JsonlStoreError, ValueError):
    """A record or sidecar row does not satisfy the JSONL engine contract."""


class JsonlCorruptionError(JsonlStoreError):
    """A store or transaction table disagrees with its committed state."""


class UncommittedTailError(JsonlCorruptionError):
    """The data file extends beyond the last committed transaction boundary."""

    def __init__(self, *, committed_bytes: int, actual_bytes: int) -> None:
        self.committed_bytes = committed_bytes
        self.actual_bytes = actual_bytes
        super().__init__(
            "JSONL store has "
            f"{actual_bytes - committed_bytes} uncommitted bytes beyond offset "
            f"{committed_bytes}"
        )
