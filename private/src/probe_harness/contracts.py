"""Versioned records shared by the probe controller and session store."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4

from .errors import ProbeHarnessError


EVENT_SCHEMA = "https://aipithicus.org/schemas/probe-harness/session-event-v1"
EVENT_TYPES = frozenset(
    {
        "direction_bundle.loaded",
        "observation.artifact_saved",
        "session.started",
        "trial.started",
        "trial.completed",
        "trial.failed",
        "trial.annotated",
        "observation.completed",
    }
)
EVIDENCE_KINDS = frozenset(
    {
        "behavioral",
        "observational",
        "geometric-estimate",
        "interventional",
        "weight-modified",
    }
)
EVENT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "event_id",
        "session_id",
        "recorded_at",
        "event_type",
        "payload",
    }
)


def utc_now() -> str:
    """Return a stable UTC timestamp for a durable record."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    """Return a readable, collision-resistant local identifier."""

    return f"{prefix}-{uuid4().hex[:16]}"


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically, rejecting non-finite measurements."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def json_sha256(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def make_event(
    *,
    session_id: str,
    event_type: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise ProbeHarnessError(f"unsupported session event type: {event_type}")
    if not isinstance(payload, Mapping):
        raise ProbeHarnessError("session event payload must be an object")
    return {
        "schema": EVENT_SCHEMA,
        "schema_version": 1,
        "event_id": new_id("event"),
        "session_id": session_id,
        "recorded_at": utc_now(),
        "event_type": event_type,
        "payload": dict(payload),
    }


def validate_event(record: Mapping[str, Any], *, session_id: str | None = None) -> None:
    """Validate the strict v1 envelope without interpreting its domain payload."""

    if set(record) != EVENT_KEYS:
        raise ProbeHarnessError("session event keys do not match the v1 contract")
    if record.get("schema") != EVENT_SCHEMA or record.get("schema_version") != 1:
        raise ProbeHarnessError("unsupported probe-session event schema")
    if record.get("event_type") not in EVENT_TYPES:
        raise ProbeHarnessError(f"unsupported session event type: {record.get('event_type')!r}")
    for key in ("event_id", "session_id", "recorded_at"):
        if not isinstance(record.get(key), str) or not str(record[key]).strip():
            raise ProbeHarnessError(f"session event {key} must be a non-empty string")
    if session_id is not None and record["session_id"] != session_id:
        raise ProbeHarnessError("session event belongs to a different session")
    if not isinstance(record.get("payload"), Mapping):
        raise ProbeHarnessError("session event payload must be an object")
    canonical_json_bytes(record)
