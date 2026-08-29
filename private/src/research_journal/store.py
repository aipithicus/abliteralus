"""Portable append, query, verification, and recovery for research journals."""

from __future__ import annotations

import math
import re
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonl_engine import (
    JsonlCorruptionError,
    JsonlFormatError,
    JsonlStore,
    TransactionState,
    UncommittedTailError,
)

from .canonical import calculate_record_hash, encode_record
from .errors import JournalCorruptionError, JournalFormatError

SCHEMA_ID = "https://aipithicus.org/schemas/research-journal/record-v1"
SCHEMA_VERSION = 1
HEADER_RECORD_TYPE = "research-journal/header"
ENTRY_RECORD_TYPE = "research-journal/entry"
SHARING_STATES = frozenset({"private", "candidate", "approved"})
MAX_RECORD_BYTES = 1024 * 1024
MAX_BODY_CHARACTERS = 64 * 1024
_ENGINE_METADATA_KEY = "research-journal/v1"

_NAME = re.compile(r"^[a-z][a-z0-9._/-]{0,79}$")
_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,79}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:authorization|credentials?|password|passwd|secret|api[_-]?key|"
    r"access[_-]?token|refresh[_-]?token|private[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_EXACT_KEYS = frozenset(
    {
        "token",
        "hf_token",
        "lightning_api_key",
        "lightning_user_id",
        "openai_api_key",
    }
)
_SENSITIVE_TEXT = re.compile(
    r"(?:"
    r"\bhf_[A-Za-z0-9]{10,}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{12,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{12,}\b|"
    r"\bsk-[A-Za-z0-9_-]{12,}\b|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|"
    r"\bBearer\s+\S{12,}|"
    r"https?://[^/\s:@]+:[^/\s@]+@|"
    r"\b(?:HF_TOKEN|LIGHTNING_API_KEY|OPENAI_API_KEY|API_KEY|ACCESS_TOKEN|"
    r"REFRESH_TOKEN|TOKEN|PASSWORD|PASSWD|SECRET)\s*[:=]\s*\S{6,}"
    r")",
    re.IGNORECASE,
)

_HEADER_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "record_type",
        "journal_id",
        "created_at",
        "previous_hash",
        "record_hash",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "record_type",
        "journal_id",
        "entry_id",
        "recorded_at",
        "kind",
        "title",
        "body",
        "actor",
        "tags",
        "relations",
        "data",
        "sharing",
        "previous_hash",
        "record_hash",
    }
)


@dataclass(frozen=True, slots=True)
class JournalInspection:
    path: Path
    exists: bool
    initialized: bool
    valid: bool
    journal_id: str | None
    record_count: int
    entry_count: int
    size_bytes: int
    valid_prefix_bytes: int
    head_hash: str | None
    error_line: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "exists": self.exists,
            "initialized": self.initialized,
            "valid": self.valid,
            "journal_id": self.journal_id,
            "record_count": self.record_count,
            "entry_count": self.entry_count,
            "size_bytes": self.size_bytes,
            "valid_prefix_bytes": self.valid_prefix_bytes,
            "head_hash": self.head_hash,
            "error_line": self.error_line,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class RepairResult:
    path: Path
    needed: bool
    applied: bool
    valid_prefix_bytes: int
    removed_bytes: int
    backup: Path | None
    error_line: int | None
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "needed": self.needed,
            "applied": self.applied,
            "valid_prefix_bytes": self.valid_prefix_bytes,
            "removed_bytes": self.removed_bytes,
            "backup": str(self.backup) if self.backup is not None else None,
            "error_line": self.error_line,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _Scan:
    inspection: JournalInspection
    records: tuple[dict[str, Any], ...]


def _journal_head_from_transaction(
    state: TransactionState,
) -> tuple[str, str, int] | None:
    """Return the journal cursor recorded by a committed engine transaction."""

    if state.record_count == 0:
        return None
    raw = state.metadata.get(_ENGINE_METADATA_KEY)
    if not isinstance(raw, Mapping) or set(raw) != {
        "journal_id",
        "head_hash",
        "entry_count",
    }:
        return None
    try:
        journal_id = _normalize_uuid(raw.get("journal_id"), "transaction journal_id")
    except JournalFormatError:
        return None
    head_hash = raw.get("head_hash")
    if not isinstance(head_hash, str) or _HASH.fullmatch(head_hash) is None:
        return None
    entry_count = raw.get("entry_count")
    if (
        isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count < 0
        or state.record_count != entry_count + 1
    ):
        return None
    return journal_id, head_hash, entry_count


class ResearchJournal:
    """Journal-domain rules composed over one authoritative :class:`JsonlStore`."""

    def __init__(self, path: str | Path, *, lock_timeout: float = 30.0) -> None:
        self.store = JsonlStore(
            path,
            lock_timeout=lock_timeout,
            maximum_record_bytes=MAX_RECORD_BYTES,
            recover_uncommitted=False,
        )

    @property
    def path(self) -> Path:
        """Return the path owned and normalized by the physical store."""

        return self.store.path

    @property
    def lock_path(self) -> Path:
        """Return the cross-process lease path owned by the physical store."""

        return self.store.lock_path

    @property
    def lock_timeout(self) -> float:
        """Return the lease timeout configured on the physical store."""

        return self.store.lock_timeout

    def append(
        self,
        *,
        kind: str,
        title: str,
        body: str = "",
        actor: str | None = None,
        tags: Sequence[str] = (),
        relations: Sequence[Mapping[str, str]] = (),
        data: Mapping[str, Any] | None = None,
        sharing: str = "private",
    ) -> dict[str, Any]:
        """Validate and durably append one immutable entry."""

        normalized = _normalize_entry_input(
            kind=kind,
            title=title,
            body=body,
            actor=actor,
            tags=tags,
            relations=relations,
            data=data,
            sharing=sharing,
        )
        try:
            with self.store.transaction() as transaction:
                state = transaction.state
                cached = _journal_head_from_transaction(state)
                if cached is None:
                    scan = self._scan(collect_records=False)
                    if not scan.inspection.valid:
                        raise JournalCorruptionError(_append_refusal(scan.inspection))
                    journal_id = scan.inspection.journal_id
                    previous_hash = scan.inspection.head_hash
                    entry_count = scan.inspection.entry_count
                    initialized = scan.inspection.initialized
                else:
                    journal_id, previous_hash, entry_count = cached
                    initialized = True

                pending: list[dict[str, Any]] = []
                if not initialized:
                    journal_id = str(uuid.uuid4())
                    header = _signed_record(
                        {
                            "schema": SCHEMA_ID,
                            "schema_version": SCHEMA_VERSION,
                            "record_type": HEADER_RECORD_TYPE,
                            "journal_id": journal_id,
                            "created_at": _utc_now(),
                            "previous_hash": None,
                        }
                    )
                    pending.append(header)
                    previous_hash = header["record_hash"]

                assert journal_id is not None
                entry: dict[str, Any] = {
                    "schema": SCHEMA_ID,
                    "schema_version": SCHEMA_VERSION,
                    "record_type": ENTRY_RECORD_TYPE,
                    "journal_id": journal_id,
                    "entry_id": str(uuid.uuid4()),
                    "recorded_at": _utc_now(),
                    **normalized,
                    "previous_hash": previous_hash,
                }
                entry = _signed_record(entry)
                if len(encode_record(entry)) > MAX_RECORD_BYTES:
                    raise JournalFormatError(
                        f"encoded journal entry exceeds {MAX_RECORD_BYTES} bytes"
                    )
                pending.append(entry)

                metadata = dict(state.metadata)
                metadata[_ENGINE_METADATA_KEY] = {
                    "journal_id": journal_id,
                    "head_hash": entry["record_hash"],
                    "entry_count": entry_count + 1,
                }
                transaction.commit(pending, metadata=metadata)
                return entry
        except (UncommittedTailError, JsonlCorruptionError) as error:
            raise JournalCorruptionError(_engine_append_refusal(error)) from error
        except JsonlFormatError as error:
            raise JournalFormatError(str(error)) from error

    def inspect(self) -> JournalInspection:
        """Return structural and integrity facts without changing the journal."""

        return self._scan(collect_records=False).inspection

    def list_entries(
        self,
        *,
        limit: int = 20,
        kind: str | None = None,
        tags: Sequence[str] = (),
        relations: Sequence[Mapping[str, str]] = (),
        newest_first: bool = True,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        expected_kind = _normalize_name(kind, "kind") if kind is not None else None
        expected_tags = set(_normalize_tags(tags))
        expected_relations = {
            (item["type"], item["target"]) for item in _normalize_relations(relations)
        }
        scan = self._read_scan()
        entries = [record for record in scan.records if record["record_type"] == ENTRY_RECORD_TYPE]
        if expected_kind is not None:
            entries = [entry for entry in entries if entry["kind"] == expected_kind]
        if expected_tags:
            entries = [entry for entry in entries if expected_tags.issubset(entry["tags"])]
        if expected_relations:
            entries = [
                entry
                for entry in entries
                if expected_relations.issubset(
                    {(item["type"], item["target"]) for item in entry["relations"]}
                )
            ]
        if newest_first:
            entries.reverse()
        return entries[:limit]

    def get_entry(self, entry_id: str) -> dict[str, Any]:
        expected = _normalize_uuid(entry_id, "entry_id")
        scan = self._read_scan()
        for record in scan.records:
            if record.get("entry_id") == expected:
                return record
        raise JournalFormatError(f"journal entry does not exist: {expected}")

    def repair(self, *, apply: bool = False) -> RepairResult:
        """Preview or repair an invalid suffix, preserving the original as a backup."""

        with self.store.lease():
            scan = self._scan(collect_records=False)
            inspection = scan.inspection
            if not inspection.exists:
                return RepairResult(self.path, False, False, 0, 0, None, None, None)
            if inspection.valid:
                return RepairResult(
                    self.path,
                    False,
                    False,
                    inspection.valid_prefix_bytes,
                    0,
                    None,
                    None,
                    None,
                )
            if inspection.valid_prefix_bytes >= inspection.size_bytes:
                raise JournalCorruptionError(
                    inspection.error or "journal repair requires a regular JSONL data file"
                )
            removed = inspection.size_bytes - inspection.valid_prefix_bytes
            if not apply:
                return RepairResult(
                    self.path,
                    True,
                    False,
                    inspection.valid_prefix_bytes,
                    removed,
                    None,
                    inspection.error_line,
                    inspection.error,
                )

            try:
                receipt = self.store.repair_prefix(
                    inspection.valid_prefix_bytes,
                    backup_label="corrupt",
                )
            except JsonlCorruptionError as error:
                raise JournalCorruptionError(str(error)) from error
            backup = receipt.backup_path
            repaired = self._scan(collect_records=False).inspection
            if not repaired.valid:
                raise JournalCorruptionError(
                    f"journal remained invalid after repair; original is preserved at {backup}"
                )
            return RepairResult(
                self.path,
                True,
                True,
                inspection.valid_prefix_bytes,
                removed,
                backup,
                inspection.error_line,
                inspection.error,
            )

    def _read_scan(self) -> _Scan:
        scan = self._scan(collect_records=True)
        if not scan.inspection.valid:
            raise JournalCorruptionError(_append_refusal(scan.inspection))
        return scan

    def _scan(self, *, collect_records: bool) -> _Scan:
        previous_hash: str | None = None
        journal_id: str | None = None
        seen_entry_ids: set[str] = set()
        entry_count = 0

        def validate(record: dict[str, Any], line_number: int) -> None:
            nonlocal journal_id, previous_hash, entry_count
            journal_id, previous_hash, is_entry = _validate_record(
                record,
                line_number=line_number,
                expected_journal_id=journal_id,
                expected_previous_hash=previous_hash,
                seen_entry_ids=seen_entry_ids,
            )
            entry_count += int(is_entry)

        physical = self.store.inspect_prefix(
            validator=validate,
            collect_records=collect_records,
        )

        return _Scan(
            JournalInspection(
                path=self.path,
                exists=physical.exists,
                initialized=physical.record_count > 0,
                valid=physical.valid,
                journal_id=journal_id,
                record_count=physical.record_count,
                entry_count=entry_count,
                size_bytes=physical.size_bytes,
                valid_prefix_bytes=physical.valid_prefix_bytes,
                head_hash=previous_hash,
                error_line=physical.error_line,
                error=physical.error,
            ),
            physical.records,
        )


def _normalize_entry_input(
    *,
    kind: str,
    title: str,
    body: str,
    actor: str | None,
    tags: Sequence[str],
    relations: Sequence[Mapping[str, str]],
    data: Mapping[str, Any] | None,
    sharing: str,
) -> dict[str, Any]:
    normalized_title = _normalize_text(title, "title", maximum=240, required=True)
    normalized_body = _normalize_text(
        body,
        "body",
        maximum=MAX_BODY_CHARACTERS,
        required=False,
        strip=False,
    )
    normalized_actor = (
        _normalize_text(actor, "actor", maximum=160, required=True) if actor is not None else None
    )
    normalized_sharing = _normalize_text(sharing, "sharing", maximum=20, required=True).lower()
    if normalized_sharing not in SHARING_STATES:
        raise JournalFormatError(f"sharing must be one of: {', '.join(sorted(SHARING_STATES))}")
    if data is not None and not isinstance(data, Mapping):
        raise JournalFormatError("data must be a JSON object")
    normalized_data = dict(data or {})
    _validate_json_value(normalized_data, path="data")
    payload = {
        "kind": _normalize_name(kind, "kind"),
        "title": normalized_title,
        "body": normalized_body,
        "actor": normalized_actor,
        "tags": _normalize_tags(tags),
        "relations": _normalize_relations(relations),
        "data": normalized_data,
        "sharing": normalized_sharing,
    }
    _reject_sensitive(payload)
    return payload


def _normalize_name(value: str, label: str) -> str:
    text = _normalize_text(value, label, maximum=80, required=True).lower()
    if _NAME.fullmatch(text) is None:
        raise JournalFormatError(
            f"{label} must begin with a lowercase letter and contain only lowercase names, "
            "digits, '.', '_', '/', or '-'"
        )
    return text


def _normalize_tags(values: Sequence[str]) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise JournalFormatError("tags must be a sequence of tag strings")
    tags = []
    for value in values:
        tag = _normalize_text(value, "tag", maximum=80, required=True)
        if _TAG.fullmatch(tag) is None:
            raise JournalFormatError(f"unsupported journal tag: {tag!r}")
        tags.append(tag)
    if len(tags) > 64:
        raise JournalFormatError("a journal entry may contain at most 64 tags")
    return sorted(set(tags))


def _normalize_relations(values: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise JournalFormatError("relations must be a sequence of relation objects")
    pairs: set[tuple[str, str]] = set()
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {"type", "target"}:
            raise JournalFormatError(f"relation {index} must contain exactly 'type' and 'target'")
        relation_type = _normalize_name(value["type"], f"relation {index} type")
        target = _normalize_text(
            value["target"], f"relation {index} target", maximum=1024, required=True
        )
        pairs.add((relation_type, target))
    if len(pairs) > 128:
        raise JournalFormatError("a journal entry may contain at most 128 relations")
    return [{"type": relation_type, "target": target} for relation_type, target in sorted(pairs)]


def _normalize_text(
    value: Any,
    label: str,
    *,
    maximum: int,
    required: bool,
    strip: bool = True,
) -> str:
    if not isinstance(value, str):
        raise JournalFormatError(f"{label} must be a string")
    text = unicodedata.normalize("NFC", value)
    if strip:
        text = text.strip()
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise JournalFormatError(f"{label} contains invalid Unicode") from error
    if required and not text:
        raise JournalFormatError(f"{label} must not be blank")
    if len(text) > maximum:
        raise JournalFormatError(f"{label} exceeds {maximum} characters")
    return text


def _validate_json_value(value: Any, *, path: str, depth: int = 0) -> None:
    if depth > 16:
        raise JournalFormatError(f"{path} exceeds the maximum nesting depth")
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise JournalFormatError(f"{path} contains invalid Unicode") from error
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise JournalFormatError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise JournalFormatError(f"{path} contains a non-string object key")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as error:
                raise JournalFormatError(
                    f"{path} contains an invalid Unicode object key"
                ) from error
            _validate_json_value(item, path=f"{path}.{key}", depth=depth + 1)
        return
    raise JournalFormatError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _reject_sensitive(value: Any, *, path: str = "entry") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            if normalized in _SENSITIVE_EXACT_KEYS or _SENSITIVE_KEY.search(normalized):
                raise JournalFormatError(f"{path}.{key} looks like a sensitive-data field")
            _reject_sensitive(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_sensitive(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and _SENSITIVE_TEXT.search(value):
        raise JournalFormatError(f"{path} appears to contain a credential or private key")


def _signed_record(record: dict[str, Any]) -> dict[str, Any]:
    signed = dict(record)
    signed["record_hash"] = calculate_record_hash(signed)
    return signed


def _validate_record(
    record: dict[str, Any],
    *,
    line_number: int,
    expected_journal_id: str | None,
    expected_previous_hash: str | None,
    seen_entry_ids: set[str],
) -> tuple[str, str, bool]:
    if record.get("schema") != SCHEMA_ID or record.get("schema_version") != SCHEMA_VERSION:
        raise JournalFormatError(f"line {line_number} declares an unsupported schema")
    record_type = record.get("record_type")
    expected_fields = _HEADER_FIELDS if line_number == 1 else _ENTRY_FIELDS
    if set(record) != expected_fields:
        missing = sorted(expected_fields - set(record))
        extra = sorted(set(record) - expected_fields)
        raise JournalFormatError(
            f"line {line_number} has incorrect fields (missing={missing}, extra={extra})"
        )
    if line_number == 1 and record_type != HEADER_RECORD_TYPE:
        raise JournalFormatError("line 1 must be the journal header")
    if line_number > 1 and record_type != ENTRY_RECORD_TYPE:
        raise JournalFormatError(f"line {line_number} must be a journal entry")

    journal_id = _normalize_uuid(record.get("journal_id"), f"line {line_number} journal_id")
    if expected_journal_id is not None and journal_id != expected_journal_id:
        raise JournalFormatError(f"line {line_number} changes the journal_id")
    previous_hash = record.get("previous_hash")
    if previous_hash != expected_previous_hash:
        raise JournalFormatError(f"line {line_number} breaks the record hash chain")
    record_hash = record.get("record_hash")
    if not isinstance(record_hash, str) or _HASH.fullmatch(record_hash) is None:
        raise JournalFormatError(f"line {line_number} has an invalid record_hash")
    if calculate_record_hash(record) != record_hash:
        raise JournalFormatError(f"line {line_number} record_hash does not match its contents")

    if record_type == HEADER_RECORD_TYPE:
        _validate_timestamp(record.get("created_at"), f"line {line_number} created_at")
        return journal_id, record_hash, False

    entry_id = _normalize_uuid(record.get("entry_id"), f"line {line_number} entry_id")
    if entry_id in seen_entry_ids:
        raise JournalFormatError(f"line {line_number} repeats entry_id {entry_id}")
    seen_entry_ids.add(entry_id)
    _validate_timestamp(record.get("recorded_at"), f"line {line_number} recorded_at")
    normalized = _normalize_entry_input(
        kind=record.get("kind"),
        title=record.get("title"),
        body=record.get("body"),
        actor=record.get("actor"),
        tags=record.get("tags"),
        relations=record.get("relations"),
        data=record.get("data"),
        sharing=record.get("sharing"),
    )
    for field, value in normalized.items():
        if record[field] != value:
            raise JournalFormatError(f"line {line_number} has a non-canonical {field}")
    return journal_id, record_hash, True


def _normalize_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise JournalFormatError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise JournalFormatError(f"{label} must be a UUID string") from error
    if str(parsed) != value:
        raise JournalFormatError(f"{label} must use canonical lowercase UUID form")
    return value


def _validate_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JournalFormatError(f"{label} must be a UTC timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise JournalFormatError(f"{label} is not a valid timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise JournalFormatError(f"{label} must be UTC")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _append_refusal(inspection: JournalInspection) -> str:
    return (
        f"journal is invalid at line {inspection.error_line}: {inspection.error}; "
        "run verify, then preview repair before appending"
    )


def _engine_append_refusal(error: Exception) -> str:
    return (
        f"journal transaction state is invalid: {error}; "
        "run verify, then preview repair before appending"
    )
