"""Transactional in-place append over canonical JSONL stores."""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock, Timeout

from .canonical import canonical_bytes, decode_record, encode_record
from .errors import JsonlCorruptionError, JsonlFormatError, UncommittedTailError
from .index import (
    JSOI_VERSION,
    Jidx,
    append_index_durable,
    calculate_index_hash,
    expected_jidx_bytes,
    extend_index_hash,
    jidx_path_for,
    mtime_ns_to_dotnet_ticks,
    read_index,
    read_index_header,
    replace_index_durable,
    restore_index_durable,
)

TRANSACTION_SCHEMA_ID = "https://aipithicus.org/schemas/jsonl-engine/transaction-v1"
DEFAULT_MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_TRANSACTION_BYTES = 1024 * 1024
HASH_PREFIX = "sha256:"

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_REPAIR_LABEL = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_DATA_SEED = HASH_PREFIX + hashlib.sha256(
    b"aipithicus/jsonl-engine/data-chain/v1"
).hexdigest()
_DATA_RECORD_DOMAIN = b"aipithicus/jsonl-engine/data-record/v1\0"
_TRANSACTION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "transaction_id",
        "generation",
        "committed_at",
        "previous_transaction_hash",
        "start_offset",
        "end_offset",
        "start_record_count",
        "appended_records",
        "record_count",
        "data_hash",
        "jidx_version",
        "jidx_bytes",
        "jidx_hash",
        "source_mtime_ns",
        "partition",
        "metadata",
        "transaction_hash",
    }
)


@dataclass(frozen=True, slots=True)
class TransactionState:
    """The last committed boundary of one JSONL store."""

    generation: int
    committed_bytes: int
    record_count: int
    data_hash: str
    jidx_bytes: int
    jidx_hash: str
    metadata: dict[str, Any]
    transaction_hash: str
    source_mtime_ns: int | None


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """Facts published by one successful append transaction."""

    transaction_id: str
    generation: int
    start_offset: int
    committed_bytes: int
    appended_records: int
    record_count: int
    data_hash: str
    jidx_bytes: int
    jidx_hash: str
    partition: dict[str, Any] | None
    transaction_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DataScan:
    size_bytes: int
    record_count: int
    data_hash: str
    mtime_ns: int | None
    offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PartitionRange:
    """One committed record range carrying a caller-owned partition descriptor."""

    transaction_id: str
    generation: int
    start_record: int
    stop_record: int
    start_offset: int
    end_offset: int
    partition: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StorePrefixScan:
    """The longest valid canonical/domain prefix of one physical JSONL store."""

    path: Path
    exists: bool
    valid: bool
    size_bytes: int
    valid_prefix_bytes: int
    record_count: int
    records: tuple[dict[str, Any], ...]
    error_line: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StoreRepairReceipt:
    """Facts from one durable prefix repair owned by :class:`JsonlStore`."""

    path: Path
    backup_path: Path
    original_bytes: int
    committed_bytes: int
    removed_bytes: int
    state: TransactionState


class JsonlTransaction:
    """One exclusive append transaction created by :class:`JsonlStore`."""

    def __init__(self, store: JsonlStore, state: TransactionState) -> None:
        self._store = store
        self._state = state
        self._active = True
        self._committed = False

    @property
    def state(self) -> TransactionState:
        """Return the boundary captured after this transaction acquired the lease."""

        self._require_active()
        return self._state

    def commit(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        partition: Mapping[str, Any] | None = None,
    ) -> CommitReceipt:
        """Append a bounded record batch and publish one transaction row."""

        self._require_active()
        if self._committed:
            raise RuntimeError("JSONL transaction has already committed")
        receipt, state = self._store._commit_unlocked(  # noqa: SLF001 - paired type
            self._state,
            records,
            metadata=metadata,
            partition=partition,
        )
        self._state = state
        self._committed = True
        return receipt

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("JSONL transaction is no longer active")

    def _close(self) -> None:
        self._active = False


class JsonlStore:
    """Canonical JSONL data, incremental JIDX, and an append-only transaction table."""

    def __init__(
        self,
        path: str | Path,
        *,
        lock_timeout: float = 30.0,
        maximum_record_bytes: int = DEFAULT_MAX_RECORD_BYTES,
        recover_uncommitted: bool = True,
    ) -> None:
        if not math.isfinite(lock_timeout) or lock_timeout <= 0:
            raise ValueError("lock_timeout must be a positive finite number")
        if (
            isinstance(maximum_record_bytes, bool)
            or not isinstance(maximum_record_bytes, int)
            or maximum_record_bytes <= 0
        ):
            raise ValueError("maximum_record_bytes must be a positive integer")

        self.path = Path(os.path.abspath(os.fspath(path)))
        self.transaction_path = self.path.with_name(f"{self.path.name}.transactions.jsonl")
        self.jidx_path = jidx_path_for(self.path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.lock_timeout = lock_timeout
        self.maximum_record_bytes = maximum_record_bytes
        self.recover_uncommitted = recover_uncommitted
        self._lock = FileLock(str(self.lock_path), timeout=lock_timeout)
        self._index_cache: tuple[str, Jidx] | None = None

    @contextmanager
    def lease(self) -> Iterator[None]:
        """Hold the store's cross-process write lease."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire()
        except Timeout as error:
            raise TimeoutError(
                f"could not acquire JSONL store lease within {self.lock_timeout}s: "
                f"{self.lock_path}"
            ) from error
        try:
            yield
        finally:
            self._lock.release()

    @contextmanager
    def transaction(self) -> Iterator[JsonlTransaction]:
        """Open one exclusive transaction at the latest committed generation."""

        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            transaction = JsonlTransaction(self, state)
            try:
                yield transaction
            finally:
                transaction._close()  # noqa: SLF001 - lifecycle owner

    def append(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        partition: Mapping[str, Any] | None = None,
    ) -> CommitReceipt:
        """Append one record batch under a single lease and commit row."""

        with self.transaction() as transaction:
            return transaction.commit(
                records,
                metadata=metadata,
                partition=partition,
            )

    def read_records(self) -> list[dict[str, Any]]:
        """Read the current committed population under the store lease."""

        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            return self._read_records_unlocked(state)

    def inspect_prefix(
        self,
        *,
        validator: Callable[[dict[str, Any], int], None] | None = None,
        collect_records: bool = False,
    ) -> StorePrefixScan:
        """Inspect the longest valid prefix without changing data or derived state."""

        with self.lease():
            return self._inspect_prefix_unlocked(
                validator=validator,
                collect_records=collect_records,
            )

    def repair_prefix(
        self,
        committed_bytes: int,
        *,
        backup_label: str = "corrupt",
    ) -> StoreRepairReceipt:
        """Back up the full store, publish a valid prefix, and rebuild derived state."""

        if (
            isinstance(committed_bytes, bool)
            or not isinstance(committed_bytes, int)
            or committed_bytes < 0
        ):
            raise ValueError("committed_bytes must be a nonnegative integer")
        if not isinstance(backup_label, str) or _REPAIR_LABEL.fullmatch(backup_label) is None:
            raise ValueError("backup_label must be a lowercase filesystem-safe name")

        with self.lease():
            self._validate_paths_unlocked()
            if not os.path.lexists(self.path):
                raise FileNotFoundError(f"JSONL data does not exist: {self.path}")
            size = self.path.stat().st_size
            if committed_bytes >= size:
                raise ValueError(
                    f"repair prefix must remove at least one byte from a {size}-byte store"
                )
            if committed_bytes > 0:
                with self.path.open("rb") as handle:
                    handle.seek(committed_bytes - 1)
                    if handle.read(1) != b"\n":
                        raise JsonlFormatError(
                            f"repair offset {committed_bytes} is not a complete-record boundary"
                        )

            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            backup = self.path.with_name(
                f"{self.path.name}.{backup_label}-{stamp}.bak"
            )
            _copy_prefix_new_durable(self.path, backup, size)
            self._invalidate_derived_state_unlocked()
            _replace_with_prefix_durable(self.path, committed_bytes)
            scan = self._scan_data_unlocked()
            state = self._publish_checkpoint_unlocked(scan)
            return StoreRepairReceipt(
                path=self.path,
                backup_path=backup,
                original_bytes=size,
                committed_bytes=committed_bytes,
                removed_bytes=size - committed_bytes,
                state=state,
            )

    def record_count(self) -> int:
        """Return the committed population size without scanning the data file."""

        with self.lease():
            return self._load_or_checkpoint_unlocked().record_count

    def read_record(self, index: int) -> dict[str, Any]:
        """Read one record by ordinal through the committed JIDX."""

        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            jidx = self._load_verified_index_unlocked(state, repair=True)
            normalized = _normalize_record_index(index, state.record_count)
            return self._read_indexed_range_unlocked(jidx, normalized, normalized + 1)[0]

    def read_range(self, start: int = 0, stop: int | None = None) -> list[dict[str, Any]]:
        """Read a half-open ordinal range with one indexed seek and a bounded walk."""

        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            normalized_start, normalized_stop = _normalize_record_range(
                start,
                stop,
                state.record_count,
            )
            if normalized_start == normalized_stop:
                return []
            jidx = self._load_verified_index_unlocked(state, repair=True)
            return self._read_indexed_range_unlocked(
                jidx,
                normalized_start,
                normalized_stop,
            )

    def partition_ranges(
        self, selector: Mapping[str, Any] | None = None
    ) -> list[PartitionRange]:
        """Return committed batch ranges whose partition contains ``selector``."""

        expected = _normalize_partition(selector)
        with self.lease():
            self._load_or_checkpoint_unlocked()
            rows = self._read_transaction_rows_unlocked()
            return _partition_ranges_from_rows(rows, expected)

    def read_partition(
        self, selector: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Read all records in matching committed partitions through JIDX ranges."""

        expected = _normalize_partition(selector)
        if expected is None:
            raise JsonlFormatError("partition selector cannot be null")
        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            rows = self._read_transaction_rows_unlocked()
            ranges = _partition_ranges_from_rows(rows, expected)
            if not ranges:
                return []
            jidx = self._load_verified_index_unlocked(state, repair=True)
            records: list[dict[str, Any]] = []
            for item in ranges:
                records.extend(
                    self._read_indexed_range_unlocked(
                        jidx,
                        item.start_record,
                        item.stop_record,
                    )
                )
            return records

    def rebuild_index(self) -> Jidx:
        """Deterministically rebuild JIDX from the committed authoritative JSONL bytes."""

        with self.lease():
            state = self._load_or_checkpoint_unlocked()
            scan = self._scan_data_unlocked()
            self._require_scan_matches_state(scan, state)
            self._replace_index_unlocked(scan)
            return self._require_index_matches_state(read_index(self.jidx_path, self.path), state)

    def verify(self) -> TransactionState:
        """Verify the complete transaction chain and committed data population."""

        with self.lease():
            self._validate_paths_unlocked()
            if not os.path.lexists(self.transaction_path):
                raise JsonlCorruptionError(
                    f"transaction table does not exist: {self.transaction_path}"
                )
            state, valid_prefix = self._scan_transactions_unlocked()
            transaction_size = self.transaction_path.stat().st_size
            if valid_prefix != transaction_size:
                raise JsonlCorruptionError(
                    "transaction table has an invalid suffix beginning at byte "
                    f"{valid_prefix}: {self.transaction_path}"
                )
            scan = self._scan_data_unlocked()
            self._require_scan_matches_state(scan, state)
            if not os.path.lexists(self.jidx_path):
                raise JsonlCorruptionError(f"JIDX does not exist: {self.jidx_path}")
            jidx = read_index(self.jidx_path, self.path)
            self._require_index_matches_state(jidx, state)
            if tuple(scan.offsets) != jidx.offsets:
                raise JsonlCorruptionError("JIDX offsets disagree with JSONL record boundaries")
            return state

    def invalidate_derived_state(self) -> None:
        """Discard transaction and JIDX state after an authoritative data repair."""

        with self.lease():
            self._invalidate_derived_state_unlocked()

    def invalidate_transaction_table(self) -> None:
        """Compatibility alias for invalidating every derived store structure."""

        self.invalidate_derived_state()

    def _load_or_checkpoint_unlocked(self) -> TransactionState:
        self._validate_paths_unlocked()
        if (
            not os.path.lexists(self.transaction_path)
            or self.transaction_path.stat().st_size == 0
        ):
            scan = self._scan_data_unlocked()
            return self._publish_checkpoint_unlocked(scan)

        try:
            row = _read_last_transaction(self.transaction_path)
            state = _state_from_transaction(row)
        except (JsonlFormatError, JsonlCorruptionError):
            state, valid_prefix = self._scan_transactions_unlocked()
            if valid_prefix < self.transaction_path.stat().st_size:
                _truncate_durable(self.transaction_path, valid_prefix)

        actual_size, actual_mtime = _source_facts(self.path)
        if actual_size > state.committed_bytes:
            if not self.recover_uncommitted:
                raise UncommittedTailError(
                    committed_bytes=state.committed_bytes,
                    actual_bytes=actual_size,
                )
            _truncate_durable(self.path, state.committed_bytes)
            actual_size, actual_mtime = _source_facts(self.path)
        if actual_size < state.committed_bytes:
            raise JsonlCorruptionError(
                f"JSONL data was truncated to {actual_size} bytes below committed boundary "
                f"{state.committed_bytes}: {self.path}"
            )
        verified_scan: _DataScan | None = None
        if state.source_mtime_ns != actual_mtime:
            verified_scan = self._scan_data_unlocked()
            self._require_scan_matches_state(verified_scan, state)
        self._ensure_index_unlocked(state, scan=verified_scan)
        return state

    def _publish_checkpoint_unlocked(self, scan: _DataScan) -> TransactionState:
        self._replace_index_unlocked(scan)
        row = _build_transaction_record(
            kind="checkpoint",
            generation=0,
            previous_transaction_hash=None,
            start_offset=0,
            end_offset=scan.size_bytes,
            start_record_count=0,
            appended_records=scan.record_count,
            record_count=scan.record_count,
            data_hash=scan.data_hash,
            jidx_hash=calculate_index_hash(scan.offsets),
            source_mtime_ns=scan.mtime_ns,
            partition=None,
            metadata={},
        )
        _append_durable(self.transaction_path, encode_record(row))
        return _state_from_transaction(row)

    def _commit_unlocked(
        self,
        state: TransactionState,
        records: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None,
        partition: Mapping[str, Any] | None,
    ) -> tuple[CommitReceipt, TransactionState]:
        lines = [encode_record(record) for record in records]
        for index, line in enumerate(lines):
            if len(line) > self.maximum_record_bytes:
                raise JsonlFormatError(
                    f"record {index} exceeds {self.maximum_record_bytes} bytes"
                )

        next_metadata = _normalize_metadata(state.metadata if metadata is None else metadata)
        next_partition = _normalize_partition(partition)
        expected_jidx_bytes(state.record_count + len(lines))
        next_hash = state.data_hash
        for line in lines:
            next_hash = _extend_data_hash(next_hash, line)

        payload = b"".join(lines)
        expected_size = state.committed_bytes + len(payload)
        cursor = state.committed_bytes
        offsets: list[int] = []
        for line in lines:
            offsets.append(cursor)
            cursor += len(line)
        next_jidx_hash = extend_index_hash(state.jidx_hash, offsets)
        transaction_start = _existing_size(self.transaction_path)
        data_mutated = False
        index_mutated = False
        transaction_mutated = False
        try:
            if payload:
                _append_durable(self.path, payload)
                data_mutated = True
            actual_size, actual_mtime = _source_facts(self.path)
            if actual_size != expected_size:
                raise OSError(
                    f"append published {actual_size} data bytes; expected {expected_size}"
                )

            if offsets:
                append_index_durable(
                    self.jidx_path,
                    offsets,
                    previous_line_count=state.record_count,
                    previous_source_length=state.committed_bytes,
                    source_length=actual_size,
                    source_mtime_ns=actual_mtime,
                )
                index_mutated = True
                self._index_cache = None

            row = _build_transaction_record(
                kind="append",
                generation=state.generation + 1,
                previous_transaction_hash=state.transaction_hash,
                start_offset=state.committed_bytes,
                end_offset=actual_size,
                start_record_count=state.record_count,
                appended_records=len(lines),
                record_count=state.record_count + len(lines),
                data_hash=next_hash,
                jidx_hash=next_jidx_hash,
                source_mtime_ns=actual_mtime,
                partition=next_partition,
                metadata=next_metadata,
            )
            _append_durable(self.transaction_path, encode_record(row))
            transaction_mutated = True
        except BaseException as error:
            rollback_errors: list[str] = []
            if data_mutated or _existing_size(self.path) > state.committed_bytes:
                try:
                    _truncate_durable(self.path, state.committed_bytes)
                except OSError as rollback_error:
                    rollback_errors.append(f"data rollback failed: {rollback_error}")
            if transaction_mutated or _existing_size(self.transaction_path) > transaction_start:
                try:
                    _truncate_durable(self.transaction_path, transaction_start)
                except OSError as rollback_error:
                    rollback_errors.append(f"transaction rollback failed: {rollback_error}")
            if index_mutated or _existing_size(self.jidx_path) != state.jidx_bytes:
                try:
                    _, rolled_back_mtime = _source_facts(self.path)
                    restore_index_durable(
                        self.jidx_path,
                        line_count=state.record_count,
                        source_length=state.committed_bytes,
                        source_mtime_ns=rolled_back_mtime,
                    )
                except Exception as rollback_error:
                    rollback_errors.append(f"JIDX rollback failed: {rollback_error}")
            if rollback_errors:
                raise JsonlCorruptionError(
                    "JSONL commit failed and did not roll back cleanly ("
                    + "; ".join(rollback_errors)
                    + ")"
                ) from error
            raise

        next_state = _state_from_transaction(row)
        receipt = CommitReceipt(
            transaction_id=row["transaction_id"],
            generation=next_state.generation,
            start_offset=state.committed_bytes,
            committed_bytes=next_state.committed_bytes,
            appended_records=len(lines),
            record_count=next_state.record_count,
            data_hash=next_state.data_hash,
            jidx_bytes=next_state.jidx_bytes,
            jidx_hash=next_state.jidx_hash,
            partition=deepcopy(next_partition),
            transaction_hash=next_state.transaction_hash,
            metadata=deepcopy(next_state.metadata),
        )
        return receipt, next_state

    def _read_records_unlocked(self, state: TransactionState) -> list[dict[str, Any]]:
        bound = state.committed_bytes
        if bound == 0:
            self._require_scan_matches_state(
                _DataScan(0, 0, _DATA_SEED, None, ()), state
            )
            return []
        records: list[dict[str, Any]] = []
        data_hash = _DATA_SEED
        offset = 0
        with self.path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if offset + len(line) > bound:
                    raise JsonlCorruptionError(
                        f"record {line_number} crosses committed boundary {bound}"
                    )
                record = decode_record(
                    line,
                    label=f"record {line_number}",
                    maximum_bytes=self.maximum_record_bytes,
                )
                records.append(record)
                data_hash = _extend_data_hash(data_hash, line)
                offset += len(line)
                if offset == bound:
                    break
        if offset != bound:
            raise JsonlCorruptionError(
                f"committed boundary {bound} is not a complete JSONL prefix"
            )
        self._require_scan_matches_state(
            _DataScan(offset, len(records), data_hash, None, ()), state
        )
        return records

    def _inspect_prefix_unlocked(
        self,
        *,
        validator: Callable[[dict[str, Any], int], None] | None,
        collect_records: bool,
    ) -> StorePrefixScan:
        if not os.path.lexists(self.path):
            return StorePrefixScan(self.path, False, True, 0, 0, 0, ())
        if self.path.is_symlink():
            return StorePrefixScan(
                self.path,
                True,
                False,
                0,
                0,
                0,
                (),
                error="JSONL data path must not be a symbolic link",
            )
        if not self.path.is_file():
            return StorePrefixScan(
                self.path,
                True,
                False,
                0,
                0,
                0,
                (),
                error="JSONL data path is not a regular file",
            )

        size = self.path.stat().st_size
        records: list[dict[str, Any]] = []
        valid_prefix = 0
        record_count = 0
        with self.path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    record = decode_record(
                        line,
                        label=f"line {line_number}",
                        maximum_bytes=self.maximum_record_bytes,
                    )
                    if validator is not None:
                        validator(record, line_number)
                except ValueError as error:
                    return StorePrefixScan(
                        self.path,
                        True,
                        False,
                        size,
                        valid_prefix,
                        record_count,
                        tuple(records),
                        error_line=line_number,
                        error=str(error),
                    )
                record_count += 1
                valid_prefix += len(line)
                if collect_records:
                    records.append(record)

        if self.path.stat().st_size != size:
            raise JsonlCorruptionError(f"JSONL data changed while it was inspected: {self.path}")
        return StorePrefixScan(
            self.path,
            True,
            True,
            size,
            valid_prefix,
            record_count,
            tuple(records),
        )

    def _invalidate_derived_state_unlocked(self) -> None:
        for label, path in (
            ("transaction", self.transaction_path),
            ("JIDX", self.jidx_path),
        ):
            if not os.path.lexists(path):
                continue
            if path.is_symlink() or not path.is_file():
                raise JsonlCorruptionError(
                    f"{label} path is not a regular file: {path}"
                )
            path.unlink()
            _fsync_parent(path.parent)
        self._index_cache = None

    def _ensure_index_unlocked(
        self,
        state: TransactionState,
        *,
        scan: _DataScan | None,
    ) -> None:
        """Ensure the derived JIDX header names the committed data generation."""

        _, actual_mtime = _source_facts(self.path)
        expected_ticks = mtime_ns_to_dotnet_ticks(actual_mtime)
        try:
            header = read_index_header(self.jidx_path)
            current = (
                header.version == JSOI_VERSION
                and header.line_count == state.record_count
                and header.source_length == state.committed_bytes
                and header.source_last_write_ticks == expected_ticks
                and self.jidx_path.stat().st_size == state.jidx_bytes
            )
        except (FileNotFoundError, JsonlFormatError, JsonlCorruptionError):
            current = False
        if current:
            return

        verified = scan or self._scan_data_unlocked()
        self._require_scan_matches_state(verified, state)
        if calculate_index_hash(verified.offsets) != state.jidx_hash:
            raise JsonlCorruptionError(
                "committed JIDX hash disagrees with authoritative JSONL record boundaries"
            )
        self._replace_index_unlocked(verified)

    def _replace_index_unlocked(self, scan: _DataScan) -> None:
        replace_index_durable(
            self.jidx_path,
            scan.offsets,
            source_length=scan.size_bytes,
            source_mtime_ns=scan.mtime_ns,
        )
        self._index_cache = None

    def _load_verified_index_unlocked(
        self,
        state: TransactionState,
        *,
        repair: bool,
    ) -> Jidx:
        if self._index_cache is not None:
            transaction_hash, cached = self._index_cache
            if transaction_hash == state.transaction_hash and cached.is_current():
                return cached
        try:
            loaded = self._require_index_matches_state(
                read_index(self.jidx_path, self.path),
                state,
            )
            self._index_cache = (state.transaction_hash, loaded)
            return loaded
        except (FileNotFoundError, JsonlFormatError, JsonlCorruptionError):
            if not repair:
                raise
        scan = self._scan_data_unlocked()
        self._require_scan_matches_state(scan, state)
        if calculate_index_hash(scan.offsets) != state.jidx_hash:
            raise JsonlCorruptionError(
                "committed JIDX hash disagrees with authoritative JSONL record boundaries"
            )
        self._replace_index_unlocked(scan)
        loaded = self._require_index_matches_state(
            read_index(self.jidx_path, self.path),
            state,
        )
        self._index_cache = (state.transaction_hash, loaded)
        return loaded

    def _require_index_matches_state(
        self,
        jidx: Jidx,
        state: TransactionState,
    ) -> Jidx:
        disagreements: list[str] = []
        if jidx.version != JSOI_VERSION:
            disagreements.append(f"version={jidx.version} (expected {JSOI_VERSION})")
        if jidx.line_count != state.record_count:
            disagreements.append(
                f"records={jidx.line_count} (committed {state.record_count})"
            )
        if jidx.source_length != state.committed_bytes:
            disagreements.append(
                f"source bytes={jidx.source_length} (committed {state.committed_bytes})"
            )
        if jidx.index_path.stat().st_size != state.jidx_bytes:
            disagreements.append(
                f"index bytes={jidx.index_path.stat().st_size} "
                f"(committed {state.jidx_bytes})"
            )
        if calculate_index_hash(jidx.offsets) != state.jidx_hash:
            disagreements.append("offset hash differs from committed transaction")
        if not jidx.is_current():
            disagreements.append("source length or timestamp is stale")
        if disagreements:
            raise JsonlCorruptionError(
                f"JIDX disagrees with transaction table: {', '.join(disagreements)}"
            )
        return jidx

    def _read_indexed_range_unlocked(
        self,
        jidx: Jidx,
        start: int,
        stop: int,
    ) -> list[dict[str, Any]]:
        if not 0 <= start <= stop <= jidx.line_count:
            raise IndexError(
                f"record range [{start}, {stop}) is outside [0, {jidx.line_count})"
            )
        if start == stop:
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            handle.seek(jidx.offsets[start])
            for ordinal in range(start, stop):
                record_start = jidx.offsets[ordinal]
                record_stop = (
                    jidx.offsets[ordinal + 1]
                    if ordinal + 1 < jidx.line_count
                    else jidx.source_length
                )
                if handle.tell() != record_start:
                    raise JsonlCorruptionError(
                        f"indexed read did not land at record {ordinal} offset {record_start}"
                    )
                line = handle.read(record_stop - record_start)
                if len(line) != record_stop - record_start:
                    raise JsonlCorruptionError(
                        f"record {ordinal} ended before indexed boundary {record_stop}"
                    )
                records.append(
                    decode_record(
                        line,
                        label=f"record {ordinal + 1}",
                        maximum_bytes=self.maximum_record_bytes,
                    )
                )
        return records

    def _read_transaction_rows_unlocked(self) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        with self.transaction_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = decode_record(
                    line,
                    label=f"transaction row {line_number}",
                    maximum_bytes=MAX_TRANSACTION_BYTES,
                )
                _validate_transaction_record(row)
                _validate_transaction_sequence(previous, row)
                rows.append(row)
                previous = row
        if not rows:
            raise JsonlCorruptionError(
                f"transaction table has no valid checkpoint: {self.transaction_path}"
            )
        return tuple(rows)

    def _scan_data_unlocked(self) -> _DataScan:
        if not os.path.lexists(self.path):
            return _DataScan(0, 0, _DATA_SEED, None, ())
        if self.path.is_symlink() or not self.path.is_file():
            raise JsonlCorruptionError(f"JSONL data path is not a regular file: {self.path}")

        data_hash = _DATA_SEED
        size_bytes = 0
        record_count = 0
        offsets: list[int] = []
        with self.path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                offsets.append(size_bytes)
                decode_record(
                    line,
                    label=f"record {line_number}",
                    maximum_bytes=self.maximum_record_bytes,
                )
                data_hash = _extend_data_hash(data_hash, line)
                size_bytes += len(line)
                record_count += 1
        stat = self.path.stat()
        if stat.st_size != size_bytes:
            raise JsonlCorruptionError(
                f"JSONL data changed while it was scanned: {self.path}"
            )
        return _DataScan(
            size_bytes,
            record_count,
            data_hash,
            stat.st_mtime_ns,
            tuple(offsets),
        )

    def _scan_transactions_unlocked(self) -> tuple[TransactionState, int]:
        previous: dict[str, Any] | None = None
        valid_prefix = 0
        with self.transaction_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = decode_record(
                        line,
                        label=f"transaction row {line_number}",
                        maximum_bytes=MAX_TRANSACTION_BYTES,
                    )
                    _validate_transaction_record(row)
                    _validate_transaction_sequence(previous, row)
                except (JsonlFormatError, JsonlCorruptionError):
                    break
                previous = row
                valid_prefix += len(line)
        if previous is None:
            raise JsonlCorruptionError(
                f"transaction table has no valid checkpoint: {self.transaction_path}"
            )
        return _state_from_transaction(previous), valid_prefix

    def _require_scan_matches_state(
        self, scan: _DataScan, state: TransactionState
    ) -> None:
        disagreements: list[str] = []
        if scan.size_bytes != state.committed_bytes:
            disagreements.append(
                f"bytes={scan.size_bytes} (committed {state.committed_bytes})"
            )
        if scan.record_count != state.record_count:
            disagreements.append(
                f"records={scan.record_count} (committed {state.record_count})"
            )
        if scan.data_hash != state.data_hash:
            disagreements.append("data hash differs from committed transaction")
        if disagreements:
            raise JsonlCorruptionError(
                f"JSONL data disagrees with transaction table: {', '.join(disagreements)}"
            )

    def _validate_paths_unlocked(self) -> None:
        for label, path in (
            ("JSONL data", self.path),
            ("transaction", self.transaction_path),
            ("JIDX", self.jidx_path),
        ):
            if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
                raise JsonlCorruptionError(f"{label} path is not a regular file: {path}")


def _build_transaction_record(
    *,
    kind: str,
    generation: int,
    previous_transaction_hash: str | None,
    start_offset: int,
    end_offset: int,
    start_record_count: int,
    appended_records: int,
    record_count: int,
    data_hash: str,
    jidx_hash: str,
    source_mtime_ns: int | None,
    partition: Mapping[str, Any] | None,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": TRANSACTION_SCHEMA_ID,
        "kind": kind,
        "transaction_id": str(uuid.uuid4()),
        "generation": generation,
        "committed_at": _utc_now(),
        "previous_transaction_hash": previous_transaction_hash,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "start_record_count": start_record_count,
        "appended_records": appended_records,
        "record_count": record_count,
        "data_hash": data_hash,
        "jidx_version": JSOI_VERSION,
        "jidx_bytes": expected_jidx_bytes(record_count),
        "jidx_hash": jidx_hash,
        "source_mtime_ns": source_mtime_ns,
        "partition": _normalize_partition(partition),
        "metadata": _normalize_metadata(metadata),
    }
    row["transaction_hash"] = _calculate_transaction_hash(row)
    _validate_transaction_record(row)
    return row


def _validate_transaction_record(row: Mapping[str, Any]) -> None:
    if set(row) != _TRANSACTION_FIELDS:
        missing = sorted(_TRANSACTION_FIELDS - set(row))
        extra = sorted(set(row) - _TRANSACTION_FIELDS)
        raise JsonlFormatError(
            f"transaction row has incorrect fields (missing={missing}, extra={extra})"
        )
    if row.get("schema") != TRANSACTION_SCHEMA_ID:
        raise JsonlFormatError("transaction row declares an unsupported schema")
    kind = row.get("kind")
    if kind not in {"checkpoint", "append"}:
        raise JsonlFormatError("transaction kind must be checkpoint or append")
    _canonical_uuid(row.get("transaction_id"), "transaction_id")
    generation = _nonnegative_integer(row.get("generation"), "generation")
    _validate_timestamp(row.get("committed_at"))
    previous_hash = row.get("previous_transaction_hash")
    if previous_hash is not None:
        _require_hash(previous_hash, "previous_transaction_hash")
    start_offset = _nonnegative_integer(row.get("start_offset"), "start_offset")
    end_offset = _nonnegative_integer(row.get("end_offset"), "end_offset")
    start_count = _nonnegative_integer(
        row.get("start_record_count"), "start_record_count"
    )
    appended = _nonnegative_integer(row.get("appended_records"), "appended_records")
    count = _nonnegative_integer(row.get("record_count"), "record_count")
    if end_offset < start_offset:
        raise JsonlFormatError("transaction end_offset precedes start_offset")
    if count != start_count + appended:
        raise JsonlFormatError("transaction record counts do not reconcile")
    if kind == "checkpoint":
        if generation != 0 or previous_hash is not None or start_offset != 0 or start_count != 0:
            raise JsonlFormatError("checkpoint must establish generation zero from an empty base")
    elif generation == 0 or previous_hash is None:
        raise JsonlFormatError("append transaction must follow an earlier generation")
    _require_hash(row.get("data_hash"), "data_hash")
    if row.get("jidx_version") != JSOI_VERSION:
        raise JsonlFormatError(f"jidx_version must be {JSOI_VERSION}")
    jidx_bytes = _nonnegative_integer(row.get("jidx_bytes"), "jidx_bytes")
    if jidx_bytes != expected_jidx_bytes(count):
        raise JsonlFormatError("jidx_bytes does not match the committed record count")
    _require_hash(row.get("jidx_hash"), "jidx_hash")
    mtime = row.get("source_mtime_ns")
    if mtime is not None:
        _nonnegative_integer(mtime, "source_mtime_ns")
    partition = _normalize_partition(row.get("partition"))
    if kind == "checkpoint" and partition is not None:
        raise JsonlFormatError("checkpoint cannot declare a partition")
    if appended == 0 and partition is not None:
        raise JsonlFormatError("an empty append cannot declare a partition")
    _normalize_metadata(row.get("metadata"))
    transaction_hash = row.get("transaction_hash")
    _require_hash(transaction_hash, "transaction_hash")
    if _calculate_transaction_hash(row) != transaction_hash:
        raise JsonlFormatError("transaction_hash does not match transaction contents")


def _validate_transaction_sequence(
    previous: Mapping[str, Any] | None, row: Mapping[str, Any]
) -> None:
    if previous is None:
        if row["kind"] != "checkpoint":
            raise JsonlCorruptionError("transaction table must begin with a checkpoint")
        return
    if row["kind"] != "append":
        raise JsonlCorruptionError("transaction table contains a second checkpoint")
    if row["generation"] != previous["generation"] + 1:
        raise JsonlCorruptionError("transaction generation is not contiguous")
    if row["previous_transaction_hash"] != previous["transaction_hash"]:
        raise JsonlCorruptionError("transaction hash chain is broken")
    if row["start_offset"] != previous["end_offset"]:
        raise JsonlCorruptionError("transaction byte boundaries are not contiguous")
    if row["start_record_count"] != previous["record_count"]:
        raise JsonlCorruptionError("transaction record counts are not contiguous")
    if row["appended_records"] == 0 and row["jidx_hash"] != previous["jidx_hash"]:
        raise JsonlCorruptionError("empty transaction changed the committed JIDX hash")


def _state_from_transaction(row: Mapping[str, Any]) -> TransactionState:
    _validate_transaction_record(row)
    return TransactionState(
        generation=int(row["generation"]),
        committed_bytes=int(row["end_offset"]),
        record_count=int(row["record_count"]),
        data_hash=str(row["data_hash"]),
        jidx_bytes=int(row["jidx_bytes"]),
        jidx_hash=str(row["jidx_hash"]),
        metadata=deepcopy(dict(row["metadata"])),
        transaction_hash=str(row["transaction_hash"]),
        source_mtime_ns=row["source_mtime_ns"],
    )


def _read_last_transaction(path: Path) -> dict[str, Any]:
    line = _read_last_line(path, maximum_bytes=MAX_TRANSACTION_BYTES)
    row = decode_record(
        line,
        label="last transaction row",
        maximum_bytes=MAX_TRANSACTION_BYTES,
    )
    _validate_transaction_record(row)
    return row


def _read_last_line(path: Path, *, maximum_bytes: int) -> bytes:
    size = path.stat().st_size
    if size == 0:
        raise JsonlFormatError(f"JSONL file is empty: {path}")
    window_size = min(size, maximum_bytes + 1)
    with path.open("rb") as handle:
        handle.seek(size - window_size)
        window = handle.read(window_size)
    if not window.endswith(b"\n"):
        raise JsonlFormatError(f"JSONL file has an unterminated final row: {path}")
    if window_size == size:
        separator = window[:-1].rfind(b"\n")
        return window[separator + 1 :]
    separator = window[:-1].rfind(b"\n")
    if separator < 0:
        raise JsonlFormatError(f"final JSONL row exceeds {maximum_bytes} bytes: {path}")
    return window[separator + 1 :]


def _extend_data_hash(previous: str, line: bytes) -> str:
    _require_hash(previous, "previous data hash")
    digest = hashlib.sha256()
    digest.update(_DATA_RECORD_DOMAIN)
    digest.update(bytes.fromhex(previous[len(HASH_PREFIX) :]))
    digest.update(struct.pack(">Q", len(line)))
    digest.update(line)
    return HASH_PREFIX + digest.hexdigest()


def _calculate_transaction_hash(row: Mapping[str, Any]) -> str:
    unsigned = dict(row)
    unsigned.pop("transaction_hash", None)
    return HASH_PREFIX + hashlib.sha256(canonical_bytes(unsigned)).hexdigest()


def _normalize_metadata(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JsonlFormatError("transaction metadata must be a JSON object")
    normalized = deepcopy(dict(value))
    try:
        encoded = canonical_bytes(normalized)
    except (RecursionError, TypeError, ValueError) as error:
        raise JsonlFormatError("transaction metadata contains a non-portable JSON value") from error
    if len(encoded) > MAX_TRANSACTION_BYTES // 2:
        raise JsonlFormatError("transaction metadata is too large")
    return normalized


def _normalize_partition(
    value: Mapping[str, Any] | Any | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise JsonlFormatError("transaction partition must be a JSON object or null")
    normalized = deepcopy(dict(value))
    try:
        encoded = canonical_bytes(normalized)
    except (RecursionError, TypeError, ValueError) as error:
        raise JsonlFormatError(
            "transaction partition contains a non-portable JSON value"
        ) from error
    if len(encoded) > MAX_TRANSACTION_BYTES // 4:
        raise JsonlFormatError("transaction partition is too large")
    return normalized


def _partition_ranges_from_rows(
    rows: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any] | None,
) -> list[PartitionRange]:
    ranges: list[PartitionRange] = []
    for row in rows:
        partition = row["partition"]
        if not isinstance(partition, Mapping) or row["appended_records"] == 0:
            continue
        if selector is not None and not _mapping_contains(partition, selector):
            continue
        ranges.append(
            PartitionRange(
                transaction_id=str(row["transaction_id"]),
                generation=int(row["generation"]),
                start_record=int(row["start_record_count"]),
                stop_record=int(row["record_count"]),
                start_offset=int(row["start_offset"]),
                end_offset=int(row["end_offset"]),
                partition=deepcopy(dict(partition)),
                metadata=deepcopy(dict(row["metadata"])),
            )
        )
    return ranges


def _mapping_contains(actual: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    for key, expected in selector.items():
        if key not in actual:
            return False
        observed = actual[key]
        if isinstance(expected, Mapping):
            if not isinstance(observed, Mapping) or not _mapping_contains(observed, expected):
                return False
        elif observed != expected:
            return False
    return True


def _normalize_record_index(index: int, record_count: int) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("record index must be an integer")
    normalized = index if index >= 0 else record_count + index
    if normalized < 0 or normalized >= record_count:
        raise IndexError(f"record index {index} is outside [0, {record_count})")
    return normalized


def _normalize_record_range(
    start: int,
    stop: int | None,
    record_count: int,
) -> tuple[int, int]:
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("range start must be an integer")
    if stop is not None and (isinstance(stop, bool) or not isinstance(stop, int)):
        raise TypeError("range stop must be an integer or null")
    normalized_start, normalized_stop, _ = slice(start, stop).indices(record_count)
    return normalized_start, normalized_stop


def _canonical_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise JsonlFormatError(f"{label} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise JsonlFormatError(f"{label} must be a UUID string") from error
    if str(parsed) != value:
        raise JsonlFormatError(f"{label} must use canonical lowercase UUID form")
    return value


def _validate_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JsonlFormatError("committed_at must be a UTC timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise JsonlFormatError("committed_at is not a valid timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise JsonlFormatError("committed_at must be UTC")


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JsonlFormatError(f"{label} must be a nonnegative integer")
    return value


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise JsonlFormatError(f"{label} must be a sha256 digest")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _source_facts(path: Path) -> tuple[int, int | None]:
    if not os.path.lexists(path):
        return 0, None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _existing_size(path: Path) -> int:
    return path.stat().st_size if os.path.lexists(path) else 0


def _append_durable(path: Path, payload: bytes) -> None:
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    existed = os.path.lexists(path)
    descriptor = os.open(path, flags, 0o666)
    try:
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not existed:
        _fsync_parent(path.parent)


def _truncate_durable(path: Path, length: int) -> None:
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        os.ftruncate(descriptor, length)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_prefix_new_durable(source: Path, target: Path, length: int) -> None:
    """Durably copy exactly ``length`` source bytes into one new adjacent file."""

    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_BINARY", 0)
    descriptor = -1
    created = False
    try:
        descriptor = os.open(target, flags, 0o666)
        created = True
        with source.open("rb") as handle:
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError(
                        f"source ended before the requested {length}-byte prefix: {source}"
                    )
                _write_all(descriptor, chunk)
                remaining -= len(chunk)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _fsync_parent(target.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        if created and os.path.lexists(target):
            target.unlink()
            _fsync_parent(target.parent)
        raise


def _replace_with_prefix_durable(path: Path, length: int) -> None:
    """Publish an adjacent, durable prefix copy over ``path`` atomically."""

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.repair.tmp")
    _copy_prefix_new_durable(path, temporary, length)
    try:
        os.replace(temporary, path)
        _fsync_parent(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
            _fsync_parent(temporary.parent)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while appending JSONL transaction bytes")
        view = view[written:]


def _fsync_parent(parent: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
