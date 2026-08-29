"""JSOI v2 byte-offset indexes for canonical JSONL stores.

The historical JIDX contract is deliberately retained: one compact binary sidecar maps record
ordinals to byte offsets.  ABLITERALUS extends it in place as data batches commit; rebuilding is a
recovery operation, not the normal append path.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .errors import JsonlCorruptionError, JsonlFormatError

JSOI_MAGIC = b"JSOI"
JSOI_VERSION = 2
JSOI_HEADER = struct.Struct("<4siiqq")
JSOI_OFFSET = struct.Struct("<q")
DOTNET_TICKS_OFFSET = 621_355_968_000_000_000
HASH_PREFIX = "sha256:"

_INT32_MAX = (1 << 31) - 1
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_INDEX_SEED = HASH_PREFIX + hashlib.sha256(
    b"aipithicus/jsonl-engine/jidx-chain/v1"
).hexdigest()
_INDEX_OFFSET_DOMAIN = b"aipithicus/jsonl-engine/jidx-offset/v1\0"


@dataclass(frozen=True, slots=True)
class JidxHeader:
    """Fixed JSOI v2 header facts readable in constant time."""

    version: int
    line_count: int
    source_length: int
    source_last_write_ticks: int


@dataclass(frozen=True, slots=True)
class Jidx:
    """A parsed and structurally validated JSOI v2 index."""

    index_path: Path
    jsonl_path: Path
    version: int
    line_count: int
    source_length: int
    source_last_write_ticks: int
    offsets: tuple[int, ...]

    def is_current(self) -> bool:
        """Return whether the source length and timestamp still match this index."""

        if not os.path.lexists(self.jsonl_path):
            return (
                self.line_count == 0
                and self.source_length == 0
                and self.source_last_write_ticks == 0
            )
        if self.jsonl_path.is_symlink() or not self.jsonl_path.is_file():
            return False
        stat = self.jsonl_path.stat()
        return (
            stat.st_size == self.source_length
            and mtime_ns_to_dotnet_ticks(stat.st_mtime_ns)
            == self.source_last_write_ticks
        )


def jidx_path_for(jsonl_path: str | Path) -> Path:
    """Derive the user-preferred ``{stem}.jidx`` path for a JSONL store."""

    path = Path(jsonl_path)
    if path.suffix.lower() == ".jsonl":
        return path.with_suffix(".jidx")
    return path.with_name(f"{path.name}.jidx")


def expected_jidx_bytes(line_count: int) -> int:
    """Return the exact JSOI v2 byte length for ``line_count`` offsets."""

    _require_line_count(line_count)
    return JSOI_HEADER.size + (JSOI_OFFSET.size * line_count)


def mtime_ns_to_dotnet_ticks(mtime_ns: int | None) -> int:
    """Convert a source mtime to the integer UTC ticks carried by JSOI v2."""

    if mtime_ns is None:
        return 0
    if isinstance(mtime_ns, bool) or not isinstance(mtime_ns, int) or mtime_ns < 0:
        raise JsonlFormatError("source mtime must be a nonnegative integer or null")
    return (mtime_ns // 100) + DOTNET_TICKS_OFFSET


def calculate_index_hash(offsets: Iterable[int]) -> str:
    """Hash an ordered offset population under the transaction-table chain policy."""

    return extend_index_hash(_INDEX_SEED, offsets)


def extend_index_hash(previous: str, offsets: Iterable[int]) -> str:
    """Extend an existing offset-chain digest without rereading earlier offsets."""

    if not isinstance(previous, str) or _HASH.fullmatch(previous) is None:
        raise JsonlFormatError("previous JIDX hash must be a sha256 digest")
    current = previous
    for offset in offsets:
        _require_offset(offset)
        digest = hashlib.sha256()
        digest.update(_INDEX_OFFSET_DOMAIN)
        digest.update(bytes.fromhex(current[len(HASH_PREFIX) :]))
        digest.update(JSOI_OFFSET.pack(offset))
        current = HASH_PREFIX + digest.hexdigest()
    return current


def read_index_header(index_path: str | Path) -> JidxHeader:
    """Read and validate a JSOI v2 header without loading its offset table."""

    path = Path(index_path)
    _require_regular_index(path)
    with path.open("rb") as handle:
        raw = handle.read(JSOI_HEADER.size)
    if len(raw) != JSOI_HEADER.size:
        raise JsonlFormatError(f"truncated JSOI header: {path}")
    magic, version, line_count, source_length, source_ticks = JSOI_HEADER.unpack(raw)
    if magic != JSOI_MAGIC:
        raise JsonlFormatError(f"invalid JSOI index magic bytes: {magic!r}")
    if version != JSOI_VERSION:
        raise JsonlFormatError(
            f"unsupported JSOI index version {version}; expected {JSOI_VERSION}"
        )
    _require_line_count(line_count)
    _require_nonnegative(source_length, "JSOI source length")
    _require_nonnegative(source_ticks, "JSOI source last-write ticks")
    expected = expected_jidx_bytes(line_count)
    actual = path.stat().st_size
    if actual != expected:
        raise JsonlFormatError(
            f"JSOI index has {actual} bytes; expected exactly {expected}: {path}"
        )
    return JidxHeader(version, line_count, source_length, source_ticks)


def read_index(index_path: str | Path, jsonl_path: str | Path) -> Jidx:
    """Parse and structurally validate one complete JSOI v2 index."""

    path = Path(index_path)
    source = Path(jsonl_path)
    header = read_index_header(path)
    with path.open("rb") as handle:
        handle.seek(JSOI_HEADER.size)
        offset_bytes = handle.read(JSOI_OFFSET.size * header.line_count)
    if len(offset_bytes) != JSOI_OFFSET.size * header.line_count:
        raise JsonlFormatError(f"truncated JSOI offset table: {path}")
    offsets = (
        tuple(struct.unpack(f"<{header.line_count}q", offset_bytes))
        if header.line_count
        else ()
    )
    _validate_offsets(offsets, header.source_length)
    return Jidx(
        index_path=path,
        jsonl_path=source,
        version=header.version,
        line_count=header.line_count,
        source_length=header.source_length,
        source_last_write_ticks=header.source_last_write_ticks,
        offsets=offsets,
    )


def replace_index_durable(
    index_path: Path,
    offsets: Sequence[int],
    *,
    source_length: int,
    source_mtime_ns: int | None,
) -> None:
    """Atomically publish a complete deterministic JIDX rebuild."""

    normalized = tuple(offsets)
    _validate_offsets(normalized, source_length)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    scratch = index_path.with_name(f".{index_path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(scratch, flags, 0o666)
    try:
        _write_all(
            descriptor,
            _header_bytes(len(normalized), source_length, source_mtime_ns),
        )
        _write_offsets(descriptor, normalized)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        try:
            scratch.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        os.replace(scratch, index_path)
        _fsync_parent(index_path.parent)
    finally:
        try:
            scratch.unlink()
        except FileNotFoundError:
            pass


def append_index_durable(
    index_path: Path,
    offsets: Sequence[int],
    *,
    previous_line_count: int,
    previous_source_length: int,
    source_length: int,
    source_mtime_ns: int | None,
) -> None:
    """Extend JIDX in place, then advance its fixed header after offsets are durable."""

    _require_line_count(previous_line_count)
    normalized = tuple(offsets)
    _validate_appended_offsets(
        normalized,
        previous_line_count,
        previous_source_length,
        source_length,
    )
    expected_before = expected_jidx_bytes(previous_line_count)
    _require_regular_index(index_path)
    actual_before = index_path.stat().st_size
    if actual_before != expected_before:
        raise JsonlCorruptionError(
            f"JIDX size changed before append: {actual_before} bytes, expected {expected_before}"
        )

    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    descriptor = os.open(index_path, flags)
    try:
        os.lseek(descriptor, 0, os.SEEK_END)
        _write_offsets(descriptor, normalized)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(
            descriptor,
            _header_bytes(
                previous_line_count + len(normalized),
                source_length,
                source_mtime_ns,
            ),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def restore_index_durable(
    index_path: Path,
    *,
    line_count: int,
    source_length: int,
    source_mtime_ns: int | None,
) -> None:
    """Roll an interrupted in-process extension back to a prior committed boundary."""

    _require_regular_index(index_path)
    target_size = expected_jidx_bytes(line_count)
    flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    descriptor = os.open(index_path, flags)
    try:
        os.ftruncate(descriptor, target_size)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        _write_all(
            descriptor,
            _header_bytes(line_count, source_length, source_mtime_ns),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _header_bytes(
    line_count: int, source_length: int, source_mtime_ns: int | None
) -> bytes:
    _require_line_count(line_count)
    _require_nonnegative(source_length, "JSOI source length")
    return JSOI_HEADER.pack(
        JSOI_MAGIC,
        JSOI_VERSION,
        line_count,
        source_length,
        mtime_ns_to_dotnet_ticks(source_mtime_ns),
    )


def _validate_offsets(offsets: Sequence[int], source_length: int) -> None:
    _require_line_count(len(offsets))
    _require_nonnegative(source_length, "JSOI source length")
    if not offsets:
        if source_length != 0:
            raise JsonlFormatError("a non-empty JSOI source must have at least one offset")
        return
    if offsets[0] != 0:
        raise JsonlFormatError("the first JSOI record offset must be zero")
    previous = -1
    for offset in offsets:
        _require_offset(offset)
        if offset <= previous:
            raise JsonlFormatError("JSOI record offsets must be strictly increasing")
        if offset >= source_length:
            raise JsonlFormatError("JSOI record offset lies outside the source")
        previous = offset


def _validate_appended_offsets(
    offsets: Sequence[int],
    previous_line_count: int,
    previous_source_length: int,
    source_length: int,
) -> None:
    _require_line_count(previous_line_count + len(offsets))
    _require_nonnegative(previous_source_length, "previous JSOI source length")
    _require_nonnegative(source_length, "JSOI source length")
    if not offsets:
        return
    if offsets[0] != previous_source_length:
        raise JsonlFormatError(
            "the first appended JSOI offset must equal the prior source length"
        )
    if previous_line_count == 0 and previous_source_length != 0:
        raise JsonlFormatError("an empty JSOI index cannot describe non-empty source bytes")
    previous = -1
    for offset in offsets:
        _require_offset(offset)
        if offset <= previous:
            raise JsonlFormatError("appended JSOI offsets must be strictly increasing")
        if offset >= source_length:
            raise JsonlFormatError("appended JSOI offset lies outside the source")
        previous = offset


def _require_regular_index(path: Path) -> None:
    if not os.path.lexists(path):
        raise FileNotFoundError(f"JIDX does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise JsonlCorruptionError(f"JIDX path is not a regular file: {path}")


def _require_line_count(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _INT32_MAX:
        raise JsonlFormatError(f"JSOI line count must be between 0 and {_INT32_MAX}")


def _require_offset(value: int) -> None:
    _require_nonnegative(value, "JSOI record offset")
    if value > (1 << 63) - 1:
        raise JsonlFormatError("JSOI record offset exceeds int64")


def _require_nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JsonlFormatError(f"{label} must be a nonnegative integer")


def _write_offsets(descriptor: int, offsets: Sequence[int]) -> None:
    chunk_size = 8192
    for start in range(0, len(offsets), chunk_size):
        chunk = offsets[start : start + chunk_size]
        _write_all(descriptor, struct.pack(f"<{len(chunk)}q", *chunk))


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write while publishing JIDX bytes")
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
