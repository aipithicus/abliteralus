"""Workspace storage census, history, and conservative reclamation.

The storage boundary is explicit: disposable runs, caches, and known tool state
may be reclaimed; scientific outputs and managed runtimes are report-only.
Snapshot files contain aggregate metadata, never commands, environments, model
prompts, or secret values.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from abliteralus.run_paths import list_run_workspaces, stale_run_workspaces

STORAGE_REPORT_SCHEMA_VERSION = 1
STORAGE_HISTORY_DIRECTORY = Path(".scratch/storage/snapshots")
RECLAIMABLE_CATEGORIES = frozenset({"runs", "huggingface-cache", "uv-cache", "tool-state"})
_CACHE_ROOTS = {
    "huggingface-cache": (Path(".scratch/cache/huggingface"),),
    "uv-cache": (Path(".scratch/cache/uv"), Path(".scratch/uv-cache")),
}
_TOOL_STATE_DIRECTORY_NAMES = frozenset(
    {
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        "test-results",
    }
)
_TOOL_STATE_FILE_NAMES = frozenset(
    {
        "coverage.xml",
    }
)
_TOOL_DISCOVERY_EXCLUSIONS = frozenset(
    {
        ".git",
        ".scratch",
        ".venv",
        "deps",
        "outputs",
    }
)
_LEGACY_TOOL_STATE_NAMES = (
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage.xml",
    "htmlcov",
    "test-results",
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


class StorageError(ValueError):
    """Storage state could not be measured or safely reclaimed."""


@dataclass(slots=True)
class _ScanStats:
    bytes: int = 0
    files: int = 0
    directories: int = 0
    newest_mtime: float | None = None
    oldest_mtime: float | None = None
    scan_errors: int = 0
    skipped_links: int = 0
    error_paths: list[Path] = field(default_factory=list)

    def observe_mtime(self, value: float) -> None:
        self.newest_mtime = value if self.newest_mtime is None else max(self.newest_mtime, value)
        self.oldest_mtime = value if self.oldest_mtime is None else min(self.oldest_mtime, value)

    def merge(self, other: _ScanStats) -> None:
        self.bytes += other.bytes
        self.files += other.files
        self.directories += other.directories
        self.scan_errors += other.scan_errors
        self.skipped_links += other.skipped_links
        for path in other.error_paths:
            if len(self.error_paths) >= 10:
                break
            self.error_paths.append(path)
        if other.newest_mtime is not None:
            self.observe_mtime(other.newest_mtime)
        if other.oldest_mtime is not None:
            self.oldest_mtime = (
                other.oldest_mtime
                if self.oldest_mtime is None
                else min(self.oldest_mtime, other.oldest_mtime)
            )


@dataclass(frozen=True, slots=True)
class StorageCategory:
    name: str
    roots: tuple[str, ...]
    bytes: int
    files: int
    directories: int
    newest_modified_at: str | None
    oldest_modified_at: str | None
    scan_errors: int
    scan_error_paths: tuple[str, ...]
    skipped_links: int
    reclaimable: bool
    reclaim_policy: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "roots": list(self.roots),
            "bytes": self.bytes,
            "files": self.files,
            "directories": self.directories,
            "newest_modified_at": self.newest_modified_at,
            "oldest_modified_at": self.oldest_modified_at,
            "scan_errors": self.scan_errors,
            "scan_error_paths": list(self.scan_error_paths),
            "skipped_links": self.skipped_links,
            "reclaimable": self.reclaimable,
            "reclaim_policy": self.reclaim_policy,
        }


@dataclass(frozen=True, slots=True)
class StorageReport:
    measured_at: str
    categories: tuple[StorageCategory, ...]
    volume_total_bytes: int
    volume_used_bytes: int
    volume_free_bytes: int
    schema_version: int = STORAGE_REPORT_SCHEMA_VERSION

    @property
    def total_bytes(self) -> int:
        return sum(category.bytes for category in self.categories)

    @property
    def total_files(self) -> int:
        return sum(category.files for category in self.categories)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "measured_at": self.measured_at,
            "measurement": "logical file bytes; links and reparse points are not followed",
            "total_bytes": self.total_bytes,
            "total_files": self.total_files,
            "volume": {
                "total_bytes": self.volume_total_bytes,
                "used_bytes": self.volume_used_bytes,
                "free_bytes": self.volume_free_bytes,
            },
            "categories": [category.to_dict() for category in self.categories],
        }


@dataclass(frozen=True, slots=True)
class ReclaimCandidate:
    category: str
    path: Path = field(repr=False)
    bytes: int
    files: int
    newest_modified_at: str | None
    reason: str

    def to_dict(self, repository: Path) -> dict[str, object]:
        return {
            "category": self.category,
            "path": _relative_display(repository, self.path),
            "bytes": self.bytes,
            "files": self.files,
            "newest_modified_at": self.newest_modified_at,
            "reason": self.reason,
        }


def measure_storage(
    repository: str | os.PathLike[str],
    *,
    now: datetime | None = None,
) -> StorageReport:
    """Measure every managed top-level storage class without following links."""

    root = _repository(repository)
    measured = _as_utc(now or datetime.now(timezone.utc))
    tool_paths = _tool_state_paths(root)
    paths = {
        "runs": (root / ".scratch" / "runs",),
        "huggingface-cache": tuple(root / path for path in _CACHE_ROOTS["huggingface-cache"]),
        "uv-cache": tuple(root / path for path in _CACHE_ROOTS["uv-cache"]),
        "storage-ledger": (root / ".scratch" / "storage",),
        "outputs": (root / "outputs",),
        "python-environment": (root / ".venv",),
        "portable-dependencies": (root / "deps",),
        "git": (root / ".git",),
        "tool-state": tuple(tool_paths),
    }
    scratch_exclusions = {
        _lexical_key(path)
        for name in ("runs", "huggingface-cache", "uv-cache", "storage-ledger")
        for path in paths[name]
    }
    paths["scratch-other"] = (root / ".scratch",)
    repository_exclusions = {
        _lexical_key(path)
        for path in (
            root / ".scratch",
            root / "outputs",
            root / ".venv",
            root / "deps",
            root / ".git",
            *tool_paths,
        )
    }
    paths["repository-other"] = (root,)

    policies = {
        "runs": (True, "stale run leaves only"),
        "huggingface-cache": (True, "whole cache root only"),
        "uv-cache": (True, "whole cache roots only"),
        "tool-state": (True, "exact known tool paths only"),
        "storage-ledger": (False, "retained history"),
        "outputs": (False, "durable scientific data; never generic-cleaned"),
        "python-environment": (False, "managed runtime; rebuild explicitly"),
        "portable-dependencies": (False, "managed tooling; update explicitly"),
        "git": (False, "source-control data"),
        "scratch-other": (False, "unclassified; inspect before handling"),
        "repository-other": (False, "source and unclassified repository data"),
    }
    order = (
        "runs",
        "huggingface-cache",
        "uv-cache",
        "tool-state",
        "storage-ledger",
        "scratch-other",
        "outputs",
        "python-environment",
        "portable-dependencies",
        "git",
        "repository-other",
    )
    categories: list[StorageCategory] = []
    for name in order:
        exclusions: set[str] = set()
        if name == "scratch-other":
            exclusions = scratch_exclusions
        elif name == "repository-other":
            exclusions = repository_exclusions
        stats = _scan_paths(paths[name], exclusions=exclusions)
        reclaimable, policy = policies[name]
        categories.append(
            StorageCategory(
                name=name,
                roots=tuple(_relative_display(root, path) for path in paths[name]),
                bytes=stats.bytes,
                files=stats.files,
                directories=stats.directories,
                newest_modified_at=_timestamp(stats.newest_mtime),
                oldest_modified_at=_timestamp(stats.oldest_mtime),
                scan_errors=stats.scan_errors,
                scan_error_paths=tuple(_relative_display(root, path) for path in stats.error_paths),
                skipped_links=stats.skipped_links,
                reclaimable=reclaimable,
                reclaim_policy=policy,
            )
        )
    volume = shutil.disk_usage(root)
    return StorageReport(
        measured_at=measured.isoformat().replace("+00:00", "Z"),
        categories=tuple(categories),
        volume_total_bytes=volume.total,
        volume_used_bytes=volume.used,
        volume_free_bytes=volume.free,
    )


def record_storage_snapshot(
    repository: str | os.PathLike[str],
    report: StorageReport,
) -> Path:
    """Persist one immutable aggregate snapshot beneath ignored scratch state."""

    root = _repository(repository)
    destination = root / STORAGE_HISTORY_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    try:
        measured = datetime.fromisoformat(report.measured_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise StorageError("storage report has an invalid measurement timestamp") from error
    stem = measured.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    payload = report.to_dict()
    for attempt in range(100):
        path = destination / f"{stem}-{os.getpid()}-{attempt:02d}.json"
        temporary = destination / f".{path.name}.tmp"
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            continue
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        os.replace(temporary, path)
        return path
    raise StorageError(f"could not allocate a unique storage snapshot beneath {destination}")


def read_storage_history(
    repository: str | os.PathLike[str],
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    """Read recent snapshots with total and per-category byte deltas."""

    if limit <= 0:
        raise StorageError("storage history limit must be positive")
    root = _repository(repository)
    directory = root / STORAGE_HISTORY_DIRECTORY
    if not directory.is_dir():
        return []
    files = sorted(path for path in directory.iterdir() if path.suffix == ".json")
    selected = files[-(limit + 1) :]
    payloads = [_read_snapshot(path) for path in selected]
    rows: list[dict[str, object]] = []
    for index, payload in enumerate(payloads):
        if index == 0 and len(payloads) > limit:
            continue
        previous = payloads[index - 1] if index > 0 else None
        current_categories = _category_bytes(payload)
        previous_categories = _category_bytes(previous) if previous is not None else {}
        total = _snapshot_total(payload)
        previous_total = _snapshot_total(previous) if previous is not None else None
        volume_free = _snapshot_volume_free(payload)
        previous_volume_free = _snapshot_volume_free(previous) if previous is not None else None
        rows.append(
            {
                "measured_at": payload["measured_at"],
                "total_bytes": total,
                "delta_bytes": None if previous_total is None else total - previous_total,
                "volume_free_bytes": volume_free,
                "volume_free_delta_bytes": (
                    None if previous_volume_free is None else volume_free - previous_volume_free
                ),
                "category_bytes": current_categories,
                "category_delta_bytes": {
                    name: value - previous_categories.get(name, 0)
                    for name, value in current_categories.items()
                }
                if previous is not None
                else None,
            }
        )
    return rows


def reclaim_storage(
    repository: str | os.PathLike[str],
    *,
    categories: Sequence[str],
    older_than: timedelta,
    apply: bool = False,
    now: datetime | None = None,
) -> list[ReclaimCandidate]:
    """Preview or apply reclamation for explicitly selected disposable classes."""

    root = _repository(repository)
    selected = _validate_reclaim_categories(categories)
    if older_than.total_seconds() <= 0:
        raise StorageError("storage reclaim age must be positive")
    instant = _as_utc(now or datetime.now(timezone.utc))
    cutoff = instant - older_than
    candidates = _plan_reclaim(root, selected, cutoff=cutoff, now=instant)
    if not apply:
        return candidates
    if selected & {"huggingface-cache", "uv-cache"}:
        active = [
            record
            for record in list_run_workspaces(root)
            if record.status == "active" and record.controller_running is not False
        ]
        if active:
            raise StorageError("cache reclaim is blocked while a managed run is active")
    _validate_candidates_before_apply(root, candidates, cutoff=cutoff)
    for candidate in candidates:
        _remove_reclaimable_path(candidate.path)
    return candidates


def _plan_reclaim(
    root: Path,
    categories: set[str],
    *,
    cutoff: datetime,
    now: datetime,
) -> list[ReclaimCandidate]:
    candidates: list[ReclaimCandidate] = []
    if "runs" in categories:
        age = now - cutoff
        for record in stale_run_workspaces(root, older_than=age, now=now):
            stats = _scan_paths((record.path,))
            candidates.append(_candidate("runs", record.path, stats, cutoff))
    for category in categories & {"huggingface-cache", "uv-cache"}:
        for relative in _CACHE_ROOTS[category]:
            path = root / relative
            candidate = _old_path_candidate(category, path, cutoff)
            if candidate is not None:
                candidates.append(candidate)
    if "tool-state" in categories:
        for path in _tool_state_paths(root):
            candidate = _old_path_candidate("tool-state", path, cutoff)
            if candidate is not None:
                candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: (candidate.category, str(candidate.path)))


def _old_path_candidate(
    category: str,
    path: Path,
    cutoff: datetime,
) -> ReclaimCandidate | None:
    if not path.exists():
        return None
    stats = _scan_paths((path,))
    if stats.scan_errors:
        # The report identifies the bounded error path.  Do not let one poisoned
        # legacy cache prevent preview/reclaim of other independently safe roots.
        return None
    if stats.newest_mtime is None:
        return None
    if datetime.fromtimestamp(stats.newest_mtime, timezone.utc) > cutoff:
        return None
    return _candidate(category, path, stats, cutoff)


def _candidate(
    category: str,
    path: Path,
    stats: _ScanStats,
    cutoff: datetime,
) -> ReclaimCandidate:
    return ReclaimCandidate(
        category=category,
        path=path,
        bytes=stats.bytes,
        files=stats.files,
        newest_modified_at=_timestamp(stats.newest_mtime),
        reason=f"no observed changes after {cutoff.isoformat().replace('+00:00', 'Z')}",
    )


def _validate_candidates_before_apply(
    root: Path,
    candidates: Sequence[ReclaimCandidate],
    *,
    cutoff: datetime,
) -> None:
    cache_paths = {
        _path_key(root / relative) for roots in _CACHE_ROOTS.values() for relative in roots
    }
    tool_paths = {_path_key(path) for path in _tool_state_paths(root)}
    run_root = root / ".scratch" / "runs"
    for candidate in candidates:
        key = _path_key(candidate.path)
        if candidate.category == "runs":
            if candidate.path.parent.parent != run_root:
                raise StorageError("refusing to reclaim a path outside the run workspace root")
        elif candidate.category in {"huggingface-cache", "uv-cache"}:
            if key not in cache_paths:
                raise StorageError("refusing to reclaim an unrecognized cache path")
        elif candidate.category == "tool-state":
            if key not in tool_paths:
                raise StorageError("refusing to reclaim an unrecognized tool-state path")
        else:  # pragma: no cover - candidates are constructed internally
            raise StorageError(f"refusing unsupported reclaim category: {candidate.category}")
        refreshed = _old_path_candidate(candidate.category, candidate.path, cutoff)
        if refreshed is None:
            raise StorageError(f"reclaim candidate changed after preview: {candidate.path}")


def _scan_paths(paths: Sequence[Path], *, exclusions: set[str] | None = None) -> _ScanStats:
    combined = _ScanStats()
    excluded = exclusions or set()
    for path in paths:
        combined.merge(_scan_path(path, exclusions=excluded))
    return combined


def _scan_path(path: Path, *, exclusions: set[str]) -> _ScanStats:
    result = _ScanStats()
    if not path.exists():
        return result
    stack = [path]
    while stack:
        current = stack.pop()
        if _lexical_key(current) in exclusions:
            continue
        try:
            metadata = os.lstat(current)
        except OSError:
            result.scan_errors += 1
            if len(result.error_paths) < 10:
                result.error_paths.append(current)
            continue
        if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
            result.skipped_links += 1
            continue
        result.observe_mtime(metadata.st_mtime)
        if stat.S_ISDIR(metadata.st_mode):
            result.directories += 1
            try:
                with os.scandir(current) as entries:
                    stack.extend(Path(entry.path) for entry in entries)
            except OSError:
                result.scan_errors += 1
                if len(result.error_paths) < 10:
                    result.error_paths.append(current)
            continue
        if stat.S_ISREG(metadata.st_mode):
            result.files += 1
            result.bytes += metadata.st_size
    return result


def _tool_state_paths(root: Path) -> list[Path]:
    # Retain missing conventional root paths in reports so the policy surface is
    # stable, then discover nested cache directories without entering them.
    paths = [root / name for name in _LEGACY_TOOL_STATE_NAMES]
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
                        continue
                    if stat.S_ISDIR(metadata.st_mode):
                        if directory == root and entry.name in _TOOL_DISCOVERY_EXCLUSIONS:
                            continue
                        if entry.name in _TOOL_STATE_DIRECTORY_NAMES:
                            paths.append(path)
                        else:
                            stack.append(path)
                    elif stat.S_ISREG(metadata.st_mode) and (
                        entry.name in _TOOL_STATE_FILE_NAMES
                        or entry.name == ".coverage"
                        or entry.name.startswith(".coverage.")
                    ):
                        paths.append(path)
        except OSError:
            continue
    return sorted(set(paths), key=str)


def _read_snapshot(path: Path) -> dict[str, object]:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise StorageError(f"storage snapshot is unexpectedly large: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StorageError(f"storage snapshot is unreadable: {path.name}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != STORAGE_REPORT_SCHEMA_VERSION
    ):
        raise StorageError(f"storage snapshot has an unsupported schema: {path.name}")
    if not isinstance(payload.get("measured_at"), str):
        raise StorageError(f"storage snapshot has no measurement time: {path.name}")
    _snapshot_total(payload)
    _snapshot_volume_free(payload)
    _category_bytes(payload)
    return payload


def _snapshot_total(payload: dict[str, object] | None) -> int:
    if payload is None:
        raise StorageError("storage snapshot is missing")
    value = payload.get("total_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StorageError("storage snapshot has an invalid total")
    return value


def _category_bytes(payload: dict[str, object] | None) -> dict[str, int]:
    if payload is None:
        return {}
    values = payload.get("categories")
    if not isinstance(values, list):
        raise StorageError("storage snapshot has invalid categories")
    result: dict[str, int] = {}
    for value in values:
        if not isinstance(value, dict):
            raise StorageError("storage snapshot contains an invalid category")
        name = value.get("name")
        size = value.get("bytes")
        if (
            not isinstance(name, str)
            or not name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise StorageError("storage snapshot contains invalid category metadata")
        result[name] = size
    return result


def _snapshot_volume_free(payload: dict[str, object] | None) -> int:
    if payload is None:
        raise StorageError("storage snapshot is missing")
    volume = payload.get("volume")
    if not isinstance(volume, dict):
        raise StorageError("storage snapshot has invalid volume metadata")
    value = volume.get("free_bytes")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StorageError("storage snapshot has invalid free-space metadata")
    return value


def _remove_reclaimable_path(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or _is_reparse(metadata):
        raise StorageError(f"refusing to reclaim a link or reparse point: {path}")
    if stat.S_ISDIR(metadata.st_mode):
        shutil.rmtree(path, onerror=_make_writable_and_retry)
        return
    try:
        path.unlink()
    except PermissionError:
        path.chmod(stat.S_IREAD | stat.S_IWRITE)
        path.unlink()


def _make_writable_and_retry(function, name, exception_info) -> None:
    error = exception_info[1]
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(name, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    function(name)


def _validate_reclaim_categories(categories: Sequence[str]) -> set[str]:
    selected = {str(category).strip() for category in categories if str(category).strip()}
    if not selected:
        raise StorageError("at least one reclaim category is required")
    unknown = sorted(selected - RECLAIMABLE_CATEGORIES)
    if unknown:
        raise StorageError(f"unsupported reclaim categories: {', '.join(unknown)}")
    return selected


def _repository(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise StorageError(f"repository does not exist: {path}")
    return path


def _relative_display(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return "." if not relative.parts else relative.as_posix()


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _lexical_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def _is_reparse(metadata: os.stat_result) -> bool:
    return bool(getattr(metadata, "st_file_attributes", 0) & _REPARSE_POINT)


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise StorageError("storage time must include a timezone")
    return value.astimezone(timezone.utc)
