"""Run-scoped ephemeral paths for local experiment controllers.

Durable experiment data belongs under ``outputs``.  This module owns the
disposable side of a run: temporary files, logs produced only for controller
diagnostics, and test-process state.  Each leaf is allocated with one atomic
``mkdir`` so a stale or concurrently-created workspace cannot be reused.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

RUN_CONTEXT_FILENAME = "run-context.json"
RUN_CONTEXT_SCHEMA_VERSION = 1
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_AUTO_ALLOCATION_ATTEMPTS = 100


class RunPathError(ValueError):
    """A run workspace could not be safely allocated or interpreted."""


@dataclass(frozen=True, slots=True)
class RunWorkspace:
    """One controller-owned workspace beneath ``.scratch/runs``."""

    repository: Path
    kind: str
    run_id: str
    path: Path
    temp_dir: Path
    logs_dir: Path
    reports_dir: Path
    created_at: str

    @property
    def context_path(self) -> Path:
        return self.path / RUN_CONTEXT_FILENAME

    def child_environment(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        coverage: bool = False,
    ) -> dict[str, str]:
        """Build, but do not install, the environment for a child process.

        ``tempfile.gettempdir()`` caches its answer, so these values must be in
        the environment before the child interpreter starts.
        """

        environment = dict(os.environ if environ is None else environ)
        temporary = str(self.temp_dir)
        environment.update(
            {
                "TEMP": temporary,
                "TMP": temporary,
                "TMPDIR": temporary,
                "ABLITERALUS_RUN_ID": self.run_id,
                "ABLITERALUS_WORK_DIR": str(self.path),
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        if coverage:
            coverage_dir = self.path / "coverage"
            coverage_dir.mkdir(exist_ok=True)
            environment["COVERAGE_FILE"] = str(coverage_dir / ".coverage")
        return environment

    def record_result(self, returncode: int) -> None:
        """Record the child result for kept or crash-recovered workspaces."""

        status = "completed" if returncode == 0 else "failed"
        _write_context(self, status=status, returncode=returncode)

    def record_interrupted(self) -> None:
        """Record controller interruption before cleanup is attempted."""

        _write_context(self, status="interrupted", returncode=None)

    def cleanup(self) -> None:
        """Remove this exact workspace without accepting a broader target."""

        expected_parent = self.repository / ".scratch" / "runs" / self.kind
        if self.path.parent != expected_parent or self.path.name != self.run_id:
            raise RunPathError("refusing to clean a workspace outside its run-path scope")
        if self.path.exists():
            _remove_tree(self.path)


@dataclass(frozen=True, slots=True)
class RunWorkspaceInfo:
    """Non-secret metadata used by run inspection and stale cleanup."""

    kind: str
    run_id: str
    path: Path
    status: str
    controller_pid: int | None
    controller_running: bool | None
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "run_id": self.run_id,
            "path": str(self.path),
            "status": self.status,
            "controller_pid": self.controller_pid,
            "controller_running": self.controller_running,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


def allocate_run_workspace(
    repository: str | os.PathLike[str],
    kind: str,
    *,
    run_id: str | None = None,
    now: datetime | None = None,
) -> RunWorkspace:
    """Atomically allocate a normal inherited-ACL workspace.

    Explicit identifiers are never silently changed.  Automatically generated
    identifiers receive a bounded numeric suffix when two controllers allocate
    during the same UTC second.
    """

    root = Path(repository).expanduser().resolve()
    if not root.is_dir():
        raise RunPathError(f"repository does not exist: {root}")
    validated_kind = _validate_segment(kind, "run kind")
    parent = root / ".scratch" / "runs" / validated_kind
    # Deliberately use ordinary mkdir rather than tempfile.mkdtemp.  On Windows,
    # the latter's restrictive mode can produce a non-inheriting DACL that a
    # later sandboxed process cannot traverse.
    parent.mkdir(parents=True, exist_ok=True)

    if run_id is not None:
        selected_id = _validate_segment(run_id, "run_id")
        path = parent / selected_id
        try:
            path.mkdir()
        except FileExistsError as error:
            raise RunPathError(f"run workspace already exists: {path}") from error
    else:
        instant = now or datetime.now(timezone.utc)
        if instant.tzinfo is None:
            raise RunPathError("run allocation time must include a timezone")
        stem = instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = parent / stem
        selected_id = stem
        for attempt in range(_AUTO_ALLOCATION_ATTEMPTS):
            selected_id = stem if attempt == 0 else f"{stem}-{attempt:02d}"
            path = parent / selected_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            break
        else:
            raise RunPathError(f"could not allocate a unique run workspace beneath {parent}")

    created_at = _utc_now()
    try:
        temp_dir = path / "temp"
        logs_dir = path / "logs"
        reports_dir = path / "reports"
        for child in (temp_dir, logs_dir, reports_dir):
            child.mkdir()
        workspace = RunWorkspace(
            repository=root,
            kind=validated_kind,
            run_id=selected_id,
            path=path,
            temp_dir=temp_dir,
            logs_dir=logs_dir,
            reports_dir=reports_dir,
            created_at=created_at,
        )
        _write_context(workspace, status="active", returncode=None)
        return workspace
    except BaseException:
        if path.exists():
            try:
                _remove_tree(path)
            except OSError:
                pass
        raise


def list_run_workspaces(
    repository: str | os.PathLike[str],
) -> list[RunWorkspaceInfo]:
    """Inspect exact two-level run leaves without following links."""

    repository_path = Path(repository).expanduser().resolve()
    runs_root = repository_path / ".scratch" / "runs"
    if not runs_root.is_dir():
        return []
    records: list[RunWorkspaceInfo] = []
    for kind_path in sorted(runs_root.iterdir(), key=lambda candidate: candidate.name):
        if kind_path.is_symlink() or not kind_path.is_dir():
            continue
        if _SEGMENT.fullmatch(kind_path.name) is None:
            continue
        for run_path in sorted(kind_path.iterdir(), key=lambda candidate: candidate.name):
            if run_path.is_symlink() or not run_path.is_dir():
                continue
            if _SEGMENT.fullmatch(run_path.name) is None:
                continue
            try:
                records.append(_read_workspace_info(kind_path.name, run_path))
            except FileNotFoundError:
                # A controller may finish and remove its leaf during inspection.
                continue
    return records


def stale_run_workspaces(
    repository: str | os.PathLike[str],
    *,
    older_than: timedelta,
    now: datetime | None = None,
) -> list[RunWorkspaceInfo]:
    """Return old workspaces, excluding a controller that is still alive."""

    if older_than.total_seconds() <= 0:
        raise RunPathError("stale-run age must be positive")
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise RunPathError("stale-run comparison time must include a timezone")
    cutoff = instant.astimezone(timezone.utc) - older_than
    return [
        record
        for record in list_run_workspaces(repository)
        if record.updated_at <= cutoff
        and not (record.status == "active" and record.controller_running is not False)
    ]


def clean_stale_run_workspaces(
    repository: str | os.PathLike[str],
    *,
    older_than: timedelta,
    apply: bool = False,
    now: datetime | None = None,
) -> list[RunWorkspaceInfo]:
    """Preview or remove stale exact run leaves.

    ``apply=False`` is intentionally the default; callers must opt into deletion.
    """

    candidates = stale_run_workspaces(repository, older_than=older_than, now=now)
    if not apply:
        return candidates
    repository_path = Path(repository).expanduser().resolve()
    runs_root = repository_path / ".scratch" / "runs"
    for candidate in candidates:
        if candidate.path.parent.parent != runs_root:
            raise RunPathError("refusing to clean a path outside .scratch/runs")
        try:
            _remove_tree(candidate.path)
        except FileNotFoundError:
            # Another cleanup process already handled the exact same candidate.
            continue
    return candidates


def _write_context(
    workspace: RunWorkspace,
    *,
    status: str,
    returncode: int | None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": RUN_CONTEXT_SCHEMA_VERSION,
        "kind": workspace.kind,
        "run_id": workspace.run_id,
        "status": status,
        "controller_pid": os.getpid(),
        "repository": str(workspace.repository),
        "work_dir": str(workspace.path),
        "created_at": workspace.created_at,
        "updated_at": _utc_now(),
    }
    if returncode is not None:
        payload["returncode"] = returncode
    temporary = workspace.path / f".{RUN_CONTEXT_FILENAME}.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, workspace.context_path)


def _remove_tree(path: Path) -> None:
    """Remove an exact run leaf, retrying Windows read-only children once."""

    def make_writable_and_retry(function, name, exception_info) -> None:
        error = exception_info[1]
        if not isinstance(error, PermissionError):
            raise error
        # Git object stores and a few model tools deliberately create read-only
        # files.  Clear only that DOS/POSIX mode bit; do not rewrite the DACL.
        os.chmod(name, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
        function(name)

    # ``onerror`` retains Python 3.10/3.11 support.  The callback is scoped to
    # the already-validated run leaf and re-raises anything except PermissionError.
    shutil.rmtree(path, onerror=make_writable_and_retry)


def _read_workspace_info(kind: str, path: Path) -> RunWorkspaceInfo:
    stat = path.stat()
    fallback = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    status = "unrecognized"
    controller_pid: int | None = None
    created_at = fallback
    updated_at = fallback
    context = path / RUN_CONTEXT_FILENAME
    try:
        if context.is_symlink():
            raise RunPathError("run context must not be a symbolic link")
        if context.stat().st_size > 64 * 1024:
            raise RunPathError("run context is unexpectedly large")
        payload = json.loads(context.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RunPathError("run context is not an object")
        raw_status = payload.get("status")
        if isinstance(raw_status, str) and raw_status:
            status = raw_status
        raw_pid = payload.get("controller_pid")
        if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0:
            controller_pid = raw_pid
        created_at = _parse_utc(payload.get("created_at"), fallback)
        updated_at = _parse_utc(payload.get("updated_at"), fallback)
    except (OSError, UnicodeError, json.JSONDecodeError, RunPathError):
        status = "unrecognized"
    running = _pid_is_running(controller_pid) if controller_pid is not None else None
    return RunWorkspaceInfo(
        kind=kind,
        run_id=path.name,
        path=path,
        status=status,
        controller_pid=controller_pid,
        controller_running=running,
        created_at=created_at,
        updated_at=updated_at,
    )


def _pid_is_running(pid: int) -> bool | None:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_pid_is_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


def _windows_pid_is_running(pid: int) -> bool | None:
    # os.kill(pid, 0) is not a harmless existence probe on Windows: Python maps
    # non-console signals to TerminateProcess.  Query the process handle instead.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    process_query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        if error == 5:  # Access denied still proves that a process owns the PID.
            return True
        if error == 87:  # Invalid parameter: no such process.
            return False
        return None
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return None
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _validate_segment(value: str, label: str) -> str:
    normalized = str(value)
    if (
        normalized != normalized.strip()
        or _SEGMENT.fullmatch(normalized) is None
        or normalized in {".", ".."}
    ):
        raise RunPathError(f"{label} contains unsupported characters")
    return normalized


def _parse_utc(value: object, fallback: datetime) -> datetime:
    if not isinstance(value, str):
        return fallback
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        return fallback
    return parsed.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
