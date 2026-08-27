from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from abliteralus.run_paths import (
    RunPathError,
    allocate_run_workspace,
    clean_stale_run_workspaces,
    list_run_workspaces,
    stale_run_workspaces,
)


def test_workspace_is_atomic_scoped_and_self_describing(tmp_path: Path) -> None:
    workspace = allocate_run_workspace(
        tmp_path,
        "local-surgery",
        run_id="experiment-001",
    )

    assert workspace.path == tmp_path / ".scratch/runs/local-surgery/experiment-001"
    assert workspace.temp_dir.is_dir()
    assert workspace.logs_dir.is_dir()
    assert workspace.reports_dir.is_dir()
    context = json.loads(workspace.context_path.read_text(encoding="utf-8"))
    assert context["status"] == "active"
    assert context["controller_pid"] == os.getpid()
    assert "environment" not in context
    assert "command" not in context


def test_automatic_ids_suffix_same_second_collisions(tmp_path: Path) -> None:
    instant = datetime(2026, 8, 27, 3, 0, tzinfo=timezone.utc)

    first = allocate_run_workspace(tmp_path, "pytest", now=instant)
    second = allocate_run_workspace(tmp_path, "pytest", now=instant)

    assert first.run_id == "20260827T030000Z"
    assert second.run_id == "20260827T030000Z-01"


def test_explicit_id_collision_and_unsafe_segments_fail(tmp_path: Path) -> None:
    allocate_run_workspace(tmp_path, "pytest", run_id="fixed")

    with pytest.raises(RunPathError, match="already exists"):
        allocate_run_workspace(tmp_path, "pytest", run_id="fixed")
    with pytest.raises(RunPathError, match="unsupported characters"):
        allocate_run_workspace(tmp_path, "../escape")
    with pytest.raises(RunPathError, match="unsupported characters"):
        allocate_run_workspace(tmp_path, "pytest", run_id="../escape")
    with pytest.raises(RunPathError, match="unsupported characters"):
        allocate_run_workspace(tmp_path, "pytest", run_id=" padded ")


def test_child_environment_is_ephemeral_and_coverage_is_scoped(tmp_path: Path) -> None:
    workspace = allocate_run_workspace(tmp_path, "pytest")
    parent = {"PATH": "portable"}

    environment = workspace.child_environment(parent, coverage=True)

    assert parent == {"PATH": "portable"}
    assert environment["TEMP"] == str(workspace.temp_dir)
    assert environment["TMP"] == str(workspace.temp_dir)
    assert environment["TMPDIR"] == str(workspace.temp_dir)
    assert environment["ABLITERALUS_RUN_ID"] == workspace.run_id
    assert environment["ABLITERALUS_WORK_DIR"] == str(workspace.path)
    assert Path(environment["COVERAGE_FILE"]).parent == workspace.path / "coverage"


def test_result_is_visible_when_kept_and_cleanup_removes_only_leaf(tmp_path: Path) -> None:
    workspace = allocate_run_workspace(tmp_path, "local-inference", run_id="chat-001")
    sibling = workspace.path.parent / "keep-me"
    sibling.mkdir()

    workspace.record_result(3)
    records = list_run_workspaces(tmp_path)

    assert [(record.run_id, record.status) for record in records] == [
        ("chat-001", "failed"),
        ("keep-me", "unrecognized"),
    ]
    workspace.cleanup()
    assert not workspace.path.exists()
    assert sibling.is_dir()


def test_cleanup_handles_read_only_children_without_rewriting_acl(tmp_path: Path) -> None:
    workspace = allocate_run_workspace(tmp_path, "pytest", run_id="read-only")
    protected = workspace.temp_dir / "git-object"
    protected.write_bytes(b"object")
    protected.chmod(stat.S_IREAD)

    workspace.cleanup()

    assert not workspace.path.exists()


def test_stale_cleanup_is_preview_first_and_skips_live_active_run(tmp_path: Path) -> None:
    instant = datetime.now(timezone.utc)
    workspace = allocate_run_workspace(tmp_path, "pytest", run_id="old-run")
    future = instant + timedelta(days=2)

    assert (
        stale_run_workspaces(
            tmp_path,
            older_than=timedelta(days=1),
            now=future,
        )
        == []
    )

    workspace.record_result(0)
    preview = clean_stale_run_workspaces(
        tmp_path,
        older_than=timedelta(days=1),
        now=future,
    )
    assert [record.path for record in preview] == [workspace.path]
    assert workspace.path.is_dir()

    removed = clean_stale_run_workspaces(
        tmp_path,
        older_than=timedelta(days=1),
        now=future,
        apply=True,
    )
    assert [record.path for record in removed] == [workspace.path]
    assert not workspace.path.exists()


def test_stale_cleanup_rejects_nonpositive_age(tmp_path: Path) -> None:
    with pytest.raises(RunPathError, match="must be positive"):
        stale_run_workspaces(tmp_path, older_than=timedelta())
