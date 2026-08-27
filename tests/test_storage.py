from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from abliteralus.run_paths import allocate_run_workspace
from abliteralus.storage import (
    StorageError,
    measure_storage,
    read_storage_history,
    reclaim_storage,
    record_storage_snapshot,
)


def _category_map(report) -> dict[str, object]:
    return {category.name: category for category in report.categories}


def test_report_partitions_managed_storage_without_double_counting(tmp_path: Path) -> None:
    files = {
        ".scratch/cache/huggingface/model.bin": b"h" * 10,
        ".scratch/uv-cache/archive.bin": b"u" * 20,
        "outputs/study/result.bin": b"o" * 30,
        ".venv/runtime.bin": b"v" * 40,
        "deps/uv/uv.exe": b"d" * 50,
        ".coverage": b"c" * 6,
        "private/.pytest_cache/state": b"p" * 8,
        "source.dat": b"s" * 7,
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    report = measure_storage(tmp_path)
    categories = _category_map(report)

    assert categories["huggingface-cache"].bytes == 10
    assert categories["uv-cache"].bytes == 20
    assert categories["outputs"].bytes == 30
    assert categories["python-environment"].bytes == 40
    assert categories["portable-dependencies"].bytes == 50
    assert categories["tool-state"].bytes == 14
    assert categories["repository-other"].bytes == 7
    assert report.total_bytes == sum(len(content) for content in files.values())
    assert report.volume_total_bytes >= report.volume_free_bytes >= 0
    assert categories["outputs"].reclaimable is False
    assert categories["huggingface-cache"].reclaimable is True
    assert all(not root.startswith(str(tmp_path)) for root in categories["outputs"].roots)


def test_snapshot_history_tracks_total_and_category_deltas(tmp_path: Path) -> None:
    first_output = tmp_path / "outputs/first.bin"
    first_output.parent.mkdir(parents=True)
    first_output.write_bytes(b"a" * 10)
    first = measure_storage(
        tmp_path,
        now=datetime(2026, 8, 27, 4, 0, tzinfo=timezone.utc),
    )
    first_path = record_storage_snapshot(tmp_path, first)

    (tmp_path / "outputs/second.bin").write_bytes(b"b" * 20)
    second = measure_storage(
        tmp_path,
        now=datetime(2026, 8, 27, 4, 1, tzinfo=timezone.utc),
    )
    second_path = record_storage_snapshot(tmp_path, second)
    history = read_storage_history(tmp_path, limit=2)

    assert first_path != second_path
    assert len(history) == 2
    assert history[0]["delta_bytes"] is None
    assert history[1]["delta_bytes"] > 0
    assert history[1]["volume_free_bytes"] >= 0
    assert history[1]["category_delta_bytes"]["outputs"] == 20
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert "command" not in payload
    assert "environment" not in payload


def test_reclaim_is_preview_first_and_never_selects_outputs(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    future = now + timedelta(days=2)
    cache = tmp_path / ".scratch/cache/huggingface/model.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    coverage = tmp_path / ".coverage"
    coverage.write_bytes(b"coverage")
    nested_tool_state = tmp_path / "private/.ruff_cache/state"
    nested_tool_state.parent.mkdir(parents=True)
    nested_tool_state.write_bytes(b"tool")
    output = tmp_path / "outputs/study/result.bin"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"durable")
    workspace = allocate_run_workspace(tmp_path, "pytest", run_id="finished")
    workspace.record_result(0)

    selected = ("runs", "huggingface-cache", "tool-state")
    preview = reclaim_storage(
        tmp_path,
        categories=selected,
        older_than=timedelta(days=1),
        now=future,
    )
    preview_paths = {candidate.path for candidate in preview}

    assert workspace.path in preview_paths
    assert cache.parent in preview_paths
    assert coverage in preview_paths
    assert nested_tool_state.parent in preview_paths
    assert output not in preview_paths
    assert all(path.exists() for path in preview_paths)

    applied = reclaim_storage(
        tmp_path,
        categories=selected,
        older_than=timedelta(days=1),
        now=future,
        apply=True,
    )
    assert {candidate.path for candidate in applied} == preview_paths
    assert all(not path.exists() for path in preview_paths)
    assert output.read_bytes() == b"durable"


def test_cache_apply_is_blocked_while_managed_run_is_active(tmp_path: Path) -> None:
    cache = tmp_path / ".scratch/cache/huggingface/model.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    workspace = allocate_run_workspace(tmp_path, "local-surgery", run_id="active")
    future = datetime.now(timezone.utc) + timedelta(days=2)

    preview = reclaim_storage(
        tmp_path,
        categories=("huggingface-cache",),
        older_than=timedelta(days=1),
        now=future,
    )
    assert [candidate.path for candidate in preview] == [cache.parent]

    with pytest.raises(StorageError, match="managed run is active"):
        reclaim_storage(
            tmp_path,
            categories=("huggingface-cache",),
            older_than=timedelta(days=1),
            now=future,
            apply=True,
        )
    assert cache.is_file()
    workspace.cleanup()


def test_recent_cache_is_not_selected(tmp_path: Path) -> None:
    cache = tmp_path / ".scratch/cache/huggingface/model.bin"
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")

    candidates = reclaim_storage(
        tmp_path,
        categories=("huggingface-cache",),
        older_than=timedelta(days=30),
    )

    assert candidates == []


def test_reclaim_requires_an_explicit_supported_category(tmp_path: Path) -> None:
    with pytest.raises(StorageError, match="at least one"):
        reclaim_storage(tmp_path, categories=(), older_than=timedelta(days=1))
    with pytest.raises(StorageError, match="unsupported"):
        reclaim_storage(
            tmp_path,
            categories=("outputs",),
            older_than=timedelta(days=1),
        )


def test_history_rejects_corrupt_snapshots(tmp_path: Path) -> None:
    history = tmp_path / ".scratch/storage/snapshots"
    history.mkdir(parents=True)
    (history / "broken.json").write_text("not json", encoding="utf-8")

    with pytest.raises(StorageError, match="unreadable"):
        read_storage_history(tmp_path)
