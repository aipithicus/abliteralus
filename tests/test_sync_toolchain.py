"""Regression tests for repository-owned sync tooling."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from private.sync import shadow_sync, writeback


pytestmark = pytest.mark.cpu


def test_shadow_export_invokes_only_the_supplied_uv(monkeypatch, tmp_path):
    exact_uv = tmp_path / "repository-tools" / "uv.exe"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout="package-a==1.2.3\ntorch==2.9.0\nproject-under-test==0.1.0\n",
            stderr="",
        )

    monkeypatch.setattr(shadow_sync.subprocess, "run", run)
    torch_versions: list[str] = []

    constraints = shadow_sync.constraints_from_fork(
        tmp_path,
        "project-under-test",
        torch_versions,
        exact_uv,
    )

    assert constraints == ["package-a==1.2.3", "pillow>=12.2.0", "torch==2.9.0"]
    assert torch_versions == ["2.9.0"]
    assert calls[0][0][0] == str(exact_uv)
    assert Path(calls[0][1]["env"]["TEMP"]).is_relative_to(shadow_sync.LAB_ROOT / ".scratch")


def test_writeback_verification_invokes_only_the_supplied_uv(monkeypatch, tmp_path):
    exact_uv = tmp_path / "repository-tools" / "uv.exe"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, stdout="1 passed\n", stderr="")

    monkeypatch.setattr(writeback.subprocess, "run", run)

    assert writeback.verify(tmp_path, ["tests/test_example.py"], exact_uv) is True
    assert len(calls) == 2
    assert all(call[0][0] == str(exact_uv) for call in calls)
    assert all(
        Path(call[1]["env"]["TEMP"]).is_relative_to(writeback.LAB_ROOT / ".scratch")
        for call in calls
    )


def test_sync_tools_have_no_ambient_uv_or_global_temp_fallbacks():
    for module in (shadow_sync, writeback):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "shutil.which" not in source
        assert '["uv"' not in source
        assert "tempfile." not in source


def test_conflicted_sync_worktree_is_retained_inside_repository(monkeypatch, tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()

    monkeypatch.setattr(shadow_sync, "git", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        shadow_sync.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    merged, checkout = shadow_sync.merge_via_worktree(lab, "fixes", "shadow")

    assert merged is False
    assert checkout is not None
    assert checkout.is_relative_to(lab / ".scratch" / "runs" / "sync-worktree")
    assert (checkout.parent / "run-context.json").is_file()


def test_failed_worktree_creation_cleans_its_run_workspace(monkeypatch, tmp_path):
    lab = tmp_path / "lab"
    lab.mkdir()
    calls: list[list[str]] = []

    def git(arguments, *_args, **_kwargs):
        calls.append(arguments)
        if arguments[:2] == ["worktree", "add"]:
            raise RuntimeError("worktree add failed")
        return ""

    monkeypatch.setattr(shadow_sync, "git", git)

    with pytest.raises(RuntimeError, match="worktree add failed"):
        shadow_sync.merge_via_worktree(lab, "fixes", "shadow")

    workspace_root = lab / ".scratch" / "runs" / "sync-worktree"
    assert workspace_root.is_dir()
    assert list(workspace_root.iterdir()) == []
    assert any(arguments[:2] == ["worktree", "remove"] for arguments in calls)
