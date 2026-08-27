"""Contract tests for the non-installing uv command runner."""

from __future__ import annotations

import subprocess

import pytest

from deps.uv import run_uv


pytestmark = pytest.mark.cpu


def test_runner_uses_verified_executable_and_local_environment(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    working_directory = repository / "private"
    working_directory.mkdir(parents=True)
    executable = repository / "deps/uv/uv.exe"
    environment = {
        "TEMP": str(repository / ".scratch/temp"),
        "UV_CACHE_DIR": str(repository / ".scratch/cache/uv"),
        "UV_PYTHON": str(repository / "deps/python/python.exe"),
    }
    observed = {}

    monkeypatch.setattr(run_uv, "REPOSITORY", repository)
    monkeypatch.setattr(run_uv, "resolve_uv", lambda selected: executable)
    monkeypatch.setattr(
        run_uv,
        "repository_tool_environment",
        lambda selected: environment,
    )
    monkeypatch.chdir(working_directory)

    def run(arguments, **kwargs):
        observed.update(arguments=arguments, **kwargs)
        return subprocess.CompletedProcess(arguments, 23)

    monkeypatch.setattr(run_uv.subprocess, "run", run)

    assert run_uv.main(["lock", "--check"]) == 23
    assert observed["arguments"] == [str(executable), "lock", "--check"]
    assert observed["cwd"] == working_directory
    assert observed["env"] is environment
    assert observed["check"] is False


def test_runner_rejects_working_directory_outside_repository(monkeypatch, tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(run_uv, "REPOSITORY", repository)
    monkeypatch.chdir(outside)

    assert run_uv.main(["--version"]) == 2
