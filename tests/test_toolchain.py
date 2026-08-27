"""Tests for the repository-owned uv toolchain contract."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import abliteralus.toolchain as toolchain
from abliteralus.toolchain import (
    UvToolchainError,
    load_uv_pin,
    repository_tool_environment,
    resolve_uv,
    restore_uv,
)


pytestmark = pytest.mark.cpu


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _archive(payload: bytes) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("uv.exe", payload)
    return target.getvalue()


def _repository(tmp_path: Path, *, executable: bytes = b"fake-uv") -> Path:
    repo = tmp_path / "repo"
    (repo / "deps/uv").mkdir(parents=True)
    portable_python = repo / toolchain.WINDOWS_PYTHON_PATH
    portable_python.parent.mkdir(parents=True)
    portable_python.write_bytes(b"fake-python")
    archive = _archive(executable)
    pin = {
        "schema_version": 1,
        "tool": "uv",
        "version": "9.8.7",
        "release_url": "https://github.com/astral-sh/uv/releases/tag/9.8.7",
        "cache_path": ".scratch/cache/uv",
        "platforms": {
            "windows-x86_64": {
                "archive_url": ("https://github.com/astral-sh/uv/releases/download/9.8.7/uv.zip"),
                "archive_sha256": _sha256(archive),
                "archive_size": len(archive),
                "primary": "deps/uv/uv.exe",
                "files": [
                    {
                        "archive_path": "uv.exe",
                        "install_path": "deps/uv/uv.exe",
                        "sha256": _sha256(executable),
                        "size": len(executable),
                        "executable": True,
                    }
                ],
            }
        },
    }
    (repo / "deps/uv/pin.json").write_text(json.dumps(pin), encoding="utf-8")
    return repo


def _successful_version(command, **kwargs):
    return subprocess.CompletedProcess(command, 0, stdout="uv 9.8.7 (test)\n", stderr="")


def test_load_uv_pin_is_strict_and_repository_relative(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    pin = load_uv_pin(repo)

    assert pin.version == "9.8.7"
    assert pin.cache_path == Path(".scratch/cache/uv")
    assert pin.platform("windows-x86_64").primary == Path("deps/uv/uv.exe")

    payload = json.loads((repo / "deps/uv/pin.json").read_text(encoding="utf-8"))
    payload["surprise"] = True
    (repo / "deps/uv/pin.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(UvToolchainError, match="unknown: surprise"):
        load_uv_pin(repo)


def test_load_uv_pin_rejects_mutable_state_outside_canonical_cache(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    pin_path = repo / "deps/uv/pin.json"
    payload = json.loads(pin_path.read_text(encoding="utf-8"))
    payload["cache_path"] = "deps/uv/cache"
    pin_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UvToolchainError, match="canonical repository cache"):
        load_uv_pin(repo)


def test_tool_environment_rejects_redirected_scratch_state(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    redirected = repo / "redirected-state"
    redirected.mkdir()
    try:
        (repo / ".scratch").symlink_to(redirected, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    with pytest.raises(UvToolchainError, match="redirected from its declared path"):
        repository_tool_environment(repo)


def test_resolve_uv_never_falls_back_to_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    ambient = tmp_path / "ambient"
    ambient.mkdir()
    (ambient / "uv.exe").write_bytes(b"ambient")
    monkeypatch.setenv("PATH", str(ambient))

    with pytest.raises(UvToolchainError, match="payload is missing"):
        resolve_uv(repo, platform_key="windows-x86_64")


@pytest.mark.skipif(os.name != "nt", reason="Windows portable-Python contract")
def test_tool_environment_never_falls_back_to_ambient_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    (repo / toolchain.WINDOWS_PYTHON_PATH).unlink()
    monkeypatch.setenv("PATH", str(tmp_path / "ambient-python"))

    with pytest.raises(UvToolchainError, match="repository Python is missing"):
        repository_tool_environment(repo)


def test_resolve_uv_verifies_hash_version_and_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    executable = repo / "deps/uv/uv.exe"
    executable.write_bytes(b"fake-uv")
    monkeypatch.setattr(toolchain.subprocess, "run", _successful_version)

    assert resolve_uv(repo, platform_key="windows-x86_64") == executable
    environment = repository_tool_environment(
        repo,
        environ={
            "APPDATA": "ambient-appdata",
            "LOCALAPPDATA": "ambient-localappdata",
            "PATH": "ambient",
            "PIP_INDEX_URL": "https://ambient.invalid/simple",
            "PROGRAMDATA": "ambient-programdata",
            "PYTHONPATH": "ambient-python",
            "UV_INDEX_URL": "https://ambient.invalid/simple",
            "UV_NO_CONFIG": "1",
            "VIRTUAL_ENV": "ambient-venv",
            "XDG_CONFIG_DIRS": "ambient-system-config",
            "XDG_CONFIG_HOME": "ambient-user-config",
        },
    )
    assert environment["UV_CACHE_DIR"] == str(repo / ".scratch/cache/uv")
    assert "UV_NO_CONFIG" not in environment
    assert environment["TEMP"] == str(repo / ".scratch/temp")
    assert environment["PATH"] == "ambient"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONUTF8"] == "1"
    for name in (
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "XDG_BIN_HOME",
        "XDG_CONFIG_DIRS",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        assert Path(environment[name]).is_relative_to(repo / ".scratch")
    system_configuration = Path(environment["PROGRAMDATA"]) / "uv/uv.toml"
    assert system_configuration.read_text(encoding="utf-8") == ""
    for name in ("PIP_INDEX_URL", "PYTHONPATH", "UV_INDEX_URL", "VIRTUAL_ENV"):
        assert name not in environment
    if os.name == "nt":
        assert environment["UV_PYTHON"] == str(repo / toolchain.WINDOWS_PYTHON_PATH)
    else:
        assert environment["UV_PYTHON"] == str(Path(sys.executable).resolve())

    executable.write_bytes(b"tampered")
    with pytest.raises(UvToolchainError, match="SHA-256"):
        resolve_uv(repo, platform_key="windows-x86_64")


def test_restore_uv_verifies_archive_and_deploys_exact_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"fake-uv"
    repo = _repository(tmp_path, executable=payload)
    archive = _archive(payload)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    observed: list[str] = []

    def opener(url: str, **_kwargs):
        observed.append(url)
        return Response(archive)

    monkeypatch.setattr(toolchain.subprocess, "run", _successful_version)
    executable = restore_uv(
        repo,
        platform_key="windows-x86_64",
        opener=opener,
    )

    assert executable.read_bytes() == payload
    assert observed == ["https://github.com/astral-sh/uv/releases/download/9.8.7/uv.zip"]
    restore_workspace = repo / ".scratch/toolchain/uv/9.8.7/windows-x86_64"
    assert not list(restore_workspace.glob("staging-*"))


def test_restore_uv_rejects_corrupt_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    with pytest.raises(UvToolchainError, match="does not match"):
        restore_uv(
            repo,
            platform_key="windows-x86_64",
            opener=lambda *_args, **_kwargs: Response(b"wrong"),
        )
    assert not list((repo / "deps/uv").glob("*.exe"))


def test_restore_uv_stops_oversized_download_before_writing_the_remainder(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    pin = load_uv_pin(repo).platform("windows-x86_64")
    oversized = b"x" * (pin.archive_size + 1)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    with pytest.raises(UvToolchainError, match="exceeded its declared size"):
        restore_uv(
            repo,
            platform_key="windows-x86_64",
            opener=lambda *_args, **_kwargs: Response(oversized),
        )
