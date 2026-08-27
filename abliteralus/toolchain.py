"""Repository-owned external toolchain contracts.

This module is intentionally standard-library only. It resolves and restores the
uv executable declared by ``deps/uv/pin.json`` without consulting ``PATH`` or a
user-global cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse


UV_PIN_PATH = Path("deps/uv/pin.json")
UV_CACHE_PATH = Path(".scratch/cache/uv")
WINDOWS_PYTHON_PATH = Path("deps/python/cpython-3.12-windows-x86_64-none/python.exe")
_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)")
_DOWNLOAD_CHUNK = 1024 * 1024
_AMBIENT_ENVIRONMENT_PREFIXES = ("UV_", "PYTHON")
_AMBIENT_ENVIRONMENT_NAMES = {
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "PIP_CONFIG_FILE",
    "PIP_EXTRA_INDEX_URL",
    "PIP_INDEX_URL",
    "PIP_REQUIRE_VIRTUALENV",
    "PIP_TRUSTED_HOST",
    "PIP_USER",
    "VIRTUAL_ENV",
}


class UvToolchainError(RuntimeError):
    """The repository uv declaration or deployed payload is invalid."""


@dataclass(frozen=True, slots=True)
class UvPinnedFile:
    archive_path: str
    install_path: Path
    sha256: str
    size: int
    executable: bool


@dataclass(frozen=True, slots=True)
class UvPlatform:
    key: str
    archive_url: str
    archive_sha256: str
    archive_size: int
    primary: Path
    files: tuple[UvPinnedFile, ...]

    def primary_file(self) -> UvPinnedFile:
        for pinned in self.files:
            if pinned.install_path == self.primary:
                return pinned
        raise UvToolchainError(f"{self.key}.primary does not name a declared file")


@dataclass(frozen=True, slots=True)
class UvPin:
    path: Path
    version: str
    release_url: str
    cache_path: Path
    platforms: Mapping[str, UvPlatform]

    def platform(self, key: str) -> UvPlatform:
        try:
            return self.platforms[key]
        except KeyError as error:
            available = ", ".join(sorted(self.platforms))
            raise UvToolchainError(
                f"uv platform {key!r} is not pinned; available: {available}"
            ) from error


def load_uv_pin(repository: str | os.PathLike[str]) -> UvPin:
    """Load and strictly validate the tracked uv manifest."""

    root = _repository(repository)
    pin_path = _inside(root, UV_PIN_PATH, label="uv pin")
    try:
        raw = json.loads(pin_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise UvToolchainError(f"tracked uv pin is missing: {pin_path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UvToolchainError(f"cannot read uv pin {pin_path}: {error}") from error

    table = _object(raw, "uv pin")
    _keys(
        table,
        {
            "schema_version",
            "tool",
            "version",
            "release_url",
            "cache_path",
            "platforms",
        },
        "uv pin",
    )
    if table["schema_version"] != _SCHEMA_VERSION:
        raise UvToolchainError(f"unsupported uv pin schema: {table['schema_version']!r}")
    if table["tool"] != "uv":
        raise UvToolchainError("uv pin tool must be 'uv'")
    version = _string(table["version"], "version")
    if _VERSION.fullmatch(version) is None:
        raise UvToolchainError(f"invalid uv version: {version!r}")
    release_url = _https_url(table["release_url"], "release_url")
    expected_release = f"https://github.com/astral-sh/uv/releases/tag/{version}"
    if release_url != expected_release:
        raise UvToolchainError(f"release_url must match the pinned uv version: {expected_release}")
    cache_path = _relative_path(table["cache_path"], "cache_path")
    if cache_path != UV_CACHE_PATH:
        raise UvToolchainError(
            f"cache_path must be the canonical repository cache: {UV_CACHE_PATH}"
        )
    _inside(root, cache_path, label="uv cache")

    platform_table = _object(table["platforms"], "platforms")
    if not platform_table:
        raise UvToolchainError("uv pin must declare at least one platform")
    parsed_platforms: dict[str, UvPlatform] = {}
    for key, value in platform_table.items():
        if not isinstance(key, str) or not key:
            raise UvToolchainError("uv platform keys must be non-empty strings")
        parsed_platforms[key] = _parse_platform(root, version, key, value)

    return UvPin(
        path=pin_path,
        version=version,
        release_url=release_url,
        cache_path=cache_path,
        platforms=parsed_platforms,
    )


def current_uv_platform() -> str:
    """Return the manifest key for the current supported host."""

    machine = platform.machine().lower().replace("amd64", "x86_64")
    if machine != "x86_64":
        raise UvToolchainError(f"unsupported uv host architecture: {platform.machine()!r}")
    if os.name == "nt":
        return "windows-x86_64"
    if sys.platform.startswith("linux"):
        return "linux-x86_64-gnu"
    raise UvToolchainError(f"unsupported uv host platform: {sys.platform!r}")


def repository_tool_environment(
    repository: str | os.PathLike[str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a child environment whose mutable tool state stays in the repository."""

    root = _repository(repository)
    pin = load_uv_pin(root)
    temporary = _inside(root, Path(".scratch/temp"), label="tool temp")
    uv_cache = _inside(root, pin.cache_path, label="uv cache")
    xdg_cache = _inside(root, Path(".scratch/cache/xdg"), label="XDG cache")
    user_configuration = _inside(
        root,
        Path(".scratch/configuration/user"),
        label="user tool configuration",
    )
    system_configuration = _inside(
        root,
        Path(".scratch/configuration/system"),
        label="system tool configuration",
    )
    local_data = _inside(root, Path(".scratch/data/local"), label="local tool data")
    roaming_data = _inside(root, Path(".scratch/data/roaming"), label="roaming tool data")
    xdg_data = _inside(root, Path(".scratch/data/xdg"), label="XDG data")
    xdg_bin = _inside(root, Path(".scratch/bin"), label="XDG executable directory")
    bytecode = _inside(
        root,
        Path(".scratch/cache/python-bytecode"),
        label="Python bytecode cache",
    )
    for directory in (
        temporary,
        uv_cache,
        xdg_cache,
        user_configuration,
        system_configuration,
        local_data,
        roaming_data,
        xdg_data,
        xdg_bin,
        bytecode,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    system_uv = _inside(
        root,
        Path(".scratch/configuration/system/uv"),
        label="system uv configuration",
    )
    system_uv.mkdir(exist_ok=True)
    system_uv_file = system_uv / "uv.toml"
    if system_uv_file.is_symlink():
        raise UvToolchainError(f"system uv configuration must not be a symlink: {system_uv_file}")
    system_uv_file.write_text("", encoding="utf-8")
    python = _uv_python(root)
    child = dict(os.environ if environ is None else environ)
    for name in tuple(child):
        if name.startswith(_AMBIENT_ENVIRONMENT_PREFIXES) or name in _AMBIENT_ENVIRONMENT_NAMES:
            child.pop(name)
    child.update(
        {
            "TEMP": str(temporary),
            "TMP": str(temporary),
            "TMPDIR": str(temporary),
            "APPDATA": str(roaming_data),
            "LOCALAPPDATA": str(local_data),
            "PROGRAMDATA": str(system_configuration),
            "UV_CACHE_DIR": str(uv_cache),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_PYTHON": str(python),
            "XDG_CACHE_HOME": str(xdg_cache),
            "XDG_BIN_HOME": str(xdg_bin),
            "XDG_CONFIG_DIRS": str(system_configuration),
            "XDG_CONFIG_HOME": str(user_configuration),
            "XDG_DATA_HOME": str(xdg_data),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": str(bytecode),
            "PYTHONUTF8": "1",
        }
    )
    return child


def resolve_uv(
    repository: str | os.PathLike[str],
    *,
    platform_key: str | None = None,
) -> Path:
    """Return the exact verified repository uv executable; never search ``PATH``."""

    root = _repository(repository)
    pin = load_uv_pin(root)
    selected = pin.platform(platform_key or current_uv_platform())
    for pinned in selected.files:
        _verify_deployed_file(root, pinned)
    executable = _inside(root, selected.primary, label="uv executable")
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            cwd=root,
            env=repository_tool_environment(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise UvToolchainError(f"cannot execute repository uv: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise UvToolchainError(f"repository uv version probe failed: {detail}")
    if re.match(rf"^uv {re.escape(pin.version)}(?:\s|$)", result.stdout.strip()) is None:
        raise UvToolchainError(
            f"repository uv reports {result.stdout.strip()!r}; expected uv {pin.version}"
        )
    return executable


def restore_uv(
    repository: str | os.PathLike[str],
    *,
    force: bool = False,
    platform_key: str | None = None,
    opener: Callable[..., Any] | None = None,
) -> Path:
    """Download, verify, and deploy the pinned uv payload beneath ``deps/uv``."""

    root = _repository(repository)
    pin = load_uv_pin(root)
    key = platform_key or current_uv_platform()
    selected = pin.platform(key)
    if not force:
        try:
            return resolve_uv(root, platform_key=key)
        except UvToolchainError:
            pass

    scratch = _inside(
        root,
        Path(".scratch/toolchain/uv") / pin.version / key,
        label="uv restore workspace",
    )
    scratch.mkdir(parents=True, exist_ok=True)
    archive_name = Path(urlparse(selected.archive_url).path).name
    if not archive_name:
        raise UvToolchainError("uv archive URL has no filename")
    archive = _inside(scratch, Path(archive_name), label="uv archive")
    if not _matches_file(archive, selected.archive_size, selected.archive_sha256):
        _download(
            selected.archive_url,
            archive,
            selected.archive_size,
            selected.archive_sha256,
            opener=opener,
        )

    stage = _inside(
        scratch,
        Path(f"staging-{os.getpid()}"),
        label="uv staging directory",
    )
    _remove_scoped_directory(stage, scratch)
    stage.mkdir()
    try:
        _extract_pinned_files(archive, selected, stage)
        for pinned in selected.files:
            staged = _inside(stage, pinned.install_path, label="staged uv executable")
            destination = _inside(root, pinned.install_path, label="uv executable")
            destination.parent.mkdir(parents=True, exist_ok=True)
            replacement = destination.with_name(f".{destination.name}.new-{os.getpid()}")
            try:
                replacement.unlink(missing_ok=True)
                shutil.copyfile(staged, replacement)
                if not _matches_file(replacement, pinned.size, pinned.sha256):
                    raise UvToolchainError(
                        f"staged uv copy failed size/SHA-256 verification: {replacement}"
                    )
                if pinned.executable:
                    replacement.chmod(0o755)
                _fsync_file(replacement)
                os.replace(replacement, destination)
            finally:
                if replacement.exists():
                    replacement.unlink()
    finally:
        _remove_scoped_directory(stage, scratch)
    return resolve_uv(root, platform_key=key)


def _parse_platform(root: Path, version: str, key: str, value: Any) -> UvPlatform:
    table = _object(value, f"platforms.{key}")
    _keys(
        table,
        {"archive_url", "archive_sha256", "archive_size", "primary", "files"},
        f"platforms.{key}",
    )
    archive_url = _https_url(table["archive_url"], f"platforms.{key}.archive_url")
    expected_prefix = f"/astral-sh/uv/releases/download/{version}/"
    parsed_url = urlparse(archive_url)
    if (
        parsed_url.netloc != "github.com"
        or parsed_url.query
        or parsed_url.fragment
        or not parsed_url.path.startswith(expected_prefix)
    ):
        raise UvToolchainError(
            f"platforms.{key}.archive_url must be an official uv {version} release asset"
        )
    archive_sha256 = _digest(table["archive_sha256"], f"platforms.{key}.archive_sha256")
    archive_size = _positive_integer(table["archive_size"], f"platforms.{key}.archive_size")
    primary = _relative_path(table["primary"], f"platforms.{key}.primary")
    _inside(root, primary, label=f"platforms.{key}.primary")

    values = table["files"]
    if not isinstance(values, list) or not values:
        raise UvToolchainError(f"platforms.{key}.files must be a non-empty array")
    files: list[UvPinnedFile] = []
    archive_paths: set[str] = set()
    install_paths: set[Path] = set()
    for index, entry in enumerate(values):
        label = f"platforms.{key}.files[{index}]"
        item = _object(entry, label)
        _keys(item, {"archive_path", "install_path", "sha256", "size", "executable"}, label)
        archive_path = _string(item["archive_path"], f"{label}.archive_path")
        if Path(archive_path).is_absolute() or ".." in Path(archive_path).parts:
            raise UvToolchainError(f"{label}.archive_path must be archive-relative")
        install_path = _relative_path(item["install_path"], f"{label}.install_path")
        if install_path.parts[:2] != ("deps", "uv"):
            raise UvToolchainError(f"{label}.install_path must remain beneath deps/uv")
        _inside(root, install_path, label=f"{label}.install_path")
        if archive_path in archive_paths or install_path in install_paths:
            raise UvToolchainError(f"{label} duplicates an archive or install path")
        archive_paths.add(archive_path)
        install_paths.add(install_path)
        executable = item["executable"]
        if not isinstance(executable, bool):
            raise UvToolchainError(f"{label}.executable must be boolean")
        files.append(
            UvPinnedFile(
                archive_path=archive_path,
                install_path=install_path,
                sha256=_digest(item["sha256"], f"{label}.sha256"),
                size=_positive_integer(item["size"], f"{label}.size"),
                executable=executable,
            )
        )
    if primary not in install_paths:
        raise UvToolchainError(f"platforms.{key}.primary must name one declared install_path")
    primary_file = next(pinned for pinned in files if pinned.install_path == primary)
    if not primary_file.executable:
        raise UvToolchainError(f"platforms.{key}.primary must be executable")
    return UvPlatform(
        key=key,
        archive_url=archive_url,
        archive_sha256=archive_sha256,
        archive_size=archive_size,
        primary=primary,
        files=tuple(files),
    )


def _download(
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    *,
    opener: Callable[..., Any] | None,
) -> None:
    partial = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    digest = hashlib.sha256()
    size = 0
    open_url = opener or urllib.request.urlopen
    partial.unlink(missing_ok=True)
    try:
        try:
            response = open_url(url, timeout=120)
            with response, partial.open("wb") as output:
                while True:
                    chunk = response.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                    if size > expected_size:
                        raise UvToolchainError("uv archive exceeded its declared size")
                output.flush()
                os.fsync(output.fileno())
        except (OSError, ValueError) as error:
            raise UvToolchainError(f"uv archive download failed: {error}") from error
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise UvToolchainError(
                "downloaded uv archive does not match its declared size and SHA-256"
            )
        os.replace(partial, destination)
    finally:
        if partial.exists():
            partial.unlink()


def _extract_pinned_files(archive: Path, selected: UvPlatform, stage: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as container:
            for pinned in selected.files:
                try:
                    info = container.getinfo(pinned.archive_path)
                except KeyError as error:
                    raise UvToolchainError(
                        f"uv archive is missing {pinned.archive_path}"
                    ) from error
                if info.is_dir():
                    raise UvToolchainError(f"uv archive member is not a file: {info.filename}")
                with container.open(info) as source:
                    destination = _inside(stage, pinned.install_path, label="staged uv executable")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _write_verified_member(source, destination, pinned)
        return
    if archive.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive, mode="r:gz") as container:
            for pinned in selected.files:
                try:
                    info = container.getmember(pinned.archive_path)
                except KeyError as error:
                    raise UvToolchainError(
                        f"uv archive is missing {pinned.archive_path}"
                    ) from error
                if not info.isfile():
                    raise UvToolchainError(f"uv archive member is not a file: {info.name}")
                source = container.extractfile(info)
                if source is None:
                    raise UvToolchainError(f"cannot read uv archive member: {info.name}")
                with source:
                    destination = _inside(stage, pinned.install_path, label="staged uv executable")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    _write_verified_member(source, destination, pinned)
        return
    raise UvToolchainError(f"unsupported uv archive format: {archive.name}")


def _write_verified_member(source: BinaryIO, destination: Path, pinned: UvPinnedFile) -> None:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > pinned.size:
                raise UvToolchainError(
                    f"uv archive member {pinned.archive_path} exceeded its declared size"
                )
        output.flush()
        os.fsync(output.fileno())
    if size != pinned.size or digest.hexdigest() != pinned.sha256:
        raise UvToolchainError(
            f"uv archive member {pinned.archive_path} failed size/SHA-256 verification"
        )


def _verify_deployed_file(root: Path, pinned: UvPinnedFile) -> None:
    path = _inside(root, pinned.install_path, label="uv executable")
    if path.is_symlink() or not path.is_file():
        raise UvToolchainError(
            f"repository uv payload is missing: {path}; run deps/uv/restore-uv.ps1"
        )
    if not _matches_file(path, pinned.size, pinned.sha256):
        raise UvToolchainError(
            f"repository uv payload failed size/SHA-256 verification: {path}; "
            "run deps/uv/restore-uv.ps1 -Force"
        )


def _matches_file(path: Path, expected_size: int, expected_sha256: str) -> bool:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_size:
            return False
        return _file_sha256(path) == expected_sha256
    except OSError:
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows' ``_commit`` rejects a read-only descriptor even though POSIX fsync
    # accepts one, so use a non-truncating read/write handle on every platform.
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _repository(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise UvToolchainError(f"repository does not exist: {root}")
    return root


def _uv_python(root: Path) -> Path:
    if os.name == "nt":
        python = _inside(
            root,
            WINDOWS_PYTHON_PATH,
            label="repository Python",
            allow_internal_redirect=True,
        )
        if not python.is_file():
            raise UvToolchainError(
                f"repository Python is missing: {WINDOWS_PYTHON_PATH}; "
                "restore the portable runtime beneath deps/python"
            )
        return python
    if not sys.executable:
        raise UvToolchainError("the current Python interpreter has no executable path")
    python = Path(sys.executable).resolve()
    if not python.is_file():
        raise UvToolchainError(f"the current Python interpreter is unavailable: {python}")
    return python


def _inside(
    root: Path,
    relative: Path,
    *,
    label: str,
    allow_internal_redirect: bool = False,
) -> Path:
    if relative.is_absolute():
        raise UvToolchainError(f"{label} must be repository-relative")
    nominal = (root / relative).absolute()
    candidate = nominal.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise UvToolchainError(f"{label} escapes the repository: {relative}") from error
    if not allow_internal_redirect and candidate != nominal:
        raise UvToolchainError(f"{label} is redirected from its declared path: {relative}")
    return candidate


def _remove_scoped_directory(path: Path, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved = path.resolve()
    if resolved.parent != resolved_parent:
        raise UvToolchainError(f"refusing to remove unscoped restore directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise UvToolchainError(f"{label} must be an object")
    return value


def _keys(table: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(table)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        unknown = ", ".join(sorted(actual - expected)) or "none"
        raise UvToolchainError(f"{label} keys are invalid; missing: {missing}; unknown: {unknown}")


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UvToolchainError(f"{label} must be a non-empty string")
    return value


def _https_url(value: Any, label: str) -> str:
    url = _string(value, label)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise UvToolchainError(f"{label} must be an HTTPS URL without credentials")
    return url


def _relative_path(value: Any, label: str) -> Path:
    raw = _string(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise UvToolchainError(f"{label} must be a repository-relative path without '..'")
    return path


def _digest(value: Any, label: str) -> str:
    digest = _string(value, label)
    if _SHA256.fullmatch(digest) is None:
        raise UvToolchainError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise UvToolchainError(f"{label} must be a positive integer")
    return value
