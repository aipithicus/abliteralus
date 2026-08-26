"""Lightning AI Studio orchestration for surgery experiments.

The launcher uploads a narrow, reproducible runtime bundle, executes the same
``surgery_bench`` contract used locally, collects requested artifacts, and
releases compute that it started. It includes only the artifact backend from
``private/`` and excludes Git metadata, caches, tests, and unrelated worktree
content from the upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from abliteralus.surgery_bench import SurgeryExperimentSpec, load_experiment_spec


UV_VERSION = "0.12.4"
RUNTIME_SCHEMA_VERSION = 1
DEFAULT_REMOTE_ROOT = ".abliteralus"
DEFAULT_ALLOCATION_TIMEOUT_SECONDS = 900.0
DEFAULT_ALLOCATION_RETRY_SECONDS = 30.0
_START_OUT_OF_CAPACITY = 75
_START_NOT_SUPPORTED = 69
_START_STATE_CHANGED = 70
_BASE_RUNTIME_IMPORTS = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
    "yaml",
    "rich",
    "matplotlib",
    "seaborn",
    "pandas",
    "numpy",
    "sklearn",
    "tqdm",
    "huggingface_hub",
)
_GGUF_RUNTIME_IMPORTS = ("google.protobuf", "sentencepiece")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_MACHINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH_SEGMENT = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_RUNTIME_ROOTS = ("abliteralus/", "private/surgery_artifacts/src/")
_RUNTIME_FILES = {"pyproject.toml", "uv.lock", "README.md"}


class LightningConfigError(ValueError):
    """Raised before any paid Lightning resource is started."""


class LightningAllocationError(RuntimeError):
    """Raised when a bounded Studio allocation cannot reach Running."""


def _require_identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise LightningConfigError(f"{label} contains unsupported characters")
    return value


def _require_teamspace(value: str) -> str:
    if value.count("/") != 1:
        raise LightningConfigError("teamspace must use OWNER/TEAMSPACE form")
    owner, name = value.split("/", 1)
    _require_identifier(owner, "teamspace owner")
    _require_identifier(name, "teamspace name")
    return value


def _require_remote_root(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise LightningConfigError("remote_root must be a relative POSIX path without '..'")
    for part in path.parts:
        if _PATH_SEGMENT.fullmatch(part) is None:
            raise LightningConfigError("remote_root segment contains unsupported characters")
    return str(path)


def _git(repo_root: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _runtime_paths(paths: Iterable[str]) -> list[str]:
    """Select only package/runtime files from an already tracked-file census."""
    selected = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized in _RUNTIME_FILES or normalized.startswith(_RUNTIME_ROOTS):
            selected.append(normalized)
    return sorted(set(selected))


def _tracked_runtime_paths(repo_root: Path) -> list[str]:
    result = _git(repo_root, ["ls-files", "--", *_RUNTIME_ROOTS, *_RUNTIME_FILES])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    paths = _runtime_paths(result.stdout.splitlines())
    if not paths:
        raise RuntimeError("no tracked runtime files were found")
    return paths


def _assert_runtime_clean(repo_root: Path) -> None:
    result = _git(
        repo_root,
        ["status", "--porcelain", "--untracked-files=all", "--", *_RUNTIME_ROOTS, *_RUNTIME_FILES],
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    if result.stdout.strip():
        raise RuntimeError(
            "runtime bundle inputs have uncommitted changes; commit them before launching "
            "so the remote run has a durable source identity"
        )


def _git_commit(repo_root: Path) -> str:
    result = _git(repo_root, ["rev-parse", "HEAD"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git rev-parse failed")
    return result.stdout.strip()


def _zip_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _write_deterministic_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def build_runtime_bundle(
    repo_root: Path,
    spec: SurgeryExperimentSpec,
    destination: Path,
    *,
    require_clean: bool = True,
) -> dict[str, Any]:
    """Build the privacy-scoped Studio upload and return its provenance record."""
    repo_root = repo_root.resolve()
    if require_clean:
        _assert_runtime_clean(repo_root)
    paths = _tracked_runtime_paths(repo_root)
    commit = _git_commit(repo_root)
    file_hashes: dict[str, str] = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w") as archive:
        for relative in paths:
            source = repo_root / relative
            if not source.is_file():
                raise RuntimeError(f"tracked runtime file is missing: {relative}")
            content = _zip_bytes(source)
            file_hashes[relative] = hashlib.sha256(content).hexdigest()
            _write_deterministic_member(archive, relative, content)

        config_content = spec.source_path.read_bytes()
        file_hashes["experiment.yaml"] = hashlib.sha256(config_content).hexdigest()
        _write_deterministic_member(archive, "experiment.yaml", config_content)
        bundle_manifest = {
            "schema_version": 1,
            "git_commit": commit,
            "experiment": spec.name,
            "files": file_hashes,
            "excludes": [
                ".git/",
                "private/* except private/surgery_artifacts/src/",
                "ci/",
                "tests/",
                ".codex/",
                ".scratch/",
                ".venv/",
            ],
        }
        _write_deterministic_member(
            archive,
            "bundle-manifest.json",
            (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    return {
        **bundle_manifest,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class LightningRuntime:
    remote_root: str
    lock_sha256: str
    variant: str
    key: str
    uv_version: str
    uv_environment: str
    uv_executable: str
    environment: str
    marker: str
    uv_cache: str
    hf_home: str
    xdg_cache: str
    required_imports: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "remote_root": self.remote_root,
            "lock_sha256": self.lock_sha256,
            "variant": self.variant,
            "key": self.key,
            "uv_version": self.uv_version,
            "uv_environment": self.uv_environment,
            "uv_executable": self.uv_executable,
            "environment": self.environment,
            "marker": self.marker,
            "uv_cache": self.uv_cache,
            "hf_home": self.hf_home,
            "xdg_cache": self.xdg_cache,
            "required_imports": list(self.required_imports),
        }


def build_runtime_layout(
    repo_root: Path,
    spec: SurgeryExperimentSpec,
    *,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    skip_gguf: bool = False,
) -> LightningRuntime:
    """Describe a persistent Studio runtime without contacting Lightning."""

    remote_root = _require_remote_root(remote_root)
    lock_path = repo_root.resolve() / "uv.lock"
    if not lock_path.is_file():
        raise LightningConfigError(f"uv.lock is unavailable: {lock_path}")
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    variant = "gguf" if spec.gguf_enabled and not skip_gguf else "base"
    key = f"{lock_sha256[:20]}-{variant}"
    root = PurePosixPath(remote_root)
    uv_environment = root / "tools" / "uv" / UV_VERSION
    environment = root / "runtimes" / key
    required_imports = _BASE_RUNTIME_IMPORTS + (_GGUF_RUNTIME_IMPORTS if variant == "gguf" else ())
    return LightningRuntime(
        remote_root=remote_root,
        lock_sha256=lock_sha256,
        variant=variant,
        key=key,
        uv_version=UV_VERSION,
        uv_environment=str(uv_environment),
        uv_executable=str(uv_environment / "bin" / "uv"),
        environment=str(environment),
        marker=str(root / "runtime-manifests" / f"{key}.json"),
        uv_cache=str(root / "cache" / "uv"),
        hf_home=str(root / "cache" / "huggingface"),
        xdg_cache=str(root / "cache" / "xdg"),
        required_imports=required_imports,
    )


def _home_path(path: str) -> str:
    return f'"$HOME"/{shlex.quote(path)}'


def _runtime_expected(runtime: LightningRuntime) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "lock_sha256": runtime.lock_sha256,
        "variant": runtime.variant,
        "key": runtime.key,
        "uv_version": runtime.uv_version,
    }


def _runtime_probe_script(runtime: LightningRuntime, *, write_marker: bool) -> str:
    expected = repr(_runtime_expected(runtime))
    required = repr(list(runtime.required_imports))
    common = [
        "import importlib.util, json, os, pathlib, platform, subprocess, sys",
        f"expected = {expected}",
        f"required = {required}",
        "marker = pathlib.Path(os.environ['ABLITERALUS_RUNTIME_MARKER'])",
        "uv = os.environ['ABLITERALUS_UV_EXECUTABLE']",
        "def available(name):",
        "    try:",
        "        return importlib.util.find_spec(name) is not None",
        "    except (ImportError, AttributeError, ValueError):",
        "        return False",
        "missing = [name for name in required if not available(name)]",
        "try:",
        "    uv_check = subprocess.run([uv, '--version'], capture_output=True, text=True, check=False)",
        "    uv_output = uv_check.stdout.strip()",
        "except OSError:",
        "    uv_check = None",
        "    uv_output = ''",
        "uv_ok = bool(uv_check and uv_check.returncode == 0 and uv_output.split()[:2] == ['uv', expected['uv_version']])",
    ]
    if write_marker:
        common.extend(
            [
                "ready = not missing and uv_ok",
                "payload = {**expected, 'ready': ready, 'python_version': platform.python_version(), 'python_executable': sys.executable, 'required_imports': required}",
                "if not ready:",
                "    print(json.dumps({**payload, 'missing_imports': missing, 'uv_output': uv_output}, sort_keys=True))",
                "    raise SystemExit(4)",
                "marker.parent.mkdir(parents=True, exist_ok=True)",
                "temporary = marker.with_suffix(marker.suffix + '.tmp')",
                "temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n', encoding='utf-8')",
                "temporary.replace(marker)",
                "print(json.dumps(payload, sort_keys=True))",
            ]
        )
    else:
        common.extend(
            [
                "try:",
                "    payload = json.loads(marker.read_text(encoding='utf-8'))",
                "except (OSError, json.JSONDecodeError):",
                "    payload = {}",
                "mismatches = {name: {'expected': value, 'actual': payload.get(name)} for name, value in expected.items() if payload.get(name) != value}",
                "ready = not missing and uv_ok and not mismatches and payload.get('ready') is True",
                "report = {'ready': ready, 'runtime': expected, 'python_version': platform.python_version(), 'missing_imports': missing, 'metadata_mismatches': mismatches, 'uv_output': uv_output}",
                "print(json.dumps(report, sort_keys=True))",
                "raise SystemExit(0 if ready else 5)",
            ]
        )
    return "\n".join(common)


def _runtime_probe_shell(runtime: LightningRuntime, *, write_marker: bool = False) -> str:
    python = _home_path(str(PurePosixPath(runtime.environment) / "bin" / "python"))
    uv = _home_path(runtime.uv_executable)
    marker = _home_path(runtime.marker)
    script = shlex.quote(_runtime_probe_script(runtime, write_marker=write_marker))
    return "; ".join(
        [
            f"test -x {python}",
            f"test -x {uv}",
            (
                f"env ABLITERALUS_RUNTIME_MARKER={marker} "
                f"ABLITERALUS_UV_EXECUTABLE={uv} {python} -c {script}"
            ),
        ]
    )


@dataclass(frozen=True)
class LightningAllocationPolicy:
    timeout_seconds: float = DEFAULT_ALLOCATION_TIMEOUT_SECONDS
    retry_seconds: float = DEFAULT_ALLOCATION_RETRY_SECONDS
    fallback_machines: tuple[str, ...] = ()
    pending_policy: str = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "retry_seconds": self.retry_seconds,
            "fallback_machines": list(self.fallback_machines),
            "pending_policy": self.pending_policy,
        }


@dataclass(frozen=True)
class LightningAllocationResult:
    requested_machine: str
    selected_machine: str
    mode: str
    started_at: str
    completed_at: str
    duration_seconds: float
    attempts: tuple[dict[str, Any], ...]

    @property
    def started_here(self) -> bool:
        return self.mode in {"started", "started_after_pending_stop"}

    @property
    def owns_compute(self) -> bool:
        return self.mode != "reused_running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_machine": self.requested_machine,
            "selected_machine": self.selected_machine,
            "mode": self.mode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "attempts": list(self.attempts),
        }


def _normalize_machine(value: str, label: str = "machine") -> str:
    if _MACHINE.fullmatch(value) is None:
        raise LightningConfigError(f"{label} contains unsupported characters")
    return value.upper()


def _allocation_policy(
    *,
    primary_machine: str,
    allocation_timeout_seconds: float,
    allocation_retry_seconds: float,
    fallback_machines: Sequence[str],
    pending_policy: str,
) -> LightningAllocationPolicy:
    if not math.isfinite(allocation_timeout_seconds) or allocation_timeout_seconds <= 0:
        raise LightningConfigError("allocation_timeout_seconds must be positive")
    if not math.isfinite(allocation_retry_seconds) or allocation_retry_seconds <= 0:
        raise LightningConfigError("allocation_retry_seconds must be positive")
    if pending_policy not in {"fail", "adopt", "stop"}:
        raise LightningConfigError("pending_policy must be fail, adopt, or stop")
    seen = {_normalize_machine(primary_machine)}
    normalized: list[str] = []
    for machine in fallback_machines:
        candidate = _normalize_machine(machine, "fallback machine")
        if candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)
    return LightningAllocationPolicy(
        timeout_seconds=float(allocation_timeout_seconds),
        retry_seconds=float(allocation_retry_seconds),
        fallback_machines=tuple(normalized),
        pending_policy=pending_policy,
    )


@dataclass(frozen=True)
class LightningProvisionPlan:
    experiment: str
    teamspace: str
    studio: str
    machine: str
    interruptible: bool
    max_runtime: int | None
    runtime: LightningRuntime
    remote_bundle: str
    remote_source: str
    prepare_command: str
    command: str
    allocation: LightningAllocationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "teamspace": self.teamspace,
            "studio": self.studio,
            "machine": self.machine,
            "interruptible": self.interruptible,
            "max_runtime": self.max_runtime,
            "runtime": self.runtime.to_dict(),
            "remote_bundle": self.remote_bundle,
            "remote_source": self.remote_source,
            "prepare_command": self.prepare_command,
            "command": self.command,
            "allocation": self.allocation.to_dict(),
        }


def build_provision_plan(
    spec: SurgeryExperimentSpec,
    *,
    repo_root: Path,
    teamspace: str,
    studio: str,
    machine: str,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    interruptible: bool = False,
    max_runtime: int | None = None,
    skip_gguf: bool = False,
    allocation_timeout_seconds: float = DEFAULT_ALLOCATION_TIMEOUT_SECONDS,
    allocation_retry_seconds: float = DEFAULT_ALLOCATION_RETRY_SECONDS,
    fallback_machines: Sequence[str] = (),
    pending_policy: str = "fail",
) -> LightningProvisionPlan:
    """Build the explicit, infrequent Studio provisioning operation."""

    teamspace = _require_teamspace(teamspace)
    studio = _require_identifier(studio, "studio")
    machine = _normalize_machine(machine)
    if max_runtime is not None and max_runtime <= 0:
        raise LightningConfigError("max_runtime must be positive")
    runtime = build_runtime_layout(
        repo_root,
        spec,
        remote_root=remote_root,
        skip_gguf=skip_gguf,
    )
    allocation = _allocation_policy(
        primary_machine=machine,
        allocation_timeout_seconds=allocation_timeout_seconds,
        allocation_retry_seconds=allocation_retry_seconds,
        fallback_machines=fallback_machines,
        pending_policy=pending_policy,
    )
    root = PurePosixPath(runtime.remote_root)
    remote_bundle = str(root / "provision" / "bundles" / f"{runtime.key}.zip")
    remote_source = str(root / "provision" / "sources" / runtime.key)
    required_directories = [
        str(PurePosixPath(remote_bundle).parent),
        remote_source,
        str(PurePosixPath(runtime.uv_environment).parent),
        str(PurePosixPath(runtime.environment).parent),
        str(PurePosixPath(runtime.marker).parent),
        runtime.uv_cache,
        runtime.hf_home,
        runtime.xdg_cache,
        str(root / "bundles"),
        str(root / "runs"),
    ]
    mkdir = "mkdir -p " + " ".join(_home_path(path) for path in required_directories)
    prepare_command = "bash -lc " + shlex.quote("; ".join(["set -euo pipefail", mkdir]))

    uv_environment = _home_path(runtime.uv_environment)
    uv_python = _home_path(str(PurePosixPath(runtime.uv_environment) / "bin" / "python"))
    uv = _home_path(runtime.uv_executable)
    runtime_environment = _home_path(runtime.environment)
    sync_arguments = ["sync", "--frozen", "--no-dev", "--no-install-project"]
    if runtime.variant == "gguf":
        sync_arguments.extend(["--extra", "gguf"])
    sync_command = " ".join([uv, *map(shlex.quote, sync_arguments)])
    commands = [
        "set -euo pipefail",
        mkdir,
        f"python -m venv {uv_environment}",
        (
            f"{uv_python} -m pip install --disable-pip-version-check "
            f"{shlex.quote(f'uv=={runtime.uv_version}')}"
        ),
        (f"python -m zipfile -e {_home_path(remote_bundle)} {_home_path(remote_source)}"),
        f"cd {_home_path(remote_source)}",
        (
            f"env UV_CACHE_DIR={_home_path(runtime.uv_cache)} "
            f"UV_PROJECT_ENVIRONMENT={runtime_environment} "
            f"UV_PYTHON_DOWNLOADS=never {sync_command}"
        ),
        _runtime_probe_shell(runtime, write_marker=True),
    ]
    return LightningProvisionPlan(
        experiment=spec.name,
        teamspace=teamspace,
        studio=studio,
        machine=machine,
        interruptible=interruptible,
        max_runtime=max_runtime,
        runtime=runtime,
        remote_bundle=remote_bundle,
        remote_source=remote_source,
        prepare_command=prepare_command,
        command="bash -lc " + shlex.quote("; ".join(commands)),
        allocation=allocation,
    )


@dataclass(frozen=True)
class LightningRunPlan:
    experiment: str
    run_id: str
    teamspace: str
    studio: str
    machine: str
    remote_root: str
    remote_bundle: str
    remote_source: str
    remote_output: str
    runtime: LightningRuntime
    doctor_command: str
    command: str
    forwarded_environment: tuple[str, ...]
    interruptible: bool
    max_runtime: int | None
    skip_gguf: bool
    allocation: LightningAllocationPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "teamspace": self.teamspace,
            "studio": self.studio,
            "machine": self.machine,
            "interruptible": self.interruptible,
            "max_runtime": self.max_runtime,
            "remote_root": self.remote_root,
            "remote_bundle": self.remote_bundle,
            "remote_source": self.remote_source,
            "remote_output": self.remote_output,
            "runtime": self.runtime.to_dict(),
            "doctor_command": self.doctor_command,
            "forwarded_environment": list(self.forwarded_environment),
            "skip_gguf": self.skip_gguf,
            "command": self.command,
            "allocation": self.allocation.to_dict(),
        }


def build_lightning_plan(
    spec: SurgeryExperimentSpec,
    *,
    teamspace: str,
    studio: str,
    machine: str,
    repo_root: Path | None = None,
    run_id: str | None = None,
    remote_root: str = DEFAULT_REMOTE_ROOT,
    forwarded_environment: Sequence[str] = (),
    interruptible: bool = False,
    max_runtime: int | None = None,
    skip_gguf: bool = False,
    allocation_timeout_seconds: float = DEFAULT_ALLOCATION_TIMEOUT_SECONDS,
    allocation_retry_seconds: float = DEFAULT_ALLOCATION_RETRY_SECONDS,
    fallback_machines: Sequence[str] = (),
    pending_policy: str = "fail",
) -> LightningRunPlan:
    """Build a shell-safe, credential-free launch plan without contacting Lightning."""
    if Path(spec.model["source"]).expanduser().exists():
        raise LightningConfigError(
            "Lightning experiments require a remote-addressable OWNER/MODEL source; "
            "local checkpoint directories are not included in the runtime bundle"
        )
    teamspace = _require_teamspace(teamspace)
    studio = _require_identifier(studio, "studio")
    machine = _normalize_machine(machine)
    if max_runtime is not None and max_runtime <= 0:
        raise LightningConfigError("max_runtime must be positive")
    run_id = _require_identifier(
        run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "run_id"
    )
    runtime = build_runtime_layout(
        repo_root or _repo_root(),
        spec,
        remote_root=remote_root,
        skip_gguf=skip_gguf,
    )
    allocation = _allocation_policy(
        primary_machine=machine,
        allocation_timeout_seconds=allocation_timeout_seconds,
        allocation_retry_seconds=allocation_retry_seconds,
        fallback_machines=fallback_machines,
        pending_policy=pending_policy,
    )
    env_names: list[str] = []
    for name in forwarded_environment:
        if _ENV_NAME.fullmatch(name) is None:
            raise LightningConfigError(f"invalid environment variable name: {name!r}")
        if name not in env_names:
            env_names.append(name)

    root = PurePosixPath(runtime.remote_root)
    remote_run = root / "runs" / run_id
    remote_bundle = str(root / "bundles" / f"{run_id}.zip")
    remote_source = str(remote_run / "source")
    remote_output_root = str(remote_run / "outputs")
    remote_output = str(PurePosixPath(remote_output_root) / spec.name / run_id)
    runtime_python = _home_path(str(PurePosixPath(runtime.environment) / "bin" / "python"))

    surgery_arguments = [
        runtime_python,
        "-m",
        "surgery_artifacts.integration",
        "run",
        "--config",
        "experiment.yaml",
        "--output-root",
        _home_path(remote_output_root),
        "--run-id",
        run_id,
    ]
    if skip_gguf:
        surgery_arguments.append("--skip-gguf")
    surgery_command = " ".join(
        token if token.startswith('"$HOME"/') else shlex.quote(token) for token in surgery_arguments
    )
    doctor_shell = _runtime_probe_shell(runtime)
    commands = [
        "set -euo pipefail",
        doctor_shell,
        f"mkdir -p {_home_path(remote_source)} {_home_path(runtime.hf_home)} {_home_path(runtime.xdg_cache)}",
        (f"python -m zipfile -e {_home_path(remote_bundle)} {_home_path(remote_source)}"),
        f"cd {_home_path(remote_source)}",
        'export PYTHONPATH="$PWD/private/surgery_artifacts/src${PYTHONPATH:+:$PYTHONPATH}"',
        f"export HF_HOME={_home_path(runtime.hf_home)}",
        f"export XDG_CACHE_HOME={_home_path(runtime.xdg_cache)}",
        surgery_command,
    ]
    return LightningRunPlan(
        experiment=spec.name,
        run_id=run_id,
        teamspace=teamspace,
        studio=studio,
        machine=machine,
        remote_root=runtime.remote_root,
        remote_bundle=remote_bundle,
        remote_source=remote_source,
        remote_output=remote_output,
        runtime=runtime,
        doctor_command="bash -lc " + shlex.quote("; ".join(["set -euo pipefail", doctor_shell])),
        command="bash -lc " + shlex.quote("; ".join(commands)),
        forwarded_environment=tuple(env_names),
        interruptible=interruptible,
        max_runtime=max_runtime,
        skip_gguf=skip_gguf,
        allocation=allocation,
    )


def _machine_value(machine_class: Any, name: str) -> Any:
    if hasattr(machine_class, "from_str"):
        return machine_class.from_str(name)
    try:
        return getattr(machine_class, name.upper())
    except AttributeError as error:
        raise LightningConfigError(f"unknown Lightning machine: {name}") from error


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration(started: float, completed: float) -> float:
    return round(max(0.0, completed - started), 3)


def _status_name(status: object) -> str:
    return str(status).rsplit(".", 1)[-1].lower()


def _status_is_running(status: object) -> bool:
    return _status_name(status) == "running"


def _status_is_pending(status: object) -> bool:
    return _status_name(status) == "pending"


def _status_is_stopped(status: object) -> bool:
    return _status_name(status) == "stopped"


def _machine_identity(machine: object) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(machine).rsplit(".", 1)[-1]).upper()


def _actual_machine(studio: Any) -> str | None:
    machine = getattr(studio, "machine", None)
    if machine is None:
        return None
    return str(machine).rsplit(".", 1)[-1].upper()


def _machine_matches(machine: object, candidates: Sequence[str]) -> bool:
    actual = _machine_identity(machine)
    return any(actual == _machine_identity(candidate) for candidate in candidates)


def _start_studio(
    studio: Any,
    machine_class: Any,
    machine: str,
    *,
    interruptible: bool,
    max_runtime: int | None,
) -> None:
    arguments: dict[str, Any] = {
        "machine": _machine_value(machine_class, machine),
        "interruptible": interruptible,
    }
    if max_runtime is not None:
        arguments["max_runtime"] = max_runtime
    studio.start(**arguments)


def _direct_start_attempt(
    studio: Any,
    machine_class: Any,
    machine: str,
    *,
    interruptible: bool,
    max_runtime: int | None,
    timeout_seconds: float,
    status_callback: Callable[[dict[str, Any]], None],
) -> str:
    del timeout_seconds, status_callback
    try:
        _start_studio(
            studio,
            machine_class,
            machine,
            interruptible=interruptible,
            max_runtime=max_runtime,
        )
    except Exception as error:
        error_name = type(error).__name__
        if error_name == "OutOfCapacityError":
            return "out_of_capacity"
        if error_name == "NotSupportedError":
            return "not_supported"
        raise
    return "running"


def _subprocess_start_attempt(
    remote: Any,
    teamspace: str,
    studio: str,
    machine: str,
    *,
    interruptible: bool,
    max_runtime: int | None,
    timeout_seconds: float,
    poll_seconds: float,
    status_callback: Callable[[dict[str, Any]], None],
) -> str:
    command = [
        sys.executable,
        "-m",
        "abliteralus.lightning_surgery",
        "_start-once",
        "--teamspace",
        teamspace,
        "--studio",
        studio,
        "--machine",
        machine,
    ]
    if interruptible:
        command.append("--interruptible")
    if max_runtime is not None:
        command.extend(["--max-runtime", str(max_runtime)])
    started_clock = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    observed_pending = False
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                if return_code == 0:
                    return "running"
                if return_code == _START_OUT_OF_CAPACITY:
                    return "out_of_capacity"
                if return_code == _START_NOT_SUPPORTED:
                    return "not_supported"
                if return_code == _START_STATE_CHANGED:
                    return "state_changed"
                return "error"

            try:
                status = _status_name(remote.status)
            except Exception:
                status = "unavailable"
            status_callback(
                {
                    "status": status,
                    "observed_at": _utc_timestamp(),
                    "elapsed_seconds": _duration(started_clock, time.monotonic()),
                }
            )
            if status == "running":
                return "running"
            if status == "pending":
                observed_pending = True
            elif status == "stopped" and observed_pending:
                return "error"

            remaining = timeout_seconds - (time.monotonic() - started_clock)
            if remaining <= 0:
                return "timeout"
            try:
                process.wait(timeout=min(poll_seconds, remaining))
            except subprocess.TimeoutExpired:
                pass
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def _write_allocation_journal(
    path: Path | None,
    *,
    requested_machine: str,
    policy: LightningAllocationPolicy,
    state: str,
    attempts: Sequence[dict[str, Any]],
) -> None:
    if path is None:
        return
    payload = {
        "requested_machine": requested_machine,
        "policy": policy.to_dict(),
        "state": state,
        "attempts": list(attempts),
        "updated_at": _utc_timestamp(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _allocation_result(
    *,
    requested_machine: str,
    selected_machine: str,
    mode: str,
    started_at: str,
    started_clock: float,
    attempts: Sequence[dict[str, Any]],
    clock: Callable[[], float],
) -> LightningAllocationResult:
    completed_clock = clock()
    return LightningAllocationResult(
        requested_machine=requested_machine,
        selected_machine=selected_machine,
        mode=mode,
        started_at=started_at,
        completed_at=_utc_timestamp(),
        duration_seconds=_duration(started_clock, completed_clock),
        attempts=tuple(attempts),
    )


def _ensure_studio_running(
    studio: Any,
    *,
    requested_machine: str,
    policy: LightningAllocationPolicy,
    reuse_running: bool,
    start_attempt: Callable[[str, float, Callable[[dict[str, Any]], None]], str],
    journal_path: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> LightningAllocationResult:
    candidates = (requested_machine, *policy.fallback_machines)
    started_at = _utc_timestamp()
    started_clock = clock()
    deadline = started_clock + policy.timeout_seconds
    attempts: list[dict[str, Any]] = []
    pending_stopped = False

    def persist(state: str) -> None:
        _write_allocation_journal(
            journal_path,
            requested_machine=requested_machine,
            policy=policy,
            state=state,
            attempts=attempts,
        )

    status = _status_name(studio.status)
    if status == "running":
        if not reuse_running:
            persist("running_requires_reuse")
            raise RuntimeError(
                "Studio is already running; pass --reuse-running to use it without "
                "changing or automatically stopping its machine"
            )
        actual = getattr(studio, "machine", None)
        if actual is not None and not _machine_matches(actual, candidates):
            persist("running_machine_mismatch")
            raise LightningAllocationError(
                "running Studio machine is outside the requested allocation policy"
            )
        result = _allocation_result(
            requested_machine=requested_machine,
            selected_machine=_actual_machine(studio) or requested_machine,
            mode="reused_running",
            started_at=started_at,
            started_clock=started_clock,
            attempts=attempts,
            clock=clock,
        )
        persist("running")
        return result

    if status == "pending":
        if policy.pending_policy == "fail":
            persist("pending_requires_policy")
            raise LightningAllocationError(
                "Studio is Pending; choose --pending-policy adopt or stop for unattended recovery"
            )
        reconcile_started = clock()
        reconcile_started_at = _utc_timestamp()
        if policy.pending_policy == "stop":
            studio.stop()
            pending_stopped = True
        while True:
            status = _status_name(studio.status)
            if status == "running" and policy.pending_policy == "adopt":
                actual = getattr(studio, "machine", None)
                if actual is not None and not _machine_matches(actual, candidates):
                    persist("adopted_machine_mismatch")
                    raise LightningAllocationError(
                        "adopted Studio machine is outside the requested allocation policy"
                    )
                completed = clock()
                attempts.append(
                    {
                        "machine": _actual_machine(studio) or requested_machine,
                        "started_at": reconcile_started_at,
                        "completed_at": _utc_timestamp(),
                        "duration_seconds": _duration(reconcile_started, completed),
                        "outcome": "adopted_pending",
                    }
                )
                result = _allocation_result(
                    requested_machine=requested_machine,
                    selected_machine=_actual_machine(studio) or requested_machine,
                    mode="adopted_pending",
                    started_at=started_at,
                    started_clock=started_clock,
                    attempts=attempts,
                    clock=clock,
                )
                try:
                    persist("running")
                except OSError:
                    _stop_owned_studio(studio)
                    raise
                return result
            if status == "stopped":
                attempts.append(
                    {
                        "machine": requested_machine,
                        "started_at": reconcile_started_at,
                        "completed_at": _utc_timestamp(),
                        "duration_seconds": _duration(reconcile_started, clock()),
                        "outcome": (
                            "stopped_pending" if policy.pending_policy == "stop" else "pending_ended"
                        ),
                    }
                )
                persist("retrying")
                break
            if status != "pending":
                persist("pending_reconciliation_failed")
                raise LightningAllocationError(
                    f"Studio left Pending in unsupported state {status!r}"
                )
            remaining = deadline - clock()
            if remaining <= 0:
                _stop_owned_studio(studio)
                persist("allocation_timeout")
                raise LightningAllocationError(
                    "Studio remained Pending beyond the allocation timeout"
                )
            sleeper(min(policy.retry_seconds, remaining))
    elif status != "stopped":
        persist("unsupported_initial_state")
        raise LightningAllocationError(f"Studio cannot be allocated from state {status!r}")

    unsupported: set[str] = set()
    while True:
        for machine in candidates:
            if machine in unsupported:
                continue
            remaining = deadline - clock()
            if remaining <= 0:
                persist("allocation_timeout")
                raise LightningAllocationError(
                    "no requested Studio machine became available before the allocation timeout"
                )
            attempt_started = clock()
            attempt_started_at = _utc_timestamp()
            attempt: dict[str, Any] = {
                "machine": machine,
                "started_at": attempt_started_at,
                "completed_at": None,
                "duration_seconds": None,
                "outcome": "starting",
                "status_polls": [],
            }
            attempts.append(attempt)
            persist("starting")

            def record_status(observation: dict[str, Any]) -> None:
                attempt["status_polls"].append(observation)
                persist("starting")

            try:
                outcome = start_attempt(machine, remaining, record_status)
            except KeyboardInterrupt:
                attempt.update(
                    completed_at=_utc_timestamp(),
                    duration_seconds=_duration(attempt_started, clock()),
                    outcome="interrupted",
                )
                _stop_owned_studio(studio)
                persist("interrupted")
                raise
            except Exception:
                attempt.update(
                    completed_at=_utc_timestamp(),
                    duration_seconds=_duration(attempt_started, clock()),
                    outcome="error",
                )
                _stop_owned_studio(studio)
                persist("error")
                raise
            attempt.update(
                completed_at=_utc_timestamp(),
                duration_seconds=_duration(attempt_started, clock()),
                outcome=outcome,
            )
            if outcome == "running":
                actual = getattr(studio, "machine", None)
                if actual is not None and not _machine_matches(actual, (machine,)):
                    persist("started_machine_mismatch")
                    raise LightningAllocationError(
                        "Studio reached Running on a machine other than the selected candidate"
                    )
                result = _allocation_result(
                    requested_machine=requested_machine,
                    selected_machine=_actual_machine(studio) or machine,
                    mode="started_after_pending_stop" if pending_stopped else "started",
                    started_at=started_at,
                    started_clock=started_clock,
                    attempts=attempts,
                    clock=clock,
                )
                try:
                    persist("running")
                except OSError:
                    _stop_owned_studio(studio)
                    raise
                return result
            persist("retrying")
            if outcome == "not_supported":
                unsupported.add(machine)
                continue
            if outcome == "timeout":
                _stop_owned_studio(studio)
                persist("allocation_timeout")
                raise LightningAllocationError(
                    "Studio start did not complete before the allocation timeout"
                )
            if outcome == "state_changed":
                persist("state_changed")
                raise LightningAllocationError(
                    "Studio state changed during allocation; refusing to assume ownership"
                )
            if outcome == "error":
                _stop_owned_studio(studio)
                persist("error")
                raise LightningAllocationError("Lightning Studio start worker failed")
            if outcome != "out_of_capacity":
                persist("error")
                raise LightningAllocationError("Lightning Studio start returned an unknown outcome")
        if len(unsupported) == len(candidates):
            persist("not_supported")
            raise LightningAllocationError("none of the requested Studio machines are supported")
        remaining = deadline - clock()
        if remaining <= 0:
            persist("allocation_timeout")
            raise LightningAllocationError(
                "no requested Studio machine became available before the allocation timeout"
            )
        sleeper(min(policy.retry_seconds, remaining))


def _stop_owned_studio(studio: Any) -> bool:
    if _status_name(studio.status) not in {"running", "pending"}:
        return False
    studio.stop()
    return True


def _forward_environment(studio: Any, names: Sequence[str]) -> dict[str, tuple[bool, str | None]]:
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
        else:
            values[name] = value
    if missing:
        raise LightningConfigError(
            "requested forwarded environment variables are unset: " + ", ".join(missing)
        )
    if not values:
        return {}
    current = dict(studio.env)
    previous = {name: (name in current, current.get(name)) for name in values}
    studio.set_env(values, partial=True)
    return previous


def _restore_environment(
    studio: Any,
    previous: dict[str, tuple[bool, str | None]],
) -> None:
    restore = {
        name: value for name, (existed, value) in previous.items() if existed and value is not None
    }
    if restore:
        studio.set_env(restore, partial=True)
    for name, (existed, _value) in previous.items():
        if not existed:
            studio.delete_env(name)


def _download_summary(studio: Any, plan: LightningRunPlan, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for relative in (
        "run-manifest.json",
        "surgery.log",
        "failure-traceback.log",
        "gguf/smoke-results.json",
    ):
        remote = str(PurePosixPath(plan.remote_output) / relative)
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            studio.download_file(remote, str(local))
        except (OSError, RuntimeError, FileNotFoundError):
            continue
        downloaded.append(relative)
    return downloaded


def _download_capsule(
    studio: Any,
    plan: LightningRunPlan,
    destination: Path,
) -> tuple[list[str], dict[str, Any]]:
    downloaded = _download_summary(studio, plan, destination)
    capsule_destination = destination / "artifact"
    remote_capsule = str(PurePosixPath(plan.remote_output) / "artifact")
    studio.download_folder(remote_capsule, str(capsule_destination))
    try:
        from surgery_artifacts.validation import validate_capsule
    except ImportError as error:
        raise RuntimeError(
            "capsule collection requires the private surgery_artifacts package on PYTHONPATH"
        ) from error
    validation = validate_capsule(capsule_destination)
    downloaded.append("artifact/")
    remote_exports = str(PurePosixPath(plan.remote_output) / "artifact-exports")
    export_destination = destination / "artifact-exports"
    try:
        studio.download_folder(remote_exports, str(export_destination))
    except (OSError, RuntimeError, FileNotFoundError):
        pass
    else:
        downloaded.append("artifact-exports/")
    return downloaded, {
        "surgery_id": validation.surgery_id,
        "operation_count": validation.operation_count,
        "file_count": validation.file_count,
        "total_bytes": validation.total_bytes,
        "verified": True,
    }


def execute_provision_plan(
    plan: LightningProvisionPlan,
    *,
    spec: SurgeryExperimentSpec,
    repo_root: Path,
    local_output: Path,
    keep_running: bool = False,
    reuse_running: bool = False,
    studio_class: Any | None = None,
    machine_class: Any | None = None,
) -> dict[str, Any]:
    """Materialize one persistent lock-addressed runtime and release owned compute."""

    use_start_subprocess = studio_class is None or machine_class is None
    if studio_class is None or machine_class is None:
        try:
            from lightning_sdk import Machine, Studio
        except ImportError as error:
            raise RuntimeError(
                "Lightning support is not installed; use the 'lightning' optional dependency"
            ) from error
        studio_class = Studio
        machine_class = Machine

    local_output = local_output.resolve()
    local_output.mkdir(parents=True, exist_ok=True)
    (local_output / "provision-plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started_here = False
    adopted_pending = False
    owns_compute = False
    allocation_ready = False
    studio_stopped = False
    allocation_result: LightningAllocationResult | None = None
    remote_exit_code: int | None = None
    remote_output = ""
    stage = "prepare"
    operation_started_at = _utc_timestamp()
    operation_started_clock = time.monotonic()
    remote_started_at: str | None = None
    remote_completed_at: str | None = None
    remote_started_clock: float | None = None
    remote_completed_clock: float | None = None
    with tempfile.TemporaryDirectory(
        prefix=".abliteralus-provision-", dir=local_output
    ) as temporary:
        bundle_path = Path(temporary) / f"{plan.runtime.key}.zip"
        bundle = build_runtime_bundle(repo_root, spec, bundle_path)
        (local_output / "bundle-manifest.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        studio = studio_class(
            name=plan.studio,
            teamspace=plan.teamspace,
            create_ok=True,
        )
        if use_start_subprocess:
            start_attempt = lambda machine, timeout, status_callback: _subprocess_start_attempt(
                studio,
                plan.teamspace,
                plan.studio,
                machine,
                interruptible=plan.interruptible,
                max_runtime=plan.max_runtime,
                timeout_seconds=timeout,
                poll_seconds=plan.allocation.retry_seconds,
                status_callback=status_callback,
            )
        else:
            start_attempt = lambda machine, timeout, status_callback: _direct_start_attempt(
                studio,
                machine_class,
                machine,
                interruptible=plan.interruptible,
                max_runtime=plan.max_runtime,
                timeout_seconds=timeout,
                status_callback=status_callback,
            )
        try:
            allocation_result = _ensure_studio_running(
                studio,
                requested_machine=plan.machine,
                policy=plan.allocation,
                reuse_running=reuse_running,
                start_attempt=start_attempt,
                journal_path=local_output / "allocation.json",
            )
            allocation_ready = True
            owns_compute = allocation_result.owns_compute
            started_here = allocation_result.started_here
            adopted_pending = allocation_result.mode == "adopted_pending"
            if use_start_subprocess and allocation_result.mode != "reused_running":
                # The start worker owns its SDK keep-alive thread. Re-resolving a
                # now-Running Studio starts one in this long-lived controller too.
                studio = studio_class(
                    name=plan.studio,
                    teamspace=plan.teamspace,
                    create_ok=False,
                )
            remote_started_at = _utc_timestamp()
            remote_started_clock = time.monotonic()
            remote_output, remote_exit_code = studio.run_with_exit_code(plan.prepare_command)
            (local_output / "studio-prepare.log").write_text(
                remote_output, encoding="utf-8", errors="replace"
            )
            if remote_exit_code == 0:
                studio.upload_file(str(bundle_path), remote_path=plan.remote_bundle)
                stage = "provision"
                remote_output, remote_exit_code = studio.run_with_exit_code(plan.command)
                (local_output / "studio-provision.log").write_text(
                    remote_output, encoding="utf-8", errors="replace"
                )
            remote_completed_clock = time.monotonic()
            remote_completed_at = _utc_timestamp()
        finally:
            should_stop = owns_compute and (not keep_running or not allocation_ready)
            if should_stop:
                studio_stopped = _stop_owned_studio(studio)

    operation_completed_clock = time.monotonic()
    result = {
        "runtime_key": plan.runtime.key,
        "runtime_variant": plan.runtime.variant,
        "remote_exit_code": remote_exit_code,
        "failed_stage": stage if remote_exit_code != 0 else None,
        "local_output": str(local_output),
        "studio_started_here": started_here,
        "studio_adopted_pending": adopted_pending,
        "studio_stopped": studio_stopped,
        "allocation": allocation_result.to_dict() if allocation_result is not None else None,
        "timing": {
            "started_at": operation_started_at,
            "completed_at": _utc_timestamp(),
            "total_seconds": _duration(operation_started_clock, operation_completed_clock),
            "remote_started_at": remote_started_at,
            "remote_completed_at": remote_completed_at,
            "remote_seconds": (
                _duration(remote_started_clock, remote_completed_clock)
                if remote_started_clock is not None and remote_completed_clock is not None
                else None
            ),
        },
    }
    (local_output / "provision-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if remote_exit_code != 0:
        raise RuntimeError(
            f"Studio runtime {stage} failed with exit code {remote_exit_code}; see {local_output}"
        )
    return result


def execute_runtime_doctor(
    runtime: LightningRuntime,
    *,
    teamspace: str,
    studio: str,
    local_output: Path | None = None,
    studio_class: Any | None = None,
) -> dict[str, Any]:
    """Validate a provisioned runtime on an already-running Studio without changing it."""

    teamspace = _require_teamspace(teamspace)
    studio = _require_identifier(studio, "studio")
    if studio_class is None:
        try:
            from lightning_sdk import Studio
        except ImportError as error:
            raise RuntimeError(
                "Lightning support is not installed; use the 'lightning' optional dependency"
            ) from error
        studio_class = Studio
    remote = studio_class(name=studio, teamspace=teamspace, create_ok=False)
    if not _status_is_running(remote.status):
        raise RuntimeError("Studio must be running before runtime doctor can inspect it")
    command = "bash -lc " + shlex.quote(
        "; ".join(["set -euo pipefail", _runtime_probe_shell(runtime)])
    )
    output, exit_code = remote.run_with_exit_code(command)
    result = {
        "runtime_key": runtime.key,
        "runtime_variant": runtime.variant,
        "remote_exit_code": exit_code,
        "ready": exit_code == 0,
        "remote_output": output,
    }
    if local_output is not None:
        destination = local_output.resolve()
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "runtime-doctor.log").write_text(output, encoding="utf-8", errors="replace")
        (destination / "doctor-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if exit_code != 0:
        raise RuntimeError(
            f"Studio runtime {runtime.key} is not ready; run the matching provision command"
        )
    return result


def execute_studio_start(
    *,
    teamspace: str,
    studio: str,
    machine: str,
    allocation: LightningAllocationPolicy,
    interruptible: bool = False,
    max_runtime: int | None = None,
    local_output: Path | None = None,
    studio_class: Any | None = None,
    machine_class: Any | None = None,
) -> dict[str, Any]:
    """Start or reconcile a Studio while bounding local allocation wait."""

    teamspace = _require_teamspace(teamspace)
    studio = _require_identifier(studio, "studio")
    machine = _normalize_machine(machine)
    if max_runtime is not None and max_runtime <= 0:
        raise LightningConfigError("max_runtime must be positive")
    use_start_subprocess = studio_class is None or machine_class is None
    if studio_class is None or machine_class is None:
        try:
            from lightning_sdk import Machine, Studio
        except ImportError as error:
            raise RuntimeError(
                "Lightning support is not installed; use the 'lightning' optional dependency"
            ) from error
        studio_class = Studio
        machine_class = Machine

    destination = local_output.resolve() if local_output is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    remote = studio_class(name=studio, teamspace=teamspace, create_ok=True)
    if use_start_subprocess:
        start_attempt = lambda candidate, timeout, status_callback: _subprocess_start_attempt(
            remote,
            teamspace,
            studio,
            candidate,
            interruptible=interruptible,
            max_runtime=max_runtime,
            timeout_seconds=timeout,
            poll_seconds=allocation.retry_seconds,
            status_callback=status_callback,
        )
    else:
        start_attempt = lambda candidate, timeout, status_callback: _direct_start_attempt(
            remote,
            machine_class,
            candidate,
            interruptible=interruptible,
            max_runtime=max_runtime,
            timeout_seconds=timeout,
            status_callback=status_callback,
        )
    allocation_result = _ensure_studio_running(
        remote,
        requested_machine=machine,
        policy=allocation,
        reuse_running=True,
        start_attempt=start_attempt,
        journal_path=destination / "allocation.json" if destination is not None else None,
    )

    result = {
        "teamspace": teamspace,
        "studio": studio,
        "status": _status_name(remote.status),
        "machine": _actual_machine(remote) or allocation_result.selected_machine,
        "allocation": allocation_result.to_dict(),
    }
    if destination is not None:
        (destination / "studio-start-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def execute_lightning_plan(
    plan: LightningRunPlan,
    *,
    spec: SurgeryExperimentSpec,
    repo_root: Path,
    local_output: Path,
    collect: str = "summary",
    keep_running: bool = False,
    reuse_running: bool = False,
    studio_class: Any | None = None,
    machine_class: Any | None = None,
) -> dict[str, Any]:
    """Execute a blocking Studio run and release only compute started here."""
    if collect not in {"none", "summary", "capsule", "all"}:
        raise LightningConfigError("collect must be none, summary, capsule, or all")
    use_start_subprocess = studio_class is None or machine_class is None
    if studio_class is None or machine_class is None:
        try:
            from lightning_sdk import Machine, Studio
        except ImportError as error:
            raise RuntimeError(
                "Lightning support is not installed; use the 'lightning' optional dependency"
            ) from error
        studio_class = Studio
        machine_class = Machine

    local_output = local_output.resolve()
    local_output.mkdir(parents=True, exist_ok=True)
    (local_output / "launch-plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started_here = False
    adopted_pending = False
    owns_compute = False
    allocation_ready = False
    studio_stopped = False
    allocation_result: LightningAllocationResult | None = None
    remote_exit_code: int | None = None
    remote_output = ""
    downloaded: list[str] = []
    capsule_validation: dict[str, Any] | None = None
    previous_environment: dict[str, tuple[bool, str | None]] = {}
    operation_started_at = _utc_timestamp()
    operation_started_clock = time.monotonic()
    remote_started_at: str | None = None
    remote_completed_at: str | None = None
    remote_started_clock: float | None = None
    remote_completed_clock: float | None = None
    with tempfile.TemporaryDirectory(
        prefix=".abliteralus-lightning-", dir=local_output
    ) as temporary:
        bundle_path = Path(temporary) / f"{plan.run_id}.zip"
        bundle = build_runtime_bundle(repo_root, spec, bundle_path)
        (local_output / "bundle-manifest.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        studio = studio_class(
            name=plan.studio,
            teamspace=plan.teamspace,
            create_ok=True,
        )
        if use_start_subprocess:
            start_attempt = lambda machine, timeout, status_callback: _subprocess_start_attempt(
                studio,
                plan.teamspace,
                plan.studio,
                machine,
                interruptible=plan.interruptible,
                max_runtime=plan.max_runtime,
                timeout_seconds=timeout,
                poll_seconds=plan.allocation.retry_seconds,
                status_callback=status_callback,
            )
        else:
            start_attempt = lambda machine, timeout, status_callback: _direct_start_attempt(
                studio,
                machine_class,
                machine,
                interruptible=plan.interruptible,
                max_runtime=plan.max_runtime,
                timeout_seconds=timeout,
                status_callback=status_callback,
            )
        try:
            allocation_result = _ensure_studio_running(
                studio,
                requested_machine=plan.machine,
                policy=plan.allocation,
                reuse_running=reuse_running,
                start_attempt=start_attempt,
                journal_path=local_output / "allocation.json",
            )
            allocation_ready = True
            owns_compute = allocation_result.owns_compute
            started_here = allocation_result.started_here
            adopted_pending = allocation_result.mode == "adopted_pending"
            if use_start_subprocess and allocation_result.mode != "reused_running":
                studio = studio_class(
                    name=plan.studio,
                    teamspace=plan.teamspace,
                    create_ok=False,
                )
            remote_started_at = _utc_timestamp()
            remote_started_clock = time.monotonic()
            doctor_output, doctor_exit_code = studio.run_with_exit_code(plan.doctor_command)
            (local_output / "runtime-doctor.log").write_text(
                doctor_output, encoding="utf-8", errors="replace"
            )
            if doctor_exit_code != 0:
                raise RuntimeError(
                    f"Studio runtime {plan.runtime.key} is not ready; run provision before surgery"
                )
            previous_environment = _forward_environment(studio, plan.forwarded_environment)
            studio.upload_file(str(bundle_path), remote_path=plan.remote_bundle)
            remote_output, remote_exit_code = studio.run_with_exit_code(plan.command)
            (local_output / "studio-command.log").write_text(
                remote_output, encoding="utf-8", errors="replace"
            )
            if collect == "summary":
                downloaded = _download_summary(studio, plan, local_output)
            elif collect == "capsule":
                if remote_exit_code == 0:
                    downloaded, capsule_validation = _download_capsule(
                        studio, plan, local_output
                    )
                else:
                    downloaded = _download_summary(studio, plan, local_output)
            elif collect == "all":
                destination = local_output / "artifacts"
                destination.mkdir(parents=True, exist_ok=True)
                studio.download_folder(plan.remote_output, str(destination))
                downloaded = ["artifacts/"]
            remote_completed_clock = time.monotonic()
            remote_completed_at = _utc_timestamp()
        finally:
            try:
                _restore_environment(studio, previous_environment)
            finally:
                should_stop = owns_compute and (not keep_running or not allocation_ready)
                if should_stop:
                    studio_stopped = _stop_owned_studio(studio)

    operation_completed_clock = time.monotonic()
    result = {
        "run_id": plan.run_id,
        "runtime_key": plan.runtime.key,
        "runtime_variant": plan.runtime.variant,
        "remote_exit_code": remote_exit_code,
        "downloaded": downloaded,
        "capsule_validation": capsule_validation,
        "local_output": str(local_output),
        "studio_started_here": started_here,
        "studio_adopted_pending": adopted_pending,
        "studio_stopped": studio_stopped,
        "allocation": allocation_result.to_dict() if allocation_result is not None else None,
        "timing": {
            "started_at": operation_started_at,
            "completed_at": _utc_timestamp(),
            "total_seconds": _duration(operation_started_clock, operation_completed_clock),
            "remote_started_at": remote_started_at,
            "remote_completed_at": remote_completed_at,
            "remote_seconds": (
                _duration(remote_started_clock, remote_completed_clock)
                if remote_started_clock is not None and remote_completed_clock is not None
                else None
            ),
        },
    }
    (local_output / "lightning-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if remote_exit_code != 0:
        raise RuntimeError(
            f"remote surgery failed with exit code {remote_exit_code}; see {local_output}"
        )
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abliteralus-lightning",
        description="Provision, inspect, plan, or run ABLITERALUS work in a Lightning Studio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_target(
        command: argparse.ArgumentParser,
        *,
        machine: str | None = None,
        remote_root: bool = True,
    ) -> None:
        command.add_argument(
            "--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"), required=False
        )
        command.add_argument("--studio", required=True)
        if machine is not None:
            command.add_argument("--machine", default=machine)
            command.add_argument("--max-runtime", type=_positive_integer)
        if remote_root:
            command.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)

    def add_allocation(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--allocation-timeout",
            type=_positive_float,
            default=DEFAULT_ALLOCATION_TIMEOUT_SECONDS,
            help="maximum local seconds for capacity retries and Pending reconciliation",
        )
        command.add_argument(
            "--allocation-retry",
            type=_positive_float,
            default=DEFAULT_ALLOCATION_RETRY_SECONDS,
            help="seconds between capacity checks",
        )
        command.add_argument(
            "--fallback-machine",
            action="append",
            default=[],
            help="explicit ordered fallback; repeat for more than one",
        )
        command.add_argument(
            "--pending-policy",
            choices=("fail", "adopt", "stop"),
            default="fail",
            help="handling for a Studio already Pending when the command starts",
        )

    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        add_target(command, machine="L40S")
        command.add_argument("--run-id")
        command.add_argument("--forward-env", action="append", default=[])
        command.add_argument("--interruptible", action="store_true")
        command.add_argument("--skip-gguf", action="store_true")
        add_allocation(command)
        if name == "run":
            command.add_argument("--local-output", type=Path)
            command.add_argument(
                "--collect",
                choices=("none", "summary", "capsule", "all"),
                default="summary",
            )
            command.add_argument("--keep-running", action="store_true")
            command.add_argument("--reuse-running", action="store_true")

    provision = subparsers.add_parser(
        "provision", help="install the pinned tool and materialize a lock-addressed runtime"
    )
    provision.add_argument("--config", required=True, type=Path)
    add_target(provision, machine="CPU-4")
    provision.add_argument("--interruptible", action="store_true")
    provision.add_argument("--skip-gguf", action="store_true")
    add_allocation(provision)
    provision.add_argument("--local-output", type=Path)
    provision.add_argument("--keep-running", action="store_true")
    provision.add_argument("--reuse-running", action="store_true")
    provision.add_argument("--dry-run", action="store_true")

    start = subparsers.add_parser(
        "start", help="start or reconcile a Studio with bounded capacity retries"
    )
    add_target(start, machine="L40S", remote_root=False)
    start.add_argument("--interruptible", action="store_true")
    start.add_argument("--local-output", type=Path)
    add_allocation(start)

    doctor = subparsers.add_parser(
        "doctor", help="inspect one provisioned runtime on an already-running Studio"
    )
    doctor.add_argument("--config", required=True, type=Path)
    add_target(doctor)
    doctor.add_argument("--skip-gguf", action="store_true")
    doctor.add_argument("--local-output", type=Path)
    return parser


def _start_once_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teamspace", required=True)
    parser.add_argument("--studio", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--interruptible", action="store_true")
    parser.add_argument("--max-runtime", type=_positive_integer)
    return parser


def _start_once_main(argv: Sequence[str]) -> int:
    args = _start_once_parser().parse_args(argv)
    from lightning_sdk import Machine, Studio
    from lightning_sdk.exceptions import NotSupportedError, OutOfCapacityError

    remote = Studio(name=args.studio, teamspace=args.teamspace, create_ok=False)
    if not _status_is_stopped(remote.status):
        return _START_STATE_CHANGED
    try:
        _start_studio(
            remote,
            Machine,
            _normalize_machine(args.machine),
            interruptible=args.interruptible,
            max_runtime=args.max_runtime,
        )
    except OutOfCapacityError:
        return _START_OUT_OF_CAPACITY
    except NotSupportedError:
        return _START_NOT_SUPPORTED
    return 0


def _main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["_start-once"]:
        return _start_once_main(arguments[1:])
    args = _parser().parse_args(arguments)
    if not args.teamspace:
        raise LightningConfigError(
            "--teamspace OWNER/TEAMSPACE is required (or set LIGHTNING_TEAMSPACE)"
        )
    repo_root = _repo_root()
    if args.command in {"plan", "run"}:
        spec = load_experiment_spec(args.config)
        plan = build_lightning_plan(
            spec,
            teamspace=args.teamspace,
            studio=args.studio,
            machine=args.machine,
            repo_root=repo_root,
            run_id=args.run_id,
            remote_root=args.remote_root,
            forwarded_environment=args.forward_env,
            interruptible=args.interruptible,
            max_runtime=args.max_runtime,
            skip_gguf=args.skip_gguf,
            allocation_timeout_seconds=args.allocation_timeout,
            allocation_retry_seconds=args.allocation_retry,
            fallback_machines=args.fallback_machine,
            pending_policy=args.pending_policy,
        )
        if args.command == "plan":
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        local_output = args.local_output or Path("outputs/lightning") / plan.run_id
        result = execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=repo_root,
            local_output=local_output,
            collect=args.collect,
            keep_running=args.keep_running,
            reuse_running=args.reuse_running,
        )
    elif args.command == "provision":
        spec = load_experiment_spec(args.config)
        provision_plan = build_provision_plan(
            spec,
            repo_root=repo_root,
            teamspace=args.teamspace,
            studio=args.studio,
            machine=args.machine,
            remote_root=args.remote_root,
            interruptible=args.interruptible,
            max_runtime=args.max_runtime,
            skip_gguf=args.skip_gguf,
            allocation_timeout_seconds=args.allocation_timeout,
            allocation_retry_seconds=args.allocation_retry,
            fallback_machines=args.fallback_machine,
            pending_policy=args.pending_policy,
        )
        if args.dry_run:
            print(json.dumps(provision_plan.to_dict(), indent=2, sort_keys=True))
            return 0
        local_output = (
            args.local_output or Path("outputs/lightning/provision") / provision_plan.runtime.key
        )
        result = execute_provision_plan(
            provision_plan,
            spec=spec,
            repo_root=repo_root,
            local_output=local_output,
            keep_running=args.keep_running,
            reuse_running=args.reuse_running,
        )
    elif args.command == "start":
        allocation = _allocation_policy(
            primary_machine=args.machine,
            allocation_timeout_seconds=args.allocation_timeout,
            allocation_retry_seconds=args.allocation_retry,
            fallback_machines=args.fallback_machine,
            pending_policy=args.pending_policy,
        )
        result = execute_studio_start(
            teamspace=args.teamspace,
            studio=args.studio,
            machine=args.machine,
            allocation=allocation,
            interruptible=args.interruptible,
            max_runtime=args.max_runtime,
            local_output=args.local_output,
        )
    else:
        spec = load_experiment_spec(args.config)
        runtime = build_runtime_layout(
            repo_root,
            spec,
            remote_root=args.remote_root,
            skip_gguf=args.skip_gguf,
        )
        result = execute_runtime_doctor(
            runtime,
            teamspace=args.teamspace,
            studio=args.studio,
            local_output=args.local_output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(argv)
    except LightningConfigError as error:
        print(f"abliteralus-lightning: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        # SDK exceptions can embed response headers. Keep provider detail out of
        # credential-bearing console output; durable operation logs remain local.
        print(
            f"abliteralus-lightning: operation failed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
