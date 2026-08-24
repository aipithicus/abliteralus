"""Lightning AI Studio orchestration for surgery experiments.

The launcher uploads a narrow, reproducible runtime bundle, executes the same
``surgery_bench`` contract used locally, collects requested artifacts, and
releases compute that it started. It deliberately excludes ``private/``, Git
metadata, caches, tests, and unrelated worktree content from the upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from abliteralus.surgery_bench import SurgeryExperimentSpec, load_experiment_spec


UV_VERSION = "0.11.23"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_MACHINE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PATH_SEGMENT = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_RUNTIME_ROOTS = ("abliteralus/",)
_RUNTIME_FILES = {"pyproject.toml", "uv.lock", "README.md"}


class LightningConfigError(ValueError):
    """Raised before any paid Lightning resource is started."""


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
            "excludes": [".git/", "private/", "ci/", "tests/", ".codex/", ".venv/"],
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
    command: str
    forwarded_environment: tuple[str, ...]
    interruptible: bool
    skip_gguf: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment,
            "run_id": self.run_id,
            "teamspace": self.teamspace,
            "studio": self.studio,
            "machine": self.machine,
            "interruptible": self.interruptible,
            "remote_root": self.remote_root,
            "remote_bundle": self.remote_bundle,
            "remote_source": self.remote_source,
            "remote_output": self.remote_output,
            "forwarded_environment": list(self.forwarded_environment),
            "skip_gguf": self.skip_gguf,
            "command": self.command,
        }


def build_lightning_plan(
    spec: SurgeryExperimentSpec,
    *,
    teamspace: str,
    studio: str,
    machine: str,
    run_id: str | None = None,
    remote_root: str = ".abliteralus/runs",
    forwarded_environment: Sequence[str] = (),
    interruptible: bool = False,
    skip_gguf: bool = False,
) -> LightningRunPlan:
    """Build a shell-safe, credential-free launch plan without contacting Lightning."""
    if Path(spec.model["source"]).expanduser().exists():
        raise LightningConfigError(
            "Lightning experiments require a remote-addressable OWNER/MODEL source; "
            "local checkpoint directories are not included in the runtime bundle"
        )
    teamspace = _require_teamspace(teamspace)
    studio = _require_identifier(studio, "studio")
    if _MACHINE.fullmatch(machine) is None:
        raise LightningConfigError("machine contains unsupported characters")
    run_id = _require_identifier(
        run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), "run_id"
    )
    remote_root = _require_remote_root(remote_root)
    env_names: list[str] = []
    for name in forwarded_environment:
        if _ENV_NAME.fullmatch(name) is None:
            raise LightningConfigError(f"invalid environment variable name: {name!r}")
        if name not in env_names:
            env_names.append(name)

    remote_run = PurePosixPath(remote_root) / run_id
    remote_bundle = str(PurePosixPath(remote_root) / "bundles" / f"{run_id}.zip")
    remote_source = str(remote_run / "source")
    remote_output_root = str(remote_run / "outputs")
    remote_output = str(PurePosixPath(remote_output_root) / spec.name / run_id)
    home = '"$HOME"'

    surgery_arguments = [
        ".venv/bin/python",
        "-m",
        "abliteralus.surgery_bench",
        "run",
        "--config",
        "experiment.yaml",
        "--output-root",
        f"$HOME/{remote_output_root}",
        "--run-id",
        run_id,
    ]
    if skip_gguf:
        surgery_arguments.append("--skip-gguf")
    surgery_command = " ".join(
        token if token.startswith("$HOME/") else shlex.quote(token) for token in surgery_arguments
    )
    sync_command = "python -m uv sync --frozen --no-dev"
    if spec.gguf_enabled and not skip_gguf:
        sync_command += " --extra gguf"
    commands = [
        "set -euo pipefail",
        f"mkdir -p {home}/{shlex.quote(remote_source)}",
        (
            f"python -m zipfile -e {home}/{shlex.quote(remote_bundle)} "
            f"{home}/{shlex.quote(remote_source)}"
        ),
        f"cd {home}/{shlex.quote(remote_source)}",
        f"python -m pip install --user --disable-pip-version-check uv=={UV_VERSION}",
        sync_command,
        surgery_command,
    ]
    command = "bash -lc " + shlex.quote("; ".join(commands))
    return LightningRunPlan(
        experiment=spec.name,
        run_id=run_id,
        teamspace=teamspace,
        studio=studio,
        machine=machine.upper(),
        remote_root=remote_root,
        remote_bundle=remote_bundle,
        remote_source=remote_source,
        remote_output=remote_output,
        command=command,
        forwarded_environment=tuple(env_names),
        interruptible=interruptible,
        skip_gguf=skip_gguf,
    )


def _machine_value(machine_class: Any, name: str) -> Any:
    if hasattr(machine_class, "from_str"):
        return machine_class.from_str(name)
    try:
        return getattr(machine_class, name.upper())
    except AttributeError as error:
        raise LightningConfigError(f"unknown Lightning machine: {name}") from error


def _status_is_running(status: object) -> bool:
    return str(status).rsplit(".", 1)[-1].lower() == "running"


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
    if collect not in {"none", "summary", "all"}:
        raise LightningConfigError("collect must be none, summary, or all")
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
    remote_exit_code: int | None = None
    remote_output = ""
    downloaded: list[str] = []
    previous_environment: dict[str, tuple[bool, str | None]] = {}
    with tempfile.TemporaryDirectory(prefix="abliteralus-lightning-") as temporary:
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
        already_running = _status_is_running(studio.status)
        if already_running and not reuse_running:
            raise RuntimeError(
                "Studio is already running; pass --reuse-running to use it without "
                "changing or automatically stopping its machine"
            )
        if not already_running:
            studio.start(
                machine=_machine_value(machine_class, plan.machine),
                interruptible=plan.interruptible,
            )
            started_here = True

        try:
            previous_environment = _forward_environment(studio, plan.forwarded_environment)
            studio.upload_file(str(bundle_path), remote_path=plan.remote_bundle)
            remote_output, remote_exit_code = studio.run_with_exit_code(plan.command)
            (local_output / "studio-command.log").write_text(
                remote_output, encoding="utf-8", errors="replace"
            )
            if collect == "summary":
                downloaded = _download_summary(studio, plan, local_output)
            elif collect == "all":
                destination = local_output / "artifacts"
                destination.mkdir(parents=True, exist_ok=True)
                studio.download_folder(plan.remote_output, str(destination))
                downloaded = ["artifacts/"]
        finally:
            try:
                _restore_environment(studio, previous_environment)
            finally:
                if started_here and not keep_running:
                    studio.stop()

    result = {
        "run_id": plan.run_id,
        "remote_exit_code": remote_exit_code,
        "downloaded": downloaded,
        "local_output": str(local_output),
        "studio_started_here": started_here,
        "studio_stopped": started_here and not keep_running,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abliteralus-lightning",
        description="Plan or run a reproducible ABLITERALUS surgery in a Lightning Studio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument(
            "--teamspace", default=os.environ.get("LIGHTNING_TEAMSPACE"), required=False
        )
        command.add_argument("--studio", required=True)
        command.add_argument("--machine", default="L40S")
        command.add_argument("--run-id")
        command.add_argument("--remote-root", default=".abliteralus/runs")
        command.add_argument("--forward-env", action="append", default=[])
        command.add_argument("--interruptible", action="store_true")
        command.add_argument("--skip-gguf", action="store_true")
        if name == "run":
            command.add_argument("--local-output", type=Path)
            command.add_argument("--collect", choices=("none", "summary", "all"), default="summary")
            command.add_argument("--keep-running", action="store_true")
            command.add_argument("--reuse-running", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.teamspace:
        raise LightningConfigError(
            "--teamspace OWNER/TEAMSPACE is required (or set LIGHTNING_TEAMSPACE)"
        )
    spec = load_experiment_spec(args.config)
    plan = build_lightning_plan(
        spec,
        teamspace=args.teamspace,
        studio=args.studio,
        machine=args.machine,
        run_id=args.run_id,
        remote_root=args.remote_root,
        forwarded_environment=args.forward_env,
        interruptible=args.interruptible,
        skip_gguf=args.skip_gguf,
    )
    if args.command == "plan":
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    local_output = args.local_output or Path("outputs/lightning") / plan.run_id
    result = execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=_repo_root(),
        local_output=local_output,
        collect=args.collect,
        keep_running=args.keep_running,
        reuse_running=args.reuse_running,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
