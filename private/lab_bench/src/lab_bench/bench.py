"""Composition layer for local surgery, Lightning control, and OpenSSH."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from secret_manager import SecretManager
from surgery_artifacts import ArtifactRegistry, rehydrate_capsule, validate_capsule

from .config import LabBenchConfig
from .errors import LabBenchError


class LabBench:
    """Keep secret delivery separate from experiment and transport mechanics."""

    def __init__(self, config: LabBenchConfig, *, manager: SecretManager | None = None) -> None:
        self.config = config
        self._manager = manager

    @property
    def manager(self) -> SecretManager:
        if self._manager is None:
            self._manager = SecretManager.from_config(self.config.secret_config)
        return self._manager

    def inventory(self) -> dict:
        """Return non-secret bench metadata without loading the secret mapping."""

        return {
            "schema_version": self.config.schema_version,
            "config": str(self.config.path),
            "repository": str(self.config.repository),
            "python": str(self.config.python),
            "secret_config": str(self.config.secret_config),
            "profiles": {
                "local_surgery": self.config.profiles.local_surgery,
                "local_inference": self.config.profiles.local_inference,
                "lightning_control": self.config.profiles.lightning_control,
                "lightning_surgery": self.config.profiles.lightning_surgery,
                "lightning_inference": self.config.profiles.lightning_inference,
            },
            "lightning": {
                "teamspace": self.config.lightning.teamspace,
                "studio": self.config.lightning.studio,
                "control_machine": self.config.lightning.control_machine,
                "surgery_machine": self.config.lightning.surgery_machine,
                "inference_machine": self.config.lightning.inference_machine,
                "remote_root": self.config.lightning.remote_root,
                "forward_environment": list(self.config.lightning.forward_environment),
                "allocation_timeout_seconds": (
                    self.config.lightning.allocation_timeout_seconds
                ),
                "allocation_retry_seconds": self.config.lightning.allocation_retry_seconds,
                "pending_policy": self.config.lightning.pending_policy,
                "surgery_fallback_machines": list(
                    self.config.lightning.surgery_fallback_machines
                ),
                "inference_fallback_machines": list(
                    self.config.lightning.inference_fallback_machines
                ),
                "control_max_runtime_seconds": (
                    self.config.lightning.control_max_runtime_seconds
                ),
                "surgery_max_runtime_seconds": (
                    self.config.lightning.surgery_max_runtime_seconds
                ),
                "inference_max_runtime_seconds": (
                    self.config.lightning.inference_max_runtime_seconds
                ),
            },
            "artifacts": {"registry": str(self.config.artifacts.registry)},
            "ssh": {
                "executable": self.config.ssh.executable,
                "destination": self.config.ssh.destination,
                "port": self.config.ssh.port,
                "identity_file": (
                    str(self.config.ssh.identity_file)
                    if self.config.ssh.identity_file is not None
                    else None
                ),
            },
        }

    def local_surgery(self, arguments: Sequence[str]) -> int:
        surgery_arguments = _arguments(arguments, "local-surgery requires a surgery command")
        module = (
            "surgery_artifacts.integration"
            if surgery_arguments[0] == "run"
            else "abliteralus.surgery_bench"
        )
        command = self._repository_python(module, surgery_arguments)
        environment = self._repository_environment()
        # Postprocessing, smoke tests, and explicitly offline runs do not need Hub access.
        # This conservative literal check can withhold Hub credentials if
        # ``--offline`` appears as another option's value. That safe-direction
        # false positive is preferable to injecting a token into an offline run.
        needs_hub = surgery_arguments[0] == "run" and "--offline" not in surgery_arguments
        if needs_hub:
            return self.manager.run(
                self.config.profiles.local_surgery,
                command,
                environ=environment,
                cwd=self.config.repository,
            ).returncode
        return self._direct(command, environ=environment)

    def local_inference(self, command: Sequence[str]) -> int:
        child = _arguments(command, "local-inference requires a child command")
        return self.manager.run(
            self.config.profiles.local_inference,
            child,
            cwd=self.config.repository,
        ).returncode

    def with_secrets(self, profile: str, command: Sequence[str]) -> int:
        child = _arguments(command, "with-secrets requires a child command")
        return self.manager.run(profile, child, cwd=self.config.repository).returncode

    def lightning_surgery(self, arguments: Sequence[str]) -> int:
        self._require_lightning_target()
        surgery_arguments = _arguments(
            arguments, "lightning-surgery requires an operation and arguments"
        )
        operation = surgery_arguments[0]
        if operation not in {"plan", "run", "provision", "doctor"}:
            raise LabBenchError("lightning-surgery begins with plan, run, provision, or doctor")
        expanded = list(surgery_arguments)
        _append_default(expanded, "--teamspace", self.config.lightning.teamspace)
        _append_default(expanded, "--studio", self.config.lightning.studio)
        _append_default(expanded, "--remote-root", self.config.lightning.remote_root)
        if operation in {"plan", "run", "provision"}:
            self._append_allocation_defaults(expanded)
        if operation in {"plan", "run"}:
            _append_default(expanded, "--machine", self.config.lightning.surgery_machine)
            if self.config.lightning.surgery_max_runtime_seconds is not None:
                _append_default(
                    expanded,
                    "--max-runtime",
                    str(self.config.lightning.surgery_max_runtime_seconds),
                )
            self._append_fallback_defaults(
                expanded, self.config.lightning.surgery_fallback_machines
            )
            forwarded = set(_option_values(expanded, "--forward-env"))
            for name in self.config.lightning.forward_environment:
                if name not in forwarded:
                    expanded.extend(["--forward-env", name])
        elif operation == "provision":
            _append_default(expanded, "--machine", self.config.lightning.control_machine)
            if self.config.lightning.control_max_runtime_seconds is not None:
                _append_default(
                    expanded,
                    "--max-runtime",
                    str(self.config.lightning.control_max_runtime_seconds),
                )
        command = self._repository_python("abliteralus.lightning_surgery", expanded)
        if operation == "plan" or (operation == "provision" and "--dry-run" in expanded):
            return self._direct(command)
        profile = (
            self.config.profiles.lightning_surgery
            if operation == "run"
            else self.config.profiles.lightning_control
        )
        return self.manager.run(
            profile,
            command,
            environ=self._repository_environment(),
            cwd=self.config.repository,
        ).returncode

    def artifact_verify(self, capsule: str | Path) -> dict[str, object]:
        validation = validate_capsule(capsule)
        return {
            "path": str(validation.path),
            "surgery_id": validation.surgery_id,
            "operation_count": validation.operation_count,
            "file_count": validation.file_count,
            "total_bytes": validation.total_bytes,
        }

    def artifact_register(
        self,
        capsule: str | Path,
        *,
        ref: str | None = None,
    ) -> dict[str, object]:
        entry = ArtifactRegistry(self.config.artifacts.registry).add(capsule, ref=ref)
        return {
            "surgery_id": entry.surgery_id,
            "path": str(entry.path),
            "ref": entry.ref,
            "created": entry.created,
        }

    def artifact_resolve(self, name: str) -> Path:
        return ArtifactRegistry(self.config.artifacts.registry).resolve(name)

    def artifact_rehydrate(
        self,
        capsule_or_ref: str | Path,
        *,
        base: str | Path,
        output: str | Path,
    ) -> Path:
        candidate = Path(capsule_or_ref).expanduser()
        capsule = candidate.resolve() if candidate.is_dir() else self.artifact_resolve(str(capsule_or_ref))
        return rehydrate_capsule(capsule, base, output)

    def studio_provision(
        self,
        *,
        experiment_config: str | Path,
        machine: str | None = None,
        interruptible: bool = False,
        max_runtime: int | None = None,
        skip_gguf: bool = False,
        local_output: str | Path | None = None,
        keep_running: bool = False,
        reuse_running: bool = False,
        dry_run: bool = False,
    ) -> int:
        arguments = ["provision", "--config", str(experiment_config)]
        if machine is not None:
            arguments.extend(["--machine", machine])
        if interruptible:
            arguments.append("--interruptible")
        if max_runtime is not None:
            arguments.extend(["--max-runtime", str(max_runtime)])
        if skip_gguf:
            arguments.append("--skip-gguf")
        if local_output is not None:
            arguments.extend(["--local-output", str(local_output)])
        if keep_running:
            arguments.append("--keep-running")
        if reuse_running:
            arguments.append("--reuse-running")
        if dry_run:
            arguments.append("--dry-run")
        return self.lightning_surgery(arguments)

    def studio_doctor(
        self,
        *,
        experiment_config: str | Path,
        skip_gguf: bool = False,
        local_output: str | Path | None = None,
    ) -> int:
        arguments = ["doctor", "--config", str(experiment_config)]
        if skip_gguf:
            arguments.append("--skip-gguf")
        if local_output is not None:
            arguments.extend(["--local-output", str(local_output)])
        return self.lightning_surgery(arguments)

    def studio_status(self) -> int:
        return self._studio_control("control", ["status"])

    def studio_start(
        self,
        *,
        purpose: str,
        machine: str | None = None,
        interruptible: bool = False,
        max_runtime: int | None = None,
    ) -> int:
        self._require_lightning_target()
        selected_machine = machine or self._machine_for_purpose(purpose)
        arguments = [
            "start",
            "--teamspace",
            self.config.lightning.teamspace,
            "--studio",
            self.config.lightning.studio,
            "--machine",
            selected_machine,
        ]
        self._append_allocation_defaults(arguments)
        self._append_fallback_defaults(arguments, self._fallback_machines_for_purpose(purpose))
        if interruptible:
            arguments.append("--interruptible")
        selected_max_runtime = (
            max_runtime if max_runtime is not None else self._max_runtime_for_purpose(purpose)
        )
        if selected_max_runtime is not None:
            arguments.extend(["--max-runtime", str(selected_max_runtime)])
        command = self._repository_python("abliteralus.lightning_surgery", arguments)
        return self.manager.run(
            self.config.profiles.lightning_control,
            command,
            cwd=self.config.repository,
        ).returncode

    def studio_stop(self) -> int:
        return self._studio_control("control", ["stop"])

    def studio_ports(self, additions: Sequence[int]) -> int:
        arguments = ["ports"]
        for port in additions:
            arguments.extend(["--add", str(port)])
        return self._studio_control("control", arguments)

    def studio_exec(self, command: Sequence[str], *, detached: bool, wait_seconds: float) -> int:
        remote = _arguments(command, "studio execution requires a remote command")
        operation = "detach" if detached else "exec"
        arguments = [operation, "--remote-command", shlex.join(remote)]
        if detached:
            arguments.extend(["--wait-seconds", str(wait_seconds)])
        for name in self.config.lightning.forward_environment:
            arguments.extend(["--forward-env", name])
        return self._studio_control("inference", arguments)

    def ssh(
        self,
        remote_command: Sequence[str],
        *,
        destination: str | None = None,
    ) -> int:
        selected_destination = destination or self.config.ssh.destination
        if not selected_destination:
            raise LabBenchError(
                "SSH destination is unset; copy the Studio SSH destination into lab.local.toml"
            )
        executable = _resolve_executable(self.config.ssh.executable, "SSH")
        arguments = [executable]
        if self.config.ssh.port != 22:
            arguments.extend(["-p", str(self.config.ssh.port)])
        if self.config.ssh.identity_file is not None:
            if not self.config.ssh.identity_file.is_file():
                raise LabBenchError("configured SSH identity file is unavailable")
            arguments.extend(["-i", str(self.config.ssh.identity_file)])
        arguments.append(selected_destination)
        remote = _strip_separator(remote_command)
        if remote:
            arguments.append(shlex.join(remote))
        try:
            completed = subprocess.run(
                arguments,
                cwd=self.config.repository,
                check=False,
                close_fds=True,
            )
        except OSError as error:
            raise LabBenchError("SSH could not start") from error
        return completed.returncode

    def _studio_control(self, purpose: str, operation: Sequence[str]) -> int:
        self._require_lightning_target()
        command = [
            sys.executable,
            "-m",
            "lab_bench._studio_worker",
            "--teamspace",
            self.config.lightning.teamspace,
            "--studio",
            self.config.lightning.studio,
            *operation,
        ]
        profile = (
            self.config.profiles.lightning_inference
            if purpose == "inference"
            else self.config.profiles.lightning_control
        )
        return self.manager.run(profile, command, cwd=self.config.repository).returncode

    def _require_lightning_target(self) -> None:
        segments = self.config.lightning.teamspace.split("/", 1)
        placeholders = {
            "OWNER",
            "TEAMSPACE",
            "SET_ME",
            "REPLACE_WITH_OWNER",
            "REPLACE_WITH_TEAMSPACE",
        }
        if len(segments) != 2 or any(segment in placeholders for segment in segments):
            raise LabBenchError(
                "Lightning teamspace is not configured; replace SET_ME/SET_ME in lab.local.toml"
            )

    def _machine_for_purpose(self, purpose: str) -> str:
        if purpose == "control":
            return self.config.lightning.control_machine
        if purpose == "surgery":
            return self.config.lightning.surgery_machine
        if purpose == "inference":
            return self.config.lightning.inference_machine
        raise LabBenchError(f"unknown Studio purpose: {purpose}")

    def _fallback_machines_for_purpose(self, purpose: str) -> tuple[str, ...]:
        if purpose == "surgery":
            return self.config.lightning.surgery_fallback_machines
        if purpose == "inference":
            return self.config.lightning.inference_fallback_machines
        if purpose == "control":
            return ()
        raise LabBenchError(f"unknown Studio purpose: {purpose}")

    def _max_runtime_for_purpose(self, purpose: str) -> int | None:
        if purpose == "control":
            return self.config.lightning.control_max_runtime_seconds
        if purpose == "surgery":
            return self.config.lightning.surgery_max_runtime_seconds
        if purpose == "inference":
            return self.config.lightning.inference_max_runtime_seconds
        raise LabBenchError(f"unknown Studio purpose: {purpose}")

    def _append_allocation_defaults(self, arguments: list[str]) -> None:
        _append_default(
            arguments,
            "--allocation-timeout",
            str(self.config.lightning.allocation_timeout_seconds),
        )
        _append_default(
            arguments,
            "--allocation-retry",
            str(self.config.lightning.allocation_retry_seconds),
        )
        _append_default(arguments, "--pending-policy", self.config.lightning.pending_policy)

    @staticmethod
    def _append_fallback_defaults(arguments: list[str], machines: Sequence[str]) -> None:
        if _has_option(arguments, "--fallback-machine"):
            return
        for machine in machines:
            arguments.extend(["--fallback-machine", machine])

    def _repository_python(self, module: str, arguments: Sequence[str]) -> list[str]:
        if not self.config.repository.is_dir():
            raise LabBenchError("configured ABLITERALUS repository is unavailable")
        if not self.config.python.is_file():
            raise LabBenchError("configured ABLITERALUS Python executable is unavailable")
        return [str(self.config.python), "-m", module, *arguments]

    def _direct(
        self,
        command: Sequence[str],
        *,
        environ: dict[str, str] | None = None,
    ) -> int:
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.repository,
                env=environ,
                check=False,
                close_fds=True,
            )
        except OSError as error:
            raise LabBenchError("lab child command could not start") from error
        return completed.returncode

    def _repository_environment(self) -> dict[str, str]:
        environment = dict(os.environ)
        source = self.config.repository / "private" / "surgery_artifacts" / "src"
        existing = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(source) if not existing else os.pathsep.join((str(source), existing))
        )
        return environment


def _arguments(arguments: Sequence[str], message: str) -> list[str]:
    values = _strip_separator(arguments)
    if not values:
        raise LabBenchError(message)
    return values


def _strip_separator(arguments: Sequence[str]) -> list[str]:
    values = [str(argument) for argument in arguments]
    if values and values[0] == "--":
        values.pop(0)
    return values


def _has_option(arguments: Sequence[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in arguments)


def _append_default(arguments: list[str], option: str, value: str) -> None:
    if not _has_option(arguments, option):
        arguments.extend([option, value])


def _option_values(arguments: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, argument in enumerate(arguments):
        if argument.startswith(f"{option}="):
            values.append(argument.split("=", 1)[1])
        elif argument == option and index + 1 < len(arguments):
            values.append(arguments[index + 1])
    return values


def _resolve_executable(configured: str, label: str) -> str:
    path = Path(configured).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        resolved = str(path.resolve())
    else:
        resolved = shutil.which(configured) or ""
    if not resolved or not Path(resolved).is_file():
        raise LabBenchError(f"{label} executable is unavailable")
    if os.name != "nt" and not os.access(resolved, os.X_OK):
        raise LabBenchError(f"{label} executable is not executable")
    return resolved
