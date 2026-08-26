"""Strict parsing for the lab bench's non-secret machine configuration."""

from __future__ import annotations

import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .errors import LabBenchError

CONFIG_ENVIRONMENT_VARIABLE = "LAB_BENCH_CONFIG"
_PROFILE_ALIAS = re.compile(r"^[a-z][a-z0-9_-]*$")
_MACHINE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TEAMSPACE = re.compile(r"^[^/\s]+/[^/\s]+$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_REMOTE_PATH_SEGMENT = re.compile(r"^\.?[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


@dataclass(frozen=True, slots=True)
class ProfileNames:
    local_surgery: str
    local_inference: str
    lightning_control: str
    lightning_surgery: str
    lightning_inference: str


@dataclass(frozen=True, slots=True)
class LightningDefaults:
    teamspace: str
    studio: str
    control_machine: str
    surgery_machine: str
    inference_machine: str
    remote_root: str
    forward_environment: tuple[str, ...]
    allocation_timeout_seconds: float
    allocation_retry_seconds: float
    pending_policy: str
    surgery_fallback_machines: tuple[str, ...]
    inference_fallback_machines: tuple[str, ...]
    control_max_runtime_seconds: int | None
    surgery_max_runtime_seconds: int | None
    inference_max_runtime_seconds: int | None


@dataclass(frozen=True, slots=True)
class SshDefaults:
    executable: str
    destination: str | None
    port: int
    identity_file: Path | None


@dataclass(frozen=True, slots=True)
class ArtifactDefaults:
    registry: Path


@dataclass(frozen=True, slots=True)
class LabBenchConfig:
    path: Path
    repository: Path
    python: Path
    secret_config: Path
    profiles: ProfileNames
    lightning: LightningDefaults
    artifacts: ArtifactDefaults
    ssh: SshDefaults
    schema_version: int = 1


def discover_config_path(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    candidate = str(path).strip() if path is not None else ""
    if not candidate:
        candidate = environment.get(CONFIG_ENVIRONMENT_VARIABLE, "").strip()
    if candidate:
        return Path(candidate).expanduser().resolve()
    app_data = environment.get("APPDATA", "").strip()
    if app_data:
        return (Path(app_data) / "Aipithicus" / "lab-bench.toml").resolve()
    xdg_config = environment.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg_config).expanduser() if xdg_config else Path.home() / ".config"
    return (root / "aipithicus" / "lab-bench.toml").resolve()


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> LabBenchConfig:
    resolved = discover_config_path(path, environ=environ)
    try:
        with resolved.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as error:
        raise LabBenchError(f"lab-bench config does not exist: {resolved}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise LabBenchError(f"lab-bench config is unreadable or invalid: {resolved}") from error
    return parse_config(raw, path=resolved)


def parse_config(raw: Mapping[str, Any], *, path: Path | None = None) -> LabBenchConfig:
    table = _table(raw, "config")
    _unknown(
        table,
        {
            "schema_version",
            "repository",
            "python",
            "secret_config",
            "profiles",
            "lightning",
            "artifacts",
            "ssh",
        },
        "config",
    )
    if table.get("schema_version") != 1:
        raise LabBenchError("schema_version must be 1")
    config_path = (path or Path("lab-bench.toml")).resolve()
    base = config_path.parent
    repository = _resolved_path(_string(table, "repository", "config"), base)
    python = _resolved_path(_string(table, "python", "config"), base)
    secret_config = _resolved_path(_string(table, "secret_config", "config"), base)
    profiles = _parse_profiles(_required_table(table, "profiles", "config"))
    lightning = _parse_lightning(_required_table(table, "lightning", "config"))
    artifacts = _parse_artifacts(table.get("artifacts", {}), base, repository)
    ssh = _parse_ssh(_required_table(table, "ssh", "config"), base)
    return LabBenchConfig(
        path=config_path,
        repository=repository,
        python=python,
        secret_config=secret_config,
        profiles=profiles,
        lightning=lightning,
        artifacts=artifacts,
        ssh=ssh,
    )


def _parse_profiles(table: Mapping[str, Any]) -> ProfileNames:
    keys = {
        "local_surgery",
        "local_inference",
        "lightning_control",
        "lightning_surgery",
        "lightning_inference",
    }
    _unknown(table, keys, "profiles")
    values = {key: _string(table, key, "profiles") for key in keys}
    for key, value in values.items():
        if _PROFILE_ALIAS.fullmatch(value) is None:
            raise LabBenchError(f"profiles.{key} must be a lowercase profile alias")
    return ProfileNames(**values)


def _parse_lightning(table: Mapping[str, Any]) -> LightningDefaults:
    keys = {
        "teamspace",
        "studio",
        "control_machine",
        "surgery_machine",
        "inference_machine",
        "remote_root",
        "forward_environment",
        "allocation_timeout_seconds",
        "allocation_retry_seconds",
        "pending_policy",
        "surgery_fallback_machines",
        "inference_fallback_machines",
        "control_max_runtime_seconds",
        "surgery_max_runtime_seconds",
        "inference_max_runtime_seconds",
    }
    _unknown(table, keys, "lightning")
    teamspace = _string(table, "teamspace", "lightning")
    if _TEAMSPACE.fullmatch(teamspace) is None:
        raise LabBenchError("lightning.teamspace must use OWNER/TEAMSPACE syntax")
    studio = _string(table, "studio", "lightning")
    machines = {
        key: _string(table, key, "lightning")
        for key in ("control_machine", "surgery_machine", "inference_machine")
    }
    for key, value in machines.items():
        if _MACHINE.fullmatch(value) is None:
            raise LabBenchError(f"lightning.{key} contains unsupported characters")
    remote_root = _optional_string(table, "remote_root", ".abliteralus", "lightning")
    remote_path = PurePosixPath(remote_root)
    if (
        not remote_root
        or remote_path.is_absolute()
        or ".." in remote_path.parts
        or any(_REMOTE_PATH_SEGMENT.fullmatch(part) is None for part in remote_path.parts)
    ):
        raise LabBenchError("lightning.remote_root must be a relative POSIX path without '..'")
    forward = table.get("forward_environment", [])
    if not isinstance(forward, list) or not all(isinstance(name, str) for name in forward):
        raise LabBenchError("lightning.forward_environment must be an array of names")
    if len(set(forward)) != len(forward):
        raise LabBenchError("lightning.forward_environment must not contain duplicates")
    if any(_ENVIRONMENT_NAME.fullmatch(name) is None for name in forward):
        raise LabBenchError(
            "lightning.forward_environment entries must use uppercase environment syntax"
        )
    allocation_timeout = _positive_number(
        table.get("allocation_timeout_seconds", 900.0),
        "lightning.allocation_timeout_seconds",
    )
    allocation_retry = _positive_number(
        table.get("allocation_retry_seconds", 30.0),
        "lightning.allocation_retry_seconds",
    )
    pending_policy = table.get("pending_policy", "fail")
    if pending_policy not in {"fail", "adopt", "stop"}:
        raise LabBenchError("lightning.pending_policy must be fail, adopt, or stop")
    surgery_fallback = _machine_list(table, "surgery_fallback_machines")
    inference_fallback = _machine_list(table, "inference_fallback_machines")
    max_runtimes = {
        key: _optional_positive_integer(table, key)
        for key in (
            "control_max_runtime_seconds",
            "surgery_max_runtime_seconds",
            "inference_max_runtime_seconds",
        )
    }
    return LightningDefaults(
        teamspace=teamspace,
        studio=studio,
        control_machine=machines["control_machine"],
        surgery_machine=machines["surgery_machine"],
        inference_machine=machines["inference_machine"],
        remote_root=str(remote_path),
        forward_environment=tuple(forward),
        allocation_timeout_seconds=allocation_timeout,
        allocation_retry_seconds=allocation_retry,
        pending_policy=pending_policy,
        surgery_fallback_machines=surgery_fallback,
        inference_fallback_machines=inference_fallback,
        control_max_runtime_seconds=max_runtimes["control_max_runtime_seconds"],
        surgery_max_runtime_seconds=max_runtimes["surgery_max_runtime_seconds"],
        inference_max_runtime_seconds=max_runtimes["inference_max_runtime_seconds"],
    )


def _parse_ssh(table: Mapping[str, Any], base: Path) -> SshDefaults:
    _unknown(table, {"executable", "destination", "port", "identity_file"}, "ssh")
    executable = _string(table, "executable", "ssh")
    destination_raw = _optional_string(table, "destination", "", "ssh")
    destination = destination_raw or None
    if destination is not None:
        if any(character.isspace() for character in destination):
            raise LabBenchError("ssh.destination must not contain whitespace")
        if destination.startswith("-"):
            raise LabBenchError("ssh.destination must not begin with an option prefix")
    port = table.get("port", 22)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise LabBenchError("ssh.port must be an integer between 1 and 65535")
    identity_raw = _optional_string(table, "identity_file", "", "ssh")
    identity = _resolved_path(identity_raw, base) if identity_raw else None
    return SshDefaults(executable, destination, port, identity)


def _parse_artifacts(
    value: Any,
    base: Path,
    repository: Path,
) -> ArtifactDefaults:
    table = _table(value, "artifacts")
    _unknown(table, {"registry"}, "artifacts")
    raw = _optional_string(table, "registry", "outputs/artifact-registry", "artifacts")
    path = Path(raw).expanduser()
    if path.is_absolute():
        registry = path.resolve()
    elif "registry" in table:
        registry = (base / path).resolve()
    else:
        registry = (repository / path).resolve()
    return ArtifactDefaults(registry=registry)


def _resolved_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabBenchError(f"{label} must be a positive number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise LabBenchError(f"{label} must be a positive finite number")
    return parsed


def _machine_list(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    raw = table.get(key, [])
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise LabBenchError(f"lightning.{key} must be an array of machine names")
    normalized = tuple(value.strip().upper() for value in raw)
    if any(not value or _MACHINE.fullmatch(value) is None for value in normalized):
        raise LabBenchError(f"lightning.{key} contains an invalid machine name")
    if len(set(normalized)) != len(normalized):
        raise LabBenchError(f"lightning.{key} must not contain duplicates")
    return normalized


def _optional_positive_integer(table: Mapping[str, Any], key: str) -> int | None:
    if key not in table:
        return None
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabBenchError(f"lightning.{key} must be a positive integer")
    return value


def _string(table: Mapping[str, Any], key: str, context: str) -> str:
    if key not in table:
        raise LabBenchError(f"{context}.{key} is required")
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        raise LabBenchError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(table: Mapping[str, Any], key: str, default: str, context: str) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str):
        raise LabBenchError(f"{context}.{key} must be a string")
    return value.strip()


def _required_table(table: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    if key not in table:
        raise LabBenchError(f"{context}.{key} is required")
    return _table(table[key], f"{context}.{key}")


def _table(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabBenchError(f"{context} must be a table")
    return value


def _unknown(table: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LabBenchError(f"{context} contains unknown keys: {', '.join(unknown)}")
