"""Strict parsing for the non-secret secret-manager mapping file."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError

CONFIG_ENVIRONMENT_VARIABLE = "SECRET_MANAGER_CONFIG"
PROFILE_ENVIRONMENT_VARIABLE = "SECRET_MANAGER_PROFILE"
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ALIAS = re.compile(r"^[a-z][a-z0-9_-]*$")
_REFERENCE_SEPARATORS = frozenset("/?#")


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """One configured secret-provider command."""

    name: str
    kind: str
    executable: str
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class CredentialConfig:
    """A stable locator for one provider field; never the field value."""

    name: str
    provider: str
    vault: str
    item: str
    field: str
    privileged: bool
    description: str


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Environment-variable names mapped to credential aliases."""

    name: str
    environment: Mapping[str, str]
    description: str


@dataclass(frozen=True, slots=True)
class BrokerConfig:
    """Profiles that may be addressed by the single-argument broker."""

    default_profile: str | None
    allowed_profiles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SecretManagerConfig:
    """Validated, non-secret secret-manager configuration."""

    path: Path
    providers: Mapping[str, ProviderConfig]
    credentials: Mapping[str, CredentialConfig]
    profiles: Mapping[str, ProfileConfig]
    broker: BrokerConfig
    schema_version: int = 1


def validate_environment_name(name: str) -> str:
    """Return a normalized credential name or fail closed."""

    if not isinstance(name, str) or _ENVIRONMENT_NAME.fullmatch(name) is None:
        raise ConfigError("secret names must use uppercase environment-variable syntax")
    return name


def discover_config_path(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve an explicit path, an environment override, or a platform default."""

    environment = os.environ if environ is None else environ
    candidate = str(path).strip() if path is not None else ""
    if not candidate:
        candidate = environment.get(CONFIG_ENVIRONMENT_VARIABLE, "").strip()
    if candidate:
        return Path(candidate).expanduser().resolve()

    app_data = environment.get("APPDATA", "").strip()
    if app_data:
        return (Path(app_data) / "Aipithicus" / "secret-manager.toml").resolve()

    xdg_config = environment.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg_config).expanduser() if xdg_config else Path.home() / ".config"
    return (root / "aipithicus" / "secret-manager.toml").resolve()


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> SecretManagerConfig:
    """Load and validate a mapping without resolving any secret values."""

    resolved = discover_config_path(path, environ=environ)
    try:
        with resolved.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"secret-manager config does not exist: {resolved}") from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"secret-manager config is unreadable or invalid: {resolved}") from error
    return parse_config(raw, path=resolved)


def parse_config(raw: Mapping[str, Any], *, path: Path | None = None) -> SecretManagerConfig:
    """Validate a decoded TOML mapping."""

    root = _require_table(raw, "config")
    _reject_unknown(
        root, {"schema_version", "providers", "credentials", "profiles", "broker"}, "config"
    )
    schema_version = root.get("schema_version")
    if schema_version != 1:
        raise ConfigError("schema_version must be 1")

    providers = _parse_providers(_required_table(root, "providers", "config"))
    credentials = _parse_credentials(_required_table(root, "credentials", "config"), providers)
    profiles = _parse_profiles(_required_table(root, "profiles", "config"), credentials)
    broker = _parse_broker(root.get("broker", {}), profiles)
    return SecretManagerConfig(
        path=(path or Path("secret-manager.toml")).resolve(),
        providers=providers,
        credentials=credentials,
        profiles=profiles,
        broker=broker,
        schema_version=1,
    )


def _parse_providers(raw: Mapping[str, Any]) -> dict[str, ProviderConfig]:
    if not raw:
        raise ConfigError("providers must contain at least one provider")
    providers: dict[str, ProviderConfig] = {}
    for name, value in raw.items():
        _validate_alias(name, "provider")
        table = _require_table(value, f"providers.{name}")
        _reject_unknown(table, {"type", "executable", "timeout_seconds"}, f"providers.{name}")
        kind = _required_string(table, "type", f"providers.{name}")
        if kind != "proton-pass":
            raise ConfigError(f"providers.{name}.type is unsupported")
        executable = _optional_string(table, "executable", "pass-cli", f"providers.{name}")
        timeout = table.get("timeout_seconds", 5.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ConfigError(f"providers.{name}.timeout_seconds must be a number")
        timeout_value = float(timeout)
        if not 0.1 <= timeout_value <= 30.0:
            raise ConfigError(
                f"providers.{name}.timeout_seconds must be between 0.1 and 30 seconds"
            )
        providers[name] = ProviderConfig(name, kind, executable, timeout_value)
    return providers


def _parse_credentials(
    raw: Mapping[str, Any], providers: Mapping[str, ProviderConfig]
) -> dict[str, CredentialConfig]:
    if not raw:
        raise ConfigError("credentials must contain at least one credential")
    credentials: dict[str, CredentialConfig] = {}
    for name, value in raw.items():
        _validate_alias(name, "credential")
        table = _require_table(value, f"credentials.{name}")
        _reject_unknown(
            table,
            {"provider", "vault", "item", "field", "privileged", "description"},
            f"credentials.{name}",
        )
        provider = _required_string(table, "provider", f"credentials.{name}")
        if provider not in providers:
            raise ConfigError(f"credentials.{name}.provider is not configured")
        vault = _reference_segment(table, "vault", f"credentials.{name}")
        item = _reference_segment(table, "item", f"credentials.{name}")
        field = _reference_segment(table, "field", f"credentials.{name}")
        privileged = table.get("privileged", False)
        if not isinstance(privileged, bool):
            raise ConfigError(f"credentials.{name}.privileged must be true or false")
        description = _optional_string(table, "description", "", f"credentials.{name}")
        credentials[name] = CredentialConfig(
            name=name,
            provider=provider,
            vault=vault,
            item=item,
            field=field,
            privileged=privileged,
            description=description,
        )
    return credentials


def _parse_profiles(
    raw: Mapping[str, Any], credentials: Mapping[str, CredentialConfig]
) -> dict[str, ProfileConfig]:
    if not raw:
        raise ConfigError("profiles must contain at least one profile")
    profiles: dict[str, ProfileConfig] = {}
    for name, value in raw.items():
        _validate_alias(name, "profile")
        table = _require_table(value, f"profiles.{name}")
        _reject_unknown(table, {"description", "environment"}, f"profiles.{name}")
        environment = _required_table(table, "environment", f"profiles.{name}")
        if not environment:
            raise ConfigError(f"profiles.{name}.environment must not be empty")
        bindings: dict[str, str] = {}
        provider_names: set[str] = set()
        for environment_name, credential_name in environment.items():
            validate_environment_name(environment_name)
            if not isinstance(credential_name, str) or credential_name not in credentials:
                raise ConfigError(
                    f"profiles.{name}.environment.{environment_name} is not a configured credential"
                )
            bindings[environment_name] = credential_name
            provider_names.add(credentials[credential_name].provider)
        if len(provider_names) != 1:
            raise ConfigError(f"profiles.{name} must use exactly one provider")
        description = _optional_string(table, "description", "", f"profiles.{name}")
        profiles[name] = ProfileConfig(name, bindings, description)
    return profiles


def _parse_broker(raw: Any, profiles: Mapping[str, ProfileConfig]) -> BrokerConfig:
    table = _require_table(raw, "broker")
    _reject_unknown(table, {"default_profile", "allowed_profiles"}, "broker")
    default = table.get("default_profile")
    if default is not None:
        if not isinstance(default, str) or default not in profiles:
            raise ConfigError("broker.default_profile is not a configured profile")
    allowed_raw = table.get("allowed_profiles", [])
    if not isinstance(allowed_raw, list) or not all(
        isinstance(value, str) for value in allowed_raw
    ):
        raise ConfigError("broker.allowed_profiles must be an array of profile names")
    if len(set(allowed_raw)) != len(allowed_raw):
        raise ConfigError("broker.allowed_profiles must not contain duplicates")
    for profile in allowed_raw:
        if profile not in profiles:
            raise ConfigError("broker.allowed_profiles contains an unknown profile")
    if default is not None and default not in allowed_raw:
        raise ConfigError("broker.default_profile must be listed in broker.allowed_profiles")
    return BrokerConfig(default, tuple(allowed_raw))


def _validate_alias(name: str, label: str) -> None:
    if _ALIAS.fullmatch(name) is None:
        raise ConfigError(f"{label} names must use lowercase kebab-case")


def _reference_segment(table: Mapping[str, Any], key: str, context: str) -> str:
    value = _required_string(table, key, context)
    if any(character in value for character in _REFERENCE_SEPARATORS):
        raise ConfigError(f"{context}.{key} contains a reserved reference character")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigError(f"{context}.{key} contains a control character")
    return value


def _required_string(table: Mapping[str, Any], key: str, context: str) -> str:
    if key not in table:
        raise ConfigError(f"{context}.{key} is required")
    value = table[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(table: Mapping[str, Any], key: str, default: str, context: str) -> str:
    if key not in table:
        return default
    value = table[key]
    if not isinstance(value, str):
        raise ConfigError(f"{context}.{key} must be a string")
    return value.strip()


def _required_table(table: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    if key not in table:
        raise ConfigError(f"{context}.{key} is required")
    return _require_table(table[key], f"{context}.{key}")


def _require_table(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a table")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{context} contains a non-string key")
    return value


def _reject_unknown(table: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{context} contains unknown keys: {', '.join(unknown)}")
