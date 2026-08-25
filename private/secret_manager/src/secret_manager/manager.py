"""Secret profiles, environment isolation, and provider dispatch."""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Mapping, Sequence

from .config import (
    PROFILE_ENVIRONMENT_VARIABLE,
    ProfileConfig,
    ProviderConfig,
    SecretManagerConfig,
    load_config,
    validate_environment_name,
)
from .errors import ConfigError, PrivilegedSecretError, ProviderError, SecretUnavailable
from .proton_pass import build_reference, resolve_reference, run_with_references


class SecretManager:
    """Resolve references at process boundaries without retaining secret values."""

    def __init__(self, config: SecretManagerConfig) -> None:
        self.config = config

    @classmethod
    def from_config(
        cls,
        path: str | os.PathLike[str] | None = None,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> "SecretManager":
        return cls(load_config(path, environ=environ))

    def profile(self, name: str) -> ProfileConfig:
        try:
            return self.config.profiles[name]
        except KeyError as error:
            raise ConfigError(f"unknown secret profile: {name}") from error

    def profile_is_privileged(self, name: str) -> bool:
        profile = self.profile(name)
        return any(
            self.config.credentials[credential].privileged
            for credential in profile.environment.values()
        )

    def references_for_profile(
        self, name: str, *, allow_privileged: bool = False
    ) -> tuple[ProviderConfig, dict[str, str]]:
        profile = self.profile(name)
        credentials = [
            self.config.credentials[credential_name]
            for credential_name in profile.environment.values()
        ]
        privileged = [credential.name for credential in credentials if credential.privileged]
        if privileged and not allow_privileged:
            raise PrivilegedSecretError(
                f"profile {name} contains privileged credentials and requires an explicit gate"
            )
        provider_names = {credential.provider for credential in credentials}
        if len(provider_names) != 1:  # also enforced while parsing; retain a runtime invariant.
            raise ConfigError(f"profile {name} must use exactly one provider")
        provider = self.config.providers[next(iter(provider_names))]
        references = {
            environment_name: build_reference(self.config.credentials[credential_name])
            for environment_name, credential_name in profile.environment.items()
        }
        return provider, references

    def isolated_environment(
        self,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Remove managed values and reject unrelated references before a provider scan."""

        environment = dict(os.environ if environ is None else environ)
        managed_names = {
            name for profile in self.config.profiles.values() for name in profile.environment
        }
        for name in managed_names:
            environment.pop(name, None)
        for name, value in environment.items():
            if "pass://" in value:
                raise ProviderError(
                    f"unmanaged Proton Pass reference found in environment variable {name}"
                )
        return environment

    def run(
        self,
        profile_name: str,
        command: Sequence[str],
        *,
        allow_privileged: bool = False,
        environ: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        capture_output: bool = False,
        timeout: float | None = None,
    ):
        """Run a command with references resolved only inside ``pass-cli run``."""

        provider, references = self.references_for_profile(
            profile_name, allow_privileged=allow_privileged
        )
        environment = self.isolated_environment(environ=environ)
        return run_with_references(
            provider,
            references,
            [str(argument) for argument in command],
            environment=environment,
            cwd=cwd,
            capture_output=capture_output,
            timeout=timeout,
        )

    def check(
        self,
        profile_name: str,
        *,
        allow_privileged: bool = False,
        environ: Mapping[str, str] | None = None,
    ) -> dict[str, bool]:
        """Resolve in an ephemeral probe and return presence booleans only."""

        provider, references = self.references_for_profile(
            profile_name, allow_privileged=allow_privileged
        )
        environment = self.isolated_environment(environ=environ)
        names = sorted(references)
        completed = run_with_references(
            provider,
            references,
            [sys.executable, "-m", "secret_manager._probe", *names],
            environment=environment,
            capture_output=True,
            timeout=provider.timeout_seconds,
        )
        if completed.returncode != 0:
            raise ProviderError(f"secret profile check failed: {profile_name}")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ProviderError(
                f"secret profile check returned invalid metadata: {profile_name}"
            ) from error
        if not isinstance(payload, dict) or set(payload) != set(names):
            raise ProviderError(
                f"secret profile check returned unexpected metadata: {profile_name}"
            )
        if not all(value is True for value in payload.values()):
            raise ProviderError(f"secret profile check failed: {profile_name}")
        return {name: True for name in names}

    def resolve_for_broker(
        self,
        environment_name: str,
        *,
        profile_name: str | None = None,
    ) -> bytes:
        """Resolve exactly one non-privileged value for a provider-neutral broker."""

        validate_environment_name(environment_name)
        selected = profile_name or self.config.broker.default_profile
        if not selected or selected not in self.config.broker.allowed_profiles:
            raise SecretUnavailable("secret profile is not broker-enabled")
        profile = self.profile(selected)
        credential_name = profile.environment.get(environment_name)
        if credential_name is None:
            raise SecretUnavailable("secret is not mapped in the selected broker profile")
        credential = self.config.credentials[credential_name]
        if credential.privileged:
            raise SecretUnavailable("privileged credentials are not broker-addressable")
        provider = self.config.providers[credential.provider]
        return resolve_reference(provider, build_reference(credential))

    def inventory(self, *, include_references: bool = False) -> dict[str, Any]:
        """Return metadata only; references require an explicit opt-in."""

        credentials: list[dict[str, Any]] = []
        for name in sorted(self.config.credentials):
            credential = self.config.credentials[name]
            record: dict[str, Any] = {
                "name": name,
                "provider": credential.provider,
                "privileged": credential.privileged,
                "description": credential.description,
            }
            if include_references:
                record["reference"] = build_reference(credential)
            credentials.append(record)
        profiles = []
        for name in sorted(self.config.profiles):
            profile = self.config.profiles[name]
            profiles.append(
                {
                    "name": name,
                    "environment": sorted(profile.environment),
                    "privileged": self.profile_is_privileged(name),
                    "broker_enabled": name in self.config.broker.allowed_profiles,
                    "description": profile.description,
                }
            )
        return {
            "schema_version": self.config.schema_version,
            "config": str(self.config.path),
            "providers": [
                {
                    "name": provider.name,
                    "type": provider.kind,
                    "executable": provider.executable,
                    "timeout_seconds": provider.timeout_seconds,
                }
                for provider in sorted(self.config.providers.values(), key=lambda item: item.name)
            ],
            "credentials": credentials,
            "profiles": profiles,
            "broker": {
                "default_profile": self.config.broker.default_profile,
                "allowed_profiles": list(self.config.broker.allowed_profiles),
                "profile_environment_variable": PROFILE_ENVIRONMENT_VARIABLE,
            },
        }
