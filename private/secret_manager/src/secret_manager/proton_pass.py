"""Narrow Proton Pass CLI adapter.

The normal execution path supplies only ``pass://`` references to ``pass-cli
run``. The direct resolver exists solely for the single-value executable broker.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .config import CredentialConfig, ProviderConfig
from .errors import ProviderError

MAX_SECRET_BYTES = 64 * 1024


def build_reference(credential: CredentialConfig) -> str:
    """Build a reference from validated, non-secret locator segments."""

    return f"pass://{credential.vault}/{credential.item}/{credential.field}"


def resolve_executable(provider: ProviderConfig) -> str:
    """Resolve the configured CLI once, without invoking a shell."""

    configured = provider.executable
    path = Path(configured).expanduser()
    has_directory = path.is_absolute() or path.parent != Path(".")
    resolved = str(path.resolve()) if has_directory else shutil.which(configured)
    if not resolved:
        raise ProviderError(f"provider executable is unavailable: {provider.name}")
    candidate = Path(resolved)
    if not candidate.is_file():
        raise ProviderError(f"provider executable is unavailable: {provider.name}")
    if os.name != "nt" and not os.access(candidate, os.X_OK):
        raise ProviderError(f"provider executable is not executable: {provider.name}")
    return str(candidate)


def run_with_references(
    provider: ProviderConfig,
    references: Mapping[str, str],
    command: Sequence[str],
    *,
    environment: Mapping[str, str],
    cwd: str | os.PathLike[str] | None = None,
    capture_output: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a child through Proton Pass's masked, ephemeral environment path."""

    if not command or not command[0]:
        raise ProviderError("a child command is required")
    executable = resolve_executable(provider)
    child_environment = dict(environment)
    child_environment.update(references)
    arguments = [executable, "run", "--", *command]
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            env=child_environment,
            stdin=None,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            check=False,
            close_fds=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise ProviderError("provider-backed child command timed out") from error
    except OSError as error:
        raise ProviderError("provider-backed child command could not start") from error


def resolve_reference(provider: ProviderConfig, reference: str) -> bytes:
    """Resolve one field for the executable-broker boundary.

    Provider stderr is discarded and never incorporated into an exception.
    """

    executable = resolve_executable(provider)
    try:
        completed = subprocess.run(
            [executable, "item", "view", reference, "--output", "human"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            close_fds=True,
            timeout=provider.timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProviderError("provider resolution failed") from error
    if completed.returncode != 0:
        raise ProviderError("provider resolution failed")
    raw = completed.stdout
    if len(raw) > MAX_SECRET_BYTES:
        raise ProviderError("provider returned an oversized value")
    value = raw.rstrip(b"\r\n")
    if not value:
        raise ProviderError("provider returned an empty value")
    if b"\x00" in value:
        raise ProviderError("provider returned an invalid value")
    try:
        value.decode("utf-8")
    except UnicodeError as error:
        raise ProviderError("provider returned a non-UTF-8 value") from error
    return value
