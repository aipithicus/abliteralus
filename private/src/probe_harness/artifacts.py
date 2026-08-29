"""Content-addressed tensor artifacts owned by one probe session."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from safetensors.torch import load_file, save_file
import torch

from .errors import ProbeHarnessError


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArtifactStore:
    """Write immutable safetensors into a session-local object directory."""

    def __init__(self, session_directory: str | Path) -> None:
        self.session_directory = Path(session_directory).resolve()
        self.root = self.session_directory / "artifacts"

    def save_activations(
        self,
        tensors: Mapping[str, torch.Tensor],
    ) -> dict[str, object] | None:
        if not tensors:
            return None
        payload: dict[str, torch.Tensor] = {}
        for name in sorted(tensors):
            value = tensors[name]
            if not isinstance(value, torch.Tensor):
                raise ProbeHarnessError(f"artifact value {name!r} is not a tensor")
            payload[name] = value.detach().float().cpu().contiguous()

        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / f".staging-{uuid4().hex}.safetensors"
        try:
            save_file(
                payload,
                str(staging),
                metadata={
                    "format": "aipithicus.probe-activations-v1",
                },
            )
            digest = _sha256_file(staging)
            destination = self.root / f"sha256-{digest}.safetensors"
            if destination.exists():
                if _sha256_file(destination) != digest:
                    raise ProbeHarnessError(
                        f"existing activation artifact does not match its name: {destination}"
                    )
                staging.unlink()
            else:
                staging.replace(destination)
        finally:
            if staging.exists():
                staging.unlink()

        return {
            "format": "safetensors",
            "sha256": digest,
            "path": destination.relative_to(self.session_directory).as_posix(),
            "bytes": destination.stat().st_size,
            "tensor_keys": sorted(payload),
        }

    def load_activations(self, reference: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Load and hash-verify one session-owned activation artifact."""

        relative = reference.get("path")
        digest = reference.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ProbeHarnessError("activation artifact has no relative path")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ProbeHarnessError("activation artifact has no valid SHA-256 digest")
        candidate = (self.session_directory / Path(relative)).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise ProbeHarnessError("activation artifact resolves outside the session object store")
        if not candidate.is_file():
            raise ProbeHarnessError(f"activation artifact does not exist: {relative}")
        if _sha256_file(candidate) != digest:
            raise ProbeHarnessError(f"activation artifact hash mismatch: {relative}")
        try:
            tensors = load_file(str(candidate), device="cpu")
        except Exception as error:
            raise ProbeHarnessError(f"could not load activation artifact {relative}: {error}") from error
        expected_keys = reference.get("tensor_keys")
        if not isinstance(expected_keys, list) or sorted(tensors) != sorted(expected_keys):
            raise ProbeHarnessError("activation artifact tensor keys do not match its reference")
        return {
            name: value.detach().float().cpu().contiguous()
            for name, value in tensors.items()
        }
