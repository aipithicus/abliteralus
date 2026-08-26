"""Read-only access to Hugging Face Safetensors checkpoint layouts."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .errors import CapsuleFormatError, UnsupportedCheckpointError


@dataclass(frozen=True, slots=True)
class CheckpointShard:
    relative_path: str
    tensor_names: tuple[str, ...]
    metadata: dict[str, str] | None


class SafeTensorCheckpoint:
    """Index a checkpoint without importing or executing model code."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise UnsupportedCheckpointError(f"checkpoint directory is unavailable: {self.root}")
        self._locations, self._shards = self._discover()

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._locations))

    @property
    def weight_files(self) -> tuple[str, ...]:
        return tuple(sorted(self._shards))

    def relative_weight_file(self, tensor_name: str) -> str:
        try:
            return self._locations[tensor_name]
        except KeyError as error:
            raise CapsuleFormatError(f"tensor is absent from checkpoint: {tensor_name}") from error

    def iter_shards(self) -> Iterator[tuple[CheckpointShard, dict[str, torch.Tensor]]]:
        for relative in sorted(self._shards):
            path = self.root / relative
            names, metadata = self._shards[relative]
            tensors = load_file(path, device="cpu")
            if set(tensors) != set(names):
                raise CapsuleFormatError(f"Safetensors keys changed while reading {path}")
            yield CheckpointShard(relative, names, metadata), tensors

    def load_tensor(self, name: str) -> torch.Tensor:
        relative = self.relative_weight_file(name)
        path = self.root / relative
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def reader(self) -> "CheckpointReader":
        return CheckpointReader(self)

    def _discover(
        self,
    ) -> tuple[
        dict[str, str],
        dict[str, tuple[tuple[str, ...], dict[str, str] | None]],
    ]:
        index_path = self.root / "model.safetensors.index.json"
        locations: dict[str, str] = {}
        expected_by_file: dict[str, set[str]] = {}
        if index_path.is_file():
            try:
                raw = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise UnsupportedCheckpointError("Safetensors index is unreadable") from error
            weight_map = raw.get("weight_map") if isinstance(raw, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise UnsupportedCheckpointError("Safetensors index has no weight_map")
            for name, relative in weight_map.items():
                if not isinstance(name, str) or not name or not isinstance(relative, str):
                    raise UnsupportedCheckpointError("Safetensors index contains an invalid entry")
                safe_relative = _safe_relative(relative)
                locations[name] = safe_relative
                expected_by_file.setdefault(safe_relative, set()).add(name)
        else:
            candidates = sorted(self.root.glob("model*.safetensors"))
            if not candidates:
                if any(self.root.glob("*.bin")):
                    raise UnsupportedCheckpointError(
                        "capsule v1 requires Safetensors; PyTorch pickle checkpoints are refused"
                    )
                raise UnsupportedCheckpointError("checkpoint contains no model Safetensors files")
            for path in candidates:
                relative = path.relative_to(self.root).as_posix()
                with safe_open(path, framework="pt", device="cpu") as handle:
                    names = set(handle.keys())
                expected_by_file[relative] = names
                for name in names:
                    if name in locations:
                        raise UnsupportedCheckpointError(f"duplicate tensor name: {name}")
                    locations[name] = relative

        shards: dict[str, tuple[tuple[str, ...], dict[str, str] | None]] = {}
        for relative, expected in expected_by_file.items():
            path = self.root / relative
            if not path.is_file():
                raise UnsupportedCheckpointError(f"indexed Safetensors shard is missing: {relative}")
            with safe_open(path, framework="pt", device="cpu") as handle:
                actual = set(handle.keys())
                metadata = handle.metadata()
            if actual != expected:
                raise UnsupportedCheckpointError(
                    f"Safetensors index does not match shard contents: {relative}"
                )
            shards[relative] = (tuple(sorted(actual)), metadata)
        return locations, shards


class CheckpointReader:
    """Keep lightweight Safetensors mappings open while tensors are streamed."""

    def __init__(self, checkpoint: SafeTensorCheckpoint) -> None:
        self.checkpoint = checkpoint
        self._stack = ExitStack()
        self._handles: dict[str, object] = {}

    def __enter__(self) -> "CheckpointReader":
        for relative in self.checkpoint.weight_files:
            handle = self._stack.enter_context(
                safe_open(
                    self.checkpoint.root / relative,
                    framework="pt",
                    device="cpu",
                )
            )
            self._handles[relative] = handle
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stack.close()

    def get_tensor(self, name: str) -> torch.Tensor:
        relative = self.checkpoint.relative_weight_file(name)
        handle = self._handles.get(relative)
        if handle is None:
            raise RuntimeError("checkpoint reader is not open")
        return handle.get_tensor(name)


def _safe_relative(value: str) -> str:
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts or not relative.parts:
        raise UnsupportedCheckpointError("checkpoint index contains an unsafe shard path")
    normalized = relative.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized
