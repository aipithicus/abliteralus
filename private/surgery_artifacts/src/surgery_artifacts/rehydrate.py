"""Rehydrate a verified capsule onto its exact Safetensors base."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors import safe_open
from safetensors.torch import save_file

from .canonical_json import pretty_text
from .checkpoint import SafeTensorCheckpoint
from .codecs import apply_operation
from .errors import CapsuleFormatError, ExactnessError
from .hashing import tensor_sha256
from .validation import validate_capsule

_UNSAFE_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}


def rehydrate_capsule(
    capsule: str | Path,
    base_checkpoint: str | Path,
    output: str | Path,
) -> Path:
    """Build an atomic runnable checkpoint and verify every resulting tensor."""

    capsule_root = validate_capsule(capsule).path
    base = SafeTensorCheckpoint(base_checkpoint)
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"rehydration destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(capsule_root / "manifest.json")
    tensor_manifest = _read_json(capsule_root / "tensor-manifest.json")
    base_records = _records_by_name(tensor_manifest["base_tensors"], "base tensor")
    changed_records = _records_by_name(tensor_manifest["changed_tensors"], "changed tensor")
    if set(base.tensor_names) != set(base_records):
        raise ExactnessError("selected base tensor set does not match capsule identity")

    operations: dict[str, list[dict[str, Any]]] = {}
    for operation in manifest["operations"]:
        name = operation["parameter"]
        operations.setdefault(name, []).append(operation)
    if set(operations) != set(changed_records):
        raise CapsuleFormatError("changed tensor records do not match capsule operations")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _copy_base_files(base.root, temporary, excluded=set(base.weight_files))
        payload_path = capsule_root / "delta.safetensors"
        with safe_open(payload_path, framework="pt", device="cpu") as payload:
            for shard, tensors in base.iter_shards():
                output_tensors = {}
                for name, before in tensors.items():
                    expected_base = base_records[name]
                    if tensor_sha256(before) != expected_base["sha256"]:
                        raise ExactnessError(f"selected base does not match capsule at {name}")
                    tensor_operations = operations.get(name)
                    if tensor_operations is None:
                        output_tensors[name] = before
                        continue
                    after = before
                    for operation in tensor_operations:
                        if tensor_sha256(after) != operation["before_sha256"]:
                            raise CapsuleFormatError(
                                f"operation sequence has a broken before hash for {name}"
                            )
                        operation_payload = {
                            logical: payload.get_tensor(record["key"])
                            for logical, record in operation["payload"].items()
                        }
                        after = apply_operation(operation, after, operation_payload)
                        if tensor_sha256(after) != operation["after_sha256"]:
                            raise ExactnessError(f"operation replay was not exact for {name}")
                    if tensor_sha256(after) != changed_records[name]["sha256"]:
                        raise CapsuleFormatError(f"target hash records disagree for {name}")
                    output_tensors[name] = after
                shard_path = temporary / shard.relative_path
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(output_tensors, shard_path, metadata=shard.metadata)

        for overlay in manifest["overlays"]:
            relative = _safe_relative(overlay["path"])
            source = capsule_root / "files" / relative
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

        (temporary / "surgery-artifact.json").write_text(
            pretty_text(
                {
                    "schema_version": 1,
                    "surgery_id": manifest["surgery_id"],
                    "capsule": str(capsule_root),
                }
            ),
            encoding="utf-8",
        )
        _validate_rehydrated(temporary, base_records, changed_records)
        os.replace(temporary, destination)
        _validate_rehydrated(destination, base_records, changed_records)
        return destination
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _copy_base_files(base_root: Path, destination: Path, *, excluded: set[str]) -> None:
    for source in sorted(path for path in base_root.rglob("*") if path.is_file()):
        relative = source.relative_to(base_root)
        relative_text = relative.as_posix()
        if relative_text in excluded or source.suffix.lower() == ".safetensors":
            continue
        if source.suffix.lower() in _UNSAFE_SUFFIXES:
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _validate_rehydrated(
    checkpoint_root: Path,
    base_records: dict[str, dict[str, Any]],
    changed_records: dict[str, dict[str, Any]],
) -> None:
    checkpoint = SafeTensorCheckpoint(checkpoint_root)
    if set(checkpoint.tensor_names) != set(base_records):
        raise ExactnessError("rehydrated checkpoint tensor set changed")
    with checkpoint.reader() as reader:
        for name in checkpoint.tensor_names:
            expected = changed_records.get(name, base_records[name])["sha256"]
            if tensor_sha256(reader.get_tensor(name)) != expected:
                raise ExactnessError(f"rehydrated tensor hash mismatch: {name}")


def _records_by_name(records: object, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise CapsuleFormatError(f"{label} records must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise CapsuleFormatError(f"invalid {label} record")
        name = record["name"]
        if name in result:
            raise CapsuleFormatError(f"duplicate {label} record: {name}")
        result[name] = record
    return result


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise CapsuleFormatError("invalid overlay path")
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
        raise CapsuleFormatError("unsafe overlay path")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CapsuleFormatError(f"expected JSON object: {path.name}")
    return value
