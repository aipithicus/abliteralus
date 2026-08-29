"""Build immutable capsule-v1 directories from two Safetensors checkpoints."""

from __future__ import annotations

import os
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from ._staging import create_staging_directory
from .canonical_json import canonical_bytes, normalize, pretty_text
from .checkpoint import SafeTensorCheckpoint
from .codecs import (
    EncodedOperation,
    apply_operation,
    encode_projection_hint,
    encode_tensor_change,
)
from .errors import ArtifactError, ExactnessError
from .hashing import sha256_bytes, sha256_file, tensor_record

SCHEMA_VERSION = 1
_UNSAFE_OVERLAY_SUFFIXES = {".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"}


@dataclass(frozen=True, slots=True)
class CapsuleResult:
    path: Path
    surgery_id: str
    operation_count: int
    changed_tensor_count: int
    unchanged_tensor_count: int
    payload_bytes: int
    total_bytes: int
    codecs: dict[str, int]


def create_capsule(
    base_checkpoint: str | Path,
    target_checkpoint: str | Path,
    output: str | Path,
    *,
    base_identity: dict[str, Any] | None = None,
    recipe: dict[str, Any] | None = None,
    require_exact: bool = True,
    max_low_rank: int = 32,
    mutation_hints: Sequence[object] = (),
) -> CapsuleResult:
    """Create and fully validate one immutable surgery capsule."""

    base = SafeTensorCheckpoint(base_checkpoint)
    target = SafeTensorCheckpoint(target_checkpoint)
    if base.tensor_names != target.tensor_names:
        missing = sorted(set(base.tensor_names) - set(target.tensor_names))
        added = sorted(set(target.tensor_names) - set(base.tensor_names))
        raise ExactnessError(
            "checkpoint tensor sets differ "
            f"(missing={missing[:3]!r}, added={added[:3]!r})"
        )

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"capsule destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    operations: list[EncodedOperation] = []
    base_records: list[dict[str, Any]] = []
    changed_records: list[dict[str, Any]] = []
    unchanged_count = 0
    hints_by_parameter: dict[str, list[object]] = {}
    for hint in mutation_hints:
        parameter = _hint_parameter(hint)
        hints_by_parameter.setdefault(parameter, []).append(hint)
    with base.reader() as base_reader, target.reader() as target_reader:
        for name in base.tensor_names:
            before = base_reader.get_tensor(name).detach().cpu().contiguous()
            after = target_reader.get_tensor(name).detach().cpu().contiguous()
            before_record = {
                "name": name,
                "shard": base.relative_weight_file(name),
                **tensor_record(before),
            }
            base_records.append(before_record)
            if torch.equal(before, after):
                unchanged_count += 1
                continue
            fallback = encode_tensor_change(
                name,
                before,
                after,
                require_exact=require_exact,
                max_low_rank=max_low_rank,
            )
            hinted = _encode_hint_sequence(
                name,
                before,
                after,
                hints_by_parameter.get(name, ()),
            )
            if hinted and sum(item.payload_bytes for item in hinted) < fallback.payload_bytes:
                operations.extend(hinted)
            else:
                operations.append(fallback)
            changed_records.append(
                {
                    "name": name,
                    "target_shard": target.relative_weight_file(name),
                    **tensor_record(after),
                }
            )

    if not operations:
        raise ArtifactError("base and target checkpoints have identical tensors")

    base_tensor_identity = {
        "schema_version": SCHEMA_VERSION,
        "tensors": _sorted_tensor_identities(base_records),
    }
    base_manifest_hash = sha256_bytes(canonical_bytes(base_tensor_identity))
    resolved_base_identity = {
        **normalize(base_identity or {}),
        "tensor_manifest_sha256": base_manifest_hash,
        "tensor_count": len(base_records),
    }
    surgery_id = sha256_bytes(
        canonical_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "base": resolved_base_identity,
                "target_tensors": _sorted_tensor_identities(changed_records),
            }
        )
    )

    temporary = create_staging_directory(
        destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    try:
        payload: dict[str, torch.Tensor] = {}
        manifest_operations: list[dict[str, Any]] = []
        for index, operation in enumerate(operations):
            keys: dict[str, str] = {}
            for name, tensor in sorted(operation.tensors.items()):
                key = f"op_{index:06d}.{name}"
                payload[key] = tensor
                keys[name] = key
            manifest_operations.append(operation.manifest_record(keys))

        overlay_records, omitted = _copy_overlays(base.root, target.root, temporary / "files")
        tensor_manifest = {
            "schema_version": SCHEMA_VERSION,
            "base_tensors": base_records,
            "changed_tensors": changed_records,
            "unchanged_tensor_count": unchanged_count,
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "aipithicus.surgery-capsule",
            "surgery_id": surgery_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base": resolved_base_identity,
            "operations": manifest_operations,
            "overlays": overlay_records,
            "omitted_unsafe_files": omitted,
        }
        recipe_payload = {
            "schema_version": SCHEMA_VERSION,
            "surgery_id": surgery_id,
            "recipe": normalize(recipe or {}),
            "codec_summary": dict(sorted(Counter(op.kind for op in operations).items())),
            "require_exact": bool(require_exact),
        }

        (temporary / "manifest.json").write_text(pretty_text(manifest), encoding="utf-8")
        (temporary / "tensor-manifest.json").write_text(
            pretty_text(tensor_manifest), encoding="utf-8"
        )
        (temporary / "recipe.json").write_text(pretty_text(recipe_payload), encoding="utf-8")
        save_file(payload, temporary / "delta.safetensors", metadata={"surgery_id": surgery_id})
        _write_checksums(temporary)

        from .validation import validate_capsule

        validation = validate_capsule(temporary)
        if validation.surgery_id != surgery_id:
            raise ArtifactError("staged capsule identity changed during validation")
        os.replace(temporary, destination)
        validation = validate_capsule(destination)
        codecs = dict(sorted(Counter(operation.kind for operation in operations).items()))
        return CapsuleResult(
            path=destination,
            surgery_id=surgery_id,
            operation_count=len(operations),
            changed_tensor_count=len(changed_records),
            unchanged_tensor_count=unchanged_count,
            payload_bytes=(destination / "delta.safetensors").stat().st_size,
            total_bytes=validation.total_bytes,
            codecs=codecs,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _copy_overlays(
    base_root: Path,
    target_root: Path,
    destination: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    overlays: list[dict[str, Any]] = []
    omitted: list[str] = []
    for source in sorted(path for path in target_root.rglob("*") if path.is_file()):
        relative = source.relative_to(target_root)
        relative_text = relative.as_posix()
        suffix = source.suffix.lower()
        if suffix == ".safetensors" or relative_text == "model.safetensors.index.json":
            continue
        if suffix in _UNSAFE_OVERLAY_SUFFIXES:
            omitted.append(relative_text)
            continue
        base_file = base_root / relative
        digest = sha256_file(source)
        if base_file.is_file() and sha256_file(base_file) == digest:
            continue
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        overlays.append(
            {"path": relative_text, "sha256": digest, "bytes": source.stat().st_size}
        )
    return overlays, omitted


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _encode_hint_sequence(
    parameter: str,
    before: torch.Tensor,
    after: torch.Tensor,
    hints: Sequence[object],
) -> list[EncodedOperation]:
    current = before
    operations: list[EncodedOperation] = []
    for hint in hints:
        operation = encode_projection_hint(parameter, current, hint)
        if operation is None:
            return []
        current = apply_operation(operation.semantic_record(), current, operation.tensors)
        operations.append(operation)
    return operations if operations and torch.equal(current, after) else []


def _hint_parameter(hint: object) -> str:
    value = hint.get("parameter") if isinstance(hint, dict) else getattr(hint, "parameter", None)
    if not isinstance(value, str) or not value:
        raise ExactnessError("mutation hint has no stable parameter name")
    return value


def _tensor_identity(record: dict[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in ("name", "dtype", "shape", "sha256")
    }


def _sorted_tensor_identities(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted((_tensor_identity(record) for record in records), key=lambda item: item["name"])
