"""Strict structural and cryptographic validation for capsule v1."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .canonical_json import canonical_bytes
from .errors import CapsuleFormatError
from .hashing import dtype_name, sha256_bytes, sha256_file, tensor_record

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DTYPE = re.compile(r"[a-z][a-z0-9_]{1,31}")
_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
_OPERATION_PAYLOADS = {
    "replace_tensor": {"target"},
    "add_dense": {"delta"},
    "add_sparse_rows": {"indices", "delta"},
    "add_sparse_elements": {"indices", "delta"},
    "add_low_rank": {"a", "b"},
}
_MANIFEST_KEYS = {
    "schema_version",
    "format",
    "surgery_id",
    "created_at",
    "base",
    "operations",
    "overlays",
    "omitted_unsafe_files",
}
_TENSOR_MANIFEST_KEYS = {
    "schema_version",
    "base_tensors",
    "changed_tensors",
    "unchanged_tensor_count",
}


@dataclass(frozen=True, slots=True)
class CapsuleValidation:
    path: Path
    surgery_id: str
    operation_count: int
    file_count: int
    total_bytes: int


def validate_capsule(path: str | Path) -> CapsuleValidation:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise CapsuleFormatError(f"capsule directory is unavailable: {root}")
    checksums = _read_checksums(root)
    actual_files = {
        file.relative_to(root).as_posix()
        for file in root.rglob("*")
        if file.is_file() and file.name != "SHA256SUMS"
    }
    if actual_files != set(checksums):
        missing = sorted(set(checksums) - actual_files)
        unexpected = sorted(actual_files - set(checksums))
        raise CapsuleFormatError(
            f"capsule file census mismatch (missing={missing!r}, unexpected={unexpected!r})"
        )
    for relative, expected in checksums.items():
        actual = sha256_file(root / relative)
        if actual != expected:
            raise CapsuleFormatError(f"capsule checksum mismatch: {relative}")

    manifest = _read_json(root / "manifest.json", "manifest")
    tensor_manifest = _read_json(root / "tensor-manifest.json", "tensor manifest")
    recipe = _read_json(root / "recipe.json", "recipe")
    if set(manifest) != _MANIFEST_KEYS:
        raise CapsuleFormatError("manifest keys do not match capsule-v1 schema")
    if set(tensor_manifest) != _TENSOR_MANIFEST_KEYS:
        raise CapsuleFormatError("tensor manifest keys do not match capsule-v1 schema")
    if manifest.get("schema_version") != 1 or tensor_manifest.get("schema_version") != 1:
        raise CapsuleFormatError("unsupported capsule schema version")
    if manifest.get("format") != "aipithicus.surgery-capsule":
        raise CapsuleFormatError("unsupported capsule format")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise CapsuleFormatError("manifest creation timestamp is invalid")
    surgery_id = manifest.get("surgery_id")
    if not isinstance(surgery_id, str) or _SHA256.fullmatch(surgery_id) is None:
        raise CapsuleFormatError("manifest surgery_id is invalid")

    base = manifest.get("base")
    operations = manifest.get("operations")
    if not isinstance(base, dict) or not isinstance(operations, list) or not operations:
        raise CapsuleFormatError("manifest base or operations are invalid")
    base_tensors = tensor_manifest.get("base_tensors")
    changed_tensors = tensor_manifest.get("changed_tensors")
    if not isinstance(base_tensors, list) or not isinstance(changed_tensors, list):
        raise CapsuleFormatError("tensor manifest tensor lists are invalid")
    base_tensor_identities = _tensor_identities(base_tensors, "base")
    changed_tensor_identities = _tensor_identities(changed_tensors, "changed")
    unchanged_count = tensor_manifest.get("unchanged_tensor_count")
    if (
        isinstance(unchanged_count, bool)
        or not isinstance(unchanged_count, int)
        or unchanged_count < 0
        or len(base_tensor_identities) != len(changed_tensor_identities) + unchanged_count
        or not changed_tensor_identities
    ):
        raise CapsuleFormatError("tensor manifest counts are inconsistent")
    base_identity_hash = sha256_bytes(
        canonical_bytes(
            {
                "schema_version": 1,
                "tensors": _ordered_identities(base_tensor_identities),
            }
        )
    )
    if base.get("tensor_manifest_sha256") != base_identity_hash:
        raise CapsuleFormatError("base tensor identity does not match tensor manifest")
    base_tensor_count = base.get("tensor_count")
    if (
        isinstance(base_tensor_count, bool)
        or not isinstance(base_tensor_count, int)
        or base_tensor_count != len(base_tensors)
    ):
        raise CapsuleFormatError("base tensor count does not match tensor manifest")

    for operation in operations:
        _semantic_operation(operation)
    expected_id = sha256_bytes(
        canonical_bytes(
            {
                "schema_version": 1,
                "base": base,
                "target_tensors": _ordered_identities(changed_tensor_identities),
            }
        )
    )
    if surgery_id != expected_id:
        raise CapsuleFormatError("surgery identity does not match target tensor identities")
    _validate_operation_chains(operations, base_tensors, changed_tensors)
    _validate_recipe(recipe, surgery_id, operations)

    payload_path = root / "delta.safetensors"
    with safe_open(payload_path, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        referenced: set[str] = set()
        for operation in operations:
            payload = operation.get("payload")
            if not isinstance(payload, dict) or not payload:
                raise CapsuleFormatError("operation payload is invalid")
            operation_payload: dict[str, torch.Tensor] = {}
            for logical_name, record in payload.items():
                if not isinstance(record, dict):
                    raise CapsuleFormatError("operation payload record is invalid")
                key = record.get("key")
                if not isinstance(key, str) or key in referenced or key not in available:
                    raise CapsuleFormatError("operation payload key is invalid or duplicated")
                tensor = handle.get_tensor(key)
                expected_record = {name: record.get(name) for name in ("dtype", "shape", "sha256")}
                if tensor_record(tensor) != expected_record:
                    raise CapsuleFormatError(f"operation payload tensor mismatch: {key}")
                operation_payload[logical_name] = tensor
                referenced.add(key)
            _validate_payload_contract(operation, operation_payload)
        if referenced != available:
            raise CapsuleFormatError("delta.safetensors contains unreferenced tensors")

    overlays = manifest.get("overlays")
    omitted = manifest.get("omitted_unsafe_files")
    if not isinstance(overlays, list) or not isinstance(omitted, list):
        raise CapsuleFormatError("manifest overlay records are invalid")
    for overlay in overlays:
        if not isinstance(overlay, dict) or set(overlay) != {"path", "sha256", "bytes"}:
            raise CapsuleFormatError("overlay record does not match capsule-v1 schema")
        relative = _safe_relative(overlay["path"])
        overlay_path = root / "files" / relative
        if (
            not overlay_path.is_file()
            or overlay.get("sha256") != sha256_file(overlay_path)
            or overlay.get("bytes") != overlay_path.stat().st_size
        ):
            raise CapsuleFormatError(f"overlay file does not match manifest: {relative}")

    total_bytes = sum(file.stat().st_size for file in root.rglob("*") if file.is_file())
    return CapsuleValidation(
        path=root,
        surgery_id=surgery_id,
        operation_count=len(operations),
        file_count=len(actual_files) + 1,
        total_bytes=total_bytes,
    )


def _semantic_operation(operation: object) -> dict[str, Any]:
    if not isinstance(operation, dict):
        raise CapsuleFormatError("operation must be an object")
    required = {
        "parameter",
        "kind",
        "metadata",
        "payload",
        "before_sha256",
        "after_sha256",
        "dtype",
        "shape",
    }
    if set(operation) != required:
        raise CapsuleFormatError("operation keys do not match capsule-v1 schema")
    parameter = operation.get("parameter")
    kind = operation.get("kind")
    dtype = operation.get("dtype")
    shape = operation.get("shape")
    if (
        not isinstance(parameter, str)
        or not parameter
        or any(ord(character) < 32 for character in parameter)
        or not isinstance(kind, str)
        or not isinstance(dtype, str)
        or _DTYPE.fullmatch(dtype) is None
        or not isinstance(shape, list)
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
        or _SHA256.fullmatch(str(operation.get("before_sha256"))) is None
        or _SHA256.fullmatch(str(operation.get("after_sha256"))) is None
    ):
        raise CapsuleFormatError("operation tensor identity is invalid")
    _validate_operation_metadata(kind, operation.get("metadata"))
    payload = operation.get("payload")
    if not isinstance(payload, dict):
        raise CapsuleFormatError("operation payload must be an object")
    semantic_payload: dict[str, Any] = {}
    for name, record in payload.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise CapsuleFormatError("operation payload entry is invalid")
        if set(record) != {"key", "dtype", "shape", "sha256"}:
            raise CapsuleFormatError("operation payload metadata is invalid")
        semantic_payload[name] = {
            field: record[field] for field in ("dtype", "shape", "sha256")
        }
    payload_names = set(semantic_payload)
    if kind == "project_direction":
        if payload_names not in ({"direction"}, {"direction", "a", "b"}):
            raise CapsuleFormatError("project_direction payload names are invalid")
        compatible = operation["metadata"]["peft_compatible"]
        if compatible != (payload_names == {"direction", "a", "b"}):
            raise CapsuleFormatError("project_direction PEFT metadata and payload disagree")
    elif kind not in _OPERATION_PAYLOADS or payload_names != _OPERATION_PAYLOADS[kind]:
        raise CapsuleFormatError(f"operation payload names are invalid for {kind!r}")
    if kind in {"add_sparse_rows", "add_sparse_elements"}:
        if payload["indices"].get("dtype") != "int64":
            raise CapsuleFormatError("sparse operation indices must use int64")
    return {**operation, "payload": semantic_payload}


def _validate_operation_metadata(kind: str, value: object) -> None:
    if not isinstance(value, dict):
        raise CapsuleFormatError("operation metadata must be an object")
    if kind == "replace_tensor":
        expected = set()
    elif kind in {"add_dense", "add_sparse_elements"}:
        expected = {"compute_dtype"}
    elif kind == "add_sparse_rows":
        expected = {"compute_dtype", "axis"}
    elif kind == "add_low_rank":
        expected = {"compute_dtype", "rank"}
    elif kind == "project_direction":
        required = {
            "norm_preserve",
            "regularization",
            "projection_row_fraction",
            "max_norm_ratio",
            "layout",
            "peft_compatible",
        }
        optional = {"compute_dtype", "rank", "peft_validation"}
        if not required.issubset(value) or set(value) - required - optional:
            raise CapsuleFormatError("project_direction metadata keys are invalid")
        if not isinstance(value["norm_preserve"], bool) or not isinstance(
            value["peft_compatible"], bool
        ):
            raise CapsuleFormatError("project_direction boolean metadata is invalid")
        regularization = _finite_number(value["regularization"], "regularization")
        row_fraction = _finite_number(
            value["projection_row_fraction"], "projection_row_fraction"
        )
        max_ratio = _finite_number(value["max_norm_ratio"], "max_norm_ratio")
        if not 0.0 <= regularization <= 1.0 or not 0.0 < row_fraction <= 1.0 or max_ratio <= 0:
            raise CapsuleFormatError("project_direction numeric metadata is out of range")
        if value["layout"] not in {"standard", "transposed"}:
            raise CapsuleFormatError("project_direction layout is invalid")
        if value["peft_compatible"]:
            if not optional.issubset(value):
                raise CapsuleFormatError("PEFT-compatible projection metadata is incomplete")
            if not isinstance(value["rank"], int) or isinstance(value["rank"], bool) or value["rank"] <= 0:
                raise CapsuleFormatError("projection rank is invalid")
            _validate_compute_dtype(value["compute_dtype"])
            validation = value["peft_validation"]
            if not isinstance(validation, dict) or set(validation) != {
                "rtol",
                "atol",
                "max_abs_error",
            }:
                raise CapsuleFormatError("PEFT validation metadata is invalid")
            for name in ("rtol", "atol", "max_abs_error"):
                if _finite_number(validation[name], name) < 0:
                    raise CapsuleFormatError("PEFT validation metric is negative")
        elif set(value) != required:
            raise CapsuleFormatError("native-only projection has unexpected PEFT metadata")
        return
    else:
        raise CapsuleFormatError(f"unsupported operation kind: {kind!r}")
    if set(value) != expected:
        raise CapsuleFormatError(f"operation metadata keys are invalid for {kind!r}")
    if "compute_dtype" in value:
        _validate_compute_dtype(value["compute_dtype"])
    if kind == "add_sparse_rows" and value["axis"] != 0:
        raise CapsuleFormatError("sparse row operation axis must be zero")
    if kind == "add_low_rank" and (
        not isinstance(value["rank"], int)
        or isinstance(value["rank"], bool)
        or value["rank"] <= 0
    ):
        raise CapsuleFormatError("low-rank operation rank is invalid")


def _validate_payload_contract(
    operation: dict[str, Any],
    payload: dict[str, torch.Tensor],
) -> None:
    kind = operation["kind"]
    shape = tuple(operation["shape"])
    dtype = operation["dtype"]
    metadata = operation["metadata"]

    if kind == "replace_tensor":
        _require_tensor(payload["target"], dtype=dtype, shape=shape, label="replacement")
        return
    if kind == "add_dense":
        _require_tensor(
            payload["delta"],
            dtype=metadata["compute_dtype"],
            shape=shape,
            label="dense delta",
        )
        return
    if kind == "add_sparse_rows":
        if not shape:
            raise CapsuleFormatError("sparse row operation requires a non-scalar tensor")
        indices = payload["indices"]
        _validate_indices(indices, upper_bound=shape[0], label="sparse row")
        _require_tensor(
            payload["delta"],
            dtype=metadata["compute_dtype"],
            shape=(indices.numel(), *shape[1:]),
            label="sparse row delta",
        )
        return
    if kind == "add_sparse_elements":
        indices = payload["indices"]
        _validate_indices(indices, upper_bound=math.prod(shape), label="sparse element")
        _require_tensor(
            payload["delta"],
            dtype=metadata["compute_dtype"],
            shape=(indices.numel(),),
            label="sparse element delta",
        )
        return
    if kind == "add_low_rank":
        _validate_low_rank_payload(operation, payload)
        return
    if kind == "project_direction":
        if dtype not in _FLOAT_DTYPES or len(shape) != 2:
            raise CapsuleFormatError("projection operation requires a floating matrix")
        direction = payload["direction"]
        expected_width = shape[1] if metadata["layout"] == "standard" else shape[0]
        if (
            not direction.is_floating_point()
            or direction.ndim != 1
            or direction.shape[0] != expected_width
        ):
            raise CapsuleFormatError("projection direction contract is invalid")
        if metadata["peft_compatible"]:
            _validate_low_rank_payload(operation, payload)
        return
    raise CapsuleFormatError(f"unsupported operation kind: {kind!r}")


def _validate_low_rank_payload(
    operation: dict[str, Any],
    payload: dict[str, torch.Tensor],
) -> None:
    shape = tuple(operation["shape"])
    metadata = operation["metadata"]
    if len(shape) != 2:
        raise CapsuleFormatError("low-rank operation requires a matrix")
    rank = metadata["rank"]
    compute_dtype = metadata["compute_dtype"]
    _require_tensor(
        payload["b"],
        dtype=compute_dtype,
        shape=(shape[0], rank),
        label="low-rank B factor",
    )
    _require_tensor(
        payload["a"],
        dtype=compute_dtype,
        shape=(rank, shape[1]),
        label="low-rank A factor",
    )


def _validate_indices(tensor: torch.Tensor, *, upper_bound: int, label: str) -> None:
    if tensor.dtype != torch.int64 or tensor.ndim != 1 or tensor.numel() == 0:
        raise CapsuleFormatError(f"{label} indices are invalid")
    if upper_bound <= 0 or int(tensor[0]) < 0 or int(tensor[-1]) >= upper_bound:
        raise CapsuleFormatError(f"{label} indices are out of range")
    if tensor.numel() > 1 and not bool(torch.all(tensor[1:] > tensor[:-1]).item()):
        raise CapsuleFormatError(f"{label} indices must be strictly increasing")


def _require_tensor(
    tensor: torch.Tensor,
    *,
    dtype: str,
    shape: tuple[int, ...],
    label: str,
) -> None:
    if dtype_name(tensor.dtype) != dtype or tuple(tensor.shape) != shape:
        raise CapsuleFormatError(f"{label} tensor contract is invalid")


def _validate_recipe(
    recipe: dict[str, Any],
    surgery_id: str,
    operations: list[object],
) -> None:
    expected = {
        "schema_version",
        "surgery_id",
        "recipe",
        "codec_summary",
        "require_exact",
    }
    if set(recipe) != expected or recipe.get("schema_version") != 1:
        raise CapsuleFormatError("recipe keys do not match capsule-v1 schema")
    if recipe.get("surgery_id") != surgery_id or not isinstance(recipe.get("recipe"), dict):
        raise CapsuleFormatError("recipe identity or payload is invalid")
    if not isinstance(recipe.get("require_exact"), bool):
        raise CapsuleFormatError("recipe exactness flag is invalid")
    actual_summary = dict(
        sorted(Counter(operation["kind"] for operation in operations).items())
    )
    if recipe.get("codec_summary") != actual_summary:
        raise CapsuleFormatError("recipe codec summary does not match operations")


def _validate_compute_dtype(value: object) -> None:
    if value not in {"float32", "float64"}:
        raise CapsuleFormatError("operation compute dtype must be float32 or float64")


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CapsuleFormatError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise CapsuleFormatError(f"{label} must be finite")
    return parsed


def _validate_operation_chains(
    operations: list[object],
    base_tensors: list[object],
    changed_tensors: list[object],
) -> None:
    base = _tensor_identities(base_tensors, "base")
    changed = _tensor_identities(changed_tensors, "changed")
    chains: dict[str, list[dict[str, Any]]] = {}
    for operation in operations:
        if not isinstance(operation, dict) or not isinstance(operation.get("parameter"), str):
            raise CapsuleFormatError("operation parameter is invalid")
        chains.setdefault(operation["parameter"], []).append(operation)
    if set(chains) != set(changed):
        raise CapsuleFormatError("operation parameters do not match changed tensor records")
    for name, chain in chains.items():
        if name not in base:
            raise CapsuleFormatError(f"operation parameter is absent from base: {name}")
        if (
            base[name]["dtype"] != changed[name]["dtype"]
            or base[name]["shape"] != changed[name]["shape"]
        ):
            raise CapsuleFormatError(f"changed tensor contract differs from base: {name}")
        expected = base[name]["sha256"]
        for operation in chain:
            if operation.get("dtype") != base[name]["dtype"] or operation.get(
                "shape"
            ) != base[name]["shape"]:
                raise CapsuleFormatError(f"operation tensor contract is inconsistent: {name}")
            if operation.get("before_sha256") != expected:
                raise CapsuleFormatError(f"operation before-hash chain is broken: {name}")
            expected = operation.get("after_sha256")
        if expected != changed[name]["sha256"]:
            raise CapsuleFormatError(f"operation after hash does not match target: {name}")


def _tensor_identities(
    records: list[object], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        expected_keys = (
            {"name", "shard", "dtype", "shape", "sha256"}
            if label == "base"
            else {"name", "target_shard", "dtype", "shape", "sha256"}
        )
        if not isinstance(record, dict) or set(record) != expected_keys:
            raise CapsuleFormatError(f"{label} tensor record keys are invalid")
        shard_key = "shard" if label == "base" else "target_shard"
        _safe_relative(record[shard_key])
        identity = _tensor_identity(record)
        name = identity["name"]
        if name in result:
            raise CapsuleFormatError(f"{label} tensor identity is invalid")
        result[name] = identity
    return result


def _ordered_identities(
    records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [records[name] for name in sorted(records)]


def _read_checksums(root: Path) -> dict[str, str]:
    path = root / "SHA256SUMS"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CapsuleFormatError("capsule SHA256SUMS is unavailable") from error
    checksums: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if separator != "  " or _SHA256.fullmatch(digest) is None:
            raise CapsuleFormatError("capsule SHA256SUMS contains an invalid line")
        safe = _safe_relative(relative)
        if safe in checksums or safe == "SHA256SUMS":
            raise CapsuleFormatError("capsule SHA256SUMS contains a duplicate or recursive path")
        checksums[safe] = digest
    required = {"manifest.json", "tensor-manifest.json", "recipe.json", "delta.safetensors"}
    if not required.issubset(checksums):
        raise CapsuleFormatError("capsule is missing one or more required files")
    return checksums


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapsuleFormatError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise CapsuleFormatError(f"{label} must be an object")
    return value


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CapsuleFormatError("capsule contains an invalid relative path")
    path = Path(value)
    if path.is_absolute() or path.drive or ".." in path.parts or not path.parts:
        raise CapsuleFormatError("capsule contains an unsafe relative path")
    return path.as_posix()


def _tensor_identity(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise CapsuleFormatError("tensor identity record is invalid")
    required = ("name", "dtype", "shape", "sha256")
    if any(field not in record for field in required):
        raise CapsuleFormatError("tensor identity record is incomplete")
    identity = {field: record[field] for field in required}
    if (
        not isinstance(identity["name"], str)
        or not identity["name"]
        or not isinstance(identity["dtype"], str)
        or _DTYPE.fullmatch(identity["dtype"]) is None
        or not isinstance(identity["shape"], list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity["shape"]
        )
        or not isinstance(identity["sha256"], str)
        or _SHA256.fullmatch(identity["sha256"]) is None
    ):
        raise CapsuleFormatError("tensor identity record is invalid")
    return identity
