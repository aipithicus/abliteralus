"""Exact tensor-operation encoders for capsule v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .errors import CapsuleFormatError, ExactnessError
from .hashing import dtype_name, tensor_record, tensor_sha256

_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_DTYPES = {
    dtype_name(dtype): dtype
    for dtype in (
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}
_PRIORITY = {
    "project_direction": 0,
    "add_low_rank": 0,
    "add_sparse_rows": 1,
    "add_sparse_elements": 2,
    "add_dense": 3,
    "replace_tensor": 4,
}


@dataclass(frozen=True, slots=True)
class EncodedOperation:
    parameter: str
    kind: str
    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    before_sha256: str
    after_sha256: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def payload_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.tensors.values())

    def semantic_record(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "kind": self.kind,
            "metadata": self.metadata,
            "payload": {
                name: tensor_record(tensor) for name, tensor in sorted(self.tensors.items())
            },
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    def manifest_record(self, payload_keys: dict[str, str]) -> dict[str, Any]:
        record = self.semantic_record()
        record["payload"] = {
            name: {**tensor_record(self.tensors[name]), "key": payload_keys[name]}
            for name in sorted(self.tensors)
        }
        return record


def encode_tensor_change(
    parameter: str,
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    require_exact: bool = True,
    max_low_rank: int = 32,
) -> EncodedOperation:
    """Choose the smallest verified representation of one changed tensor."""

    before = before.detach().cpu().contiguous()
    after = after.detach().cpu().contiguous()
    if before.shape != after.shape or before.dtype != after.dtype:
        raise ExactnessError(f"tensor contract changed for {parameter}")
    if torch.equal(before, after):
        raise ValueError(f"tensor is unchanged: {parameter}")

    candidates: list[EncodedOperation] = []
    if before.dtype in _FLOAT_DTYPES:
        compute_dtype = _compute_dtype(before.dtype)
        delta = after.to(compute_dtype) - before.to(compute_dtype)
        if before.ndim == 2 and min(before.shape) <= 256:
            low_rank = _encode_low_rank(
                parameter,
                before,
                after,
                delta,
                compute_dtype=compute_dtype,
                max_rank=max_low_rank,
            )
            if low_rank is not None:
                candidates.append(low_rank)
            sparse_rows = _encode_sparse_rows(
                parameter, before, after, delta, compute_dtype=compute_dtype
            )
            if sparse_rows is not None:
                candidates.append(sparse_rows)
        sparse_elements = _encode_sparse_elements(
            parameter, before, after, delta, compute_dtype=compute_dtype
        )
        if sparse_elements is not None:
            candidates.append(sparse_elements)
        dense = _operation(
            parameter,
            "add_dense",
            before,
            after,
            {"compute_dtype": dtype_name(compute_dtype)},
            {"delta": delta.contiguous()},
        )
        if _replays_exactly(dense, before, after):
            candidates.append(dense)

    replacement = _operation(
        parameter,
        "replace_tensor",
        before,
        after,
        {},
        {"target": after.clone()},
    )
    candidates.append(replacement)
    verified = [candidate for candidate in candidates if _replays_exactly(candidate, before, after)]
    if not verified:
        if require_exact:
            raise ExactnessError(f"no exact codec is available for {parameter}")
        return replacement
    return min(verified, key=lambda item: (item.payload_bytes, _PRIORITY[item.kind]))


def encode_projection_hint(
    parameter: str,
    before: torch.Tensor,
    hint: object,
) -> EncodedOperation | None:
    """Encode an ABLITERALUS projection recipe before its factors are lost to rounding."""

    from abliteralus.analysis.numerical_contracts import (
        project_weight_against_direction,
        select_projection_coefficients,
    )

    direction = _hint_value(hint, "direction")
    if not isinstance(direction, torch.Tensor):
        raise ExactnessError(f"projection hint direction is invalid for {parameter}")
    norm_preserve = bool(_hint_value(hint, "norm_preserve"))
    regularization = float(_hint_value(hint, "regularization"))
    projection_row_fraction = float(_hint_value(hint, "projection_row_fraction"))
    max_norm_ratio = float(_hint_value(hint, "max_norm_ratio"))
    before = before.detach().cpu().contiguous()
    direction = direction.detach().cpu().contiguous()
    result = project_weight_against_direction(
        before,
        direction,
        norm_preserve=norm_preserve,
        regularization=regularization,
        projection_row_fraction=projection_row_fraction,
        max_norm_ratio=max_norm_ratio,
    )
    if not result.projected:
        return None
    metadata: dict[str, Any] = {
        "norm_preserve": norm_preserve,
        "regularization": regularization,
        "projection_row_fraction": projection_row_fraction,
        "max_norm_ratio": max_norm_ratio,
        "layout": result.layout,
        "peft_compatible": False,
    }
    tensors = {"direction": direction}
    if not norm_preserve and before.ndim == 2:
        compute_dtype = _compute_dtype(before.dtype)
        work = before.to(compute_dtype)
        d = direction.to(compute_dtype).reshape(-1, 1)
        d = d / d.norm()
        scale = 1.0 - regularization
        if result.layout == "standard":
            coeff = select_projection_coefficients(
                work @ d, projection_row_fraction
            )
            b = -(scale * coeff)
            a = d.T
        elif result.layout == "transposed":
            coeff = select_projection_coefficients(
                d.T @ work, projection_row_fraction
            )
            b = -(scale * d)
            a = coeff
        else:
            b = a = None
        if b is not None and a is not None:
            peft_result = (work + b @ a).to(before.dtype)
            rtol, atol = _peft_tolerances(before.dtype)
            if torch.allclose(peft_result, result.weight, rtol=rtol, atol=atol):
                difference = (peft_result.to(torch.float64) - result.weight.to(torch.float64)).abs()
                tensors.update({"b": b.contiguous(), "a": a.contiguous()})
                metadata["peft_compatible"] = True
                metadata["rank"] = 1
                metadata["compute_dtype"] = dtype_name(compute_dtype)
                metadata["peft_validation"] = {
                    "rtol": rtol,
                    "atol": atol,
                    "max_abs_error": float(difference.max().item()),
                }
    operation = _operation(
        parameter,
        "project_direction",
        before,
        result.weight,
        metadata,
        tensors,
    )
    return operation if _replays_exactly(operation, before, result.weight) else None


def apply_operation(
    operation: dict[str, Any],
    base: torch.Tensor,
    payload: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply one validated operation to a base tensor."""

    kind = operation.get("kind")
    metadata = operation.get("metadata")
    if not isinstance(metadata, dict):
        raise CapsuleFormatError("operation metadata must be an object")
    base = base.detach().cpu().contiguous()
    if kind == "replace_tensor":
        result = payload["target"].to(dtype=base.dtype).contiguous()
    elif kind == "add_dense":
        compute = _parse_dtype(metadata.get("compute_dtype"))
        result = (base.to(compute) + payload["delta"].to(compute)).to(base.dtype)
    elif kind == "add_sparse_rows":
        compute = _parse_dtype(metadata.get("compute_dtype"))
        result = base.clone()
        indices = payload["indices"].to(torch.int64)
        updated = result.index_select(0, indices).to(compute) + payload["delta"].to(compute)
        result.index_copy_(0, indices, updated.to(base.dtype))
    elif kind == "add_sparse_elements":
        compute = _parse_dtype(metadata.get("compute_dtype"))
        result = base.clone()
        flat = result.reshape(-1)
        indices = payload["indices"].to(torch.int64)
        updated = flat.index_select(0, indices).to(compute) + payload["delta"].to(compute)
        flat.index_copy_(0, indices, updated.to(base.dtype))
    elif kind == "add_low_rank":
        compute = _parse_dtype(metadata.get("compute_dtype"))
        update = payload["b"].to(compute) @ payload["a"].to(compute)
        result = (base.to(compute) + update).to(base.dtype)
    elif kind == "project_direction":
        from abliteralus.analysis.numerical_contracts import project_weight_against_direction

        result = project_weight_against_direction(
            base,
            payload["direction"],
            norm_preserve=bool(metadata.get("norm_preserve")),
            regularization=float(metadata.get("regularization")),
            projection_row_fraction=float(metadata.get("projection_row_fraction")),
            max_norm_ratio=float(metadata.get("max_norm_ratio")),
        ).weight
    else:
        raise CapsuleFormatError(f"unsupported capsule operation: {kind!r}")
    return result.contiguous()


def _encode_low_rank(
    parameter: str,
    before: torch.Tensor,
    after: torch.Tensor,
    delta: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    max_rank: int,
) -> EncodedOperation | None:
    rows, columns = delta.shape
    limit = min(max_rank, rows, columns)
    if limit <= 0:
        return None
    # Greedy residual-column selection avoids a full matrix SVD for large model
    # weights. The final least-squares solve determines whether the apparent
    # low rank is an exact storage representation after casting to target dtype.
    selected: list[int] = []
    basis: torch.Tensor | None = None
    residual = delta
    for _rank in range(1, limit + 1):
        norms = torch.linalg.vector_norm(residual, dim=0)
        index = int(torch.argmax(norms).item())
        if float(norms[index]) == 0.0 or index in selected:
            break
        selected.append(index)
        basis = delta[:, selected].contiguous()
        factors = torch.linalg.lstsq(basis, delta).solution.contiguous()
        operation = _operation(
            parameter,
            "add_low_rank",
            before,
            after,
            {"compute_dtype": dtype_name(compute_dtype), "rank": len(selected)},
            {"b": basis, "a": factors},
        )
        if _replays_exactly(operation, before, after):
            return operation
        residual = delta - basis @ factors
    return None


def _encode_sparse_rows(
    parameter: str,
    before: torch.Tensor,
    after: torch.Tensor,
    delta: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> EncodedOperation | None:
    changed = torch.any(before != after, dim=tuple(range(1, before.ndim)))
    indices = torch.nonzero(changed, as_tuple=False).reshape(-1).to(torch.int64)
    if indices.numel() == 0 or indices.numel() == before.shape[0]:
        return None
    operation = _operation(
        parameter,
        "add_sparse_rows",
        before,
        after,
        {"compute_dtype": dtype_name(compute_dtype), "axis": 0},
        {"indices": indices, "delta": delta.index_select(0, indices).contiguous()},
    )
    return operation if _replays_exactly(operation, before, after) else None


def _encode_sparse_elements(
    parameter: str,
    before: torch.Tensor,
    after: torch.Tensor,
    delta: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> EncodedOperation | None:
    indices = torch.nonzero((before != after).reshape(-1), as_tuple=False).reshape(-1)
    if indices.numel() == 0 or indices.numel() * 2 >= before.numel():
        return None
    operation = _operation(
        parameter,
        "add_sparse_elements",
        before,
        after,
        {"compute_dtype": dtype_name(compute_dtype)},
        {
            "indices": indices.to(torch.int64),
            "delta": delta.reshape(-1).index_select(0, indices).contiguous(),
        },
    )
    return operation if _replays_exactly(operation, before, after) else None


def _operation(
    parameter: str,
    kind: str,
    before: torch.Tensor,
    after: torch.Tensor,
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> EncodedOperation:
    return EncodedOperation(
        parameter=parameter,
        kind=kind,
        metadata=metadata,
        tensors={name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        before_sha256=tensor_sha256(before),
        after_sha256=tensor_sha256(after),
        dtype=dtype_name(before.dtype),
        shape=tuple(before.shape),
    )


def _replays_exactly(
    operation: EncodedOperation,
    before: torch.Tensor,
    after: torch.Tensor,
) -> bool:
    return torch.equal(
        apply_operation(operation.semantic_record(), before, operation.tensors),
        after,
    )


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return torch.float64


def _parse_dtype(value: object) -> torch.dtype:
    if not isinstance(value, str) or value not in _DTYPES:
        raise CapsuleFormatError(f"unsupported operation compute dtype: {value!r}")
    return _DTYPES[value]


def _hint_value(hint: object, name: str) -> object:
    if isinstance(hint, dict):
        if name not in hint:
            raise ExactnessError(f"projection hint is missing {name}")
        return hint[name]
    try:
        return getattr(hint, name)
    except AttributeError as error:
        raise ExactnessError(f"projection hint is missing {name}") from error


def _peft_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return 1e-3, 1e-3
    if dtype == torch.float32:
        return 1e-5, 1e-6
    return 1e-12, 1e-12
