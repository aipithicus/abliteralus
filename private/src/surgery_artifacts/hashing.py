"""Hashing helpers with dtype-safe tensor byte access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return contiguous logical bytes without relying on NumPy dtype support."""

    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": dtype_name(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_sha256(tensor),
    }
