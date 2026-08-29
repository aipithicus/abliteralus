"""Numerical precision as a declared, recorded parameter of a study.

A guard margin is a difference of two logits of similar magnitude, so its error
floor is set by the magnitude of the operands rather than of the result. That
floor has to travel with the measurement: a layer reported as causally inert is
only inert relative to the resolution the run actually had.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import torch

from .errors import ContractError

INHERIT = "inherit"
COMPUTE_CHOICES = (INHERIT, "float32", "bfloat16", "float16")
READOUT_CHOICES = ("float32", "float64")

_TORCH_DTYPES: Mapping[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}


def torch_dtype(name: str) -> torch.dtype:
    dtype = _TORCH_DTYPES.get(name)
    if dtype is None:
        raise ContractError(f"unsupported dtype {name!r}")
    return dtype


def resolution_floor(dtype: torch.dtype, magnitude: float) -> float:
    """Return the representable spacing of `dtype` at the given magnitude.

    This is the quantum on which every difference of two such values must land,
    and therefore the smallest causal effect the readout can distinguish from zero.
    """

    value = torch.tensor(max(abs(float(magnitude)), 1.0), dtype=dtype)
    following = torch.nextafter(value, torch.tensor(float("inf"), dtype=dtype))
    return float((following - value).item())


@dataclass(frozen=True)
class PrecisionSpec:
    """Declared compute dtype, readout dtype, and TF32 policy for one study."""

    compute: str
    readout: str
    allow_tf32: bool

    def resolve_compute(self, inherited: str) -> str:
        """Return the dtype the model loads in, deferring to the surgery spec by default."""

        return inherited if self.compute == INHERIT else self.compute

    @property
    def readout_dtype(self) -> torch.dtype:
        return torch_dtype(self.readout)

    def summary(self, *, inherited: str | None = None) -> dict[str, Any]:
        resolved = self.resolve_compute(inherited) if inherited is not None else self.compute
        return {
            "compute": resolved,
            "compute_declared": self.compute,
            "readout": self.readout,
            "allow_tf32": self.allow_tf32,
        }


DEFAULT_PRECISION = PrecisionSpec(compute=INHERIT, readout="float32", allow_tf32=False)


@contextmanager
def pinned_matmul_precision(spec: PrecisionSpec) -> Iterator[dict[str, Any]]:
    """Pin TF32 for the duration of a run and report what was actually in force.

    On Tensor Core hardware a float32 matmul may be computed in TF32, whose
    significand is 11 bits — the same width as float16. A run that declares
    float32 and silently receives TF32 would report a resolution it never had.
    """

    matmul = torch.backends.cuda.matmul
    cudnn = torch.backends.cudnn
    previous = (matmul.allow_tf32, cudnn.allow_tf32, torch.get_float32_matmul_precision())
    matmul.allow_tf32 = spec.allow_tf32
    cudnn.allow_tf32 = spec.allow_tf32
    torch.set_float32_matmul_precision("high" if spec.allow_tf32 else "highest")
    try:
        yield {
            "allow_tf32": bool(matmul.allow_tf32),
            "cudnn_allow_tf32": bool(cudnn.allow_tf32),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
    finally:
        matmul.allow_tf32, cudnn.allow_tf32 = previous[0], previous[1]
        torch.set_float32_matmul_precision(previous[2])


def parse_precision(raw: object, *, label: str) -> PrecisionSpec:
    """Validate a precision block, defaulting to inherited compute with a float32 readout."""

    if raw is None:
        return DEFAULT_PRECISION
    if not isinstance(raw, dict):
        raise ContractError(f"{label} must be a mapping")
    unknown = sorted(set(raw) - {"compute", "readout", "allow_tf32"})
    if unknown:
        raise ContractError(f"{label} has unknown keys: {', '.join(unknown)}")

    compute = raw.get("compute", DEFAULT_PRECISION.compute)
    if compute not in COMPUTE_CHOICES:
        raise ContractError(f"{label}.compute must be one of {', '.join(COMPUTE_CHOICES)}")
    readout = raw.get("readout", DEFAULT_PRECISION.readout)
    if readout not in READOUT_CHOICES:
        raise ContractError(f"{label}.readout must be one of {', '.join(READOUT_CHOICES)}")
    allow_tf32 = raw.get("allow_tf32", DEFAULT_PRECISION.allow_tf32)
    if not isinstance(allow_tf32, bool):
        raise ContractError(f"{label}.allow_tf32 must be true or false")
    return PrecisionSpec(compute=compute, readout=readout, allow_tf32=allow_tf32)
