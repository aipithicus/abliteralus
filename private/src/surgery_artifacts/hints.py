"""Capture compact, non-private projection recipes from a running pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ProjectionHint:
    parameter: str
    direction: torch.Tensor
    norm_preserve: bool
    regularization: float
    projection_row_fraction: float
    max_norm_ratio: float


class ProjectionHintRecorder:
    """Map live module identities to stable state-dict names and retain small recipes."""

    def __init__(self) -> None:
        self._names: dict[tuple[int, str], str] = {}
        self._hints: list[ProjectionHint] = []
        self._lock = Lock()

    def bind_model(self, model: nn.Module) -> None:
        names: dict[tuple[int, str], str] = {}
        for module_name, module in model.named_modules():
            for parameter_name, _parameter in module.named_parameters(recurse=False):
                full_name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                names[(id(module), parameter_name)] = full_name
        with self._lock:
            self._names = names

    def record_projection(
        self,
        module: nn.Module,
        parameter_name: str,
        direction: torch.Tensor,
        *,
        norm_preserve: bool,
        regularization: float,
        projection_row_fraction: float,
        max_norm_ratio: float,
    ) -> None:
        with self._lock:
            name = self._names.get((id(module), parameter_name))
            if name is None:
                return
            self._hints.append(
                ProjectionHint(
                    parameter=name,
                    direction=direction.detach().cpu().contiguous().clone(),
                    norm_preserve=bool(norm_preserve),
                    regularization=float(regularization),
                    projection_row_fraction=float(projection_row_fraction),
                    max_norm_ratio=float(max_norm_ratio),
                )
            )

    def snapshot(self) -> tuple[ProjectionHint, ...]:
        with self._lock:
            return tuple(self._hints)
