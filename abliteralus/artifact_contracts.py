"""Neutral callback contract for post-surgery artifact materialization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ArtifactStageContext:
    """Immutable paths and provenance offered to an artifact backend."""

    spec: Any
    run_dir: Path
    base_checkpoint: Path
    target_checkpoint: Path
    manifest_path: Path
    mutation_hints: tuple[Any, ...] = ()


class ArtifactStage(Protocol):
    def __call__(self, context: ArtifactStageContext) -> Mapping[str, Any]: ...


class MutationHintSink(Protocol):
    def bind_model(self, model: nn.Module) -> None: ...

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
    ) -> None: ...

    def snapshot(self) -> tuple[Any, ...]: ...


_ACTIVE_MUTATION_SINK: ContextVar[MutationHintSink | None] = ContextVar(
    "abliteralus_active_mutation_sink", default=None
)


@contextmanager
def capture_mutation_hints(sink: MutationHintSink | None) -> Iterator[None]:
    token = _ACTIVE_MUTATION_SINK.set(sink)
    try:
        yield
    finally:
        _ACTIVE_MUTATION_SINK.reset(token)


def emit_projection_hint(
    module: nn.Module,
    parameter_name: str,
    direction: torch.Tensor,
    *,
    norm_preserve: bool,
    regularization: float,
    projection_row_fraction: float,
    max_norm_ratio: float,
) -> None:
    sink = _ACTIVE_MUTATION_SINK.get()
    if sink is not None:
        sink.record_projection(
            module,
            parameter_name,
            direction,
            norm_preserve=norm_preserve,
            regularization=regularization,
            projection_row_fraction=projection_row_fraction,
            max_norm_ratio=max_norm_ratio,
        )
