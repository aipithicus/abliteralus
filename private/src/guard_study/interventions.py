"""Reversible, layer-specific activation steering primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from abliteralus.analysis.steering_vectors import SteeringVectorFactory
from torch import nn

from .errors import StudyRuntimeError


@dataclass(frozen=True)
class ContrastiveDirection:
    layer_idx: int
    direction: torch.Tensor
    projection_std: float
    safe_projection_mean: float
    unsafe_projection_mean: float
    projection_gap: float


def build_contrastive_direction(
    *,
    layer_idx: int,
    unsafe_activations: Sequence[torch.Tensor],
    safe_activations: Sequence[torch.Tensor],
) -> ContrastiveDirection:
    """Build one OBLITERATUS contrastive direction and its natural steering scale."""

    if not unsafe_activations or not safe_activations:
        raise StudyRuntimeError("contrastive direction requires both classes")
    vector = SteeringVectorFactory.from_contrastive_pairs(
        positive_activations=list(unsafe_activations),
        negative_activations=list(safe_activations),
        label="guard-unsafe-minus-safe",
        alpha=1.0,
    )
    direction = vector.direction.detach().float().cpu()
    safe = torch.stack([value.detach().float().cpu().reshape(-1) for value in safe_activations])
    unsafe = torch.stack([value.detach().float().cpu().reshape(-1) for value in unsafe_activations])
    if safe.shape[1] != direction.numel() or unsafe.shape[1] != direction.numel():
        raise StudyRuntimeError("activation and direction dimensions do not agree")
    all_projections = torch.cat((safe @ direction, unsafe @ direction))
    projection_std = float(all_projections.std(unbiased=False).item())
    if not torch.isfinite(torch.tensor(projection_std)) or projection_std < 1e-8:
        raise StudyRuntimeError(f"layer {layer_idx} has a degenerate contrastive scale")
    safe_mean = float((safe @ direction).mean().item())
    unsafe_mean = float((unsafe @ direction).mean().item())
    return ContrastiveDirection(
        layer_idx=layer_idx,
        direction=direction,
        projection_std=projection_std,
        safe_projection_mean=safe_mean,
        unsafe_projection_mean=unsafe_mean,
        projection_gap=unsafe_mean - safe_mean,
    )


def orthogonal_sham(direction: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Create a deterministic unit vector orthogonal to a learned direction."""

    learned = direction.detach().float().cpu().reshape(-1)
    learned = learned / learned.norm().clamp(min=1e-10)
    if learned.numel() < 2:
        raise StudyRuntimeError("orthogonal control requires hidden dimension >= 2")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    candidate = torch.randn(learned.shape, generator=generator)
    candidate = candidate - (candidate @ learned) * learned
    if candidate.norm() < 1e-8:
        basis = torch.zeros_like(learned)
        basis[int(torch.argmin(learned.abs()).item())] = 1.0
        candidate = basis - (basis @ learned) * learned
    return candidate / candidate.norm().clamp(min=1e-10)


class LayerSteeringHooks:
    """Temporarily add one signed vector at the last position of each target layer."""

    def __init__(
        self,
        layers: Sequence[nn.Module],
        vectors: Mapping[int, torch.Tensor],
        scales: Mapping[int, float],
        *,
        dose: float,
    ) -> None:
        self.layers = layers
        self.vectors = dict(vectors)
        self.scales = dict(scales)
        self.dose = float(dose)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def install(self) -> "LayerSteeringHooks":
        if self._handles:
            raise StudyRuntimeError("steering hooks are already installed")
        if set(self.vectors) != set(self.scales):
            raise StudyRuntimeError("steering vectors and scales must target the same layers")
        if not self.vectors:
            raise StudyRuntimeError("at least one steering layer is required")
        try:
            for layer_idx in sorted(self.vectors):
                if layer_idx < 0 or layer_idx >= len(self.layers):
                    raise StudyRuntimeError(f"steering layer {layer_idx} is out of range")
                vector = self.vectors[layer_idx].detach().float().cpu().reshape(-1)
                vector = vector / vector.norm().clamp(min=1e-10)
                scale = float(self.scales[layer_idx])
                handle = self.layers[layer_idx].register_forward_hook(
                    self._make_hook(vector, self.dose * scale)
                )
                self._handles.append(handle)
        except BaseException:
            self.remove()
            raise
        return self

    @staticmethod
    def _make_hook(vector: torch.Tensor, signed_scale: float):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
                raise StudyRuntimeError("steering target did not return hidden-state tensors")
            if hidden.shape[-1] != vector.numel():
                raise StudyRuntimeError(
                    "steering vector dimension does not match target hidden dimension"
                )
            direction = vector.to(device=hidden.device, dtype=hidden.dtype)
            steered = hidden.clone()
            if steered.ndim == 2:
                steered[:, :] = steered[:, :] + signed_scale * direction
            else:
                steered[..., -1, :] = steered[..., -1, :] + signed_scale * direction
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

        return hook

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    @property
    def is_active(self) -> bool:
        return bool(self._handles)

    def __enter__(self) -> "LayerSteeringHooks":
        return self.install()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.remove()


class LastTokenPatch:
    """Temporarily replace one layer's final-token activation for causal tracing."""

    def __init__(self, layer: nn.Module, activation: torch.Tensor) -> None:
        self.layer = layer
        self.activation = activation.detach().float().cpu().reshape(-1)
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __enter__(self) -> "LastTokenPatch":
        if self._handle is not None:
            raise StudyRuntimeError("last-token patch is already installed")

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
                raise StudyRuntimeError("patch target did not return hidden-state tensors")
            if hidden.shape[-1] != self.activation.numel():
                raise StudyRuntimeError("patch activation dimension does not match target layer")
            patched = hidden.clone()
            source = self.activation.to(device=hidden.device, dtype=hidden.dtype)
            if patched.ndim == 2:
                patched[:, :] = source
            else:
                patched[..., -1, :] = source
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        self._handle = self.layer.register_forward_hook(hook)
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@dataclass(frozen=True)
class ContrastiveBasis:
    """An orthonormal rank-r basis for the unsafe-minus-safe difference subspace."""

    layer_idx: int
    basis: torch.Tensor
    singular_values: tuple[float, ...]
    explained_fraction: tuple[float, ...]

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    def summary(self) -> dict[str, object]:
        return {
            "layer_idx": self.layer_idx,
            "rank": self.rank,
            "singular_values": list(self.singular_values),
            "explained_fraction": list(self.explained_fraction),
        }


def build_contrastive_basis(
    *,
    layer_idx: int,
    unsafe_activations: Sequence[torch.Tensor],
    safe_activations: Sequence[torch.Tensor],
    rank: int,
) -> ContrastiveBasis:
    """Decompose the per-pair difference vectors into their dominant orthonormal axes.

    Rank one recovers the dominant axis of the same differences the mean-difference
    direction averages. Higher ranks expose how much of the class separation that
    single axis was leaving behind, which is the measurement a chokepoint claim
    actually rests on: a bottleneck is a statement about rank, and a full-vector
    patch cannot distinguish rank one from rank five hundred.
    """

    if rank < 1:
        raise StudyRuntimeError("contrastive basis requires rank >= 1")
    if len(unsafe_activations) != len(safe_activations):
        raise StudyRuntimeError("contrastive basis requires matched pairs")
    if not unsafe_activations:
        raise StudyRuntimeError("contrastive basis requires at least one pair")

    unsafe = torch.stack([value.detach().float().cpu().reshape(-1) for value in unsafe_activations])
    safe = torch.stack([value.detach().float().cpu().reshape(-1) for value in safe_activations])
    if unsafe.shape != safe.shape:
        raise StudyRuntimeError("paired activations do not agree in dimension")
    differences = unsafe - safe

    available = int(min(differences.shape))
    if rank > available:
        raise StudyRuntimeError(
            f"layer {layer_idx} supports rank <= {available} from "
            f"{differences.shape[0]} pairs, not {rank}"
        )
    _, singular, right = torch.linalg.svd(differences, full_matrices=False)
    total = float((singular**2).sum().item())
    if total < 1e-12:
        raise StudyRuntimeError(f"layer {layer_idx} has no usable contrastive spread")
    basis = right[:rank].contiguous()
    return ContrastiveBasis(
        layer_idx=layer_idx,
        basis=basis,
        singular_values=tuple(float(value) for value in singular[:rank]),
        explained_fraction=tuple(float(value**2 / total) for value in singular[:rank]),
    )


class ProjectionAblationHooks:
    """Temporarily project a subspace out of the residual stream at target layers.

    Steering adds a vector and slides the decision threshold. Ablation removes a
    subspace and asks whether the decision can still be reached without it, which
    is the reversible analogue of excising a direction from the weights. The
    position scope matters to what is being claimed: removing a direction only at
    the decision register is a statement about a readout, while removing it at
    every position is a statement about the computation.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        bases: Mapping[int, torch.Tensor],
        *,
        fraction: float = 1.0,
        positions: str = "all",
    ) -> None:
        if positions not in {"all", "last"}:
            raise StudyRuntimeError("ablation positions must be 'all' or 'last'")
        self.layers = layers
        self.bases = dict(bases)
        self.fraction = float(fraction)
        self.positions = positions
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _orthonormalize(basis: torch.Tensor) -> torch.Tensor:
        """Return an orthonormal row basis, since a projector needs one to be correct."""

        matrix = basis.detach().float().cpu()
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise StudyRuntimeError("ablation basis must be a matrix of row vectors")
        transposed, _ = torch.linalg.qr(matrix.transpose(0, 1))
        return transposed.transpose(0, 1).contiguous()

    def install(self) -> "ProjectionAblationHooks":
        if self._handles:
            raise StudyRuntimeError("ablation hooks are already installed")
        if not self.bases:
            raise StudyRuntimeError("at least one ablation layer is required")
        try:
            for layer_idx in sorted(self.bases):
                if layer_idx < 0 or layer_idx >= len(self.layers):
                    raise StudyRuntimeError(f"ablation layer {layer_idx} is out of range")
                basis = self._orthonormalize(self.bases[layer_idx])
                handle = self.layers[layer_idx].register_forward_hook(
                    self._make_hook(basis, self.fraction, self.positions)
                )
                self._handles.append(handle)
        except BaseException:
            self.remove()
            raise
        return self

    @staticmethod
    def _make_hook(basis: torch.Tensor, fraction: float, positions: str):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
                raise StudyRuntimeError("ablation target did not return hidden-state tensors")
            if hidden.shape[-1] != basis.shape[-1]:
                raise StudyRuntimeError(
                    "ablation basis dimension does not match target hidden dimension"
                )
            rows = basis.to(device=hidden.device, dtype=hidden.dtype)
            ablated = hidden.clone()
            if positions == "all" or ablated.ndim == 2:
                target = ablated
            else:
                target = ablated[..., -1:, :]
            component = torch.matmul(torch.matmul(target, rows.transpose(0, 1)), rows)
            target -= fraction * component
            if isinstance(output, tuple):
                return (ablated,) + output[1:]
            return ablated

        return hook

    def remove(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    @property
    def is_active(self) -> bool:
        return bool(self._handles)

    def __enter__(self) -> "ProjectionAblationHooks":
        return self.install()

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.remove()
