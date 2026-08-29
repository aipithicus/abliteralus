from __future__ import annotations

import pytest
import torch
from guard_study.errors import StudyRuntimeError
from guard_study.interventions import (
    ProjectionAblationHooks,
    build_contrastive_basis,
    build_contrastive_direction,
)
from torch import nn


class _PassThrough(nn.Module):
    def forward(self, hidden):
        return hidden


def _paired_activations(seed: int = 3, pairs: int = 6, dim: int = 8):
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(dim, generator=generator)
    safe, unsafe = [], []
    for index in range(pairs):
        base = torch.randn(dim, generator=generator)
        safe.append(base)
        # A dominant shared axis plus a smaller pair-specific component.
        unsafe.append(base + 3.0 * shared + 0.3 * torch.randn(dim, generator=generator))
    return safe, unsafe


def test_rank_one_basis_recovers_the_mean_difference_axis():
    safe, unsafe = _paired_activations()
    direction = build_contrastive_direction(
        layer_idx=0, unsafe_activations=unsafe, safe_activations=safe
    ).direction
    basis = build_contrastive_basis(
        layer_idx=0, unsafe_activations=unsafe, safe_activations=safe, rank=1
    )

    unit = direction / direction.norm()
    alignment = abs(float(torch.dot(basis.basis[0], unit)))

    assert basis.rank == 1
    assert alignment > 0.95
    assert basis.explained_fraction[0] > 0.8


def test_explained_fraction_is_ordered_and_bounded():
    safe, unsafe = _paired_activations()
    basis = build_contrastive_basis(
        layer_idx=0, unsafe_activations=unsafe, safe_activations=safe, rank=4
    )

    fractions = basis.explained_fraction
    assert list(fractions) == sorted(fractions, reverse=True)
    assert 0.0 < sum(fractions) <= 1.0 + 1e-6
    assert basis.summary()["rank"] == 4


def test_basis_rows_are_orthonormal():
    safe, unsafe = _paired_activations()
    basis = build_contrastive_basis(
        layer_idx=0, unsafe_activations=unsafe, safe_activations=safe, rank=3
    ).basis

    gram = basis @ basis.transpose(0, 1)
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)


def test_rank_beyond_the_available_pairs_is_refused():
    safe, unsafe = _paired_activations(pairs=3)

    with pytest.raises(StudyRuntimeError, match="supports rank <= 3"):
        build_contrastive_basis(
            layer_idx=0, unsafe_activations=unsafe, safe_activations=safe, rank=4
        )


def test_full_ablation_removes_the_subspace_component_entirely():
    layers = [_PassThrough()]
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hidden = torch.tensor([[[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]])

    with ProjectionAblationHooks(layers, {0: basis}, fraction=1.0, positions="all"):
        result = layers[0](hidden)

    assert torch.allclose(result, torch.tensor([[[0.0, 0.0, 5.0], [0.0, 0.0, 13.0]]]))


def test_partial_ablation_scales_the_removed_component():
    layers = [_PassThrough()]
    basis = torch.tensor([[1.0, 0.0, 0.0]])
    hidden = torch.tensor([[[4.0, 3.0, 5.0]]])

    with ProjectionAblationHooks(layers, {0: basis}, fraction=0.25, positions="all"):
        result = layers[0](hidden)

    assert torch.allclose(result, torch.tensor([[[3.0, 3.0, 5.0]]]))


def test_zero_fraction_is_the_identity():
    layers = [_PassThrough()]
    hidden = torch.tensor([[[4.0, 3.0, 5.0]]])

    with ProjectionAblationHooks(layers, {0: torch.tensor([[1.0, 0.0, 0.0]])}, fraction=0.0):
        result = layers[0](hidden)

    assert torch.allclose(result, hidden)


def test_last_position_scope_leaves_earlier_positions_untouched():
    layers = [_PassThrough()]
    basis = torch.tensor([[1.0, 0.0, 0.0]])
    hidden = torch.tensor([[[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]])

    with ProjectionAblationHooks(layers, {0: basis}, fraction=1.0, positions="last"):
        result = layers[0](hidden)

    assert torch.allclose(result, torch.tensor([[[2.0, 3.0, 5.0], [0.0, 11.0, 13.0]]]))


def test_a_non_orthonormal_basis_still_yields_a_true_projector():
    layers = [_PassThrough()]
    # Two non-orthogonal, unnormalized rows spanning the x-y plane.
    basis = torch.tensor([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    hidden = torch.tensor([[[2.0, 3.0, 5.0]]])

    with ProjectionAblationHooks(layers, {0: basis}, fraction=1.0, positions="all"):
        once = layers[0](hidden)
    with ProjectionAblationHooks(layers, {0: basis}, fraction=1.0, positions="all"):
        twice = layers[0](once)

    # Removing the x-y plane must leave only z, and a projector is idempotent.
    assert torch.allclose(once, torch.tensor([[[0.0, 0.0, 5.0]]]), atol=1e-6)
    assert torch.allclose(twice, once, atol=1e-6)


def test_hooks_are_removed_on_exit_and_out_of_range_layers_are_refused():
    layers = [_PassThrough()]
    hooks = ProjectionAblationHooks(layers, {0: torch.tensor([[1.0, 0.0, 0.0]])})

    with hooks:
        assert hooks.is_active is True
    assert hooks.is_active is False

    with pytest.raises(StudyRuntimeError, match="out of range"):
        ProjectionAblationHooks(layers, {5: torch.tensor([[1.0, 0.0, 0.0]])}).install()

    with pytest.raises(StudyRuntimeError, match="positions must be"):
        ProjectionAblationHooks(layers, {0: torch.tensor([[1.0, 0.0]])}, positions="middle")
