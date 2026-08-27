from __future__ import annotations

import pytest
import torch
from guard_study.interventions import (
    LastTokenPatch,
    LayerSteeringHooks,
    build_contrastive_direction,
    orthogonal_sham,
)


def test_contrastive_direction_uses_unsafe_minus_safe_orientation_and_scale():
    direction = build_contrastive_direction(
        layer_idx=3,
        unsafe_activations=[torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0])],
        safe_activations=[torch.tensor([-2.0, 0.0]), torch.tensor([-4.0, 0.0])],
    )

    assert torch.allclose(direction.direction, torch.tensor([1.0, 0.0]))
    assert direction.projection_gap == pytest.approx(6.0)
    assert direction.projection_std == pytest.approx(3.16227766)


def test_layer_steering_changes_only_last_position_and_always_removes_hooks():
    layer = torch.nn.Identity()
    values = torch.zeros(1, 3, 2)

    with pytest.raises(RuntimeError, match="sentinel"):
        with LayerSteeringHooks(
            [layer],
            {0: torch.tensor([1.0, 0.0])},
            {0: 2.0},
            dose=-0.5,
        ) as hooks:
            assert hooks.is_active
            steered = layer(values)
            assert torch.allclose(steered[0, :2], torch.zeros(2, 2))
            assert torch.allclose(steered[0, -1], torch.tensor([-1.0, 0.0]))
            raise RuntimeError("sentinel")

    assert torch.equal(layer(values), values)


def test_last_token_patch_is_length_independent_and_reversible():
    layer = torch.nn.Identity()
    values = torch.zeros(1, 5, 2)

    with LastTokenPatch(layer, torch.tensor([3.0, 4.0])):
        patched = layer(values)
        assert torch.allclose(patched[0, -1], torch.tensor([3.0, 4.0]))
        assert torch.equal(patched[0, :-1], values[0, :-1])

    assert torch.equal(layer(values), values)


def test_orthogonal_sham_is_deterministic_unit_and_orthogonal():
    learned = torch.tensor([1.0, 2.0, 3.0])
    first = orthogonal_sham(learned, seed=11)
    second = orthogonal_sham(learned, seed=11)

    assert torch.equal(first, second)
    assert first.norm().item() == pytest.approx(1.0)
    assert torch.dot(first, learned / learned.norm()).item() == pytest.approx(0.0, abs=1e-6)
