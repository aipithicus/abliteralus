"""Closed-form oracle tests for Grassmann subspace geometry.

## Test Context
- Code to test: `abliteralus/analysis/grassmann.py`
- Testing framework: pytest with torch CPU tensors (CUDA placements are exercised
  additionally when a device is present; nothing here requires one)
- Test types: closed-form oracles, metric axioms on adversarial triples, gauge and
  permutation invariants, rank-revealing boundary inputs, device/dtype contracts
- External dependencies to mock: none
"""

from __future__ import annotations

import math

import pytest
import torch

from abliteralus.analysis.grassmann import (
    geodesic_distance,
    max_geodesic_distance,
    mean_principal_cosine,
    orthonormal_basis,
    pairwise_geodesic_distances,
    pairwise_projection_distances,
    principal_angles,
    projection_distance,
)

torch.manual_seed(0)
E = torch.eye(8, dtype=torch.float64)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _random_subspace(k: int, d: int, generator: torch.Generator) -> torch.Tensor:
    return orthonormal_basis(torch.randn(k, d, dtype=torch.float64, generator=generator))


def _projector(basis: torch.Tensor) -> torch.Tensor:
    return basis.T @ basis


# ---------------------------------------------------------------------------
# Rank-revealing bases (review finding 1)
# ---------------------------------------------------------------------------


def test_orthonormal_basis_is_rank_revealing_and_order_independent():
    dependent = torch.stack([E[0], E[0], E[1]])
    basis = orthonormal_basis(dependent)
    assert basis.shape == (2, 8)

    reordered = orthonormal_basis(torch.stack([E[1], E[0], E[0]]))
    assert torch.allclose(_projector(basis), _projector(reordered), atol=1e-12)
    assert torch.allclose(_projector(basis), _projector(torch.stack([E[0], E[1]])), atol=1e-12)


def test_orthonormal_basis_near_dependent_rows_collapse_but_independent_rows_survive():
    nearly = torch.stack([E[0], E[0] + 1e-9 * E[2]])
    assert orthonormal_basis(nearly).shape[0] == 1
    independent = torch.stack([E[0], E[0] + 1e-3 * E[2]])
    assert orthonormal_basis(independent).shape[0] == 2


def test_orthonormal_basis_zero_input_has_rank_zero_and_single_row_keeps_orientation():
    assert orthonormal_basis(torch.zeros(1, 8)).shape == (0, 8)
    assert orthonormal_basis(torch.zeros(3, 8)).shape == (0, 8)
    direction = -3.0 * E[4]
    basis = orthonormal_basis(direction)
    assert basis.shape == (1, 8)
    assert float(basis[0] @ direction) > 0


# ---------------------------------------------------------------------------
# Principal angles and the two distances (review findings 2 and 4)
# ---------------------------------------------------------------------------


def test_principal_angles_closed_form_for_two_planes():
    a, b = math.radians(20.0), math.radians(70.0)
    y = torch.stack([E[0], E[1]])
    z = torch.stack([
        math.cos(a) * E[0] + math.sin(a) * E[2],
        math.cos(b) * E[1] + math.sin(b) * E[3],
    ])
    theta = principal_angles(y, z)
    assert theta.tolist() == pytest.approx([a, b], abs=1e-12)
    assert geodesic_distance(y, z).item() == pytest.approx(math.hypot(a, b), abs=1e-12)
    assert projection_distance(y, z).item() == pytest.approx(math.hypot(math.sin(a), math.sin(b)), abs=1e-12)


def test_geodesic_distance_is_the_canonical_two_norm_not_the_rms_angle():
    """Two planes with both principal angles pi/3: the geodesic is sqrt(2) * pi/3."""
    t = math.pi / 3
    y = torch.stack([E[0], E[1]])
    z = torch.stack([math.cos(t) * E[0] + math.sin(t) * E[2], math.cos(t) * E[1] + math.sin(t) * E[3]])
    assert geodesic_distance(y, z).item() == pytest.approx(math.sqrt(2) * t, abs=1e-12)
    assert geodesic_distance(y, z).item() == pytest.approx(1.4810, abs=1e-4)


def test_geodesic_distance_reduces_to_arccos_abs_cosine_for_single_directions():
    y, z = orthonormal_basis(torch.randn(8, dtype=torch.float64)), orthonormal_basis(torch.randn(8, dtype=torch.float64))
    expected = math.acos(abs(float(y[0] @ z[0])))
    assert geodesic_distance(y, z).item() == pytest.approx(expected, abs=1e-12)
    assert mean_principal_cosine(y, z).item() == pytest.approx(abs(float(y[0] @ z[0])), abs=1e-12)


def test_geodesic_distance_is_gauge_invariant():
    """A re-gauged basis ``Q @ Y`` is the same point of the Grassmannian."""
    generator = torch.Generator().manual_seed(5)
    y = _random_subspace(3, 16, generator)
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator))
    assert geodesic_distance(y, q @ y).item() == pytest.approx(0.0, abs=1e-10)


def test_small_angles_are_resolved_below_the_acos_floor():
    eps = 1e-9
    y = orthonormal_basis(E[0])
    z = orthonormal_basis(E[0] + eps * E[1])
    assert geodesic_distance(y, z).item() == pytest.approx(eps, rel=1e-6)


def test_mixed_rank_principal_angles_measure_containment_and_geodesic_refuses():
    line = orthonormal_basis(E[0])
    plane = torch.stack([E[0], E[1]])
    assert principal_angles(line, plane).tolist() == pytest.approx([0.0], abs=1e-12)
    with pytest.raises(ValueError, match="equal ranks"):
        geodesic_distance(line, plane)


def test_projection_distance_is_a_metric_across_ranks_on_the_adversarial_triple():
    """d(e0, plane) and d(plane, e1) must not be zero, or clustering would bridge e0 and e1
    through the plane while d(e0, e1) is maximal."""
    e0, e1 = orthonormal_basis(E[0]), orthonormal_basis(E[1])
    plane = torch.stack([E[0], E[1]])
    d_line_plane = projection_distance(e0, plane).item()
    d_plane_line = projection_distance(plane, e1).item()
    d_lines = projection_distance(e0, e1).item()
    assert d_line_plane == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert d_plane_line == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert d_lines == pytest.approx(1.0, abs=1e-12)
    assert d_lines <= d_line_plane + d_plane_line + 1e-12
    assert projection_distance(plane, plane).item() == pytest.approx(0.0, abs=1e-12)
    assert projection_distance(E[:0], plane).item() == pytest.approx(1.0, abs=1e-12)  # rank 0 vs rank 2


def test_projection_distance_equals_chordal_distance_for_equal_ranks():
    generator = torch.Generator().manual_seed(3)
    y, z = _random_subspace(3, 12, generator), _random_subspace(3, 12, generator)
    chordal = torch.sin(principal_angles(y, z)).norm().item()
    assert projection_distance(y, z).item() == pytest.approx(chordal, abs=1e-10)
    frob = (_projector(y) - _projector(z)).norm().item() / math.sqrt(2)
    assert projection_distance(y, z).item() == pytest.approx(frob, abs=1e-10)


def test_non_orthonormal_input_is_rejected():
    with pytest.raises(ValueError, match="orthonormal rows"):
        principal_angles(torch.stack([E[0], E[0] + E[1]]), torch.stack([E[0], E[1]]))


# ---------------------------------------------------------------------------
# Batched pairwise distances
# ---------------------------------------------------------------------------


def test_pairwise_geodesic_matches_pairwise_calls_and_projection_handles_mixed_rank():
    generator = torch.Generator().manual_seed(13)
    subspaces = [_random_subspace(2, 9, generator) for _ in range(5)]
    batched = pairwise_geodesic_distances(subspaces)
    for i in range(5):
        assert batched[i, i].item() == 0.0
        for j in range(5):
            assert batched[i, j].item() == pytest.approx(geodesic_distance(subspaces[i], subspaces[j]).item(), abs=1e-9)
            assert batched[i, j].item() == pytest.approx(batched[j, i].item(), abs=1e-12)

    mixed = [orthonormal_basis(E[0]), torch.stack([E[0], E[1]]), E[:0]]
    proj = pairwise_projection_distances(mixed)
    assert proj[0, 1].item() == pytest.approx(1 / math.sqrt(2), abs=1e-12)
    assert proj[0, 2].item() == pytest.approx(math.sqrt(0.5), abs=1e-12)
    assert proj[1, 2].item() == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="identical shape"):
        pairwise_geodesic_distances(mixed[:2])


def test_max_geodesic_distance_respects_the_ambient_dimension():
    y, z = torch.stack([E[0], E[1]]), torch.stack([E[2], E[3]])
    assert geodesic_distance(y, z).item() == pytest.approx(
        max_geodesic_distance(2, 8), abs=1e-12
    )
    assert max_geodesic_distance(3, 4) == pytest.approx(math.pi / 2, abs=1e-12)
    assert max_geodesic_distance(4, 4) == 0.0
    with pytest.raises(ValueError, match="between 0 and ambient_dim"):
        max_geodesic_distance(5, 4)


# ---------------------------------------------------------------------------
# Device and dtype contract (review finding 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_results_are_cpu_float64_and_inputs_are_untouched(device, dtype):
    y = orthonormal_basis(torch.stack([E[0], E[1]])).to(device=device, dtype=dtype)
    z = orthonormal_basis(torch.stack([E[0], E[2]])).to(device=device, dtype=dtype)

    theta = principal_angles(y, z)
    assert theta.device.type == "cpu" and theta.dtype == torch.float64
    assert theta.tolist() == pytest.approx([0.0, math.pi / 2], abs=1e-3)
    assert geodesic_distance(y, z).device.type == "cpu"
    assert projection_distance(y, z).item() == pytest.approx(1.0, abs=1e-3)
    assert y.device.type == device and y.dtype == dtype  # inputs untouched


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_random_low_precision_subspaces_are_accepted(device, dtype):
    generator = torch.Generator().manual_seed(17)
    y = _random_subspace(3, 16, generator).to(device=device, dtype=dtype)

    assert principal_angles(y, y).norm().item() == pytest.approx(0.0, abs=1e-10)
