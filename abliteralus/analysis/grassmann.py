"""Intrinsic geometry of refusal subspaces on the Grassmann manifold Gr(k, d).

A rank-k refusal subspace (the span of k extracted directions at one layer) is a
single point of the Grassmannian Gr(k, d), the manifold of k-dimensional linear
subspaces of R^d. Everything here is *gauge invariant*: it depends only on the
span, so a basis ``Y`` and ``Q @ Y`` (any k x k orthogonal ``Q``) describe the same
point. Single directions get this for free from ``|cos|`` (a line is a point of
Gr(1, d)); it breaks as soon as k > 1, where comparing first rows or averaging
basis matrices depends on an arbitrary SVD gauge.

Conventions
    A subspace is a ``(k, d)`` tensor with orthonormal rows, matching the
    ``directions`` tensors used by the surgery code. All computations run on CPU
    float64 copies (these are diagnostics over a handful of vectors, and MPS has
    no float64); angles and distances are returned as CPU float64 tensors.

Two notions of "distance" are provided, and they must not be mixed:

* :func:`geodesic_distance` — the canonical Grassmann geodesic ``||theta||_2``
  (2-norm of the principal angles). A metric on Gr(k, d) for one fixed rank k.
  Between subspaces of different rank the principal angles measure *containment*
  (a line inside a plane has all-zero angles), so this function requires equal
  ranks.
* :func:`projection_distance` — ``||P_Y - P_Z||_F / sqrt(2)`` with ``P = Y^T Y`` the
  orthogonal projector. A metric on the set of *all* subspaces, whatever their
  ranks, and equal to the chordal distance ``||sin theta||_2`` when ranks match.
  Use this wherever subspaces of mixed rank have to be compared or clustered.

Numerical notes
    Principal angles combine cosines (``svdvals(Y Z^T)``) and sines (``svdvals`` of
    ``Y`` projected onto the complement of ``Z``) through ``atan2``, so angles near
    zero are not lost to ``acos`` round-off. Bases are made orthonormal by an
    SVD with a relative tolerance, which is rank revealing regardless of row
    order; an unpivoted QR is not.

References
    - Edelman, Arias & Smith (1998): The Geometry of Algorithms with Orthogonality
      Constraints. SIAM J. Matrix Anal. Appl. 20(2).
    - Bendokat, Zimmermann & Absil (2020): A Grassmann Manifold Handbook
      (arXiv:2011.13699). Section 4 lists the geodesic and projection metrics.
    - Bjorck & Golub (1973): Numerical methods for computing angles between
      linear subspaces. Math. Comp. 27.
    - Knyazev & Argentati (2002): Principal angles between subspaces in an A-based
      scalar product. SIAM J. Sci. Comput. 23(6).
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "orthonormal_basis",
    "principal_angles",
    "geodesic_distance",
    "projection_distance",
    "mean_principal_cosine",
    "pairwise_geodesic_distances",
    "pairwise_projection_distances",
    "max_geodesic_distance",
]

_DTYPE = torch.float64


def _as_rows(t: torch.Tensor) -> torch.Tensor:
    """Validate and copy ``(d,)`` / ``(k, d)`` input to a CPU float64 ``(k, d)`` matrix."""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected a torch.Tensor, got {type(t).__name__}")
    if t.dim() == 1:
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"a subspace must be a (k, d) or (d,) tensor, got shape {tuple(t.shape)}")
    if t.shape[1] == 0:
        raise ValueError("a subspace needs a positive ambient dimension")
    work = t.detach().to(device="cpu", dtype=_DTYPE)
    if not torch.isfinite(work).all():
        raise ValueError("subspace contains non-finite values")
    return work


def orthonormal_basis(basis: torch.Tensor, rel_tol: float = 1e-6) -> torch.Tensor:
    """Rank-revealing orthonormal basis (as rows) of ``span(rows of basis)``.

    Right singular vectors whose singular value exceeds ``rel_tol`` times the
    largest are kept, so dependent or duplicate rows never add a dimension and
    the result does not depend on row order. Returns ``(rank, d)`` on CPU in
    float64; ``rank`` may be 0 for an all-zero input. For a single nonzero input
    row the returned direction keeps that row's orientation.
    """
    work = _as_rows(basis)
    _, singular_values, vh = torch.linalg.svd(work, full_matrices=False)
    if singular_values.numel() == 0 or singular_values[0] <= 0:
        return work[:0]
    keep = singular_values > rel_tol * singular_values[0]
    out = vh[keep].clone()
    if work.shape[0] == 1 and out.shape[0] == 1 and (out[0] @ work[0]) < 0:
        out = -out
    return out


def _check_orthonormal(y: torch.Tensor, name: str) -> torch.Tensor:
    rows = _as_rows(y)
    if rows.shape[0] == 0:
        return rows
    gram = rows @ rows.T
    atol = 1e-6
    if y.is_floating_point():
        # A basis that is orthonormal before conversion cannot remain exact in
        # float16/bfloat16. Four machine epsilons bounds ordinary storage
        # quantization while still rejecting materially non-orthonormal rows.
        atol = max(atol, 4.0 * torch.finfo(y.dtype).eps)
    if not torch.allclose(
        gram,
        torch.eye(rows.shape[0], dtype=_DTYPE),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(f"{name} must have orthonormal rows; use orthonormal_basis() first")
    # Project accepted low-precision inputs back onto the Stiefel manifold so
    # downstream angle calculations operate on an actually orthonormal basis.
    u, _, vh = torch.linalg.svd(rows, full_matrices=False)
    return u @ vh


def principal_angles(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Principal angles between ``span(y)`` and ``span(z)``, ascending, in radians.

    Both inputs must have orthonormal rows. Returns ``min(k_y, k_z)`` angles in
    ``[0, pi/2]`` as CPU float64. Between subspaces of *different* rank these are
    containment angles (a line inside a plane gives ``[0]``), not a distance.
    """
    y64, z64 = _check_orthonormal(y, "y"), _check_orthonormal(z, "z")
    if y64.shape[1] != z64.shape[1]:
        raise ValueError(f"ambient dimensions differ: {y64.shape[1]} vs {z64.shape[1]}")
    if y64.shape[0] == 0 or z64.shape[0] == 0:
        return torch.zeros(0, dtype=_DTYPE)
    if y64.shape[0] > z64.shape[0]:
        y64, z64 = z64, y64
    k = y64.shape[0]
    cos = torch.linalg.svdvals(y64 @ z64.T)[:k].clamp(0.0, 1.0)  # descending
    residual = y64 - (y64 @ z64.T) @ z64  # rows of y orthogonal to span(z)
    sin = torch.linalg.svdvals(residual)[:k].clamp(0.0, 1.0).flip(0)  # ascending
    return torch.atan2(sin, cos)


def geodesic_distance(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Canonical Grassmann geodesic distance ``||theta||_2`` for two subspaces of equal rank.

    Reduces to ``arccos|<y, z>|`` for single directions. Raises ``ValueError`` for
    different ranks, where no geodesic is defined; use :func:`projection_distance`.
    """
    ky, kz = _as_rows(y).shape[0], _as_rows(z).shape[0]
    if ky != kz:
        raise ValueError(
            f"geodesic_distance needs equal ranks, got {ky} and {kz}; use projection_distance"
        )
    return principal_angles(y, z).norm()


def projection_distance(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """``||P_Y - P_Z||_F / sqrt(2)``: a metric on subspaces of any ranks.

    Equals the chordal distance ``||sin theta||_2`` when the ranks agree, and
    ``sqrt((k_y + k_z)/2 - sum cos^2 theta)`` in general, so a line inside a plane
    is at distance ``1/sqrt(2)`` from it rather than 0. Defined (as
    ``sqrt(k/2)``) against the rank-0 subspace.
    """
    y64, z64 = _check_orthonormal(y, "y"), _check_orthonormal(z, "z")
    if y64.shape[1] != z64.shape[1]:
        raise ValueError(f"ambient dimensions differ: {y64.shape[1]} vs {z64.shape[1]}")
    overlap = (y64 @ z64.T).pow(2).sum() if (y64.shape[0] and z64.shape[0]) else torch.zeros((), dtype=_DTYPE)
    value = (y64.shape[0] + z64.shape[0]) / 2.0 - overlap
    return value.clamp(min=0.0).sqrt()


def mean_principal_cosine(y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Average cosine of the principal angles, in ``[0, 1]``; ``|cos|`` for single directions.

    A similarity for thresholding, not a metric. Zero if either subspace is empty.
    """
    theta = principal_angles(y, z)
    if theta.numel() == 0:
        return torch.zeros((), dtype=_DTYPE)
    return torch.cos(theta).mean()


def _stack_uniform(subspaces: list[torch.Tensor]) -> torch.Tensor:
    bases = [_check_orthonormal(s, f"subspaces[{i}]") for i, s in enumerate(subspaces)]
    if not bases:
        raise ValueError("at least one subspace is required")
    shape = bases[0].shape
    for b in bases:
        if b.shape != shape:
            raise ValueError("batched distances require subspaces of identical shape")
    return torch.stack(bases)  # (n, k, d)


def pairwise_geodesic_distances(subspaces: list[torch.Tensor]) -> torch.Tensor:
    """Symmetric ``(n, n)`` CPU float64 matrix of canonical geodesic distances.

    One batched Gram computation and one batched SVD over all pairs; equal ranks
    required. Uses ``acos`` of the singular values, which is adequate for the
    layer-to-layer scale this is used at; use :func:`geodesic_distance` where
    angles near zero must be resolved finely.
    """
    bases = _stack_uniform(subspaces)
    gram = torch.einsum("ikd,jld->ijkl", bases, bases)  # (n, n, k, k)
    cos = torch.linalg.svdvals(gram).clamp(0.0, 1.0)  # (n, n, k)
    theta = torch.acos(cos)
    out = theta.norm(dim=-1)
    out.fill_diagonal_(0.0)
    return 0.5 * (out + out.T)


def pairwise_projection_distances(subspaces: list[torch.Tensor]) -> torch.Tensor:
    """Symmetric ``(n, n)`` CPU float64 matrix of projection distances; ranks may differ.

    Computed from the Frobenius norms of projector differences; rank-0 inputs are
    allowed and sit at distance ``sqrt(k/2)`` from a rank-k subspace.
    """
    bases = [_check_orthonormal(s, f"subspaces[{i}]") for i, s in enumerate(subspaces)]
    n = len(bases)
    out = torch.zeros(n, n, dtype=_DTYPE)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = projection_distance(bases[i], bases[j])
    return out


def max_geodesic_distance(rank: int, ambient_dim: int) -> float:
    """Largest possible canonical geodesic distance on ``Gr(rank, ambient_dim)``.

    Two rank-k subspaces in R^d intersect in at least ``max(0, 2k - d)``
    dimensions. Therefore only ``min(k, d - k)`` principal angles can reach
    pi/2; the remainder are forced to zero when ``2k > d``.
    """
    if ambient_dim <= 0:
        raise ValueError("ambient_dim must be positive")
    if rank < 0 or rank > ambient_dim:
        raise ValueError(
            f"rank must be between 0 and ambient_dim, got {rank} and {ambient_dim}"
        )
    return math.sqrt(min(rank, ambient_dim - rank)) * math.pi / 2.0
