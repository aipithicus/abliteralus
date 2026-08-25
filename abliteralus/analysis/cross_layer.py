"""Cross-layer refusal direction alignment analysis.

A key open question in abliteration research is whether refusal is mediated
by the *same* direction propagated through the residual stream, or by
*different* directions at each layer. This module answers that question
quantitatively by computing pairwise cosine similarities between refusal
directions across all layers.

If refusal uses a single persistent direction, we expect high cosine
similarities across adjacent layers (the residual stream preserves the
direction). If different layers encode refusal independently, similarities
will be low even between adjacent layers.

This analysis also reveals "refusal direction clusters" -- groups of layers
that share similar refusal geometry, which may correspond to distinct
functional stages of refusal processing:
  - Early layers: instruction comprehension
  - Middle layers: harm assessment / refusal decision
  - Late layers: refusal token generation

Contribution: We also compute the "refusal direction flow" --
the cumulative angular drift of the refusal direction through the network,
measured as the total geodesic distance on the unit hypersphere.

Rank-k refusal subspaces
------------------------
Multi-direction extraction gives each layer a rank-k subspace rather than a
single direction. Layers are then compared as points of the Grassmann manifold
Gr(k, d) through their principal angles (:mod:`abliteralus.analysis.grassmann`):

* ``cosine_matrix`` holds the mean principal cosine of each pair -- exactly
  ``|cos|`` for single directions -- and is the similarity the cluster threshold
  applies to.
* ``angular_drift`` / ``total_geodesic_distance`` use the canonical Grassmann
  geodesic ``||theta||_2`` (the arc on the unit hypersphere when k = 1).
* ``distance_matrix`` holds projection distances ``||P_i - P_j||_F / sqrt(2)``, a
  true metric even when some layers carry no signal.

All non-zero layers must share one rank; a layer whose tensor is all zeros is
kept with rank 0, zero similarity to everything, and a missing-signal sentinel
step whose principal angles are all pi/2 (``arccos 0`` for single directions,
as before). Every quantity is gauge invariant: it depends only on each layer's
span, not on the basis given for it.

References:
    - Arditi et al. (2024): Found refusal concentrated in middle-late layers
    - Joad et al. (2026): Identified 11 geometrically distinct refusal directions
    - Anthropic Biology (2025): Default refusal circuits span specific layer ranges
    - Edelman, Arias & Smith (1998): geometry of the Grassmann manifold
    - Bendokat, Zimmermann & Absil (2020): A Grassmann Manifold Handbook (arXiv:2011.13699)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from abliteralus.analysis.grassmann import (
    orthonormal_basis,
    pairwise_projection_distances,
)


def _missing_signal_geodesic_step(rank: int) -> float:
    """Sentinel drift for an absent layer: the norm of ``max(rank, 1)`` right angles."""
    return math.sqrt(max(rank, 1)) * math.pi / 2.0


@dataclass
class CrossLayerResult:
    """Result of cross-layer alignment analysis.

    Each layer contributes a refusal *subspace*: a single direction ``(hidden_dim,)``
    or a rank-k basis ``(k, hidden_dim)``. Layers are compared as points of the
    Grassmann manifold Gr(k, d), so every quantity below depends only on the span,
    never on the basis chosen for it. For single directions the numbers coincide
    with the historical cosine-based ones.
    """

    cosine_matrix: torch.Tensor             # (n_layers, n_layers) mean principal cosine; |cos| for k=1
    layer_indices: list[int]                # which layers have refusal directions
    clusters: list[list[int]]               # groups of aligned layers
    angular_drift: list[float]              # cumulative canonical geodesic distance per layer
    total_geodesic_distance: float          # sum of adjacent-layer geodesic distances (||theta||_2)
    mean_adjacent_cosine: float             # avg cosine between consecutive layers
    direction_persistence_score: float      # 0=independent per layer, 1=single direction
    cluster_count: int                      # number of distinct direction clusters
    distance_matrix: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 0))
    # (n_layers, n_layers) projection distances ||P_i - P_j||_F / sqrt(2): a metric for any ranks
    subspace_rank: int = 1                  # common rank of the non-zero layer subspaces
    subspace_ranks: dict[int, int] = field(default_factory=dict)
    # per-layer rank after orthonormalization; 0 marks a zero (absent) refusal signal


class CrossLayerAlignmentAnalyzer:
    """Analyze how refusal directions relate across transformer layers.

    Computes a full pairwise cosine similarity matrix and identifies
    clusters of layers that share similar refusal geometry.
    """

    def __init__(self, cluster_threshold: float = 0.85):
        """
        Args:
            cluster_threshold: Minimum cosine similarity for two layers
                to be considered in the same refusal direction cluster.
        """
        self.cluster_threshold = cluster_threshold

    def analyze(
        self,
        refusal_directions: dict[int, torch.Tensor],
        strong_layers: list[int] | None = None,
    ) -> CrossLayerResult:
        """Compute cross-layer alignment analysis.

        Args:
            refusal_directions: {layer_idx: tensor} for each layer. A tensor is a
                single direction ``(hidden_dim,)`` / ``(1, hidden_dim)`` or a
                rank-k subspace basis ``(k, hidden_dim)``; rows need not be unit
                or orthogonal. Every non-zero layer must have the same rank after
                orthonormalization (dependent rows are dropped); a layer whose
                tensor is all zeros is treated as carrying no refusal signal.
            strong_layers: Optional subset of layers to analyze. If None,
                all layers with directions are included.

        Returns:
            CrossLayerResult with full alignment analysis.

        Raises:
            ValueError: if the non-zero layers have different ranks, or if the
                ambient dimensions differ.
        """
        if strong_layers is not None:
            indices = sorted(strong_layers)
        else:
            indices = sorted(refusal_directions.keys())

        if not indices:
            return CrossLayerResult(
                cosine_matrix=torch.zeros(0, 0),
                layer_indices=[],
                clusters=[],
                angular_drift=[],
                total_geodesic_distance=0.0,
                mean_adjacent_cosine=0.0,
                direction_persistence_score=0.0,
                cluster_count=0,
            )

        # Rank-revealing orthonormal basis per layer (CPU float64, gauge free).
        bases = [orthonormal_basis(refusal_directions[idx]) for idx in indices]
        n = len(indices)
        ranks = {idx: int(b.shape[0]) for idx, b in zip(indices, bases)}
        nonzero = [i for i, b in enumerate(bases) if b.shape[0] > 0]
        distinct = sorted({bases[i].shape[0] for i in nonzero})
        if len(distinct) > 1:
            detail = ", ".join(f"layer {indices[i]}: rank {bases[i].shape[0]}" for i in nonzero)
            raise ValueError(
                "refusal subspaces must have a uniform rank across layers "
                f"(dependent directions are dropped): {detail}"
            )
        widths = {b.shape[1] for b in bases}
        if len(widths) > 1:
            raise ValueError(f"refusal subspaces have different hidden sizes: {sorted(widths)}")
        rank = distinct[0] if distinct else 0

        # One batched Gram + SVD over all non-zero pairs gives every principal-angle
        # cosine at once: the mean over angles is the similarity (|cos| for k = 1),
        # and the 2-norm of the angles is the canonical geodesic distance.
        cosine_matrix = torch.zeros(n, n, dtype=torch.float64)
        geodesic_matrix = torch.full(
            (n, n),
            _missing_signal_geodesic_step(rank),
            dtype=torch.float64,
        )
        if nonzero:
            stack = torch.stack([bases[i] for i in nonzero])  # (m, k, d)
            gram = torch.einsum("ikd,jld->ijkl", stack, stack)
            cos = torch.linalg.svdvals(gram).clamp(0.0, 1.0)  # (m, m, k)
            sel = torch.tensor(nonzero)
            cosine_matrix[sel[:, None], sel[None, :]] = cos.mean(dim=-1)
            geodesic_matrix[sel[:, None], sel[None, :]] = torch.acos(cos).norm(dim=-1)
        geodesic_matrix.fill_diagonal_(0.0)
        cosine_matrix = cosine_matrix.to(torch.float32)

        # Projection distances form a metric even when some layers are zero.
        distance_matrix = pairwise_projection_distances(bases).to(torch.float32)

        # Adjacent layer cosines (for layers in sorted order)
        adjacent_cosines = []
        for i in range(n - 1):
            adjacent_cosines.append(cosine_matrix[i, i + 1].item())

        mean_adjacent = sum(adjacent_cosines) / max(len(adjacent_cosines), 1)

        # Angular drift: cumulative canonical geodesic distance from layer to layer.
        # A zero layer is not a Grassmann point. Represent the missing signal by
        # a sentinel step with every principal angle pi/2, which for single
        # directions is the historical arccos(0).
        angular_drift = [0.0]
        total_geodesic = 0.0
        for i in range(n - 1):
            total_geodesic += geodesic_matrix[i, i + 1].item()
            angular_drift.append(total_geodesic)

        # Direction persistence score:
        # 1.0 = all layers use identical direction (perfect persistence)
        # 0.0 = all layers use orthogonal directions (no persistence)
        # Computed as mean off-diagonal cosine similarity
        if n > 1:
            mask = ~torch.eye(n, dtype=torch.bool)
            persistence = cosine_matrix[mask].mean().item()
        else:
            persistence = 1.0

        # Cluster detection via greedy agglomerative approach
        clusters = self._find_clusters(cosine_matrix, indices)

        return CrossLayerResult(
            cosine_matrix=cosine_matrix,
            layer_indices=indices,
            clusters=clusters,
            angular_drift=angular_drift,
            total_geodesic_distance=total_geodesic,
            mean_adjacent_cosine=mean_adjacent,
            direction_persistence_score=persistence,
            cluster_count=len(clusters),
            distance_matrix=distance_matrix,
            subspace_rank=rank,
            subspace_ranks=ranks,
        )

    def _find_clusters(
        self, cosine_matrix: torch.Tensor, indices: list[int]
    ) -> list[list[int]]:
        """Find clusters of layers with similar refusal directions.

        Uses single-linkage clustering: two layers are in the same cluster
        if their cosine similarity exceeds the threshold. Connected
        components form the clusters.
        """
        n = len(indices)
        if n == 0:
            return []

        # Build adjacency from threshold
        adj = cosine_matrix >= self.cluster_threshold

        # Find connected components via BFS
        visited = set()
        clusters = []

        for i in range(n):
            if i in visited:
                continue
            # BFS from i
            cluster = []
            queue = [i]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                cluster.append(indices[node])
                for j in range(n):
                    if j not in visited and adj[node, j]:
                        queue.append(j)
            clusters.append(sorted(cluster))

        return sorted(clusters, key=lambda c: c[0])

    @staticmethod
    def format_report(result: CrossLayerResult) -> str:
        """Format cross-layer analysis as a human-readable report."""
        lines = []
        lines.append("Cross-Layer Refusal Direction Alignment Analysis")
        lines.append("=" * 52)
        lines.append("")

        if not result.layer_indices:
            lines.append("No layers to analyze.")
            return "\n".join(lines)

        lines.append(f"Layers analyzed: {result.layer_indices}")
        if result.subspace_rank > 1:
            zero_layers = [idx for idx, r in result.subspace_ranks.items() if r == 0]
            lines.append(f"Subspace rank: {result.subspace_rank} per layer"
                         + (f" (no signal at layers {zero_layers})" if zero_layers else ""))
        lines.append(f"Direction persistence score: {result.direction_persistence_score:.3f}")
        lines.append("  (1.0 = single direction, 0.0 = all orthogonal)")
        lines.append(f"Mean adjacent-layer cosine: {result.mean_adjacent_cosine:.3f}")
        lines.append(f"Total geodesic distance: {result.total_geodesic_distance:.3f} rad")
        lines.append(f"Number of direction clusters: {result.cluster_count}")
        lines.append("")

        # Cluster summary
        lines.append("Direction Clusters:")
        for i, cluster in enumerate(result.clusters):
            lines.append(f"  Cluster {i + 1}: layers {cluster}")
        lines.append("")

        # Angular drift
        lines.append("Cumulative Angular Drift:")
        for i, (idx, drift) in enumerate(
            zip(result.layer_indices, result.angular_drift)
        ):
            bar_len = int(drift / max(result.total_geodesic_distance, 0.01) * 20)
            lines.append(f"  layer {idx:3d}: {drift:.3f} rad {'▓' * bar_len}")
        lines.append("")

        # Cosine matrix (abbreviated for large models)
        n = len(result.layer_indices)
        if n <= 20:
            lines.append("Pairwise Cosine Similarity Matrix:")
            header = "       " + "".join(f"{idx:6d}" for idx in result.layer_indices)
            lines.append(header)
            for i, idx_i in enumerate(result.layer_indices):
                row = f"  {idx_i:3d}  "
                for j in range(n):
                    val = result.cosine_matrix[i, j].item()
                    row += f" {val:.3f}"
                lines.append(row)
        else:
            lines.append(f"(Cosine matrix too large to display: {n}x{n})")

        return "\n".join(lines)
