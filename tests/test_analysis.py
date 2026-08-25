"""Tests for the analysis techniques."""

from __future__ import annotations

import math

import pytest
import torch

from abliteralus.analysis.whitened_svd import (
    WhitenedSVDExtractor,
    WhitenedSVDResult,
    _orthonormal_row_basis,
)
from abliteralus.analysis.cross_layer import CrossLayerAlignmentAnalyzer, CrossLayerResult
from abliteralus.analysis.activation_probing import ActivationProbe, ProbeResult


# ---------------------------------------------------------------------------
# WhitenedSVDExtractor
# ---------------------------------------------------------------------------

class TestWhitenedSVD:
    @pytest.mark.parametrize("regularization_eps", [0, -1e-4, float("nan"), True])
    def test_rejects_invalid_regularization(self, regularization_eps):
        with pytest.raises(ValueError, match="finite positive"):
            WhitenedSVDExtractor(regularization_eps=regularization_eps)

    @pytest.mark.parametrize("min_variance_ratio", [-0.1, 1.0, float("inf"), True])
    def test_rejects_invalid_variance_ratio(self, min_variance_ratio):
        with pytest.raises(ValueError, match=r"interval \[0, 1\)"):
            WhitenedSVDExtractor(min_variance_ratio=min_variance_ratio)

    @pytest.mark.parametrize("n_directions", [-1, 0, 1.5, True])
    def test_rejects_invalid_direction_count(self, n_directions):
        with pytest.raises(ValueError, match="positive integer"):
            WhitenedSVDExtractor().extract(
                [torch.ones(4)],
                [torch.zeros(4)],
                n_directions=n_directions,
            )

    def test_rejects_empty_unpaired_and_mismatched_activations(self):
        extractor = WhitenedSVDExtractor()
        with pytest.raises(ValueError, match="both be non-empty"):
            extractor.extract([], [])
        with pytest.raises(ValueError, match="equal sample counts"):
            extractor.extract([torch.ones(4)] * 2, [torch.zeros(4)])
        with pytest.raises(ValueError, match="same width"):
            extractor.extract([torch.ones(4)], [torch.zeros(3)])
        with pytest.raises(ValueError, match="consistent width"):
            extractor.extract(
                [torch.ones(4), torch.ones(3)],
                [torch.zeros(4), torch.zeros(4)],
            )
        with pytest.raises(ValueError, match="must be a tensor"):
            extractor.extract([object()], [torch.zeros(4)])
        with pytest.raises(ValueError, match="non-empty vector"):
            extractor.extract([torch.ones(2, 4)], [torch.zeros(4)])

    def test_single_sample_signal_uses_regularized_zero_covariance(self):
        result = WhitenedSVDExtractor().extract(
            [torch.tensor([0.0, 2.0, 0.0])],
            [torch.zeros(3)],
            n_directions=1,
        )
        assert result.directions.shape == (1, 3)
        assert result.directions.norm() == pytest.approx(1.0)
        assert result.variance_explained == pytest.approx(1.0)

    def test_basic_extraction(self):
        """Whitened SVD should extract directions from activation differences."""
        torch.manual_seed(42)
        n_prompts, hidden_dim = 10, 32

        # Create activations with a clear refusal direction
        refusal_dir = torch.randn(hidden_dim)
        refusal_dir = refusal_dir / refusal_dir.norm()

        harmless = [torch.randn(hidden_dim) for _ in range(n_prompts)]
        harmful = [h + 2.0 * refusal_dir for h in harmless]  # shifted along refusal dir

        extractor = WhitenedSVDExtractor()
        result = extractor.extract(harmful, harmless, n_directions=3)

        assert isinstance(result, WhitenedSVDResult)
        assert result.directions.shape == (3, hidden_dim)
        assert result.singular_values.shape == (3,)
        assert result.variance_explained > 0
        assert result.condition_number > 0
        assert result.effective_rank > 0

    def test_directions_are_unit_vectors(self):
        """Extracted directions should be unit length."""
        torch.manual_seed(42)
        harmless = [torch.randn(16) for _ in range(8)]
        harmful = [h + torch.randn(16) * 0.5 for h in harmless]

        extractor = WhitenedSVDExtractor()
        result = extractor.extract(harmful, harmless, n_directions=2)

        for i in range(result.directions.shape[0]):
            assert abs(result.directions[i].norm().item() - 1.0) < 1e-4

    def test_primary_aligns_with_planted_direction(self):
        """Primary whitened direction should capture the planted refusal signal.

        Whitening rotates directions relative to the covariance structure,
        so perfect alignment with the raw direction is not expected. We verify
        the whitened direction explains substantial variance and has moderate
        alignment (whitening intentionally reweights dimensions).
        """
        torch.manual_seed(42)
        hidden_dim = 64
        n_prompts = 30

        refusal_dir = torch.randn(hidden_dim)
        refusal_dir = refusal_dir / refusal_dir.norm()

        # Isotropic harmless activations (whitening has minimal effect)
        harmless = [torch.randn(hidden_dim) * 0.1 for _ in range(n_prompts)]
        harmful = [h + 5.0 * refusal_dir for h in harmless]

        extractor = WhitenedSVDExtractor(regularization_eps=1e-3)
        result = extractor.extract(harmful, harmless, n_directions=1)

        cos_sim = (result.directions[0] @ refusal_dir).abs().item()
        # Moderate alignment expected (whitening reweights dimensions)
        assert cos_sim > 0.2, f"Expected alignment > 0.2, got {cos_sim:.3f}"
        # More importantly: the direction should explain most variance
        assert result.variance_explained > 0.5

    def test_extract_all_layers(self):
        """Should extract directions for all provided layers."""
        torch.manual_seed(42)
        harmful_acts = {}
        harmless_acts = {}
        for layer in range(4):
            harmful_acts[layer] = [torch.randn(16) for _ in range(5)]
            harmless_acts[layer] = [torch.randn(16) for _ in range(5)]

        extractor = WhitenedSVDExtractor()
        results = extractor.extract_all_layers(harmful_acts, harmless_acts, n_directions=2)

        assert len(results) == 4
        for idx in range(4):
            assert idx in results
            assert results[idx].directions.shape[0] == 2

    def test_extract_all_layers_skips_unpaired_harmful_layer(self):
        extractor = WhitenedSVDExtractor()
        harmful = {
            0: [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
            1: [torch.tensor([1.0, 1.0]), torch.tensor([2.0, 2.0])],
        }
        harmless = {
            0: [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])],
        }

        results = extractor.extract_all_layers(harmful, harmless, n_directions=1)

        assert results.keys() == {0}

    def test_compare_with_standard(self):
        """Comparison should return valid cosine similarities."""
        torch.manual_seed(42)
        harmless = [torch.randn(16) for _ in range(8)]
        harmful = [h + torch.randn(16) for h in harmless]

        extractor = WhitenedSVDExtractor()
        result = extractor.extract(harmful, harmless, n_directions=2)

        std_dir = torch.randn(16)
        std_dir = std_dir / std_dir.norm()

        comparison = WhitenedSVDExtractor.compare_with_standard(result, std_dir)
        assert "primary_direction_cosine" in comparison
        assert "subspace_principal_cosine" in comparison
        assert 0 <= comparison["primary_direction_cosine"] <= 1.0

        standard_subspace = torch.linalg.qr(torch.randn(16, 2)).Q.T
        subspace_comparison = WhitenedSVDExtractor.compare_with_standard(
            result,
            standard_subspace,
        )
        assert 0 <= subspace_comparison["subspace_principal_cosine"] <= 1.0

    @staticmethod
    def _result_with_directions(directions: torch.Tensor) -> WhitenedSVDResult:
        k = directions.shape[0]
        return WhitenedSVDResult(
            layer_idx=0,
            directions=directions,
            whitened_directions=directions,
            singular_values=torch.ones(k),
            variance_explained=1.0,
            condition_number=1.0,
            effective_rank=float(k),
        )

    def test_compare_with_standard_principal_angles_use_orthonormal_bases(self):
        """Whitened directions are unit-norm but not orthogonal; the principal
        angles must come from orthonormalized bases, not from the raw rows."""
        e = torch.eye(6)
        theta = torch.tensor(25.0).deg2rad()
        # Two unit rows spanning the e0-e1 plane at 25 degrees to each other.
        skewed = torch.stack([e[0], torch.cos(theta) * e[0] + torch.sin(theta) * e[1]])
        result = self._result_with_directions(skewed)

        same_plane = WhitenedSVDExtractor.compare_with_standard(result, torch.stack([e[0], e[1]]))
        assert same_plane["subspace_principal_cosine"] == pytest.approx(1.0, abs=1e-6)
        assert same_plane["subspace_principal_cosines"] == pytest.approx([1.0, 1.0], abs=1e-6)

        orthogonal_plane = WhitenedSVDExtractor.compare_with_standard(result, torch.stack([e[2], e[3]]))
        assert orthogonal_plane["subspace_principal_cosines"] == pytest.approx([0.0, 0.0], abs=1e-6)

        # Shares e0, second direction tilted 40 degrees off the skewed plane.
        phi = torch.tensor(40.0).deg2rad()
        tilted = torch.stack([e[0], torch.cos(phi) * e[1] + torch.sin(phi) * e[2]])
        partial = WhitenedSVDExtractor.compare_with_standard(result, tilted)
        assert partial["subspace_principal_cosines"] == pytest.approx([1.0, torch.cos(phi).item()], abs=1e-6)
        assert all(0.0 <= c <= 1.0 for c in partial["subspace_principal_cosines"])

    def test_compare_with_standard_principal_angles_for_mixed_ranks(self):
        """A single direction against a plane is still a principal angle, not the
        cosine against the plane's first basis vector."""
        e = torch.eye(5)
        line = self._result_with_directions(e[1].unsqueeze(0))

        containing_plane = WhitenedSVDExtractor.compare_with_standard(line, torch.stack([e[0], e[1]]))
        assert containing_plane["subspace_principal_cosines"] == pytest.approx([1.0], abs=1e-6)
        assert containing_plane["subspace_principal_cosine"] == pytest.approx(1.0, abs=1e-6)
        # The direction-level metric keeps its own meaning: e1 against the first row e0.
        assert containing_plane["primary_direction_cosine"] == pytest.approx(0.0, abs=1e-6)

        orthogonal_plane = WhitenedSVDExtractor.compare_with_standard(line, torch.stack([e[2], e[3]]))
        assert orthogonal_plane["subspace_principal_cosines"] == pytest.approx([0.0], abs=1e-6)

        plane = self._result_with_directions(torch.stack([e[0], e[1]]))
        single = WhitenedSVDExtractor.compare_with_standard(plane, e[1])
        assert single["subspace_principal_cosines"] == pytest.approx([1.0], abs=1e-6)

    def test_compare_with_standard_ignores_fictitious_rank(self):
        """Dependent rows must not contribute a dimension: duplicate e0 spans a
        line, so its angle to the orthogonal plane span(e1, e2) is 90 degrees."""
        e = torch.eye(5)
        duplicated = self._result_with_directions(torch.stack([e[0], e[0]]))

        orthogonal_plane = WhitenedSVDExtractor.compare_with_standard(duplicated, torch.stack([e[1], e[2]]))
        assert orthogonal_plane["subspace_principal_cosines"] == pytest.approx([0.0], abs=1e-6)
        assert orthogonal_plane["subspace_principal_cosine"] == pytest.approx(0.0, abs=1e-6)

        containing_plane = WhitenedSVDExtractor.compare_with_standard(duplicated, torch.stack([e[0], e[1]]))
        assert containing_plane["subspace_principal_cosines"] == pytest.approx([1.0], abs=1e-6)

        # Near-duplicate rows (numerically dependent) collapse the same way.
        nearly = torch.stack([e[0], e[0] + 1e-9 * e[3]])
        nearly = nearly / nearly.norm(dim=-1, keepdim=True)
        near_dup = self._result_with_directions(nearly)
        assert WhitenedSVDExtractor.compare_with_standard(
            near_dup, torch.stack([e[1], e[2]])
        )["subspace_principal_cosines"] == pytest.approx([0.0], abs=1e-6)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_principal_angle_diagnostic_runs_on_cpu_in_float64(self, dtype):
        """The rank-revealing basis is a small diagnostic: it must land on the CPU in
        float64 whatever the input dtype, since accelerators such as MPS have no
        float64 and the caller only consumes Python floats."""
        e = torch.eye(6)
        rows = torch.stack([e[0], e[0], e[2]]).to(dtype)

        basis = _orthonormal_row_basis(rows)
        assert basis.device.type == "cpu"
        assert basis.dtype == torch.float64
        assert basis.shape == (2, 6)  # duplicate row collapsed
        assert rows.dtype == dtype  # input untouched

        result = self._result_with_directions(rows)
        comparison = WhitenedSVDExtractor.compare_with_standard(result, torch.stack([e[0], e[1]]).to(dtype))
        assert comparison["subspace_principal_cosines"] == pytest.approx([1.0, 0.0], abs=1e-3)

    @pytest.mark.parametrize(
        "whitened_device, standard_device",
        [("cpu", "cpu")]
        + ([("cuda", "cpu"), ("cpu", "cuda"), ("cuda", "cuda")] if torch.cuda.is_available() else []),
    )
    def test_compare_with_standard_is_device_agnostic(self, whitened_device, standard_device):
        """All cosines are computed on CPU copies, so inputs may live on different
        devices and in any float dtype; the inputs are not moved or modified."""
        e = torch.eye(6)
        theta = torch.tensor(25.0).deg2rad()
        skewed = torch.stack([e[0], torch.cos(theta) * e[0] + torch.sin(theta) * e[1]])
        whitened = skewed.to(device=whitened_device, dtype=torch.float16)
        standard = torch.stack([e[0], e[1]]).to(device=standard_device, dtype=torch.bfloat16)
        result = self._result_with_directions(whitened)

        comparison = WhitenedSVDExtractor.compare_with_standard(result, standard)

        assert comparison["primary_direction_cosine"] == pytest.approx(1.0, abs=1e-3)
        # Row 0 matches e0 exactly; row 1's best match is e0 at cos(25 degrees).
        assert comparison["avg_max_direction_cosine"] == pytest.approx(((1 + torch.cos(theta)) / 2).item(), abs=2e-3)
        assert comparison["subspace_principal_cosines"] == pytest.approx([1.0, 1.0], abs=1e-3)
        assert result.directions.device.type == whitened_device
        assert result.directions.dtype == torch.float16
        assert standard.device.type == standard_device

    def test_handles_3d_activations(self):
        """Should handle activations with an extra batch dimension."""
        torch.manual_seed(42)
        # (1, hidden_dim) shape from hook output
        harmless = [torch.randn(1, 16) for _ in range(5)]
        harmful = [torch.randn(1, 16) for _ in range(5)]

        extractor = WhitenedSVDExtractor()
        result = extractor.extract(harmful, harmless, n_directions=2)
        assert result.directions.shape == (2, 16)

    def test_variance_explained_bounded(self):
        """Variance explained should be between 0 and 1."""
        torch.manual_seed(42)
        harmless = [torch.randn(16) for _ in range(8)]
        harmful = [torch.randn(16) for _ in range(8)]

        extractor = WhitenedSVDExtractor()
        result = extractor.extract(harmful, harmless, n_directions=3)
        assert 0 <= result.variance_explained <= 1.0


# ---------------------------------------------------------------------------
# CrossLayerAlignmentAnalyzer
# ---------------------------------------------------------------------------

class TestCrossLayerAlignment:
    def test_identical_directions(self):
        """Identical directions across layers should give persistence = 1."""
        direction = torch.randn(32)
        direction = direction / direction.norm()
        directions = {i: direction.clone() for i in range(5)}

        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)

        assert isinstance(result, CrossLayerResult)
        assert result.direction_persistence_score > 0.99
        assert result.mean_adjacent_cosine > 0.99
        assert result.total_geodesic_distance < 0.01

    def test_orthogonal_directions(self):
        """Orthogonal directions should give low persistence."""
        # Create orthogonal directions via QR decomposition
        torch.manual_seed(42)
        M = torch.randn(5, 32)
        Q, _ = torch.linalg.qr(M.T)
        directions = {i: Q[:, i] for i in range(5)}

        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)

        assert result.direction_persistence_score < 0.3
        assert result.mean_adjacent_cosine < 0.3

    def test_cluster_detection(self):
        """Should detect clusters of similar directions."""
        torch.manual_seed(42)
        # Create two clusters
        d1 = torch.randn(32)
        d1 = d1 / d1.norm()
        d2 = torch.randn(32)
        d2 = d2 / d2.norm()

        directions = {
            0: d1, 1: d1 + 0.01 * torch.randn(32),
            2: d1 + 0.01 * torch.randn(32),
            3: d2, 4: d2 + 0.01 * torch.randn(32),
        }
        # Normalize
        directions = {k: v / v.norm() for k, v in directions.items()}

        analyzer = CrossLayerAlignmentAnalyzer(cluster_threshold=0.9)
        result = analyzer.analyze(directions)

        # Should find at least 2 clusters
        assert result.cluster_count >= 2

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze({})
        assert result.layer_indices == []
        assert result.cluster_count == 0

    def test_single_layer(self):
        """Single layer should work fine."""
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze({5: torch.randn(16)})
        assert result.layer_indices == [5]
        assert result.direction_persistence_score == 1.0

    def test_strong_layers_filter(self):
        """Should only analyze specified strong layers."""
        directions = {i: torch.randn(16) for i in range(10)}
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions, strong_layers=[2, 5, 7])
        assert result.layer_indices == [2, 5, 7]
        assert result.cosine_matrix.shape == (3, 3)

    def test_cosine_matrix_symmetry(self):
        """Cosine matrix should be symmetric."""
        torch.manual_seed(42)
        directions = {i: torch.randn(16) for i in range(4)}
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)
        diff = (result.cosine_matrix - result.cosine_matrix.T).abs().max().item()
        assert diff < 1e-5

    def test_cosine_matrix_diagonal_ones(self):
        """Diagonal of cosine matrix should be 1.0."""
        torch.manual_seed(42)
        directions = {i: torch.randn(16) for i in range(4)}
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)
        for i in range(4):
            assert abs(result.cosine_matrix[i, i].item() - 1.0) < 1e-4

    def test_angular_drift_monotonic(self):
        """Angular drift should be monotonically non-decreasing."""
        torch.manual_seed(42)
        directions = {i: torch.randn(16) for i in range(6)}
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)
        for i in range(len(result.angular_drift) - 1):
            assert result.angular_drift[i + 1] >= result.angular_drift[i] - 1e-6

    def test_format_report(self):
        """Format report should produce a non-empty string."""
        torch.manual_seed(42)
        directions = {i: torch.randn(16) for i in range(4)}
        analyzer = CrossLayerAlignmentAnalyzer()
        result = analyzer.analyze(directions)
        report = CrossLayerAlignmentAnalyzer.format_report(result)
        assert "Cross-Layer" in report
        assert "persistence" in report

    # -- rank-k subspaces -------------------------------------------------------

    def test_single_direction_results_match_the_historical_cosine_formulas(self):
        """For single directions every reported number equals the pre-subspace definition."""
        torch.manual_seed(7)
        directions = {i: torch.randn(24) for i in range(6)}
        result = CrossLayerAlignmentAnalyzer(cluster_threshold=0.3).analyze(directions)

        D = torch.stack([d / d.norm() for d in directions.values()])
        legacy_cos = (D @ D.T).abs()
        assert torch.allclose(result.cosine_matrix, legacy_cos, atol=1e-5)
        drift, total = [0.0], 0.0
        for i in range(5):
            total += torch.acos(legacy_cos[i, i + 1].clamp(max=1.0)).item()
            drift.append(total)
        assert result.angular_drift == pytest.approx(drift, abs=1e-5)
        assert result.total_geodesic_distance == pytest.approx(total, abs=1e-5)
        assert result.subspace_rank == 1
        assert result.subspace_ranks == {i: 1 for i in range(6)}
        # projection distance between lines is sin(angle)
        assert result.distance_matrix[0, 1].item() == pytest.approx(
            torch.sin(torch.acos(legacy_cos[0, 1])).item(), abs=1e-5
        )

    def test_zero_signal_layers_keep_the_historical_fallback_for_both_shapes(self):
        """(1, d) and (d,) zero inputs are treated identically: rank 0, zero similarity,
        a maximal drift step, and a singleton cluster -- never an exception."""
        analyzer = CrossLayerAlignmentAnalyzer()
        alone = analyzer.analyze({0: torch.zeros(1, 4)})
        assert alone.layer_indices == [0] and alone.subspace_ranks == {0: 0}

        e = torch.eye(4)
        for zero in (torch.zeros(1, 4), torch.zeros(4)):
            result = analyzer.analyze({0: e[0], 1: zero, 2: e[0]})
            assert result.subspace_ranks == {0: 1, 1: 0, 2: 1}
            assert result.cosine_matrix[0, 1].item() == 0.0 and result.cosine_matrix[1, 1].item() == 0.0
            assert result.cosine_matrix[0, 2].item() == pytest.approx(1.0, abs=1e-6)
            assert result.angular_drift == pytest.approx([0.0, math.pi / 2, math.pi], abs=1e-6)
            assert [1] in result.clusters and [0, 2] in result.clusters
            assert result.distance_matrix[0, 1].item() == pytest.approx(math.sqrt(0.5), abs=1e-6)

    def test_rank_two_subspaces_use_the_canonical_grassmann_geodesic(self):
        """Two planes with both principal angles pi/3: drift is sqrt(2) * pi/3, the
        similarity is the mean principal cosine 0.5, and the result is gauge invariant."""
        e = torch.eye(6)
        t = math.pi / 3
        plane_a = torch.stack([e[0], e[1]])
        plane_b = torch.stack([math.cos(t) * e[0] + math.sin(t) * e[2], math.cos(t) * e[1] + math.sin(t) * e[3]])
        analyzer = CrossLayerAlignmentAnalyzer(cluster_threshold=0.85)

        result = analyzer.analyze({3: plane_a, 7: plane_b})
        assert result.subspace_rank == 2
        assert result.total_geodesic_distance == pytest.approx(math.sqrt(2) * t, abs=1e-6)
        assert result.cosine_matrix[0, 1].item() == pytest.approx(0.5, abs=1e-6)
        assert result.distance_matrix[0, 1].item() == pytest.approx(math.sqrt(2) * math.sin(t), abs=1e-6)
        assert result.clusters == [[3], [7]]

        rotation = torch.tensor([[0.6, 0.8], [-0.8, 0.6]])
        regauged = analyzer.analyze({3: rotation @ plane_a, 7: 2.5 * plane_b.flip(0)})
        assert torch.allclose(regauged.cosine_matrix, result.cosine_matrix, atol=1e-6)
        assert regauged.total_geodesic_distance == pytest.approx(result.total_geodesic_distance, abs=1e-6)

        identical = analyzer.analyze({0: plane_a, 1: rotation @ plane_a, 2: plane_a})
        assert identical.clusters == [[0, 1, 2]]
        assert identical.total_geodesic_distance == pytest.approx(0.0, abs=1e-6)

    def test_dependent_rows_are_dropped_and_mixed_ranks_are_rejected(self):
        e = torch.eye(5)
        analyzer = CrossLayerAlignmentAnalyzer()
        collapsed = analyzer.analyze({0: torch.stack([e[0], e[0]]), 1: e[0]})
        assert collapsed.subspace_rank == 1 and collapsed.cosine_matrix[0, 1].item() == pytest.approx(1.0, abs=1e-6)

        with pytest.raises(ValueError, match="uniform rank"):
            analyzer.analyze({0: e[0], 1: torch.stack([e[0], e[1]])})


# ---------------------------------------------------------------------------
# ActivationProbe
# ---------------------------------------------------------------------------

class TestActivationProbe:
    def test_clean_elimination(self):
        """After removing direction, projections should be near-zero."""
        torch.manual_seed(42)
        hidden_dim = 32
        refusal_dir = torch.randn(hidden_dim)
        refusal_dir = refusal_dir / refusal_dir.norm()

        # "Post-abliteration" activations: direction has been removed
        harmless = [torch.randn(hidden_dim) for _ in range(10)]
        harmful = [torch.randn(hidden_dim) for _ in range(10)]
        # Both sets are random, no refusal signal => gap should be small

        probe = ActivationProbe()
        result = probe.probe_layer(harmful, harmless, refusal_dir)
        assert abs(result.projection_gap) < 1.0
        assert result.separation_d_prime < 2.0

    def test_residual_detection(self):
        """Should detect residual refusal signal when direction wasn't removed."""
        torch.manual_seed(42)
        hidden_dim = 32
        refusal_dir = torch.randn(hidden_dim)
        refusal_dir = refusal_dir / refusal_dir.norm()

        harmless = [torch.randn(hidden_dim) for _ in range(10)]
        # Harmful still has strong refusal direction component
        harmful = [h + 5.0 * refusal_dir for h in harmless]

        probe = ActivationProbe()
        result = probe.probe_layer(harmful, harmless, refusal_dir)
        assert abs(result.projection_gap) > 1.0
        assert result.separation_d_prime > 2.0

    def test_probe_all_layers(self):
        """Should compute aggregate metrics across layers."""
        torch.manual_seed(42)
        hidden_dim = 16
        n_layers = 4

        harmful_acts = {}
        harmless_acts = {}
        refusal_dirs = {}

        for layer in range(n_layers):
            harmful_acts[layer] = [torch.randn(hidden_dim) for _ in range(5)]
            harmless_acts[layer] = [torch.randn(hidden_dim) for _ in range(5)]
            d = torch.randn(hidden_dim)
            refusal_dirs[layer] = d / d.norm()

        probe = ActivationProbe()
        result = probe.probe_all_layers(harmful_acts, harmless_acts, refusal_dirs)

        assert isinstance(result, ProbeResult)
        assert len(result.per_layer) == n_layers
        assert 0 <= result.refusal_elimination_score <= 1.0
        assert result.mean_projection_gap >= 0

    def test_res_score_range(self):
        """RES should always be between 0 and 1."""
        torch.manual_seed(42)
        for seed in range(5):
            torch.manual_seed(seed)
            harmful = {0: [torch.randn(8) for _ in range(3)]}
            harmless = {0: [torch.randn(8) for _ in range(3)]}
            dirs = {0: torch.randn(8)}
            dirs[0] = dirs[0] / dirs[0].norm()

            probe = ActivationProbe()
            result = probe.probe_all_layers(harmful, harmless, dirs)
            assert 0 <= result.refusal_elimination_score <= 1.0

    def test_format_report(self):
        """Format report should produce readable output."""
        torch.manual_seed(42)
        harmful = {0: [torch.randn(8) for _ in range(3)]}
        harmless = {0: [torch.randn(8) for _ in range(3)]}
        dirs = {0: torch.randn(8)}

        probe = ActivationProbe()
        result = probe.probe_all_layers(harmful, harmless, dirs)
        report = ActivationProbe.format_report(result)
        assert "Refusal Elimination Score" in report

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        probe = ActivationProbe()
        result = probe.probe_all_layers({}, {}, {})
        assert result.refusal_elimination_score == 0.0
        assert len(result.per_layer) == 0
