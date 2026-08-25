# feat(cross_layer): compare rank-k refusal subspaces on the Grassmannian

<!-- Paste into the upstream PR template; fill TBD cells from the writeback verification
output and the fork's own checks. Scrub before posting. -->

## PR package

| Field | Value |
|---|---|
| Local branch | `feat/grassmann-cross-layer` @ `86ff98a` |
| Branch base | `feat/grassmann-metrics` @ `069bf13` |
| Upstream target | `feat/grassmann-metrics` initially; retarget to `upstream/main` once that PR merges |
| Prerequisite PRs | `feat/grassmann-metrics` (hard dependency — imports `orthonormal_basis` and `pairwise_projection_distances`) |
| Dependents | `feat/cross-layer-subspace-pipeline` (not yet built) |
| Already in ABLITERALUS main | yes, via merge `4d6f9ca` |
| Fork materialization | not yet materialized |
| Source | authored fresh off the metrics branch; content quarried from archived `feat/grassmann` @ `a0877ae` (commit `a84bb12` plus its missing-signal sentinel correction) |

Exclusions — must not enter the fork branch: `uv.lock`, `.python-version`, and
everything under `private/`. The PR diff must be computed against the metrics
branch so the geometry module does not reappear as this PR's own work.

## Summary

- Problem: `CrossLayerAlignmentAnalyzer` collapsed every layer to a single direction through `squeeze()`, so multi-direction extraction could not be analyzed and any rank-k input silently lost all but one dimension.
- Change: layers are orthonormalized into rank-revealing bases and compared as points of Gr(k, d). `cosine_matrix` becomes the mean principal cosine, drift uses the canonical Grassmann geodesic, and a new `distance_matrix` carries projection distances.
- User-visible effect: `CrossLayerResult` gains `distance_matrix`, `subspace_rank`, and `subspace_ranks`; the report prints the rank when k > 1. Single-direction inputs reproduce the previous numbers exactly, so no pipeline behaviour changes (the informed pipeline still passes single directions).
- Linked issue: Closes #TBD

### Design decisions worth a reviewer's attention

1. **Rank-1 compatibility is a tested contract, not an aspiration.** `test_single_direction_results_match_the_historical_cosine_formulas` recomputes the pre-subspace formulas inline and asserts equality against `cosine_matrix`, `angular_drift`, and `total_geodesic_distance`.
2. **`cosine_matrix` is the mean principal cosine** — exactly `|cos|` at k = 1, a similarity in [0, 1] that the cluster threshold applies to, and explicitly *not* a distance. `total_geodesic_distance` is `‖θ‖₂`: two planes with both angles π/3 report √2·π/3 ≈ 1.481, not the RMS angle 1.047.
3. **`distance_matrix` uses the projection metric**, which stays a true metric when some layers carry no signal. The geodesic cannot be used here because principal angles across differing ranks measure containment, which would let a clustering bridge two orthogonal layers through a third.
4. **Zero-signal layers keep the historical fallback** for both `(hidden_dim,)` and `(1, hidden_dim)` inputs: rank 0, zero similarity, a maximal drift step (`arccos 0` for k = 1), singleton cluster — never an exception. That sentinel is computed as `√max(rank,1) · π/2` and is deliberately *not* routed through `max_geodesic_distance`: the missing-signal marker and the true manifold diameter are different quantities and diverge once `2k > d`.
5. **Uniform rank is enforced across non-zero layers**, with a per-layer rank listing in the error, because a silent mixed-rank comparison would produce containment angles masquerading as similarity.
6. **Gauge invariance is tested end to end**: re-gauging a layer's basis by an orthogonal rotation, rescaling, or row permutation leaves every reported number unchanged.

### Files

- `obliteratus/analysis/cross_layer.py`
- `tests/test_analysis.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: `CrossLayerResult` gains three fields, all defaulted; rank-1 numeric outputs are unchanged.

## Test evidence

Exact head SHA: `86ff98a` (local); fork head TBD

| Check | Result | Evidence or notes |
|---|---|---|
| Focused regression/contract tests | pass (local) | `pytest tests/test_analysis.py tests/test_grassmann.py` — 76 passed |
| Negative and boundary tests | pass (local) | mixed-rank rejection, dependent-row collapse, zero-signal layers of both shapes, single-layer input |
| `python -m ruff check --select F app.py obliteratus tests scripts` | TBD | |
| `uv lock --check` | TBD | |
| Selected PR core/risk tests | TBD | |
| Package build, when package inputs changed | not applicable | |
| Import and CLI smoke checks | TBD | |
| Applicable risk-surface checks | not applicable | |
| Conditional hardware/service gates | not applicable | |

Coverage or mutation impact:

- Changed-line: TBD

## Research or performance evidence

Not applicable.
