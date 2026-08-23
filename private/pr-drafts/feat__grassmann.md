# feat(analysis): add Grassmann subspace geometry

<!-- branch feat/grassmann, head 015f02c. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: multi-direction extraction gives each layer a rank-k refusal subspace, but the cross-layer analysis compares only the first direction of each layer with `|cos|`, and there is no gauge-invariant way to compare, cluster, or average rank-k subspaces.
- Change: a `grassmann` module (rank-revealing bases, principal angles, canonical geodesic, projection metric, log/exp, Karcher mean, batched pairwise distances) and a `CrossLayerAlignmentAnalyzer` that accepts `(k, hidden_dim)` bases per layer. Single-direction inputs reproduce the previous numbers exactly.
- User-visible effect: `CrossLayerResult` gains `distance_matrix`, `subspace_rank`, `subspace_ranks`; `angular_drift` / `total_geodesic_distance` are the canonical Grassmann geodesic (identical for k = 1); the report prints the rank. No pipeline behaviour changes (the informed pipeline still passes single directions).
- Linked issue: Closes #TBD

### Design decisions worth a reviewer's attention

1. **Two distances, deliberately.** Principal angles between subspaces of *different* rank measure containment (a line inside a plane has angle 0), so they cannot be a metric: `d(e0, span(e0,e1)) = d(span(e0,e1), e1) = 0` while `d(e0, e1) = π/2` would let a clustering bridge two orthogonal lines through a plane. `geodesic_distance` therefore requires equal ranks, and `distance_matrix` uses the projection metric `‖P_Y − P_Z‖_F/√2`, a true metric on subspaces of any rank that equals the chordal distance when ranks match. The analyzer enforces one rank across non-zero layers and reports per-layer ranks.
2. **`cosine_matrix` is the mean principal cosine** — exactly `|cos|` at k = 1, a similarity in [0, 1] for thresholding, and explicitly *not* a distance; `total_geodesic_distance` is `‖θ‖₂` (two planes with both angles π/3 report √2·π/3 ≈ 1.481, not the RMS angle 1.047).
3. **Rank detection is SVD-based.** An unpivoted QR with a diagonal threshold returns rank 1 for `[e0, e0, e1]` and changes span with row order; the SVD basis is order-independent (tested).
4. **CPU float64 for the diagnostics.** These are a handful of `k × hidden_dim` vectors; computing on CPU copies keeps MPS (no float64) and mixed-device inputs working, and results are returned on the input's device/dtype. Tested for float16/bfloat16 and, where available, CUDA placements.
5. **Zero-signal layers keep the historical fallback** for both `(hidden_dim,)` and `(1, hidden_dim)` inputs: rank 0, zero similarity, maximal drift step (`arccos 0` for k = 1), singleton cluster — no exception.
6. **PR gate**: `ci/test-risk-map.json` lists `obliteratus/analysis/grassmann.py` and `tests/test_grassmann.py` under mechanistic analysis; `scripts/check_test_risk_map.py` and `scripts/select_pr_tests.py --base-ref main` both succeed on this head.

### Rationale (from the lab commit)

Rank-k refusal subspaces are points of Gr(k, d). This module provides the
gauge-invariant vocabulary: rank-revealing orthonormal bases (SVD with a
relative tolerance; an unpivoted QR drops valid rows when an earlier row is
dependent), principal angles via atan2 of sines and cosines, the canonical
geodesic ||theta||_2 (equal ranks only), the projection metric
||P_Y - P_Z||_F / sqrt(2) (a true metric across ranks; principal angles across
ranks measure containment and cannot feed a clustering), log/exp maps, the
Karcher mean, and batched pairwise distances. All computation is on CPU
float64 copies (MPS has no float64); subspaces are returned on the input's
device and dtype. Closed-form oracle tests; risk map lists the module and its
test under mechanistic analysis.

### Follow-up commit — feat(cross_layer): compare rank-k refusal subspaces on the Grassmannian

Each layer may pass a (k, hidden_dim) basis instead of a single direction.
cosine_matrix becomes the mean principal cosine (exactly |cos| for k = 1),
angular drift and total_geodesic_distance use the canonical Grassmann geodesic
(two planes with both angles pi/3 give sqrt(2)*pi/3, not the RMS angle), and
a new distance_matrix holds projection distances, a metric even when layers
carry no signal. Non-zero layers must share one rank after dependent rows are
dropped; an all-zero layer of either shape keeps the historical fallback
(rank 0, zero similarity, maximal drift step, singleton cluster). Single
direction inputs reproduce the previous numbers exactly.

Files:
- `obliteratus/analysis/cross_layer.py`
- `tests/test_analysis.py`

### Files

- `ci/test-risk-map.json`
- `obliteratus/analysis/grassmann.py`
- `tests/test_grassmann.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `015f02c`

| Check | Result | Evidence or notes |
|---|---|---|
| Focused regression/contract tests | TBD | |
| Negative and boundary tests | TBD | |
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
