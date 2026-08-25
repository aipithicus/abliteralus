# feat(analysis): add Grassmann subspace geometry

<!-- Paste into the upstream PR template; fill TBD cells from the writeback verification
output and the fork's own checks. Scrub before posting. -->

## PR package

| Field | Value |
|---|---|
| Local branch | `feat/grassmann-metrics` @ `069bf13` |
| Branch base | `66ad774` (ABLITERALUS main) |
| Upstream target | `upstream/main` |
| Prerequisite PRs | none — submittable as soon as upstream main is current |
| Dependents | `feat/grassmann-cross-layer`, `feat/grassmann-statistics` |
| Already in ABLITERALUS main | yes, via merge `86d3d9b` |
| Fork materialization | not yet materialized |
| Source | authored fresh off main; content quarried from archived `feat/grassmann` @ `a0877ae` (commit `605843b` plus its P2/P3 corrections, minus the manifold statistics) |

Exclusions — must not enter the fork branch: `uv.lock`, `.python-version`, and
everything under `private/`.

## Summary

- Problem: multi-direction extraction gives each layer a rank-k refusal subspace, but there is no gauge-invariant vocabulary for comparing, clustering, or measuring distances between rank-k subspaces. Comparing first rows or averaging basis matrices depends on an arbitrary SVD gauge.
- Change: a `grassmann` module providing rank-revealing orthonormal bases, principal angles, the canonical geodesic, the projection metric, mean principal cosine, and batched pairwise distance matrices.
- User-visible effect: none. This PR adds a module with no callers; `feat/grassmann-cross-layer` is its first consumer.
- Linked issue: Closes #TBD

### Design decisions worth a reviewer's attention

1. **Two distances, deliberately.** Principal angles between subspaces of *different* rank measure containment (a line inside a plane has angle 0), so they cannot be a metric: `d(e0, span(e0,e1)) = d(span(e0,e1), e1) = 0` while `d(e0, e1) = π/2` would let a clustering bridge two orthogonal lines through a plane. `geodesic_distance` therefore requires equal ranks, and `projection_distance` uses `‖P_Y − P_Z‖_F/√2`, a true metric on subspaces of any rank that equals the chordal distance when ranks match.
2. **`mean_principal_cosine` is a similarity, not a distance** — exactly `|cos|` at k = 1, in [0, 1], for thresholding. `geodesic_distance` is `‖θ‖₂`; two planes with both angles π/3 report √2·π/3 ≈ 1.481, not the RMS angle 1.047.
3. **Rank detection is SVD-based.** An unpivoted QR with a diagonal threshold returns rank 1 for `[e0, e0, e1]` and changes span with row order; the SVD basis is order-independent (tested).
4. **CPU float64 for the diagnostics.** These are a handful of `k × hidden_dim` vectors; computing on CPU copies keeps MPS (no float64) and mixed-device inputs working. Angles and distances return as CPU float64. Tested for float16/bfloat16 and, where available, CUDA placements.
5. **Low-precision bases are accepted and re-projected.** A basis that is orthonormal before conversion cannot stay exact in float16/bfloat16, so the orthonormality check widens to four machine epsilons of the input dtype and then re-projects onto the Stiefel manifold, so downstream angles run on an actually orthonormal basis.
6. **`max_geodesic_distance` is ambient-dimension aware.** Two rank-k subspaces of R^d intersect in at least `max(0, 2k − d)` dimensions, so only `min(k, d − k)` principal angles can reach π/2. The naive `√k · π/2` overstates the manifold diameter whenever `2k > d`.
7. **PR gate**: `ci/test-risk-map.json` lists `obliteratus/analysis/grassmann.py` and `tests/test_grassmann.py` under mechanistic analysis; `scripts/check_test_risk_map.py` and `scripts/select_pr_tests.py --base-ref main` both need to succeed on the fork head.

### Files

- `ci/test-risk-map.json`
- `obliteratus/analysis/grassmann.py`
- `tests/test_grassmann.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none — new module, no callers

## Test evidence

Exact head SHA: `069bf13` (local); fork head TBD

| Check | Result | Evidence or notes |
|---|---|---|
| Focused regression/contract tests | pass (local) | `pytest tests/test_grassmann.py` — 24 passed |
| Negative and boundary tests | pass (local) | non-orthonormal rejection, rank-0 input, mixed-rank geodesic refusal, `max_geodesic_distance` domain |
| `python -m ruff check --select F app.py obliteratus tests scripts` | TBD | |
| `uv lock --check` | TBD | |
| Selected PR core/risk tests | TBD | |
| Package build, when package inputs changed | not applicable | |
| Import and CLI smoke checks | TBD | |
| Applicable risk-surface checks | not applicable | |
| Conditional hardware/service gates | not applicable | CUDA placements are exercised when present, never required |

Coverage or mutation impact:

- Changed-line: TBD

## Research or performance evidence

Not applicable.
