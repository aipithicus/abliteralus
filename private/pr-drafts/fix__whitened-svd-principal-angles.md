# fix(whitened_svd): take principal angles on orthonormalized bases

<!-- branch fix/whitened-svd-principal-angles, head b57e8c5. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: Singular values of Y Z^T are principal-angle cosines only when both bases
- Change: TBD
- User-visible effect: TBD
- Linked issue: Closes #TBD

### Rationale (from the lab commit)

Singular values of Y Z^T are principal-angle cosines only when both bases
are orthonormal. Whitened directions are unit-norm but orthogonal under the
inverse harmless covariance, not the Euclidean inner product, so the raw
product could yield values above one (hidden by the clamp) and a wrong
spectrum below the first. Orthonormalize both sides with
orthogonalize_subspace_rows first. subspace_principal_cosine keeps its
meaning (cosine of the smallest angle, now correct); the new
subspace_principal_cosines lists the whole spectrum in descending order.
Oracle tests cover coincident, orthogonal, and partially shared planes.

### Second commit — review findings on the first cut

- With one row on either side the first cut fell back to the cosine against the
  first standard vector; `span(e1)` vs `span(e0, e1)` returned 0.0 instead of 1.0.
  Principal angles are now computed for any pair of ranks (the number of cosines
  is `min(rank, rank)`), and the direction-level `primary_direction_cosine` keeps
  its own meaning.
- `orthogonalize_subspace_rows` is plain Householder QR and pads dependent rows
  with an arbitrary orthonormal direction; duplicate `e0` vs `span(e1, e2)`
  returned 1.0 instead of 0.0. The comparison now uses a local SVD-based
  rank-revealing basis (singular values above 1e-6 of the largest). The shared
  helper is left unchanged here because it also serves the surgery path; that
  is raised separately.
- Regression tests for both, including near-duplicate rows and a single
  standard direction against a whitened plane.

### Files

- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `b57e8c5`

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
