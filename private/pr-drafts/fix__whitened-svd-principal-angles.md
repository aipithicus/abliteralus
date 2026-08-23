# fix(whitened_svd): take principal angles on orthonormalized bases

<!-- branch fix/whitened-svd-principal-angles, head 7d7db93. Paste into the upstream PR template; fill TBD cells from the
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

### Files

- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `7d7db93`

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
