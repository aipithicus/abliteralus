# fix(whitened_svd): take principal angles on orthonormalized bases

<!-- branch fix/whitened-svd-principal-angles, head ce7524f. Paste into the upstream PR template; fill TBD cells from the
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

### Follow-up commit — fix(whitened_svd): principal angles for any ranks on rank-revealing bases

Review findings on the first cut:
- P1: with one row on either side the code fell back to the cosine against the
  first standard vector; span(e1) vs span(e0,e1) returned 0.0 instead of 1.0.
  Principal angles are now computed for any pair of ranks.
- P2: orthogonalize_subspace_rows is plain Householder QR and pads dependent
  rows with an arbitrary orthonormal direction; duplicate e0 vs span(e1,e2)
  returned 1.0 instead of 0.0. A local SVD-based rank-revealing basis
  (singular values above 1e-6 of the largest) replaces it here.

Files:
- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

### Follow-up commit — fix(whitened_svd): run the principal-angle diagnostic on the CPU

Review finding: _orthonormal_row_basis converted to float64 while keeping the
input device; MPS has no float64, so MPS-resident results failed before the
SVD. The basis is a k x hidden_dim diagnostic whose output is consumed as
Python floats, so it now moves to the CPU explicitly. Test pins the CPU /
float64 contract for float32, float16, and bfloat16 inputs.

Files:
- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

### Follow-up commit — fix(whitened_svd): compute every comparison cosine on CPU copies

Review follow-up: the direction-level cosines still assumed both inputs share
a device; a CPU standard direction against accelerator-resident whitened
directions failed. All cosines now come from detached CPU float64 copies;
inputs are never moved. Test covers every available device placement.

Files:
- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

### Files

- `obliteratus/analysis/whitened_svd.py`
- `tests/test_analysis.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `ce7524f`

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
