# fix(numerical_contracts): return zero rows for dependent subspace directions

<!-- branch fix/subspace-rank-deficiency, head c373328. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: orthogonalize_subspace_rows used a plain Householder QR, which pads a
- Change: TBD
- User-visible effect: TBD
- Linked issue: Closes #TBD

### Rationale (from the lab commit)

orthogonalize_subspace_rows used a plain Householder QR, which pads a
rank-deficient input with an arbitrary orthonormal direction. On the surgery
path (_orthogonalize_subspace before weight projection, and the shield
residualization) that fictitious direction was then projected out of the
weights. Replace with modified Gram-Schmidt (two passes) that keeps the row
count and returns zero rows for dependent directions; both projection
consumers treat a zero direction as a no-op. Full-rank behaviour, dtype
handling, and row-0 orientation are unchanged (existing contract tests pass).

### Files

- `obliteratus/analysis/numerical_contracts.py`
- `tests/test_projection_math_contracts.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `c373328`

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
