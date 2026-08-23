# fix(device): do not request expandable_segments on Windows

<!-- branch fix/cuda-alloc-windows, head 7c89f64. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: TBD
- Change: TBD
- User-visible effect: TBD
- Linked issue: Closes #TBD

### Rationale (from the lab commit)

TBD

### Files

- `obliteratus/device.py`
- `tests/test_device_boundaries.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `7c89f64`

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
