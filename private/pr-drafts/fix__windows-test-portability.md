# test: make persistence, loader, and device tests portable to Windows

<!-- branch fix/windows-test-portability, head 35c29de. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: - persistence: symlink-dependent cases skip when the user cannot create
- Change: TBD
- User-visible effect: TBD
- Linked issue: Closes #TBD

### Rationale (from the lab commit)

- persistence: symlink-dependent cases skip when the user cannot create
  symbolic links (Windows needs Developer Mode or admin); probed once at
  import via a real symlink attempt. Linux behaviour unchanged.
- loader: the permission-retry assertions compared a string suffix with a
  POSIX separator; compare the Path against tmp_path / hf_home / hub.
- device: os.sysconf does not exist on Windows; monkeypatch with
  raising=False. The production fallback already handles AttributeError.

### Files

- `tests/test_device_boundaries.py`
- `tests/test_loader_boundaries.py`
- `tests/test_persistence_contracts.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `35c29de`

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
