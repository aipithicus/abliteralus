# private/sync — shadow-branch vendoring of the public fork

The lab mirrors the fork's layout; everything outside `private/` has a fork counterpart
(existing or future) at the same relative path. `private/` is the only directory with no
counterpart. Package and identifiers are renamed `obliteratus → abliteralus` (three-case,
whole-token); write-back reverses it.

## Branches

```
shadow ──► fixes ──► main
```

- `shadow` — machine-written by `shadow_sync.py`, never edited by hand. One commit per sync,
  built from a fork **commit** (not its working tree), trailers `Fork-Commit:` and `Transform:`.
- `fixes` — one commit per bug fix you intend to send upstream (ledger entries). Nothing else.
  `git diff shadow fixes` is therefore exactly the pending-upstream patch series, and
  `git format-patch shadow..fixes` exports it one patch per ledger entry.
- `main` — research work and lab-only adaptations (things you will *not* upstream).

Sync with `--merge` = `shadow → fixes` (temporary worktree; conflicts stop with the worktree kept
for resolution), then `fixes → main`. A fix that upstream has absorbed drops out of
`diff shadow fixes` on that sync and is reported as **absorbed** — close its ledger entry.
A local fix is never blocked on upstream: it rides on `fixes` across every sync; if upstream later
fixes the same lines differently, that hunk conflicts once (take upstream's), and `rerere` replays
the resolution afterwards.

Adding a fix (never commit fixes on `main`):

```
git worktree add ../fixes-wt fixes        # or: git switch fixes, if main is clean
...edit, test...
git -C ../fixes-wt commit -am "fix(area): one-line summary   [ledger YYYY-MM-DD]"
git worktree remove ../fixes-wt
git merge fixes                           # on main
```

`git diff shadow~1 shadow` is the upstream changelog for the core, already renamed.

## Commands (run from the lab root)

```
python private/sync/shadow_sync.py --self-test        # rename involution + seams + compile; no writes
python private/sync/shadow_sync.py --dry-run          # full report; no writes
python private/sync/shadow_sync.py                    # commit a new shadow; report; no merge
python private/sync/shadow_sync.py --sync-fork        # first fast-forward the fork's main from `upstream` (ff-only)
python private/sync/shadow_sync.py --merge            # ...then shadow -> fixes -> main
python private/sync/airgap_check.py                   # sockets blocked: import every module, poke sentinels
```

The fork needs an `upstream` remote for `--sync-fork`:
`git -C D:/aipithicus/contributing/OBLITERATUS remote add upstream https://github.com/elder-plinius/OBLITERATUS.git`

## What the transform does

1. **select** by `[include]`/`[exclude]` globs in `manifest.toml`; files matching neither are
   reported **UNMAPPED** and not copied — a new upstream file forces a conscious decision.
2. **rename** contents and path segments.
3. **seams** — inline imports of `huggingface_hub` / `datasets` / `requests` are rewritten in place
   (indentation and line count preserved) into `_lab_airgap` sentinels that raise `DisabledError`
   naming the original symbol. Local-path code keeps working; hub-ID paths fail loudly.
   `credential_sources.py` is replaced by an env-only stub. Hard assertion afterwards: no
   forbidden import remains anywhere in the package.
4. **pyproject** — drops the console script, the `app` module and license metadata; swaps the
   torch index to CUDA; pins `[tool.uv] constraint-dependencies` to the fork's locked versions
   (`uv export`), so dependency parity with the fork is maintained on every sync. Adding a
   dependency for a contribution: add it on `main`; it merges cleanly and becomes part of the PR.
5. **commit** via git plumbing — the working tree is never touched; you merge when ready.

Lab-only artifacts (checkpoints, adapters, results) carry the codename in metadata keys and are
not interchangeable with fork-produced ones; cached activations (plain safetensors) are.

## Running the vendored tests

```
uv sync --extra dev                 # CUDA torch per pyproject; fresh uv.lock (never the fork's)
uv run python private/sync/airgap_check.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
uv run pytest -q                    # the vendored core tests are the extraction's acceptance test
```

No `PYTORCH_CUDA_ALLOC_CONF` workaround is needed: the `fixes` branch carries the win32
`expandable_segments` fix (identical to the fork branch `fix/cuda-alloc-windows`), merged into
`main`. When upstream absorbs it, the next sync reports it as absorbed.

Baseline on this machine (fork b0da692, transform as of 2026-08-22): **1173 passed, 10 failed,
1 skipped, 77 % coverage**. The 10 are Windows-only and not extraction artifacts — seven symlink
tests (`WinError 1314`, needs Developer Mode/admin), two loader-retry tests asserting POSIX path
separators, one `os.sysconf` call. They pass on Linux.

What the transform prunes from the vendored tests (all reported on every sync): tests that
reference a non-vendored module, a stubbed network library, an excluded top-level package such
as `scripts/`, or a module-level helper/fixture that does (one level of indirection); and
parametrize literals naming exports dropped from `__init__`.
