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

Adding a fix (never commit fixes on `main`). A persistent worktree for `fixes` lives at
`../abliteralus-fixes-wt` with its own `.venv` (synced from this repo's `uv.lock`, copied in), so
tests there exercise the worktree's code and `main`'s working tree is never disturbed:

```
git worktree add ../fixes-wt fixes        # or: git switch fixes, if main is clean
...edit, test...
git -C ../fixes-wt commit -am "fix(area): one-line summary   [ledger YYYY-MM-DD]"
git worktree remove ../fixes-wt
git merge fixes                           # on main
```

`git diff shadow~1 shadow` is the upstream changelog for the core, already renamed.

## Write-back

`writeback.py <lab-commit> --branch fix/<name>` turns one `fixes` commit into a signed branch on the
fork, based on the fork's `main`: `git format-patch` → reverse the codename (contents and paths) →
strip lab-only message lines → **scrub gate** (identity tokens and the codename must be absent;
private-work vocabulary only warns) → `git am -S --3way` → Ruff F and pytest on the touched files in
the fork's locked CPU environment. Upstream commits are **subject-only** (conventional-commit
subject, no body, no trailers), so the fork commit keeps just the subject; the lab commit's body
seeds a PR-description draft in `private/pr-drafts/<branch>.md` laid out on the upstream PR
template. `--list` shows which entries are already exported (matched by patch-id);
`--draft-only` regenerates a draft for an exported branch; `--onto` appends a follow-up `fixes`
commit (for example a review fix) to an existing, unpushed fork branch; `--squash` collapses an
unpushed branch to one signed subject-only commit (first commit's subject) and records the lab
commits it carries in `exported.toml`, since patch-id matching cannot survive a squash. One ledger entry = one
fork branch = one upstream PR; a branch may carry more than one commit. Pushing is always a
separate, manual step.

## Commands (run from the lab root)

```
python private/sync/shadow_sync.py --self-test        # rename involution + seams + compile; no writes
python private/sync/shadow_sync.py --dry-run          # full report; no writes
python private/sync/shadow_sync.py                    # commit a new shadow; report; no merge
python private/sync/shadow_sync.py --sync-fork        # first fast-forward the fork's main from `upstream` (ff-only)
python private/sync/shadow_sync.py --merge            # ...then shadow -> fixes -> main
python private/sync/airgap_check.py                   # sockets blocked: import every module, poke sentinels
python private/sync/writeback.py --list               # pending series and which entries are already in the fork
python private/sync/writeback.py <lab-commit> --branch fix/<name>   # re-author one fixes commit as a fork branch
python private/sync/writeback.py <lab-commit> --branch fix/<name> --onto   # append a follow-up to that branch
python private/sync/writeback.py --squash --branch fix/<name>    # unpushed branch -> one commit; lab mapping recorded
python private/sync/writeback.py <lab-commit> --branch feat/<name> --lab-branch feat/<name>   # contributions from a lab feature branch
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

Baseline on this machine (fork b0da692, `fixes` through 5b1c440): **1188 passed, 0 failed,
8 skipped, 77 % coverage**. The skips are the seven symlink-privilege cases (capability-probed;
they run on Linux or with Windows Developer Mode) plus one upstream skip. Before the portability
fixes on `fixes`, the same tree had 10 Windows-only failures.

What the transform prunes from the vendored tests (all reported on every sync): tests that
reference a non-vendored module, a stubbed network library, an excluded top-level package such
as `scripts/`, or a module-level helper/fixture that does (one level of indirection); and
parametrize literals naming exports dropped from `__init__`.
