# private/sync — shadow-branch vendoring of the public fork

The lab mirrors the fork's layout; everything outside `private/` has a fork counterpart
(existing or future) at the same relative path. `private/` is the only directory with no
counterpart. Package and identifiers are renamed `obliteratus → abliteralus` (three-case,
whole-token); write-back reverses it.

## Branches

- `shadow` — machine-written by `shadow_sync.py`, never edited by hand. One commit per sync,
  built from a fork **commit** (not its working tree), trailer `Fork-Commit: <sha>`.
- `main` — your work. Created from the first shadow commit. Sync = `git merge shadow`.

`git diff shadow~1 shadow` is the upstream changelog for the core, already renamed.

## Commands (run from the lab root)

```
python private/sync/shadow_sync.py --self-test        # rename involution + seams + compile; no writes
python private/sync/shadow_sync.py --dry-run          # full report; no writes
python private/sync/shadow_sync.py                    # commit a new shadow; report; no merge
python private/sync/shadow_sync.py --sync-fork        # first fast-forward the fork's main from `upstream` (ff-only)
python private/sync/shadow_sync.py --merge            # ...then merge shadow into main
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

## After the first extraction

```
uv sync --extra dev                 # CUDA torch per pyproject; fresh uv.lock (never the fork's)
python private/sync/airgap_check.py
uv run pytest -q                    # the vendored core tests are the extraction's acceptance test
git add private README.md && git commit -m "lab: tooling"
```
