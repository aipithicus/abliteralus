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

## Running the vendored tests

```
uv sync --extra dev                 # CUDA torch per pyproject; fresh uv.lock (never the fork's)
uv run python private/sync/airgap_check.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8 \
uv run pytest -q                    # the vendored core tests are the extraction's acceptance test
```

`PYTORCH_CUDA_ALLOC_CONF` must be pre-set on a Windows CUDA box: upstream `device.py` otherwise
sets `expandable_segments:True`, torch warns it is unsupported on Windows, and upstream's
`filterwarnings = ["error"]` turns that into a test failure (upstream bug; their CI has no CUDA).

Baseline on this machine (fork b0da692, transform as of 2026-08-22): **1173 passed, 10 failed,
1 skipped, 77 % coverage**. The 10 are Windows-only and not extraction artifacts — seven symlink
tests (`WinError 1314`, needs Developer Mode/admin), two loader-retry tests asserting POSIX path
separators, one `os.sysconf` call. They pass on Linux.

What the transform prunes from the vendored tests (all reported on every sync): tests that
reference a non-vendored module, a stubbed network library, an excluded top-level package such
as `scripts/`, or a module-level helper/fixture that does (one level of indirection); and
parametrize literals naming exports dropped from `__init__`.
