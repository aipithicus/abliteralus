#!/usr/bin/env python3
"""writeback — re-author a lab ``fixes`` commit as a branch in the public fork.

    python private/sync/writeback.py <lab-commit> --branch fix/<name> [--base main] [--no-verify]
    python private/sync/writeback.py --list            # commits in shadow..fixes and their export state

For one lab commit: ``git format-patch`` it, reverse the codename rename (contents and paths),
strip lab-only message lines, run the scrub gate, then ``git am -S --3way`` it onto a new branch
created from the fork's base branch. The author and message are preserved; the commit is signed by
the fork's configured key; a ``Lab-Commit:`` trailer records provenance so re-exports are detected.

Scrub gate (hard stop): identity tokens and the codename must not appear anywhere in the patch.
Soft warnings are printed for the private-work vocabulary so you can judge false positives.

Verification (default on): Ruff F on the touched Python files and pytest on the touched test
files, both through the repository-pinned uv executable in the fork's locked CPU environment.

Stdlib only. Python >= 3.11.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

from abliteralus.run_paths import allocate_run_workspace
from abliteralus.toolchain import (
    UvToolchainError,
    repository_tool_environment,
    resolve_uv,
)

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent

HARD_TOKENS = ["aghado01", "thermomapper", "markbrain", "science-facility", "command-center"]
SOFT_TOKENS = ["spc", "bars", "swendsen", "potts", "superparamagnetic"]


def die(msg: str, code: int = 2):
    print(f"writeback: error: {msg}", file=sys.stderr)
    sys.exit(code)


def repository_uv() -> Path:
    try:
        return resolve_uv(LAB_ROOT)
    except UvToolchainError as error:
        die(str(error))


def git(args: list[str], cwd: Path, *, check: bool = True, input_bytes: bytes | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, input=input_bytes)
    if check and proc.returncode != 0:
        die(f"git {' '.join(args)} (in {cwd}) failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout.decode("utf-8", errors="replace")


def load_manifest() -> dict:
    with open(HERE / "manifest.toml", "rb") as fh:
        return tomllib.load(fh)


class Renamer:
    def __init__(self, src: str, dst: str):
        pairs = [(src, dst), (src.capitalize(), dst.capitalize()), (src.upper(), dst.upper())]
        self.rev = [(re.compile(rf"(?<![A-Za-z0-9]){re.escape(b)}(?![A-Za-z0-9])"), a) for a, b in pairs]

    def reverse(self, s: str) -> str:
        for pat, rep in self.rev:
            s = pat.sub(rep, s)
        return s


_LAB_ONLY_LINES = re.compile(r"^(Ledger:|Pending upstream as fork branch).*\n?", re.M)


def export_patch(lab: Path, commit: str, ren: Renamer) -> tuple[str, str, str, list[str]]:
    """Return (patch with codename reversed and a subject-only message, subject, lab body, paths).

    Upstream commits are subject-only; the rationale belongs in the PR description. The lab
    commit's body (minus lab-only lines) is returned separately so it can seed that description.
    """
    patch = git(["format-patch", "-1", "--stdout", "--no-signature", commit], lab)
    subject = ren.reverse(git(["log", "-1", "--format=%s", commit], lab).strip())
    body = ren.reverse(git(["log", "-1", "--format=%b", commit], lab))
    body = _LAB_ONLY_LINES.sub("", body).strip()
    patch = ren.reverse(patch)
    head, sep, rest = patch.partition("\n---\n")
    if not sep:
        die("unexpected format-patch layout (no '---' separator)")
    # Keep the mail headers (From/Date/Subject), drop everything after them: subject-only message.
    headers, seen_subject = [], False
    for line in head.split("\n"):
        if seen_subject and not line.startswith((" ", "\t")):
            break  # end of a (possibly folded) Subject header
        headers.append(line)
        if line.startswith("Subject:"):
            seen_subject = True
    patch = "\n".join(headers) + "\n" + sep + rest
    touched = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))
    return patch, subject, body, touched


def scrub(patch: str, codename: str) -> None:
    hard = HARD_TOKENS + [codename]
    hits = {t: len(re.findall(rf"(?i)(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])", patch)) for t in hard}
    bad = {t: n for t, n in hits.items() if n}
    if bad:
        die(f"scrub gate: private/identity tokens present in patch: {bad}")
    soft = {t: len(re.findall(rf"(?i)\b{re.escape(t)}\b", patch)) for t in SOFT_TOKENS}
    soft = {t: n for t, n in soft.items() if n}
    if soft:
        print(f"writeback: warning: private-work vocabulary in patch (judge each): {soft}")


def _patch_id(fork: Path, patch_text: str) -> str:
    out = git(["patch-id", "--stable"], fork, input_bytes=patch_text.encode("utf-8")).strip()
    return out.split(" ")[0] if out else ""


EXPORT_RECORD = HERE / "exported.toml"


def recorded_exports() -> dict[str, str]:
    """lab commit sha -> fork branch, for exports that can no longer be matched by patch-id
    (squashed branches). Maintained by ``--squash``; plain TOML, one table per record."""
    if not EXPORT_RECORD.exists():
        return {}
    with open(EXPORT_RECORD, "rb") as fh:
        data = tomllib.load(fh)
    return {rec["lab"]: rec["branch"] for rec in data.get("exported", [])}


def record_exports(pairs: list[tuple[str, str]], how: str) -> None:
    lines = [] if not EXPORT_RECORD.exists() else [EXPORT_RECORD.read_text(encoding="utf-8").rstrip("\n")]
    for lab_sha, branch in pairs:
        lines.append(f'\n[[exported]]\nlab = "{lab_sha}"\nbranch = "{branch}"\nhow = "{how}"')
    EXPORT_RECORD.write_text("\n".join(lines).lstrip("\n") + "\n", encoding="utf-8", newline="\n")


def already_exported(fork: Path, commit: str, patch: str | None = None, base: str = "main") -> str | None:
    """Find a fork branch carrying this lab commit: by record, by Lab-Commit trailer, else by patch-id."""
    recorded = recorded_exports().get(commit)
    if recorded:
        return f"{recorded} (recorded)"
    out = git(["log", "--all", "--format=%h %D", f"--grep=Lab-Commit: {commit}"], fork).strip()
    if out:
        return out.split("\n")[0]
    if patch is None:
        return None
    want = _patch_id(fork, patch)
    if not want:
        return None
    for ref in git(["for-each-ref", "--format=%(refname:short)", "refs/heads"], fork).split():
        if ref == base:
            continue
        for sha in git(["rev-list", "--no-merges", f"{base}..{ref}"], fork).split():
            have = _patch_id(fork, git(["show", "--format=", sha], fork))
            if have == want:
                return f"{sha[:7]} {ref} (by patch-id)"
    return None


def cmd_squash(
    lab: Path,
    fork: Path,
    branch: str,
    base: str,
    ren: Renamer,
    no_verify: bool,
    uv_executable: Path | None,
) -> int:
    """Squash an unpushed fork branch to one signed, subject-only commit and record the lab mapping."""
    if not git(["rev-parse", "-q", "--verify", f"refs/heads/{branch}"], fork, check=False).strip():
        die(f"fork branch {branch!r} does not exist")
    if git(["rev-parse", "-q", "--verify", f"refs/remotes/origin/{branch}"], fork, check=False).strip():
        die(f"origin/{branch} exists; never rewrite published history")
    if git(["status", "--porcelain", "--untracked-files=no"], fork).strip():
        die("fork has uncommitted tracked changes; commit or stash first")
    shas = git(["rev-list", "--reverse", "--no-merges", f"{base}..{branch}"], fork).split()
    if len(shas) < 2:
        die(f"{branch} has {len(shas)} commit(s) on {base}; nothing to squash")
    # Which lab commits does this branch carry? Match each fork commit back by patch-id so the
    # mapping survives the squash.
    lab_log = git(["log", "--no-merges", "--format=%H", "shadow..fixes"], lab).split()
    fork_ids = {_patch_id(fork, git(["show", "--format=", s], fork)): s for s in shas}
    carried: list[tuple[str, str]] = []
    for lab_sha in lab_log:
        patch, _, _, _ = export_patch(lab, lab_sha, ren)
        if _patch_id(fork, patch) in fork_ids:
            carried.append((lab_sha, branch))
    subject = git(["log", "-1", "--format=%s", shas[0]], fork).strip()
    touched = git(["diff", "--name-only", base, branch], fork).split()
    original = git(["rev-parse", "--abbrev-ref", "HEAD"], fork).strip()
    git(["checkout", "-q", branch], fork)
    git(["reset", "-q", "--soft", base], fork)
    git(["commit", "-q", "-S", "-m", subject], fork)
    head = git(["log", "-1", "--format=%h %G? %GS"], fork).strip()
    if no_verify:
        ok = True
    else:
        if uv_executable is None:
            raise AssertionError("verified squash requires the repository uv executable")
        ok = verify(fork, touched, uv_executable)
    git(["checkout", "-q", original], fork)
    record_exports(carried, "squash")
    draft = LAB_ROOT / "private" / "pr-drafts" / f"{branch.replace('/', '__')}.md"
    if draft.exists():
        text = draft.read_text(encoding="utf-8")
        short = head.split(" ")[0]
        text = re.sub(r"(<!-- branch [^,]+, head )\w+", rf"\g<1>{short}", text, count=1)
        text = re.sub(r"(Exact head SHA: `)\w+(`)", rf"\g<1>{short}\2", text, count=1)
        draft.write_text(text, encoding="utf-8", newline="\n")
    print(f"squashed {len(shas)} commits -> {head}  [{subject}]")
    print(f"recorded {len(carried)} lab commit(s) as exported to {branch}")
    return 0 if ok else 1


def cmd_list(lab: Path, fork: Path) -> int:
    log = git(["log", "--reverse", "--no-merges", "--format=%h %s", "shadow..fixes"], lab).strip()
    if not log:
        print("no commits in shadow..fixes")
        return 0
    manifest = load_manifest()
    ren = Renamer(manifest["rename"]["from"], manifest["rename"]["to"])
    print("pending-upstream series (lab fixes branch):")
    for line in log.split("\n"):
        sha = line.split(" ", 1)[0]
        full = git(["rev-parse", sha], lab).strip()
        patch, _, _, _ = export_patch(lab, full, ren)
        state = already_exported(fork, full, patch)
        if not state:
            mark = "not exported"
        elif state.endswith("(recorded)"):
            mark = f"exported -> {state}"
        else:
            mark = f"exported -> {state.split(' ', 1)[1] or state[:12]}"
        print(f"  {line}    [{mark}]")
    return 0


def write_pr_draft(branch: str, subject: str, body: str, touched: list[str], head: str) -> Path:
    """Seed the upstream PR template from the lab commit's rationale (kept out of the commit)."""
    out_dir = LAB_ROOT / "private" / "pr-drafts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{branch.replace('/', '__')}.md"
    files = "\n".join(f"- `{t}`" for t in touched)
    problem = body.splitlines()[0] if body else "TBD"
    text = f"""# {subject}

<!-- branch {branch}, head {head}. Paste into the upstream PR template; fill TBD cells from the
writeback verification output and the fork's own checks. Scrub before posting. -->

## Summary

- Problem: {problem}
- Change: TBD
- User-visible effect: TBD
- Linked issue: Closes #TBD

### Rationale (from the lab commit)

{body or "TBD"}

### Files

{files}

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none

## Test evidence

Exact head SHA: `{head}`

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
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def append_pr_draft(branch: str, subject: str, body: str, touched: list[str], head: str) -> Path:
    """Add a follow-up-commit section to an existing draft (created if missing) and refresh the head."""
    path = LAB_ROOT / "private" / "pr-drafts" / f"{branch.replace('/', '__')}.md"
    if not path.exists():
        return write_pr_draft(branch, subject, body, touched, head)
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(<!-- branch [^,]+, head )\w+", rf"\g<1>{head}", text, count=1)
    text = re.sub(r"(Exact head SHA: `)\w+(`)", rf"\g<1>{head}\2", text, count=1)
    files = "\n".join(f"- `{t}`" for t in touched)
    section = f"\n### Follow-up commit — {subject}\n\n{body or 'TBD'}\n\nFiles:\n{files}\n"
    marker = "\n### Files\n"
    if marker in text:
        text = text.replace(marker, section + marker, 1)
    else:
        text = text.rstrip("\n") + "\n" + section
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def verify(fork: Path, touched: list[str], uv_executable: Path) -> bool:
    py = [p for p in touched if p.endswith(".py")]
    tests = [p for p in py if p.startswith("tests/")]
    ok = True
    environment = repository_tool_environment(LAB_ROOT)
    if py:
        proc = subprocess.run(
            [
                str(uv_executable),
                "run",
                "--frozen",
                "ruff",
                "check",
                "--select",
                "F",
                *py,
            ],
            cwd=str(fork),
            env=environment,
            timeout=300,
        )
        ok &= proc.returncode == 0
        print("verify: ruff F", "ok" if proc.returncode == 0 else "FAILED")
    if tests:
        proc = subprocess.run(
            [
                str(uv_executable),
                "run",
                "--frozen",
                "pytest",
                "-q",
                "--no-cov",
                "-p",
                "no:randomly",
                *tests,
            ],
            cwd=str(fork),
            env=environment,
            capture_output=True,
            text=True,
            timeout=900,
        )
        tail = [line for line in proc.stdout.strip().split("\n") if line][-1:]
        print("verify: pytest", "ok" if proc.returncode == 0 else "FAILED", "-", *tail)
        ok &= proc.returncode == 0
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commit", nargs="?", help="lab commit on the fixes branch to export")
    ap.add_argument("--branch", help="fork branch to create (required unless --list)")
    ap.add_argument("--base", default="main", help="fork branch to base on (default: main)")
    ap.add_argument("--fork", type=Path, help="fork checkout (default: manifest [fork].path)")
    ap.add_argument("--list", action="store_true", help="list shadow..fixes and export state")
    ap.add_argument("--no-verify", action="store_true", help="skip ruff/pytest in the fork")
    ap.add_argument("--draft-only", action="store_true",
                    help="only (re)write the PR description draft for an already-exported branch")
    ap.add_argument("--lab-branch", default="fixes",
                    help="lab branch the commit must belong to (default: fixes; use a feat/* branch for contributions)")
    ap.add_argument("--onto", action="store_true",
                    help="append to an existing (unpushed) fork branch instead of creating one")
    ap.add_argument("--squash", action="store_true",
                    help="squash the unpushed fork --branch to one signed subject-only commit and record the lab mapping")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    fork = (args.fork or Path(manifest["fork"]["path"])).resolve()
    lab = LAB_ROOT
    ren = Renamer(manifest["rename"]["from"], manifest["rename"]["to"])
    if args.list:
        return cmd_list(lab, fork)
    if args.squash:
        if not args.branch:
            ap.error("--squash requires --branch")
        uv_executable = None if args.no_verify else repository_uv()
        return cmd_squash(
            lab,
            fork,
            args.branch,
            args.base,
            ren,
            args.no_verify,
            uv_executable,
        )
    if not args.commit or not args.branch:
        ap.error("a lab commit and --branch are required (or use --list / --squash)")

    uv_executable = None if args.no_verify else repository_uv()

    commit = git(["rev-parse", "--verify", f"{args.commit}^{{commit}}"], lab).strip()
    if not git(["branch", "--contains", commit, args.lab_branch], lab).strip():
        die(f"{commit[:12]} is not on the lab branch {args.lab_branch!r}")
    patch, subject, body, touched = export_patch(lab, commit, ren)
    prior = already_exported(fork, commit, patch, args.base)
    if args.draft_only:
        if not git(["rev-parse", "-q", "--verify", f"refs/heads/{args.branch}"], fork, check=False).strip():
            die(f"fork branch {args.branch!r} does not exist; export first")
        head = git(["rev-parse", "--short", args.branch], fork).strip()
        if args.onto:
            print(f"PR description draft: {append_pr_draft(args.branch, subject, body, touched, head)}")
        else:
            print(f"PR description draft: {write_pr_draft(args.branch, subject, body, touched, head)}")
        return 0
    if prior:
        die(f"{commit[:12]} was already exported: {prior}")
    if git(["status", "--porcelain", "--untracked-files=no"], fork).strip():
        die("fork has uncommitted tracked changes; commit or stash first")
    branch_exists = bool(git(["rev-parse", "-q", "--verify", f"refs/heads/{args.branch}"], fork, check=False).strip())
    if branch_exists and not args.onto:
        die(f"fork branch {args.branch!r} already exists (use --onto to append a follow-up commit)")
    if args.onto and not branch_exists:
        die(f"--onto given but fork branch {args.branch!r} does not exist")
    if args.onto and git(["rev-parse", "-q", "--verify", f"refs/remotes/origin/{args.branch}"], fork, check=False).strip():
        print(f"writeback: note: origin/{args.branch} exists; appending is fine, never rewrite published history")
    email = git(["config", "user.email"], fork).strip()
    if email != "aipithicus@proton.me":
        die(f"fork identity resolves to {email!r}; refusing")

    scrub(patch, manifest["rename"]["to"])
    print(f"exporting {commit[:12]} — {subject}" + (f" (onto {args.branch})" if args.onto else ""))
    for t in touched:
        print(f"    {t}")

    original = git(["rev-parse", "--abbrev-ref", "HEAD"], fork).strip()
    if args.onto:
        git(["checkout", "-q", args.branch], fork)
    else:
        git(["checkout", "-q", "-b", args.branch, args.base], fork)
    workspace = allocate_run_workspace(lab, "sync-writeback")
    patch_path = workspace.path / "export.patch"
    patch_path.write_text(patch, encoding="utf-8", newline="\n")
    proc = subprocess.run(["git", "am", "-S", "--3way", patch_path], cwd=str(fork), capture_output=True, text=True)
    if proc.returncode != 0:
        workspace.record_result(proc.returncode)
        print(proc.stdout, proc.stderr, sep="\n")
        print(f"git am failed; branch {args.branch} left checked out in the fork for resolution "
              f"(git am --continue / --abort). Patch: {patch_path}")
        return 1
    workspace.record_result(0)
    workspace.cleanup()
    head = git(["log", "-1", "--format=%h %G? %GS"], fork).strip()
    print(f"fork {args.branch}: {head}")
    draft = (append_pr_draft if args.onto else write_pr_draft)(args.branch, subject, body, touched, head.split(" ")[0])
    print(f"PR description draft: {draft}")
    if args.no_verify:
        ok = True
    else:
        if uv_executable is None:
            raise AssertionError("verified writeback requires the repository uv executable")
        ok = verify(fork, touched, uv_executable)
    git(["checkout", "-q", original], fork)
    print(f"fork back on {original}; branch {args.branch} ready" if ok else f"verification FAILED on {args.branch}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
