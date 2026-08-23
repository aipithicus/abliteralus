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
files, both in the fork's locked CPU environment (``uv run --frozen``).

Stdlib only. Python >= 3.11.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent

HARD_TOKENS = ["aghado01", "thermomapper", "markbrain", "science-facility", "command-center"]
SOFT_TOKENS = ["spc", "bars", "swendsen", "potts", "superparamagnetic"]


def die(msg: str, code: int = 2):
    print(f"writeback: error: {msg}", file=sys.stderr)
    sys.exit(code)


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


def export_patch(lab: Path, commit: str, ren: Renamer) -> tuple[str, str, list[str]]:
    """Return (patch text with codename reversed, subject, touched fork paths)."""
    patch = git(["format-patch", "-1", "--stdout", "--no-signature", commit], lab)
    subject = git(["log", "-1", "--format=%s", commit], lab).strip()
    patch = ren.reverse(patch)
    patch = _LAB_ONLY_LINES.sub("", patch)
    # Provenance trailer: inserted before the first diff header (end of the message body).
    head, sep, rest = patch.partition("\n---\n")
    if not sep:
        die("unexpected format-patch layout (no '---' separator)")
    head = head.rstrip("\n") + f"\n\nLab-Commit: {commit}\n"
    patch = head + sep + rest
    touched = sorted(set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.M)))
    return patch, subject, touched


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


def already_exported(fork: Path, commit: str, patch: str | None = None, base: str = "main") -> str | None:
    """Find a fork branch carrying this lab commit: by Lab-Commit trailer, else by patch-id."""
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
        patch, _, _ = export_patch(lab, full, ren)
        state = already_exported(fork, full, patch)
        mark = f"exported -> {state.split(' ', 1)[1] or state[:12]}" if state else "not exported"
        print(f"  {line}    [{mark}]")
    return 0


def verify(fork: Path, touched: list[str]) -> bool:
    py = [p for p in touched if p.endswith(".py")]
    tests = [p for p in py if p.startswith("tests/")]
    ok = True
    if py:
        proc = subprocess.run(["uv", "run", "--frozen", "ruff", "check", "--select", "F", *py], cwd=str(fork))
        ok &= proc.returncode == 0
        print("verify: ruff F", "ok" if proc.returncode == 0 else "FAILED")
    if tests:
        proc = subprocess.run(["uv", "run", "--frozen", "pytest", "-q", "--no-cov", "-p", "no:randomly", *tests],
                              cwd=str(fork), capture_output=True, text=True)
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
    args = ap.parse_args(argv)

    manifest = load_manifest()
    fork = (args.fork or Path(manifest["fork"]["path"])).resolve()
    lab = LAB_ROOT
    ren = Renamer(manifest["rename"]["from"], manifest["rename"]["to"])
    if args.list:
        return cmd_list(lab, fork)
    if not args.commit or not args.branch:
        ap.error("a lab commit and --branch are required (or use --list)")

    commit = git(["rev-parse", "--verify", f"{args.commit}^{{commit}}"], lab).strip()
    if not git(["branch", "--contains", commit, "fixes"], lab).strip():
        die(f"{commit[:12]} is not on the lab fixes branch")
    patch, subject, touched = export_patch(lab, commit, ren)
    prior = already_exported(fork, commit, patch, args.base)
    if prior:
        die(f"{commit[:12]} was already exported: {prior}")
    if git(["status", "--porcelain", "--untracked-files=no"], fork).strip():
        die("fork has uncommitted tracked changes; commit or stash first")
    if git(["rev-parse", "-q", "--verify", f"refs/heads/{args.branch}"], fork, check=False).strip():
        die(f"fork branch {args.branch!r} already exists")
    email = git(["config", "user.email"], fork).strip()
    if email != "aipithicus@proton.me":
        die(f"fork identity resolves to {email!r}; refusing")

    scrub(patch, manifest["rename"]["to"])
    print(f"exporting {commit[:12]} — {subject}")
    for t in touched:
        print(f"    {t}")

    original = git(["rev-parse", "--abbrev-ref", "HEAD"], fork).strip()
    git(["checkout", "-q", "-b", args.branch, args.base], fork)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8", newline="\n") as fh:
        fh.write(patch)
        patch_path = fh.name
    proc = subprocess.run(["git", "am", "-S", "--3way", patch_path], cwd=str(fork), capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, sep="\n")
        print(f"git am failed; branch {args.branch} left checked out in the fork for resolution "
              f"(git am --continue / --abort). Patch: {patch_path}")
        return 1
    head = git(["log", "-1", "--format=%h %G? %GS"], fork).strip()
    print(f"fork {args.branch}: {head}")
    ok = True if args.no_verify else verify(fork, touched)
    git(["checkout", "-q", original], fork)
    print(f"fork back on {original}; branch {args.branch} ready" if ok else f"verification FAILED on {args.branch}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
