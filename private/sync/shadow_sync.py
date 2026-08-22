#!/usr/bin/env python3
"""shadow_sync — pull the public fork's research core into this lab as a vendor branch.

The transform is a pure function of a fork *commit* (read with ``git show <ref>:<path>``),
never of the fork's working tree, so uncommitted fork changes can never leak in.

Pipeline (see manifest.toml):
  1. select   fork files by [include]/[exclude] globs; everything else is reported UNMAPPED
  2. rename   whole-token three-case substitution of the project name, in contents and paths
  3. seams    rewrite inline network imports into airgap sentinels; replace/inject stub files
  4. pyproject drop console script / app module / license metadata, swap torch index to CUDA,
              pin [tool.uv] constraint-dependencies to the fork's locked versions (uv export)
  5. verify   no forbidden imports remain anywhere in the package
  6. commit   the tree on the local ``shadow`` branch via git plumbing (working tree untouched);
              first run also creates ``main`` from it and checks it out
  7. report   unmapped files, seam counts, files changed upstream since the last shadow, and
              which of those you have also modified on ``main`` (the merges that need eyes)

Default is report-only. ``--merge`` merges ``shadow`` into ``main`` afterwards.
``--self-test`` proves the rename is an involution on the selected set and that the seams
leave no forbidden import behind, without writing anything.

Stdlib only. Python >= 3.11 (tomllib).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent  # private/sync -> private -> lab root


# ───────────────────────────── helpers ─────────────────────────────


def die(msg: str, code: int = 2) -> NoReturn:
    print(f"shadow_sync: error: {msg}", file=sys.stderr)
    sys.exit(code)


def git(args: list[str], cwd: Path, *, check: bool = True, env: dict | None = None,
        binary: bool = False) -> str | bytes:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, env=full_env)
    if check and proc.returncode != 0:
        die(f"git {' '.join(args)} (in {cwd}) failed:\n{proc.stderr.decode(errors='replace')}")
    return proc.stdout if binary else proc.stdout.decode("utf-8", errors="replace").strip()


def glob_to_regex(pattern: str) -> re.Pattern:
    out = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
            if i < len(pattern) and pattern[i] == "/":
                i += 1  # "**/" already covered by .*
            continue
        if c == "*":
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("^" + "".join(out) + "$")


def matches_any(path: str, patterns: list[re.Pattern]) -> bool:
    return any(p.match(path) for p in patterns)


# ───────────────────────────── rename ─────────────────────────────


class Renamer:
    """Whole-token three-case substitution; underscores count as token-internal neighbours."""

    def __init__(self, src: str, dst: str):
        self.pairs = [(src, dst), (src.capitalize(), dst.capitalize()), (src.upper(), dst.upper())]
        self.fwd = [(re.compile(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])"), b) for a, b in self.pairs]
        self.rev = [(re.compile(rf"(?<![A-Za-z0-9]){re.escape(b)}(?![A-Za-z0-9])"), a) for a, b in self.pairs]

    def text(self, s: str, reverse: bool = False) -> str:
        for pat, rep in (self.rev if reverse else self.fwd):
            s = pat.sub(rep, s)
        return s

    def path(self, p: str, reverse: bool = False) -> str:
        return "/".join(self.text(seg, reverse) for seg in p.split("/"))


# ───────────────────────────── seams ─────────────────────────────

_FROM_IMPORT = re.compile(r"^(?P<ind>[ \t]*)from (?P<mod>{mods})(?P<sub>(?:\.[\w.]+)?) import (?P<names>[\w]+(?:[ \t]*,[ \t]*[\w]+)*)(?:[ \t]+as[ \t]+(?P<alias>\w+))?[ \t]*(?:#.*)?$")
_PLAIN_IMPORT = re.compile(r"^(?P<ind>[ \t]*)import (?P<mod>{mods})(?:[ \t]+as[ \t]+(?P<alias>\w+))?[ \t]*(?:#.*)?$")
_FUTURE = re.compile(r"^from __future__ import .*$")
_EXC_NAME = re.compile(r"(Error|Exception|Warning|Expired)$")


def seam_regexes(modules: list[str]) -> tuple[re.Pattern, re.Pattern]:
    alt = "|".join(re.escape(m) for m in modules)
    return (re.compile(_FROM_IMPORT.pattern.format(mods=alt), re.M),
            re.compile(_PLAIN_IMPORT.pattern.format(mods=alt), re.M))


def apply_seams(text: str, package: str, from_re: re.Pattern, plain_re: re.Pattern) -> tuple[str, int]:
    """Rewrite inline network imports to sentinels. Returns (text, replacements)."""
    lines = text.split("\n")
    count = 0
    for i, line in enumerate(lines):
        m = from_re.match(line)
        if m:
            ind, mod, sub, alias = m["ind"], m["mod"], m["sub"] or "", m["alias"]
            names = [n.strip() for n in m["names"].split(",")]
            stmts = []
            for n in names:
                bound = alias if (alias and len(names) == 1) else n
                dotted = f"{mod}{sub}.{n}"
                if _EXC_NAME.search(n):
                    stmts.append(f"{bound} = _lab_airgap.DisabledError  # airgap: {dotted}")
                else:
                    stmts.append(f"{bound} = _lab_airgap.disabled({dotted!r})")
            lines[i] = ind + "; ".join(stmts)
            count += 1
            continue
        m = plain_re.match(line)
        if m:
            ind, mod, alias = m["ind"], m["mod"], m["alias"]
            lines[i] = f"{ind}{alias or mod} = _lab_airgap.disabled_module({mod!r})"
            count += 1
    if count == 0:
        return text, 0
    # Inject the sentinel import after `from __future__` if present, else after the module
    # docstring, else at the very top.
    inject = f"from {package} import _lab_airgap  # airgap sentinels (lab-only)"
    insert_at = 0
    for i, line in enumerate(lines[:80]):
        if _FUTURE.match(line):
            insert_at = i + 1
            break
    else:
        if lines and lines[0].lstrip().startswith(('"""', "'''")):
            quote = lines[0].lstrip()[:3]
            if lines[0].count(quote) >= 2 and len(lines[0].strip()) > 3:
                insert_at = 1
            else:
                for i in range(1, len(lines)):
                    if quote in lines[i]:
                        insert_at = i + 1
                        break
    lines.insert(insert_at, inject)
    return "\n".join(lines), count


# ───────────────────────────── pruning ─────────────────────────────

_LAZY_BRANCH = re.compile(
    r"^    if name == \"(?P<name>\w+)\":\n        from (?P<mod>[\w.]+) import \w+\n        return \w+\n",
    re.M,
)


def prune_init_exports(text: str, package: str, selected_modules: set[str]) -> tuple[str, list[str]]:
    """Drop lazy exports in the package ``__init__`` whose target module was not vendored."""
    dropped: list[str] = []

    def _branch(m: re.Match) -> str:
        mod = m["mod"]
        if mod.startswith(package + ".") and mod[len(package) + 1:] not in selected_modules:
            dropped.append(m["name"])
            return ""
        return m.group(0)

    text = _LAZY_BRANCH.sub(_branch, text)
    for name in dropped:
        text = re.sub(rf'^    "{re.escape(name)}",\n', "", text, flags=re.M)
    return text, dropped


def prune_literal_exports(text: str, dropped_exports: list[str]) -> tuple[str, int]:
    """Drop ``"name",`` list entries in test parametrizations for exports the lab no longer has."""
    n = 0
    for name in dropped_exports:
        text, k = re.subn(rf'^\s*"{re.escape(name)}",[ \t]*\n', "", text, flags=re.M)
        n += k
    return text, n


def prune_tests(text: str, package: str, excluded_modules: set[str], network_modules: list[str],
                path: str, extra_refs: list[str] | None = None) -> tuple[str, list[str]]:
    """Remove test functions that reach a non-vendored module or a stubbed network library.

    AST-anchored: a test (module-level ``test_*`` or a method of a top-level class) is removed,
    decorators included, when its source mentions ``<package>.<excluded module>``, any network
    module name as a whole word, an import of an excluded top-level package (``extra_refs``), or
    the name of a module-level helper/fixture that itself does any of those (one level of
    indirection). A class emptied by pruning gets a ``pass`` body.
    """
    import ast

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text, []
    lines = text.split("\n")
    pats: list[re.Pattern] = []
    if excluded_modules:
        pats.append(re.compile(rf"\b{re.escape(package)}\.({'|'.join(map(re.escape, sorted(excluded_modules)))})\b"))
    if network_modules:
        pats.append(re.compile(rf"\b({'|'.join(map(re.escape, network_modules))})\b"))
    if extra_refs:
        alt = "|".join(map(re.escape, extra_refs))
        pats.append(re.compile(rf"(?:^|[^\w.])(?:from|import)\s+({alt})\b|[\"']({alt})[.\"']", re.M))

    def src_of(fn: ast.AST) -> tuple[int, int, str]:
        start = min([fn.lineno] + [d.lineno for d in fn.decorator_list])  # type: ignore[attr-defined]
        end = fn.end_lineno  # type: ignore[attr-defined]
        return start, end, "\n".join(lines[start - 1:end])

    def tainted(src: str) -> bool:
        return any(p.search(src) for p in pats)

    # One level of indirection: module-level non-test functions (helpers, fixtures) that are tainted.
    helpers = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("test")]
    tainted_helpers = {h.name for h in helpers if tainted(src_of(h)[2])}
    if tainted_helpers:
        pats.append(re.compile(rf"\b({'|'.join(map(re.escape, sorted(tainted_helpers)))})\b"))

    removed: list[str] = []
    spans: list[tuple[int, int]] = []  # 1-based inclusive

    def consider(fn: ast.AST, qual: str) -> None:
        start, end, src = src_of(fn)
        if tainted(src):
            spans.append((start, end))
            removed.append(f"{path}::{qual}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            consider(node, node.name)
        elif isinstance(node, ast.ClassDef):
            pruned_starts: set[int] = set()
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith("test"):
                    before = len(spans)
                    consider(m, f"{node.name}::{m.name}")
                    if len(spans) > before:
                        pruned_starts.add(spans[-1][0])
            if not pruned_starts:
                continue
            survivors = [
                n for n in node.body
                if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and min([n.lineno] + [d.lineno for d in n.decorator_list]) in pruned_starts)
            ]
            if all(isinstance(n, ast.Expr) for n in survivors):
                # Only a docstring (or nothing) would remain: give the class a ``pass`` body.
                indent = " " * node.body[0].col_offset
                anchor = node.body[0].end_lineno if survivors else node.lineno
                lines[anchor - 1] = lines[anchor - 1] + "\n" + indent + "pass"
    for start, end in sorted(spans, reverse=True):
        del lines[start - 1:end]
    out = "\n".join(lines)
    out = re.sub(r"\n{4,}", "\n\n\n", out)
    return out, removed


# ───────────────────────────── pyproject ─────────────────────────────


def _toml_str(value: str) -> str:
    """Quote a string for TOML; PEP 508 markers use single quotes, so prefer basic strings."""
    if '"' not in value and "\\" not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def transform_pyproject(text: str, cfg: dict, torch_cfg: dict, constraints: list[str],
                        old_index_name: str = "pytorch-cpu") -> str:
    lines = text.split("\n")
    out: list[str] = []
    drop_sections = set(cfg.get("drop_sections", []))
    drop_prefixes = tuple(cfg.get("drop_line_prefixes", []))
    skipping = False
    found = {"sources": 0, "index_name": 0, "index_url": 0, "constraints": 0}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[]")
            skipping = section in drop_sections
            if skipping:
                continue
        if skipping:
            continue
        if stripped.startswith(drop_prefixes):
            continue
        if f'index = "{old_index_name}"' in line:
            line = line.replace(f'index = "{old_index_name}"', f'index = "{torch_cfg["index_name"]}"')
            found["sources"] += 1
        if stripped == f'name = "{old_index_name}"':
            line = line.replace(old_index_name, torch_cfg["index_name"])
            found["index_name"] += 1
        if stripped.startswith('url = "https://download.pytorch.org/whl/'):
            line = re.sub(r'url = "[^"]+"', f'url = "{torch_cfg["index_url"]}"', line)
            found["index_url"] += 1
        if stripped.startswith("constraint-dependencies = "):
            if constraints:
                body = ",\n".join("    " + _toml_str(c) for c in constraints)
                line = "constraint-dependencies = [\n" + body + ",\n]"
            found["constraints"] += 1
        out.append(line)
    bad = {k: v for k, v in found.items() if v != 1}
    if bad:
        die(f"pyproject anchors did not match exactly once: {bad} — upstream changed pyproject.toml structure; inspect before syncing")
    # Collapse the blank line left by a dropped section header if it produced a double blank.
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text)


def constraints_from_fork(fork: Path, project_name: str, torch_version_out: list[str]) -> list[str]:
    if shutil.which("uv") is None:
        print("shadow_sync: warning: uv not found; constraint-dependencies will be left as in the fork")
        return []
    proc = subprocess.run(["uv", "export", "--frozen", "--no-hashes", "--no-emit-project",
                           "--format", "requirements-txt"], cwd=str(fork), capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"uv export failed in fork:\n{proc.stderr}")
    pins: list[str] = []
    torch_versions: set[str] = set()
    for raw in proc.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        name = re.split(r"[=<>!~ ;\[]", line, 1)[0].lower()
        if name == project_name.lower():
            continue
        if name == "torch":
            m = re.match(r"torch==([\d.]+)", line)
            if m:
                torch_versions.add(m.group(1))
            continue
        pins.append(line)
    if torch_versions:
        v = sorted(torch_versions)[-1]
        pins.append(f"torch=={v}")
        torch_version_out.append(v)
    pins.append("pillow>=12.2.0")  # keep the fork's own constraint
    return sorted(set(pins), key=str.lower)


# ───────────────────────────── transform ─────────────────────────────


def load_manifest() -> dict:
    with open(HERE / "manifest.toml", "rb") as fh:
        return tomllib.load(fh)


def transform_version() -> str:
    """Short hash of everything that defines the transform: script, manifest, stubs."""
    import hashlib

    h = hashlib.sha256()
    for p in sorted([HERE / "shadow_sync.py", HERE / "manifest.toml", *sorted((HERE / "stubs").glob("*.py"))]):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def select_files(fork: Path, ref: str, manifest: dict) -> tuple[list[str], list[str], list[str]]:
    all_files = git(["ls-tree", "-r", "--name-only", ref], fork).split("\n")
    inc = [glob_to_regex(p) for p in manifest["include"]["paths"]]
    exc = [glob_to_regex(p) for p in manifest["exclude"]["paths"]]
    included, excluded, unmapped = [], [], []
    for f in all_files:
        if matches_any(f, inc):
            included.append(f)
        elif matches_any(f, exc):
            excluded.append(f)
        else:
            unmapped.append(f)
    return included, excluded, unmapped


def build_tree(fork: Path, ref: str, manifest: dict, out: Path, *, stage_a_only: bool = False) -> dict:
    """Write the transformed tree to ``out``. Returns a report dict."""
    ren = Renamer(manifest["rename"]["from"], manifest["rename"]["to"])
    pkg_old, pkg_new = manifest["rename"]["from"], manifest["rename"]["to"]
    from_re, plain_re = seam_regexes(manifest["seams"]["network_modules"])
    replace_files = manifest["seams"].get("replace_files", {})
    inject_files = manifest["seams"].get("inject_files", {})
    included, excluded, unmapped = select_files(fork, ref, manifest)
    report = {"included": included, "excluded": excluded, "unmapped": unmapped,
              "seams": {}, "replaced": [], "injected": [], "binary": [],
              "dropped_exports": [], "pruned_tests": []}
    torch_version: list[str] = []
    constraints = [] if stage_a_only else constraints_from_fork(fork, pkg_old, torch_version)

    def _module_name(rel: str) -> str | None:
        if not (rel.startswith(pkg_old + "/") and rel.endswith(".py")):
            return None
        dotted = rel[len(pkg_old) + 1:-3].replace("/", ".")
        return dotted[:-len(".__init__")] if dotted.endswith(".__init__") else dotted

    selected_modules = {m for m in map(_module_name, included) if m}
    excluded_modules = {m for m in map(_module_name, excluded) if m} - selected_modules
    injected_modules = {m for m in map(_module_name, inject_files) if m}
    selected_modules |= injected_modules
    # Package __init__ first: its dropped exports feed the test-literal pruning below.
    for rel in sorted(included, key=lambda r: (r != f"{pkg_old}/__init__.py", r)):
        raw = git(["show", f"{ref}:{rel}"], fork, binary=True)
        dst_rel = ren.path(rel)
        dst = out / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not stage_a_only and rel in replace_files:
            stub = (HERE / replace_files[rel]).read_text(encoding="utf-8")
            dst.write_text(ren.text(stub), encoding="utf-8", newline="\n")
            report["replaced"].append(dst_rel)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            dst.write_bytes(raw)
            report["binary"].append(rel)
            continue
        text = ren.text(text)
        if not stage_a_only:
            if rel.startswith(pkg_old + "/") and rel.endswith(".py"):
                text, n = apply_seams(text, pkg_new, from_re, plain_re)
                if n:
                    report["seams"][dst_rel] = n
            if rel == f"{pkg_old}/__init__.py":
                text, dropped = prune_init_exports(text, pkg_new, selected_modules)
                report["dropped_exports"] = dropped
            if rel.startswith("tests/") and rel.endswith(".py"):
                text, pruned = prune_tests(text, pkg_new, excluded_modules,
                                           manifest["seams"]["network_modules"], dst_rel,
                                           manifest["seams"].get("prune_references", []))
                report["pruned_tests"].extend(pruned)
                text, k = prune_literal_exports(text, report["dropped_exports"])
                if k:
                    report["pruned_literals"] = report.get("pruned_literals", 0) + k
            if rel == "pyproject.toml":
                text = transform_pyproject(text, manifest["pyproject"], manifest["torch"], constraints)
        dst.write_text(text, encoding="utf-8", newline="\n")
    if not stage_a_only:
        for lab_rel, stub in inject_files.items():
            dst_rel = ren.path(lab_rel)
            dst = out / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(ren.text((HERE / stub).read_text(encoding="utf-8")), encoding="utf-8", newline="\n")
            report["injected"].append(dst_rel)
        # verify: no forbidden imports anywhere in the package
        forb = manifest["seams"]["forbidden_after"]
        forb_re = re.compile(rf"^\s*(?:from|import)\s+(?:{'|'.join(map(re.escape, forb))})(?:[\s.]|$)", re.M)
        leftovers = []
        for p in (out / pkg_new).rglob("*.py"):
            for m in forb_re.finditer(p.read_text(encoding="utf-8")):
                leftovers.append(f"{p.relative_to(out).as_posix()}: {m.group(0).strip()}")
        if leftovers:
            die("forbidden imports remain after seams:\n  " + "\n  ".join(leftovers))
        report["torch_version"] = torch_version[0] if torch_version else None
        report["constraints"] = len(constraints)
    return report


# ───────────────────────────── git plumbing in the lab ─────────────────────────────


def lab_git_init(lab: Path) -> None:
    if (lab / ".git").exists():
        return
    git(["init", "-b", "main"], lab)
    git(["config", "core.autocrlf", "false"], lab)
    print(f"shadow_sync: initialised git repository in {lab} (branch main, autocrlf=false)")


def commit_tree_to_shadow(lab: Path, tree_dir: Path, message: str) -> tuple[str | None, str | None]:
    """Commit ``tree_dir`` onto refs/heads/shadow without touching the working tree.

    Returns (new_commit, previous_commit); new_commit is None when the tree is unchanged.
    """
    git_dir = lab / ".git"
    with tempfile.TemporaryDirectory() as td:
        index = Path(td) / "index"
        env = {"GIT_INDEX_FILE": str(index)}
        git(["--git-dir", str(git_dir), "--work-tree", str(tree_dir), "add", "-A", "--", "."], tree_dir, env=env)
        tree = git(["--git-dir", str(git_dir), "write-tree"], lab, env=env)
    prev = git(["rev-parse", "-q", "--verify", "refs/heads/shadow"], lab, check=False) or None
    if prev:
        prev_tree = git(["rev-parse", f"{prev}^{{tree}}"], lab)
        if prev_tree == tree:
            return None, prev
    args = ["commit-tree", tree, "-m", message]
    if prev:
        args += ["-p", prev]
    commit = git(args, lab)
    git(["update-ref", "refs/heads/shadow", commit] + ([prev] if prev else []), lab)
    return commit, prev


def ensure_main(lab: Path, shadow_commit: str) -> bool:
    """First run: create main at the shadow commit and populate the working tree."""
    if git(["rev-parse", "-q", "--verify", "refs/heads/main"], lab, check=False):
        return False
    git(["update-ref", "refs/heads/main", shadow_commit], lab)
    git(["symbolic-ref", "HEAD", "refs/heads/main"], lab)
    git(["read-tree", "-u", "-m", shadow_commit], lab)
    return True


def changed_since(lab: Path, prev: str | None, new: str) -> list[str]:
    if not prev:
        return []
    out = git(["diff", "--name-status", prev, new], lab)
    return out.split("\n") if out else []


def main_modified_files(lab: Path, prev_shadow: str | None) -> set[str]:
    if not prev_shadow or not git(["rev-parse", "-q", "--verify", "refs/heads/main"], lab, check=False):
        return set()
    base = git(["merge-base", "refs/heads/main", prev_shadow], lab, check=False)
    if not base:
        return set()
    out = git(["diff", "--name-only", base, "refs/heads/main"], lab)
    return set(out.split("\n")) if out else set()


# ───────────────────────────── fork sync ─────────────────────────────


def sync_fork(fork: Path, remote: str, branch: str) -> None:
    remotes = git(["remote"], fork).split("\n")
    if remote not in remotes:
        die(f"fork has no remote named {remote!r}. Add it first, e.g.\n"
            f"  git -C {fork} remote add {remote} https://github.com/elder-plinius/OBLITERATUS.git")
    git(["fetch", remote, branch], fork)
    current = git(["rev-parse", "--abbrev-ref", "HEAD"], fork)
    if current == branch:
        if git(["status", "--porcelain", "--untracked-files=no"], fork):
            die(f"fork working tree has uncommitted tracked changes on {branch}; commit or stash before --sync-fork")
        git(["merge", "--ff-only", f"{remote}/{branch}"], fork)
    else:
        git(["fetch", remote, f"{branch}:{branch}"], fork)  # fast-forward the non-checked-out branch
    print(f"shadow_sync: fork {branch} fast-forwarded to {remote}/{branch} = {git(['rev-parse', '--short', branch], fork)}")


# ───────────────────────────── commands ─────────────────────────────


def cmd_self_test(fork: Path, ref: str, manifest: dict) -> int:
    ren = Renamer(manifest["rename"]["from"], manifest["rename"]["to"])
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "tree"
        rep = build_tree(fork, ref, manifest, out, stage_a_only=True)
        mismatches = []
        for rel in rep["included"]:
            original = git(["show", f"{ref}:{rel}"], fork, binary=True)
            lab_rel = ren.path(rel)
            restored_path = ren.path(lab_rel, reverse=True)
            if restored_path != rel:
                mismatches.append(f"path not involutive: {rel} -> {lab_rel} -> {restored_path}")
                continue
            data = (out / lab_rel).read_bytes()
            if rel in rep["binary"]:
                restored = data
            else:
                restored = ren.text(data.decode("utf-8"), reverse=True).encode("utf-8")
            if restored != original:
                mismatches.append(f"content not involutive: {rel}")
        print(f"self-test stage A (select+rename): {len(rep['included'])} files, {len(mismatches)} mismatches")
        for m in mismatches[:20]:
            print("  ", m)
        out_b = Path(td) / "tree_b"
        rep_b = build_tree(fork, ref, manifest, out_b)
        print(f"self-test stage B (seams+pyproject): {sum(rep_b['seams'].values())} import rewrites in {len(rep_b['seams'])} files; "
              f"replaced={rep_b['replaced']} injected={rep_b['injected']}; forbidden-import scan: clean")
        # compile every .py in the package to catch syntax damage from the rewrite
        bad = []
        for p in (out_b / manifest["rename"]["to"]).rglob("*.py"):
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except SyntaxError as exc:
                bad.append(f"{p.relative_to(out_b).as_posix()}: {exc}")
        print(f"self-test stage B pruning: dropped exports {rep_b['dropped_exports']}; pruned tests {len(rep_b['pruned_tests'])}")
        for p in (out_b / "tests").rglob("*.py"):
            try:
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            except SyntaxError as exc:
                bad.append(f"{p.relative_to(out_b).as_posix()}: {exc}")
        print(f"self-test stage B compile check: {len(bad)} syntax errors")
        for b in bad:
            print("  ", b)
        if rep["unmapped"]:
            print(f"UNMAPPED ({len(rep['unmapped'])}): decide include/exclude in manifest.toml")
            for u in rep["unmapped"]:
                print("  ", u)
        return 1 if (mismatches or bad) else 0


def print_report(rep: dict, ref_sha: str, ref_date: str, commit: str | None, prev: str | None,
                 changed: list[str], main_mod: set[str], first_run: bool, dry: bool) -> None:
    print("\n" + "=" * 72)
    print(f"shadow_sync report — fork {ref_sha[:12]} ({ref_date}){'  [DRY RUN]' if dry else ''}")
    print("=" * 72)
    print(f"selected: {len(rep['included'])} files   excluded: {len(rep['excluded'])}   unmapped: {len(rep['unmapped'])}")
    if rep["unmapped"]:
        print("UNMAPPED — new upstream files matching neither list; decide in manifest.toml:")
        for u in rep["unmapped"]:
            print("   ", u)
    print(f"seams: {sum(rep['seams'].values())} import rewrites in {len(rep['seams'])} files; "
          f"replaced {rep['replaced']}; injected {rep['injected']}")
    for f, n in sorted(rep["seams"].items()):
        print(f"    {n:2d}  {f}")
    if rep.get("dropped_exports"):
        print(f"dropped lazy exports of non-vendored modules: {rep['dropped_exports']}")
    if rep.get("pruned_tests"):
        print(f"pruned tests referencing non-vendored modules or stubbed network libs: {len(rep['pruned_tests'])}")
        for t in rep["pruned_tests"]:
            print(f"    - {t}")
    if rep.get("pruned_literals"):
        print(f"pruned parametrize literals naming dropped exports: {rep['pruned_literals']}")
    if rep.get("torch_version"):
        print(f"constraints: {rep['constraints']} pins from fork lock; torch=={rep['torch_version']} via CUDA index")
    if dry:
        print("no git writes performed")
        return
    if commit is None:
        print(f"shadow: unchanged (still {prev[:12] if prev else '?'})")
    else:
        print(f"shadow: {commit[:12]}" + (f"  (previous {prev[:12]})" if prev else "  (first shadow commit)"))
    if first_run:
        print("main: created from shadow and checked out — commit private/ and README on main when ready")
    if changed:
        prev_fork = rep.get("prev_fork_commit")
        cause = ("changed by the transform (fork unchanged)" if prev_fork == ref_sha
                 else "changed upstream since last shadow")
        print(f"{cause} ({len(changed)}):")
        for line in changed:
            status, _, path = line.partition("\t")
            flag = "  <-- also modified on main" if path in main_mod else ""
            print(f"    {status:2s} {path}{flag}")
    if commit and prev:
        print("next: review above, then  git merge shadow   (or rerun with --merge)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", type=Path, help="fork checkout (default: manifest [fork].path)")
    ap.add_argument("--lab", type=Path, default=LAB_ROOT, help="lab root (default: this repo)")
    ap.add_argument("--ref", default="main", help="fork ref to shadow (default: main)")
    ap.add_argument("--sync-fork", action="store_true", help="fast-forward the fork's branch from its upstream remote first")
    ap.add_argument("--merge", action="store_true", help="merge shadow into main after committing")
    ap.add_argument("--dry-run", action="store_true", help="build and report only; no git writes")
    ap.add_argument("--self-test", action="store_true", help="verify rename involution and seam integrity; no writes")
    args = ap.parse_args(argv)

    manifest = load_manifest()
    fork = (args.fork or Path(manifest["fork"]["path"])).resolve()
    lab = args.lab.resolve()
    if not (fork / ".git").exists():
        die(f"fork path is not a git checkout: {fork}")
    if args.sync_fork:
        sync_fork(fork, manifest["fork"]["upstream_remote"], manifest["fork"]["upstream_branch"])

    ref_sha = git(["rev-parse", args.ref], fork)
    ref_date = git(["log", "-1", "--format=%cI", ref_sha], fork)

    if args.self_test:
        return cmd_self_test(fork, ref_sha, manifest)

    with tempfile.TemporaryDirectory() as td:
        tree = Path(td) / "tree"
        rep = build_tree(fork, ref_sha, manifest, tree)
        if args.dry_run:
            print_report(rep, ref_sha, ref_date, None, None, [], set(), False, True)
            return 0
        lab_git_init(lab)
        prev_before = git(["rev-parse", "-q", "--verify", "refs/heads/shadow"], lab, check=False) or None
        main_mod = main_modified_files(lab, prev_before)
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        tv = transform_version()
        msg = (f"shadow: fork {ref_sha[:12]} ({ref_date[:10]}) — {len(rep['included'])} files, "
               f"{sum(rep['seams'].values())} seams, transform {tv}\n\n"
               f"Fork-Commit: {ref_sha}\nTransform: {tv}\nSynced: {stamp}\n")
        commit, prev = commit_tree_to_shadow(lab, tree, msg)
        if prev:
            body = git(["log", "-1", "--format=%B", prev], lab)
            m = re.search(r"^Fork-Commit: ([0-9a-f]+)", body, re.M)
            rep["prev_fork_commit"] = m.group(1) if m else None
        first_run = ensure_main(lab, commit or prev)  # type: ignore[arg-type]
        changed = changed_since(lab, prev, commit) if commit else []
        print_report(rep, ref_sha, ref_date, commit, prev, changed, main_mod, first_run, False)
        if args.merge and commit and prev and not first_run:
            if git(["status", "--porcelain", "--untracked-files=no"], lab):
                die("main has uncommitted tracked changes; commit or stash before --merge")
            cur = git(["rev-parse", "--abbrev-ref", "HEAD"], lab)
            if cur != "main":
                die(f"checked-out branch is {cur!r}, expected 'main' for --merge")
            proc = subprocess.run(["git", "merge", "--no-edit", "shadow"], cwd=str(lab))
            print("merge:", "clean" if proc.returncode == 0 else "CONFLICTS — resolve, then git commit")
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
