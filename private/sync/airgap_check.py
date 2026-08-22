#!/usr/bin/env python3
"""Prove the vendored package is airgapped: block sockets, import every module, poke sentinels.

Run from the lab root inside its environment:
    python private/sync/airgap_check.py

Exit code 0 means: every module under the package imported with sockets disabled, no import
of a forbidden network library exists anywhere in the package, and the airgap sentinels raise
DisabledError when called. Anything else prints the offending module and exits 1.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import socket
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB_ROOT = HERE.parent.parent
sys.path.insert(0, str(LAB_ROOT))

with open(HERE / "manifest.toml", "rb") as fh:
    MANIFEST = tomllib.load(fh)
PKG = MANIFEST["rename"]["to"]
FORBIDDEN = MANIFEST["seams"]["forbidden_after"]


class _SocketBlocked(RuntimeError):
    pass


def _blocked(*args, **kwargs):
    raise _SocketBlocked("network access attempted during airgap check")


def main() -> int:
    failures: list[str] = []

    # 1. static scan: no forbidden imports anywhere in the package
    forb_re = re.compile(rf"^\s*(?:from|import)\s+(?:{'|'.join(map(re.escape, FORBIDDEN))})(?:[\s.]|$)", re.M)
    for p in (LAB_ROOT / PKG).rglob("*.py"):
        for m in forb_re.finditer(p.read_text(encoding="utf-8")):
            failures.append(f"forbidden import in {p.relative_to(LAB_ROOT).as_posix()}: {m.group(0).strip()}")

    # 2. dynamic: block sockets, import every module
    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    pkg = importlib.import_module(PKG)
    imported = 0
    for info in pkgutil.walk_packages(pkg.__path__, prefix=PKG + "."):
        try:
            importlib.import_module(info.name)
            imported += 1
        except _SocketBlocked as exc:
            failures.append(f"{info.name}: {exc}")
        except Exception as exc:  # import errors are reported, not fatal to the scan
            failures.append(f"{info.name}: import failed: {type(exc).__name__}: {exc}")

    # 3. sentinels behave
    airgap = importlib.import_module(f"{PKG}._lab_airgap")
    try:
        airgap.disabled("huggingface_hub.HfApi")()
        failures.append("sentinel did not raise")
    except airgap.DisabledError:
        pass
    try:
        airgap.disabled_module("requests").post("https://example.invalid")
        failures.append("module sentinel did not raise")
    except airgap.DisabledError:
        pass
    creds = importlib.import_module(f"{PKG}.credential_sources")
    if creds.resolve_first("HF_TOKEN_DOES_NOT_EXIST_XYZ") is not None:
        failures.append("credential stub resolved a value it should not have")

    print(f"airgap check: {imported} modules imported with sockets blocked; {len(failures)} failures")
    for f in failures:
        print("  ", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
