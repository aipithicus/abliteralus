"""Contracts for the repository-local Codex managed-pytest hook."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
HOOK = REPOSITORY / ".codex/hooks/enforce_managed_pytest.py"
HOOKS_JSON = REPOSITORY / ".codex/hooks.json"
WINDOWS_WRAPPER = REPOSITORY / ".codex/hooks/invoke_enforce_managed_pytest.ps1"

SPEC = importlib.util.spec_from_file_location("enforce_managed_pytest", HOOK)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -q",
        ".venv/Scripts/pytest.exe -q private/tests/guard_study",
        "python -m pytest -q",
        ".venv/Scripts/python.exe -m pytest private/tests/guard_study",
        "uv run --frozen pytest -q",
        "pwsh -NoProfile -File deps/uv/run-uv.ps1 run pytest -q",
        "nu -c '^.venv/Scripts/python.exe -m pytest -q tests'",
        'pwsh -NoProfile -Command ".venv/Scripts/python.exe -m pytest -q"',
        "rg needle .; pytest -q",
        'python -c "import pytest; raise SystemExit(pytest.main())"',
    ],
)
def test_unmanaged_pytest_invocations_are_detected(command: str) -> None:
    assert POLICY.find_unmanaged_pytest(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "rg -n pytest .",
        'rg -n "python -m pytest" README.md',
        'python -c "import pytest; print(pytest.__version__)"',
        (
            "private/.venv/Scripts/lab-bench.exe --config "
            "private/config/lab.local.toml test -- private/tests/guard_study"
        ),
        "pwsh -NoProfile -File deps/uv/run-uv.ps1 run rg pytest README.md",
        "uv run python helper.py pytest",
        "git diff -- tests/test_codex_hook_policy.py",
    ],
)
def test_nonexecuting_mentions_and_managed_launcher_are_allowed(command: str) -> None:
    assert POLICY.find_unmanaged_pytest(command) is None


def _event(command: str) -> str:
    return json.dumps(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": command},
        }
    )


def test_hook_emits_codex_deny_contract_for_unmanaged_pytest() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(HOOK)],
        input=_event("python -m pytest -q"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    output = payload["hookSpecificOutput"]
    assert output["hookEventName"] == "PreToolUse"
    assert output["permissionDecision"] == "deny"
    assert "lab-bench.exe" in output["permissionDecisionReason"]


def test_hook_is_silent_for_managed_launcher() -> None:
    command = (
        "private/.venv/Scripts/lab-bench.exe --config "
        "private/config/lab.local.toml test -- private/tests/guard_study"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(HOOK)],
        input=_event(command),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_hook_configuration_is_pretooluse_for_unified_exec() -> None:
    payload = json.loads(HOOKS_JSON.read_text(encoding="utf-8"))
    group = payload["hooks"]["PreToolUse"][0]
    handler = group["hooks"][0]

    assert group["matcher"] == "^Bash$"
    assert handler["type"] == "command"
    assert "enforce_managed_pytest.py" in handler["command"]
    assert "invoke_enforce_managed_pytest.ps1" in handler["commandWindows"]
    assert handler["timeout"] <= 10


@pytest.mark.skipif(sys.platform != "win32", reason="Windows hook wrapper contract")
def test_windows_wrapper_forwards_stdin_and_denies_unmanaged_pytest() -> None:
    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-File", str(WINDOWS_WRAPPER)],
        cwd=REPOSITORY,
        input=_event("python -m pytest -q"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"
