"""Block unmanaged pytest execution requested through a Codex shell tool.

Codex sends one PreToolUse event as JSON on stdin. This hook is intentionally
limited to command classification; the existing lab-bench launcher remains the
owner of isolated run directories, temp variables, coverage state, and cleanup.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass


MANAGED_TEST_COMMAND = (
    "private/.venv/Scripts/lab-bench.exe --config "
    "private/lab_bench/lab.local.toml test -- <pytest arguments>"
)
BLOCK_REASON = (
    "Unmanaged pytest execution is prohibited in this repository because bare pytest "
    "inherits ambient Windows temp paths. Use the repository-managed launcher: "
    f"{MANAGED_TEST_COMMAND}"
)

_PYTHON_NAME = re.compile(r"^python(?:\d+(?:\.\d+)*)?(?:\.exe)?$", re.IGNORECASE)
_PYTEST_NAMES = {"pytest", "pytest.exe", "py.test", "py.test.exe"}
_SHELL_NAMES = {
    "bash",
    "bash.exe",
    "cmd",
    "cmd.exe",
    "nu",
    "nu.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "sh.exe",
    "zsh",
    "zsh.exe",
}
_SHELL_COMMAND_FLAGS = {"-c", "--command", "-command", "/c"}
_INVOCATION_PREFIXES = {"&", "call", "command", "exec", "sudo", "^"}
_RUNNER_OPTIONS_WITH_VALUE = {
    "--directory",
    "--extra",
    "--group",
    "--package",
    "--project",
    "--python",
    "--with",
}
_ASSIGNMENT = re.compile(r"^(?:\$env:)?[A-Za-z_][A-Za-z0-9_]*=", re.IGNORECASE)
_PYTEST_MAIN = re.compile(r"\bpytest\s*\.\s*main\s*\(", re.IGNORECASE)


@dataclass(frozen=True)
class Violation:
    invocation: str


def _basename(token: str) -> str:
    cleaned = token.strip().strip("'\"`").lstrip("^@")
    return cleaned.replace("\\", "/").rsplit("/", 1)[-1].lower()


def _split_segments(command: str) -> list[str]:
    """Split common shell command separators while preserving quoted commands."""

    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        character = command[index]
        if escaped:
            current.append(character)
            escaped = False
            index += 1
            continue
        if character in {"\\", "`"} and quote != "'":
            current.append(character)
            escaped = True
            index += 1
            continue
        if quote is not None:
            current.append(character)
            if character == quote:
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
            current.append(character)
            index += 1
            continue
        if character in {";", "\n", "\r", "|", "&"}:
            value = "".join(current).strip()
            if value:
                segments.append(value)
            current = []
            if character in {"|", "&"} and index + 1 < len(command):
                if command[index + 1] == character:
                    index += 1
            index += 1
            continue
        current.append(character)
        index += 1
    value = "".join(current).strip()
    if value:
        segments.append(value)
    return segments


def _tokenize(segment: str) -> list[str]:
    """Tokenize the shell subset needed to recognize process invocations."""

    tokens: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False
    for character in segment:
        if escaped:
            current.append(character)
            escaped = False
            continue
        if character in {"\\", "`"} and quote != "'":
            if character == "\\":
                current.append(character)
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            else:
                current.append(character)
            continue
        if character in {"'", '"'}:
            quote = character
            continue
        if character.isspace():
            if current:
                tokens.append("".join(current))
                current = []
            continue
        current.append(character)
    if current:
        tokens.append("".join(current))
    return tokens


def _drop_invocation_prefixes(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    while remaining:
        name = _basename(remaining[0])
        if remaining[0] in _INVOCATION_PREFIXES or name in _INVOCATION_PREFIXES:
            remaining.pop(0)
            continue
        if _ASSIGNMENT.match(remaining[0]):
            remaining.pop(0)
            continue
        if name == "env":
            remaining.pop(0)
            while remaining and (remaining[0].startswith("-") or _ASSIGNMENT.match(remaining[0])):
                remaining.pop(0)
            continue
        break
    return remaining


def _python_violation(tokens: list[str]) -> Violation | None:
    normalized = [_basename(token) for token in tokens]
    for index, token in enumerate(normalized[:-1]):
        if token == "-m" and normalized[index + 1] in _PYTEST_NAMES:
            return Violation("python -m pytest")
    for index, token in enumerate(normalized[:-1]):
        if token in {"-c", "-command"} and _PYTEST_MAIN.search(tokens[index + 1]):
            return Violation("python -c pytest.main(...)")
    return None


def _nested_shell_violation(tokens: list[str]) -> Violation | None:
    for index, token in enumerate(tokens[:-1]):
        if token.lower() in _SHELL_COMMAND_FLAGS:
            violation = find_unmanaged_pytest(tokens[index + 1])
            if violation is not None:
                return violation

    normalized = [_basename(token) for token in tokens]
    for index, token in enumerate(normalized):
        if token == "run-uv.ps1" and index + 1 < len(tokens):
            tail = tokens[index + 1 :]
            if "run" in [_basename(value) for value in tail]:
                violation = _runner_tail_violation(tail)
                if violation is not None:
                    return violation
    return None


def _runner_tail_violation(tokens: Iterable[str]) -> Violation | None:
    values = list(tokens)
    normalized = [_basename(token) for token in values]
    if "run" in normalized:
        values = values[normalized.index("run") + 1 :]

    while values:
        token = values[0]
        if token == "--":
            values.pop(0)
            break
        if not token.startswith("-"):
            break
        option = token.split("=", 1)[0].lower()
        values.pop(0)
        if "=" not in token and option in _RUNNER_OPTIONS_WITH_VALUE and values:
            values.pop(0)

    if not values:
        return None
    executable = _basename(values[0])
    if executable in _PYTEST_NAMES:
        return Violation("runner pytest")
    if _PYTHON_NAME.match(executable) or executable in {"py", "py.exe"}:
        return _python_violation(values[1:])
    if executable in _SHELL_NAMES:
        return _nested_shell_violation(values[1:])
    return None


def _segment_violation(segment: str) -> Violation | None:
    tokens = _drop_invocation_prefixes(_tokenize(segment))
    if not tokens:
        return None

    executable = _basename(tokens[0])
    if executable in _PYTEST_NAMES:
        return Violation(executable)
    if _PYTHON_NAME.match(executable) or executable in {"py", "py.exe"}:
        return _python_violation(tokens[1:])
    if executable in {"uv", "uv.exe", "uvx", "uvx.exe"}:
        return _runner_tail_violation(tokens[1:])
    if executable == "run-uv.ps1":
        return _runner_tail_violation(tokens[1:])
    if executable in _SHELL_NAMES:
        return _nested_shell_violation(tokens[1:])
    return None


def find_unmanaged_pytest(command: str) -> Violation | None:
    for segment in _split_segments(command):
        violation = _segment_violation(segment)
        if violation is not None:
            return violation
    return None


def _deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )
    sys.stdout.write("\n")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError) as error:
        print(f"Managed-pytest hook received invalid JSON: {error}", file=sys.stderr)
        return 2

    if payload.get("hook_event_name") != "PreToolUse" or payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str):
        print("Managed-pytest hook received no Bash command", file=sys.stderr)
        return 2

    violation = find_unmanaged_pytest(command)
    if violation is None:
        return 0
    _deny(f"{BLOCK_REASON} Detected: {violation.invocation}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
