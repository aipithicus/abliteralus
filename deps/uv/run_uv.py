#!/usr/bin/env python3
"""Run the verified repository uv with repository-local mutable state."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from abliteralus.toolchain import (  # noqa: E402
    UvToolchainError,
    repository_tool_environment,
    resolve_uv,
)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    working_directory = Path.cwd().resolve()
    try:
        working_directory.relative_to(REPOSITORY)
    except ValueError:
        print(
            f"run-uv: working directory must remain inside {REPOSITORY}: {working_directory}",
            file=sys.stderr,
        )
        return 2

    try:
        executable = resolve_uv(REPOSITORY)
        environment = repository_tool_environment(REPOSITORY)
        result = subprocess.run(
            [str(executable), *arguments],
            cwd=working_directory,
            env=environment,
            check=False,
        )
    except UvToolchainError as error:
        print(f"run-uv: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"run-uv: could not execute repository uv: {error}", file=sys.stderr)
        return 126
    except KeyboardInterrupt:
        return 130
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
