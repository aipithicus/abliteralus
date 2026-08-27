#!/usr/bin/env python3
"""Restore the repository-pinned uv payload without using an ambient uv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from abliteralus.toolchain import UvToolchainError, restore_uv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="redeploy the pinned payload even when the current files verify",
    )
    parser.add_argument(
        "--platform",
        help="manifest platform key; defaults to the current operating system",
    )
    args = parser.parse_args(argv)
    try:
        executable = restore_uv(
            REPOSITORY,
            force=args.force,
            platform_key=args.platform,
        )
    except UvToolchainError as error:
        print(f"restore-uv: {error}", file=sys.stderr)
        return 1
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
