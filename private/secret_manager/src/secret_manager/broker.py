"""Single-argument executable broker compatible with OBLITERATUS."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

from .config import PROFILE_ENVIRONMENT_VARIABLE
from .errors import SecretManagerError, SecretUnavailable
from .manager import SecretManager


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        return 2
    try:
        manager = SecretManager.from_config()
        profile = os.environ.get(PROFILE_ENVIRONMENT_VARIABLE, "").strip() or None
        value = manager.resolve_for_broker(arguments[0], profile_name=profile)
    except SecretUnavailable:
        return 2
    except SecretManagerError:
        print("secret broker failed", file=sys.stderr)
        return 1
    try:
        sys.stdout.buffer.write(value)
        sys.stdout.buffer.flush()
    except OSError:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
