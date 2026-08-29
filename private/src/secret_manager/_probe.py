"""Internal child process that reports presence metadata, never values."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence

from .config import validate_environment_name
from .errors import ConfigError


def main(argv: Sequence[str] | None = None) -> int:
    names = list(sys.argv[1:] if argv is None else argv)
    if not names:
        return 1
    try:
        validated = [validate_environment_name(name) for name in names]
    except ConfigError:
        return 1
    present = {name: bool(os.environ.get(name)) for name in validated}
    print(json.dumps(present, sort_keys=True))
    return 0 if all(present.values()) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
