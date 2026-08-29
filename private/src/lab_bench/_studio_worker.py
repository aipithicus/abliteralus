"""Short-lived Lightning SDK worker executed inside a secret profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any


class StudioWorkerError(RuntimeError):
    """A normalized Studio operation failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lab-bench-studio-worker")
    parser.add_argument("--teamspace", required=True)
    parser.add_argument("--studio", required=True)
    operations = parser.add_subparsers(dest="operation", required=True)
    operations.add_parser("status")

    start = operations.add_parser("start")
    start.add_argument("--machine", required=True)
    start.add_argument("--interruptible", action="store_true")
    start.add_argument("--max-runtime", type=int)

    operations.add_parser("stop")

    execute = operations.add_parser("exec")
    execute.add_argument("--remote-command", required=True)
    execute.add_argument("--forward-env", action="append", default=[])

    detach = operations.add_parser("detach")
    detach.add_argument("--remote-command", required=True)
    detach.add_argument("--forward-env", action="append", default=[])
    detach.add_argument("--wait-seconds", type=float, default=10.0)

    ports = operations.add_parser("ports")
    ports.add_argument("--add", action="append", type=int, default=[])
    return parser


def _status_name(status: object) -> str:
    return str(status).rsplit(".", 1)[-1].lower()


def _running(status: object) -> bool:
    return _status_name(status) == "running"


def _machine_name(machine: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(machine).rsplit(".", 1)[-1].lower())


def _metadata(studio: Any) -> dict[str, Any]:
    return {
        "name": studio.name,
        "teamspace": str(studio.teamspace),
        "status": _status_name(studio.status),
        "machine": str(studio.machine) if studio.machine is not None else None,
        "public_ip": studio.public_ip,
    }


def _forward_environment(studio: Any, names: Sequence[str]) -> dict[str, tuple[bool, str | None]]:
    values: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is None:
            missing.append(name)
        else:
            values[name] = value
    if missing:
        raise StudioWorkerError(
            "requested forwarded environment variables are unset: " + ", ".join(missing)
        )
    if not values:
        return {}
    current = dict(studio.env)
    previous = {name: (name in current, current.get(name)) for name in values}
    studio.set_env(values, partial=True)
    return previous


def _restore_environment(studio: Any, previous: Mapping[str, tuple[bool, str | None]]) -> None:
    restore = {
        name: value for name, (existed, value) in previous.items() if existed and value is not None
    }
    if restore:
        studio.set_env(restore, partial=True)
    for name, (existed, _value) in previous.items():
        if not existed:
            studio.delete_env(name)


def _endpoint(endpoint: Any) -> dict[str, Any]:
    return {
        "name": str(getattr(endpoint, "name", "")),
        "ports": _json_value(getattr(endpoint, "ports", [])),
        "urls": _json_value(getattr(endpoint, "urls", [])),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def execute(
    args: argparse.Namespace,
    *,
    studio_class: Any | None = None,
    machine_class: Any | None = None,
) -> tuple[dict[str, Any], int]:
    if studio_class is None or machine_class is None:
        from lightning_sdk import Machine, Studio

        studio_class = Studio
        machine_class = Machine
    create_ok = args.operation == "start"
    studio = studio_class(
        name=args.studio,
        teamspace=args.teamspace,
        create_ok=create_ok,
    )

    if args.operation == "status":
        return _metadata(studio), 0
    if args.operation == "start":
        if _running(studio.status):
            if _machine_name(studio.machine) != _machine_name(args.machine):
                raise StudioWorkerError(
                    "Studio is already running on a different machine; stop it before switching"
                )
            payload = _metadata(studio)
            payload["changed"] = False
            return payload, 0
        machine = (
            machine_class.from_str(args.machine)
            if hasattr(machine_class, "from_str")
            else args.machine
        )
        studio.start(
            machine=machine,
            interruptible=args.interruptible,
            max_runtime=args.max_runtime,
        )
        payload = _metadata(studio)
        payload["changed"] = True
        return payload, 0
    if args.operation == "stop":
        if _status_name(studio.status) not in {"running", "pending"}:
            payload = _metadata(studio)
            payload["changed"] = False
            return payload, 0
        studio.stop()
        payload = _metadata(studio)
        payload["changed"] = True
        return payload, 0
    if args.operation == "ports":
        if args.add:
            invalid = [port for port in args.add if not 1 <= port <= 65535]
            if invalid:
                raise StudioWorkerError("ports must be between 1 and 65535")
            studio.add_ports(args.add)
        return {"ports": [_endpoint(endpoint) for endpoint in studio.list_ports()]}, 0
    if args.operation in {"exec", "detach"}:
        if not _running(studio.status):
            raise StudioWorkerError("Studio must be running before remote execution")
        previous = _forward_environment(studio, args.forward_env)
        try:
            if args.operation == "exec":
                output, exit_code = studio.run_with_exit_code(args.remote_command)
            else:
                output, exit_code = studio.run_and_detach(
                    args.remote_command,
                    timeout=args.wait_seconds,
                )
        finally:
            _restore_environment(studio, previous)
        payload = _metadata(studio)
        payload["remote_output"] = output
        payload["detached"] = args.operation == "detach"
        payload["remote_exit_code"] = exit_code
        return payload, 0 if exit_code is None else int(exit_code)
    raise StudioWorkerError("unknown Studio operation")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload, return_code = execute(args)
    except Exception as error:  # provider masking remains active around this worker.
        print(
            f"lab-bench Studio operation failed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 1
    remote_output = payload.pop("remote_output", None)
    if remote_output:
        sys.stdout.write(str(remote_output))
        if not str(remote_output).endswith("\n"):
            sys.stdout.write("\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
