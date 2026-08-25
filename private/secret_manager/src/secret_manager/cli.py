"""Human and automation-friendly secret-manager CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .errors import SecretManagerError
from .manager import SecretManager


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="secret-manager",
        description="Run child processes with ephemeral Proton Pass secret profiles.",
    )
    parser.add_argument("--config", help="non-secret TOML mapping file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command_name", required=True)

    inventory = commands.add_parser("inventory", help="show configured metadata")
    inventory.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    inventory.add_argument(
        "--show-references",
        action="store_true",
        help="include pass:// locators (never values)",
    )

    check = commands.add_parser("check", help="resolve a profile in a boolean-only probe")
    check.add_argument("profile")
    _privilege_arguments(check)
    check.add_argument("--json", action="store_true")

    run = commands.add_parser("run", help="run a command through pass-cli run")
    run.add_argument("profile")
    _privilege_arguments(run)
    run.add_argument("child_command", nargs="+", metavar="CHILD_COMMAND")
    return parser


def _privilege_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-privileged",
        action="store_true",
        help="explicitly unlock a profile containing privileged credentials",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed privileged-profile confirmation",
    )


def _confirm_privileged(
    manager: SecretManager,
    profile: str,
    *,
    allow_privileged: bool,
    assume_yes: bool,
) -> None:
    if assume_yes and not allow_privileged:
        raise SecretManagerError("--yes requires --allow-privileged")
    if not manager.profile_is_privileged(profile):
        return
    if not allow_privileged:
        raise SecretManagerError(
            f"profile {profile} is privileged; pass --allow-privileged to continue"
        )
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise SecretManagerError("privileged non-interactive use requires --allow-privileged --yes")
    print(
        f"Privileged secret profile requested. Type {profile!r} to continue:",
        file=sys.stderr,
    )
    if sys.stdin.readline().rstrip("\r\n") != profile:
        raise SecretManagerError("privileged profile confirmation did not match")


def _child_arguments(arguments: Sequence[str]) -> list[str]:
    child = list(arguments)
    if child and child[0] == "--":
        child.pop(0)
    if not child:
        raise SecretManagerError("run requires a child command after --")
    return child


def _print_inventory(payload: dict) -> None:
    print(f"config: {payload['config']}")
    print("providers:")
    for provider in payload["providers"]:
        print(f"  {provider['name']}: {provider['type']} ({provider['executable']})")
    print("credentials:")
    for credential in payload["credentials"]:
        privilege = "privileged" if credential["privileged"] else "standard"
        reference = f" -> {credential['reference']}" if "reference" in credential else ""
        print(f"  {credential['name']}: {privilege}{reference}")
    print("profiles:")
    for profile in payload["profiles"]:
        flags = []
        if profile["privileged"]:
            flags.append("privileged")
        if profile["broker_enabled"]:
            flags.append("broker")
        suffix = f" [{' '.join(flags)}]" if flags else ""
        print(f"  {profile['name']}{suffix}: {', '.join(profile['environment'])}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manager = SecretManager.from_config(args.config)
        if args.command_name == "inventory":
            payload = manager.inventory(include_references=args.show_references)
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_inventory(payload)
            return 0
        if args.command_name == "check":
            _confirm_privileged(
                manager,
                args.profile,
                allow_privileged=args.allow_privileged,
                assume_yes=args.yes,
            )
            result = manager.check(
                args.profile,
                allow_privileged=args.allow_privileged,
            )
            if args.json:
                print(json.dumps(result, sort_keys=True))
            else:
                print(f"{args.profile}: available ({', '.join(result)})")
            return 0
        if args.command_name == "run":
            _confirm_privileged(
                manager,
                args.profile,
                allow_privileged=args.allow_privileged,
                assume_yes=args.yes,
            )
            completed = manager.run(
                args.profile,
                _child_arguments(args.child_command),
                allow_privileged=args.allow_privileged,
            )
            return completed.returncode
    except SecretManagerError as error:
        print(f"secret-manager: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
