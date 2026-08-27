"""Command-line surface for the private ABLITERALUS lab bench."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from secret_manager.errors import SecretManagerError
from surgery_artifacts.errors import ArtifactError

from . import __version__
from .bench import LabBench
from .config import load_config
from .errors import LabBenchError


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab-bench",
        description="Control local and Lightning ABLITERALUS experiment sessions.",
    )
    parser.add_argument("--config", help="non-secret lab.local.toml path")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command_name", required=True)

    show = commands.add_parser("show", help="show non-secret bench configuration")
    show.add_argument("--json", action="store_true")

    local_surgery = commands.add_parser("local-surgery", help="invoke abliteralus.surgery_bench")
    local_surgery.add_argument(
        "--anonymous-hub",
        action="store_true",
        help="run without Proton Pass or implicit Hugging Face credentials",
    )
    local_surgery.add_argument(
        "--keep-workdir",
        action="store_true",
        help="retain the disposable run workspace for debugging",
    )
    local_surgery.add_argument("arguments", nargs=argparse.REMAINDER)

    lightning_surgery = commands.add_parser(
        "lightning-surgery", help="plan or run through abliteralus.lightning_surgery"
    )
    lightning_surgery.add_argument(
        "--keep-workdir",
        action="store_true",
        help="retain the local controller workspace for debugging",
    )
    lightning_surgery.add_argument("arguments", nargs=argparse.REMAINDER)

    local_inference = commands.add_parser(
        "local-inference", help="run a local inference command with the read-only Hub profile"
    )
    local_inference.add_argument("--run-id")
    local_inference.add_argument("--keep-workdir", action="store_true")
    local_inference.add_argument("child_command", nargs=argparse.REMAINDER)

    tests = commands.add_parser("test", help="run pytest in a unique, controller-owned workspace")
    tests.add_argument("--run-id")
    tests.add_argument("--keep-workdir", action="store_true")
    tests.add_argument(
        "--cwd",
        default=".",
        help="repository-relative suite working directory",
    )
    tests.add_argument("arguments", nargs=argparse.REMAINDER)

    runs = commands.add_parser("runs", help="inspect or clean disposable run workspaces")
    run_commands = runs.add_subparsers(dest="runs_operation", required=True)
    run_commands.add_parser("list")
    clean = run_commands.add_parser("clean")
    clean.add_argument("--older-than-hours", type=_positive_float, default=24.0)
    clean.add_argument(
        "--apply",
        action="store_true",
        help="delete the displayed stale workspaces; omission is a dry run",
    )

    with_secrets = commands.add_parser(
        "with-secrets", help="run any local command with a named non-privileged profile"
    )
    with_secrets.add_argument("profile")
    with_secrets.add_argument("child_command", nargs=argparse.REMAINDER)

    studio = commands.add_parser("studio", help="Lightning Studio lifecycle and execution")
    studio_commands = studio.add_subparsers(dest="studio_operation", required=True)
    studio_commands.add_parser("status")

    provision = studio_commands.add_parser(
        "provision", help="materialize the current lock-addressed runtime"
    )
    provision.add_argument("--experiment-config", required=True)
    provision.add_argument("--machine")
    provision.add_argument("--interruptible", action="store_true")
    provision.add_argument("--max-runtime", type=_positive_integer)
    provision.add_argument("--skip-gguf", action="store_true")
    provision.add_argument("--local-output")
    provision.add_argument("--keep-running", action="store_true")
    provision.add_argument("--reuse-running", action="store_true")
    provision.add_argument("--dry-run", action="store_true")

    doctor = studio_commands.add_parser(
        "doctor", help="verify a provisioned runtime on a running Studio"
    )
    doctor.add_argument("--experiment-config", required=True)
    doctor.add_argument("--skip-gguf", action="store_true")
    doctor.add_argument("--local-output")

    start = studio_commands.add_parser("start")
    start.add_argument(
        "--purpose",
        choices=("control", "surgery", "inference"),
        default="control",
    )
    start.add_argument("--machine")
    start.add_argument("--interruptible", action="store_true")
    start.add_argument("--max-runtime", type=_positive_integer)

    studio_commands.add_parser("stop")

    execute = studio_commands.add_parser("exec", help="run one quoted remote command")
    execute.add_argument("remote_command", nargs=argparse.REMAINDER)

    detach = studio_commands.add_parser(
        "detach", help="start a long-running remote command and return"
    )
    detach.add_argument("--wait-seconds", type=_positive_float, default=10.0)
    detach.add_argument("remote_command", nargs=argparse.REMAINDER)

    ports = studio_commands.add_parser("ports", help="list or add Studio endpoints")
    ports.add_argument("--add", action="append", type=_port, default=[])

    ssh = commands.add_parser("ssh", help="open direct OpenSSH using the existing OS key or agent")
    ssh.add_argument("--destination")
    ssh.add_argument("remote_command", nargs=argparse.REMAINDER)

    artifact = commands.add_parser("artifact", help="verify and manage durable surgery capsules")
    artifact_commands = artifact.add_subparsers(dest="artifact_operation", required=True)
    verify = artifact_commands.add_parser("verify")
    verify.add_argument("capsule")
    register = artifact_commands.add_parser("register")
    register.add_argument("capsule")
    register.add_argument("--ref")
    resolve = artifact_commands.add_parser("resolve")
    resolve.add_argument("name")
    rehydrate = artifact_commands.add_parser("rehydrate")
    rehydrate.add_argument("capsule_or_ref")
    rehydrate.add_argument("--base", required=True)
    rehydrate.add_argument("--output", required=True)
    return parser


def _print_inventory(payload: dict) -> None:
    print(f"config: {payload['config']}")
    print(f"repository: {payload['repository']}")
    print(f"python: {payload['python']}")
    print(f"secret config: {payload['secret_config']}")
    lightning = payload["lightning"]
    print(f"Lightning: {lightning['teamspace']} / {lightning['studio']}")
    print(
        "machines: "
        f"control={lightning['control_machine']}, "
        f"surgery={lightning['surgery_machine']}, "
        f"inference={lightning['inference_machine']}"
    )
    print(f"remote root: {lightning['remote_root']}")
    print(
        "allocation: "
        f"timeout={lightning['allocation_timeout_seconds']}s, "
        f"retry={lightning['allocation_retry_seconds']}s, "
        f"pending={lightning['pending_policy']}"
    )
    print(
        "remote leases (seconds): "
        f"control={lightning['control_max_runtime_seconds'] or 'unset'}, "
        f"surgery={lightning['surgery_max_runtime_seconds'] or 'unset'}, "
        f"inference={lightning['inference_max_runtime_seconds'] or 'unset'}"
    )
    ssh = payload["ssh"]
    print(f"SSH: {ssh['destination'] or 'not configured'}")
    print(f"artifact registry: {payload['artifacts']['registry']}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bench = LabBench(load_config(args.config))
        if args.command_name == "show":
            payload = bench.inventory()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                _print_inventory(payload)
            return 0
        if args.command_name == "local-surgery":
            return bench.local_surgery(
                args.arguments,
                anonymous_hub=args.anonymous_hub,
                keep_workdir=args.keep_workdir,
            )
        if args.command_name == "lightning-surgery":
            return bench.lightning_surgery(
                args.arguments,
                keep_workdir=args.keep_workdir,
            )
        if args.command_name == "local-inference":
            return bench.local_inference(
                args.child_command,
                run_id=args.run_id,
                keep_workdir=args.keep_workdir,
            )
        if args.command_name == "test":
            return bench.test(
                args.arguments,
                run_id=args.run_id,
                keep_workdir=args.keep_workdir,
                working_directory=args.cwd,
            )
        if args.command_name == "runs":
            if args.runs_operation == "list":
                print(json.dumps(bench.run_workspaces(), indent=2, sort_keys=True))
                return 0
            records = bench.clean_run_workspaces(
                older_than_hours=args.older_than_hours,
                apply=args.apply,
            )
            print(json.dumps(records, indent=2, sort_keys=True))
            if records and not args.apply:
                print("lab-bench: dry run; pass --apply to delete", file=sys.stderr)
            return 0
        if args.command_name == "with-secrets":
            return bench.with_secrets(args.profile, args.child_command)
        if args.command_name == "ssh":
            return bench.ssh(args.remote_command, destination=args.destination)
        if args.command_name == "artifact":
            if args.artifact_operation == "verify":
                print(json.dumps(bench.artifact_verify(args.capsule), indent=2, sort_keys=True))
                return 0
            if args.artifact_operation == "register":
                print(
                    json.dumps(
                        bench.artifact_register(args.capsule, ref=args.ref),
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            if args.artifact_operation == "resolve":
                print(bench.artifact_resolve(args.name))
                return 0
            if args.artifact_operation == "rehydrate":
                print(
                    bench.artifact_rehydrate(
                        args.capsule_or_ref,
                        base=args.base,
                        output=args.output,
                    )
                )
                return 0
        if args.command_name == "studio":
            if args.studio_operation == "status":
                return bench.studio_status()
            if args.studio_operation == "provision":
                return bench.studio_provision(
                    experiment_config=args.experiment_config,
                    machine=args.machine,
                    interruptible=args.interruptible,
                    max_runtime=args.max_runtime,
                    skip_gguf=args.skip_gguf,
                    local_output=args.local_output,
                    keep_running=args.keep_running,
                    reuse_running=args.reuse_running,
                    dry_run=args.dry_run,
                )
            if args.studio_operation == "doctor":
                return bench.studio_doctor(
                    experiment_config=args.experiment_config,
                    skip_gguf=args.skip_gguf,
                    local_output=args.local_output,
                )
            if args.studio_operation == "start":
                return bench.studio_start(
                    purpose=args.purpose,
                    machine=args.machine,
                    interruptible=args.interruptible,
                    max_runtime=args.max_runtime,
                )
            if args.studio_operation == "stop":
                return bench.studio_stop()
            if args.studio_operation == "exec":
                return bench.studio_exec(
                    args.remote_command,
                    detached=False,
                    wait_seconds=10.0,
                )
            if args.studio_operation == "detach":
                return bench.studio_exec(
                    args.remote_command,
                    detached=True,
                    wait_seconds=args.wait_seconds,
                )
            if args.studio_operation == "ports":
                return bench.studio_ports(args.add)
    except (
        ArtifactError,
        LabBenchError,
        OSError,
        SecretManagerError,
        ValueError,
    ) as error:
        print(f"lab-bench: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    raise AssertionError("unreachable command dispatch")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
