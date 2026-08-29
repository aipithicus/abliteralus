"""Command-line surface for the private ABLITERALUS lab bench."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from abliteralus.storage import RECLAIMABLE_CATEGORIES
from secret_manager.errors import SecretManagerError
from surgery_artifacts.errors import ArtifactError

from research_journal import SHARING_STATES, ResearchJournalError

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
        "lightning-surgery", help="plan or run through the private Lightning controller"
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

    storage = commands.add_parser("storage", help="measure storage creep and reclaim safe state")
    storage_commands = storage.add_subparsers(dest="storage_operation", required=True)
    storage_report = storage_commands.add_parser("report")
    storage_report.add_argument("--record", action="store_true")
    storage_report.add_argument("--json", action="store_true")
    storage_history = storage_commands.add_parser("history")
    storage_history.add_argument("--limit", type=_positive_integer, default=20)
    storage_history.add_argument("--json", action="store_true")
    storage_clean = storage_commands.add_parser("clean")
    storage_clean.add_argument(
        "--category",
        action="append",
        choices=sorted(RECLAIMABLE_CATEGORIES),
        required=True,
    )
    storage_clean.add_argument("--older-than-days", type=_positive_float, default=7.0)
    storage_clean.add_argument("--apply", action="store_true")
    storage_clean.add_argument("--json", action="store_true")

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

    journal = commands.add_parser(
        "journal", help="append, query, verify, and recover the private research journal"
    )
    journal_commands = journal.add_subparsers(dest="journal_operation", required=True)
    journal_add = journal_commands.add_parser("add", help="append one immutable research entry")
    journal_add.add_argument("--kind", required=True)
    journal_add.add_argument("--title", required=True)
    journal_body = journal_add.add_mutually_exclusive_group()
    journal_body.add_argument("--body", default="")
    journal_body.add_argument("--body-file")
    journal_add.add_argument("--actor")
    journal_add.add_argument("--tag", action="append", default=[])
    _add_journal_relation_arguments(journal_add)
    journal_add.add_argument("--data-file", help="UTF-8 JSON object for structured results")
    journal_add.add_argument("--sharing", choices=sorted(SHARING_STATES), default="private")

    journal_list = journal_commands.add_parser("list", help="list newest matching entries")
    journal_list.add_argument("--limit", type=_positive_integer, default=20)
    journal_list.add_argument("--kind")
    journal_list.add_argument("--tag", action="append", default=[])
    _add_journal_relation_arguments(journal_list)

    journal_show = journal_commands.add_parser("show", help="show one entry by UUID")
    journal_show.add_argument("entry_id")
    journal_commands.add_parser("verify", help="verify framing, schema, hashes, and the chain")
    journal_repair = journal_commands.add_parser(
        "repair", help="preview or recover a corrupt suffix"
    )
    journal_repair.add_argument("--apply", action="store_true")

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
    print(f"research journal: {payload['journal']['path']}")


def _format_bytes(value: int, *, signed: bool = False) -> str:
    sign = ""
    amount = float(value)
    if value < 0:
        sign = "-"
        amount = -amount
    elif signed and value > 0:
        sign = "+"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if amount < 1024 or candidate == units[-1]:
            break
        amount /= 1024
    precision = 0 if unit == "B" else 1
    return f"{sign}{amount:.{precision}f} {unit}"


def _print_storage_report(payload: dict[str, object]) -> None:
    print(f"measured: {payload['measured_at']}")
    print(f"{'category':24} {'logical size':>14} {'files':>10}  policy")
    for category in payload["categories"]:
        status = category["reclaim_policy"]
        if category["scan_errors"]:
            samples = ", ".join(category["scan_error_paths"])
            status = f"INCOMPLETE ({category['scan_errors']} errors: {samples}); {status}"
        print(
            f"{category['name']:24} "
            f"{_format_bytes(category['bytes']):>14} "
            f"{category['files']:>10}  {status}"
        )
    print(f"{'total':24} {_format_bytes(payload['total_bytes']):>14} {payload['total_files']:>10}")
    print(f"volume free: {_format_bytes(payload['volume']['free_bytes'])}")
    if "snapshot" in payload:
        print(f"snapshot: {payload['snapshot']}")


def _print_storage_history(rows: list[dict[str, object]]) -> None:
    if not rows:
        print("no storage snapshots")
        return
    print(f"{'measured':28} {'managed':>14} {'change':>14} {'volume free':>14}")
    for row in rows:
        delta = row["delta_bytes"]
        formatted_delta = "n/a" if delta is None else _format_bytes(delta, signed=True)
        print(
            f"{row['measured_at']:28} "
            f"{_format_bytes(row['total_bytes']):>14} "
            f"{formatted_delta:>14} "
            f"{_format_bytes(row['volume_free_bytes']):>14}"
        )


def _print_reclaim_candidates(rows: list[dict[str, object]], *, applied: bool) -> None:
    if not rows:
        print("no reclaim candidates")
        return
    verb = "reclaimed" if applied else "candidate"
    print(f"{'category':20} {'logical size':>14}  {verb}")
    for row in rows:
        print(f"{row['category']:20} {_format_bytes(row['bytes']):>14}  {row['path']}")
    print(f"{'total':20} {_format_bytes(sum(row['bytes'] for row in rows)):>14}")


def _add_journal_relation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--relation", action="append", default=[], metavar="TYPE=TARGET")


def _parse_journal_relation(value: str) -> dict[str, str]:
    relation_type, separator, target = value.partition("=")
    if not separator or not relation_type.strip() or not target.strip():
        raise ValueError("journal relations must use TYPE=TARGET")
    return {"type": relation_type.strip(), "target": target.strip()}


def _journal_relations(args: argparse.Namespace) -> list[dict[str, str]]:
    relations = [_parse_journal_relation(value) for value in args.relation]
    for relation_type, attribute in (
        ("run", "run_id"),
        ("model", "model"),
        ("artifact", "artifact"),
    ):
        relations.extend(
            {"type": relation_type, "target": target} for target in getattr(args, attribute)
        )
    return relations


def _read_journal_body(args: argparse.Namespace) -> str:
    if args.body_file is None:
        return args.body
    if args.body_file == "-":
        return sys.stdin.read()
    return Path(args.body_file).read_text(encoding="utf-8")


def _read_journal_data(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path == "-":
        value = json.load(sys.stdin)
    else:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("--data-file must contain one JSON object")
    return value


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
        if args.command_name == "storage":
            if args.storage_operation == "report":
                payload = bench.storage_report(record=args.record)
                if args.json:
                    print(json.dumps(payload, indent=2, sort_keys=True))
                else:
                    _print_storage_report(payload)
                return 0
            if args.storage_operation == "history":
                history = bench.storage_history(limit=args.limit)
                if args.json:
                    print(json.dumps(history, indent=2, sort_keys=True))
                else:
                    _print_storage_history(history)
                return 0
            candidates = bench.storage_clean(
                categories=args.category,
                older_than_days=args.older_than_days,
                apply=args.apply,
            )
            if args.json:
                print(json.dumps(candidates, indent=2, sort_keys=True))
            else:
                _print_reclaim_candidates(candidates, applied=args.apply)
            if candidates and not args.apply:
                print("lab-bench: dry run; pass --apply to reclaim", file=sys.stderr)
            return 0
        if args.command_name == "with-secrets":
            return bench.with_secrets(args.profile, args.child_command)
        if args.command_name == "ssh":
            return bench.ssh(args.remote_command, destination=args.destination)
        if args.command_name == "journal":
            if args.journal_operation == "add":
                if args.body_file == "-" and args.data_file == "-":
                    raise ValueError("--body-file and --data-file cannot both read standard input")
                entry = bench.journal_add(
                    kind=args.kind,
                    title=args.title,
                    body=_read_journal_body(args),
                    actor=args.actor,
                    tags=args.tag,
                    relations=_journal_relations(args),
                    data=_read_journal_data(args.data_file),
                    sharing=args.sharing,
                )
                print(json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False))
                return 0
            if args.journal_operation == "list":
                entries = bench.journal_list(
                    limit=args.limit,
                    kind=args.kind,
                    tags=args.tag,
                    relations=_journal_relations(args),
                )
                print(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False))
                return 0
            if args.journal_operation == "show":
                print(
                    json.dumps(
                        bench.journal_show(args.entry_id),
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
                return 0
            if args.journal_operation == "verify":
                inspection = bench.journal_verify()
                print(json.dumps(inspection, indent=2, sort_keys=True))
                return 0 if inspection["valid"] else 2
            if args.journal_operation == "repair":
                result = bench.journal_repair(apply=args.apply)
                print(json.dumps(result, indent=2, sort_keys=True))
                if result["needed"] and not args.apply:
                    print("lab-bench: dry run; pass --apply to repair", file=sys.stderr)
                return 0
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
        ResearchJournalError,
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
