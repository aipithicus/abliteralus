"""Standalone command-line interface for portable research journals."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .errors import ResearchJournalError
from .store import SHARING_STATES, ResearchJournal


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-journal",
        description="Append, query, verify, and recover a research journal JSONL store.",
    )
    parser.add_argument("--journal", required=True, help="path to the durable journal JSONL")
    commands = parser.add_subparsers(dest="operation", required=True)

    add = commands.add_parser("add", help="append one immutable research entry")
    add.add_argument("--kind", required=True)
    add.add_argument("--title", required=True)
    body = add.add_mutually_exclusive_group()
    body.add_argument("--body", default="")
    body.add_argument("--body-file")
    add.add_argument("--actor")
    add.add_argument("--tag", action="append", default=[])
    add.add_argument("--relation", action="append", default=[], metavar="TYPE=TARGET")
    add.add_argument("--data-file", help="UTF-8 JSON object merged into the entry data field")
    add.add_argument("--sharing", choices=sorted(SHARING_STATES), default="private")

    listing = commands.add_parser("list", help="list newest entries")
    listing.add_argument("--limit", type=_positive_integer, default=20)
    listing.add_argument("--kind")
    listing.add_argument("--tag", action="append", default=[])
    listing.add_argument("--relation", action="append", default=[], metavar="TYPE=TARGET")

    show = commands.add_parser("show", help="show one entry by UUID")
    show.add_argument("entry_id")

    commands.add_parser("verify", help="verify framing, schema, hashes, and the chain")
    repair = commands.add_parser("repair", help="preview or recover an invalid suffix")
    repair.add_argument("--apply", action="store_true")
    return parser


def _relation(value: str) -> dict[str, str]:
    relation_type, separator, target = value.partition("=")
    if not separator or not relation_type.strip() or not target.strip():
        raise ValueError("relations must use TYPE=TARGET")
    return {"type": relation_type.strip(), "target": target.strip()}


def _read_body(args: argparse.Namespace) -> str:
    if args.body_file is None:
        return args.body
    if args.body_file == "-":
        return sys.stdin.read()
    return Path(args.body_file).read_text(encoding="utf-8")


def _read_data(path: str | None) -> dict[str, Any]:
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
    journal = ResearchJournal(args.journal)
    try:
        if args.operation == "add":
            if args.body_file == "-" and args.data_file == "-":
                raise ValueError("--body-file and --data-file cannot both read standard input")
            entry = journal.append(
                kind=args.kind,
                title=args.title,
                body=_read_body(args),
                actor=args.actor,
                tags=args.tag,
                relations=[_relation(value) for value in args.relation],
                data=_read_data(args.data_file),
                sharing=args.sharing,
            )
            print(json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.operation == "list":
            entries = journal.list_entries(
                limit=args.limit,
                kind=args.kind,
                tags=args.tag,
                relations=[_relation(value) for value in args.relation],
            )
            print(json.dumps(entries, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
        if args.operation == "show":
            print(
                json.dumps(
                    journal.get_entry(args.entry_id),
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.operation == "verify":
            inspection = journal.inspect()
            print(json.dumps(inspection.to_dict(), indent=2, sort_keys=True))
            return 0 if inspection.valid else 2
        if args.operation == "repair":
            result = journal.repair(apply=args.apply)
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            if result.needed and not args.apply:
                print("research-journal: dry run; pass --apply to repair", file=sys.stderr)
            return 0
    except (OSError, ResearchJournalError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"research-journal: {error}", file=sys.stderr)
        return 1
    raise AssertionError("unreachable journal command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
