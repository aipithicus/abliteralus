"""Command-line interface for capsule creation, validation, and use."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from .capsule import create_capsule
from .errors import ArtifactError
from .exporters import export_llama_cpp, export_peft
from .registry import ArtifactRegistry
from .rehydrate import rehydrate_capsule
from .validation import validate_capsule


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="surgery-artifact")
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create an exact capsule from two checkpoints")
    create.add_argument("--base", required=True, type=Path)
    create.add_argument("--target", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--source")
    create.add_argument("--revision")
    create.add_argument("--resolved-revision")
    create.add_argument("--recipe", type=Path)
    create.add_argument("--max-low-rank", type=_positive_integer, default=32)

    verify = commands.add_parser("verify", help="validate all capsule hashes and contracts")
    verify.add_argument("capsule", type=Path)

    inspect = commands.add_parser("inspect", help="print a capsule manifest")
    inspect.add_argument("capsule", type=Path)

    rehydrate = commands.add_parser("rehydrate", help="apply a capsule to its exact base")
    rehydrate.add_argument("--capsule", required=True, type=Path)
    rehydrate.add_argument("--base", required=True, type=Path)
    rehydrate.add_argument("--output", required=True, type=Path)

    registry = commands.add_parser("registry", help="manage the content-addressed registry")
    registry_commands = registry.add_subparsers(dest="registry_command", required=True)
    add = registry_commands.add_parser("add")
    add.add_argument("--registry", required=True, type=Path)
    add.add_argument("--capsule", required=True, type=Path)
    add.add_argument("--ref")
    resolve = registry_commands.add_parser("resolve")
    resolve.add_argument("--registry", required=True, type=Path)
    resolve.add_argument("name")

    peft = commands.add_parser("export-peft", help="derive a PEFT LoRA adapter")
    peft.add_argument("--capsule", required=True, type=Path)
    peft.add_argument("--output", required=True, type=Path)

    llama = commands.add_parser("export-llama", help="derive a llama.cpp GGUF LoRA")
    llama.add_argument("--capsule", required=True, type=Path)
    llama.add_argument("--output", required=True, type=Path)
    llama.add_argument("--converter", required=True, type=Path)
    llama.add_argument("--base", type=Path)
    llama.add_argument("--outtype", choices=("f32", "f16", "bf16", "q8_0", "auto"), default="f16")
    llama.add_argument("--timeout", type=_positive_float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            recipe = _read_json(args.recipe) if args.recipe is not None else {}
            base_identity = {
                key: value
                for key, value in {
                    "source": args.source,
                    "revision": args.revision,
                    "resolved_revision": args.resolved_revision,
                }.items()
                if value is not None
            }
            result = create_capsule(
                args.base,
                args.target,
                args.output,
                base_identity=base_identity,
                recipe=recipe,
                max_low_rank=args.max_low_rank,
            )
            print(json.dumps(_capsule_result(result), indent=2, sort_keys=True))
            return 0
        if args.command == "verify":
            print(json.dumps(_validation_result(validate_capsule(args.capsule)), indent=2, sort_keys=True))
            return 0
        if args.command == "inspect":
            validation = validate_capsule(args.capsule)
            print((validation.path / "manifest.json").read_text(encoding="utf-8"), end="")
            return 0
        if args.command == "rehydrate":
            print(rehydrate_capsule(args.capsule, args.base, args.output))
            return 0
        if args.command == "registry":
            registry = ArtifactRegistry(args.registry)
            if args.registry_command == "add":
                entry = registry.add(args.capsule, ref=args.ref)
                print(
                    json.dumps(
                        {
                            "surgery_id": entry.surgery_id,
                            "path": str(entry.path),
                            "ref": entry.ref,
                            "created": entry.created,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            print(registry.resolve(args.name))
            return 0
        if args.command == "export-peft":
            print(export_peft(args.capsule, args.output))
            return 0
        if args.command == "export-llama":
            print(
                export_llama_cpp(
                    args.capsule,
                    args.output,
                    converter=args.converter,
                    base_checkpoint=args.base,
                    outtype=args.outtype,
                    timeout_seconds=args.timeout,
                )
            )
            return 0
    except (ArtifactError, OSError, ValueError) as error:
        print(f"surgery-artifact: {error}", file=sys.stderr)
        return 1
    raise AssertionError("unreachable command dispatch")


def _capsule_result(result: object) -> dict[str, object]:
    return {
        "path": str(result.path),
        "surgery_id": result.surgery_id,
        "operation_count": result.operation_count,
        "changed_tensor_count": result.changed_tensor_count,
        "unchanged_tensor_count": result.unchanged_tensor_count,
        "payload_bytes": result.payload_bytes,
        "total_bytes": result.total_bytes,
        "codecs": result.codecs,
    }


def _validation_result(result: object) -> dict[str, object]:
    return {
        "path": str(result.path),
        "surgery_id": result.surgery_id,
        "operation_count": result.operation_count,
        "file_count": result.file_count,
        "total_bytes": result.total_bytes,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recipe must contain a JSON object")
    return value


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
