"""Command line entry point for guard-study contracts and runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .contracts import compute_dataset_digest, load_dataset_contract, load_study_spec
from .errors import ContractError, StudyRuntimeError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guard-study",
        description="Run reversible decision-geometry studies without saving model weights.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    digest = subparsers.add_parser(
        "digest",
        help="compute the content digest for a dataset draft",
    )
    digest.add_argument("dataset", type=Path)

    validate = subparsers.add_parser(
        "validate",
        help="strictly validate a dataset and, optionally, a study spec",
    )
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--study", type=Path)

    run = subparsers.add_parser(
        "run",
        help="run causal mapping, dev steering sweep, and locked winner evaluation",
    )
    run.add_argument("study", type=Path)
    run.add_argument(
        "--offline",
        action="store_true",
        help="refuse Hub access and require the pinned checkpoint to be cached",
    )
    run.add_argument("--run-id", help="explicit immutable run identifier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "digest":
            print(compute_dataset_digest(args.dataset))
            return 0
        if args.command == "validate":
            dataset = load_dataset_contract(args.dataset)
            result = {"dataset": dataset.summary()}
            if args.study is not None:
                study = load_study_spec(args.study)
                if study.dataset_path != dataset.path:
                    raise ContractError("the study does not reference the validated dataset")
                result["study"] = {
                    "name": study.name,
                    "path": str(study.path),
                    "sha256": study.sha256,
                    "doses": list(study.doses),
                    "output_root": str(study.output_root),
                }
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "run":
            from .runner import run_study

            study = load_study_spec(args.study)
            output = run_study(study, offline=args.offline, run_id=args.run_id)
            print(output)
            return 0
        raise StudyRuntimeError(f"unsupported command: {args.command}")
    except (ContractError, StudyRuntimeError, OSError, RuntimeError) as error:
        print(f"guard-study: {error}")
        return 2
