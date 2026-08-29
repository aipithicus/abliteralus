"""Command line for building and inspecting a stimulus corpus.

Extraction defaults to a dry run. Prose rules need tuning against the actual
source before anything is committed, and structured sources need their layout
confirmed, so both report what they kept and what they dropped before writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import StimulusCorpusError
from .sources.arxiv_taxonomy import ArxivTaxonomySource
from .sources.model_card import ModelCardSource, TermRules
from .store import StimulusStore, new_run_id

DEFAULT_CORPUS_VERSION = "v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stimulus-corpus",
        description="Build and inspect an append-only stimulus corpus.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="path to the corpus JSONL store",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="extract literal terms from a model card")
    extract.add_argument("--card", type=Path, required=True, help="model card markdown")
    extract.add_argument(
        "--source-id",
        required=True,
        help="stable dotted source identifier, e.g. card.llama-guard-3-1b",
    )
    extract.add_argument(
        "--corpus-version", default=DEFAULT_CORPUS_VERSION, help="corpus version tag"
    )
    extract.add_argument(
        "--split-commas",
        action="store_true",
        help="also split on commas (expect adjective fragments from list prose)",
    )
    extract.add_argument("--max-words", type=int, default=5)
    extract.add_argument("--min-chars", type=int, default=3)
    extract.add_argument(
        "--commit",
        action="store_true",
        help="append the extraction to the store (default is a dry run)",
    )
    extract.add_argument(
        "--sample", type=int, default=25, help="how many examples to show per list"
    )

    taxonomy = subparsers.add_parser("arxiv", help="load the arXiv category taxonomy")
    taxonomy.add_argument(
        "--html", type=Path, required=True, help="saved arxiv.org/category_taxonomy page"
    )
    taxonomy.add_argument("--source-id", default="taxonomy.arxiv")
    taxonomy.add_argument("--corpus-version", default=DEFAULT_CORPUS_VERSION)
    taxonomy.add_argument("--max-words", type=int, default=4)
    taxonomy.add_argument("--commit", action="store_true")
    taxonomy.add_argument("--sample", type=int, default=25)

    listing = subparsers.add_parser("list", help="list folded stimuli")
    listing.add_argument("--source-id", default=None, help="narrow to one source")
    listing.add_argument("--json", action="store_true", help="emit JSON")

    export = subparsers.add_parser("export", help="write a frozen battery dataset")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--name", required=True)
    export.add_argument("--description", default="")
    export.add_argument("--corpus-version", default=DEFAULT_CORPUS_VERSION)
    export.add_argument(
        "--review-state",
        action="append",
        default=None,
        help="review states to include (repeatable; default: accepted)",
    )

    subparsers.add_parser("inspect", help="report store facts")
    return parser


def _report(source_id: str, source_version: str, result, sample: int) -> None:
    """Print what one extraction pass kept and dropped."""

    print(f"source          {source_id}")
    print(f"version         {source_version}")
    for key, value in result.stats.items():
        print(f"{key:<16}{value}")

    print()
    print(f"kept ({len(result.candidates)}):")
    for candidate in result.candidates[:sample]:
        print(f"  {candidate.source_ref:<18} {candidate.surface}")
    if len(result.candidates) > sample:
        print(f"  ... {len(result.candidates) - sample} more")

    counts = result.reason_counts()
    if counts:
        print()
        print(f"rejected ({len(result.rejected)}) by reason:")
        for reason, count in counts.items():
            print(f"  {count:>5}  {reason}")
        print()
        print("rejection sample:")
        for rejection in result.rejected[:sample]:
            print(f"  {rejection.reason:<20} {rejection.surface[:60]}")


def _commit(args: argparse.Namespace, source, result) -> int:
    """Append one extraction pass, or explain that this was a dry run."""

    if not args.commit:
        print()
        print("dry run - nothing written. Re-run with --commit when it looks right.")
        return 0

    store = StimulusStore(args.store)
    run_id = new_run_id()
    receipt = store.append_candidates(
        result.candidates,
        source_id=source.source_id,
        source_version=source.source_version,
        corpus_version=args.corpus_version,
        run_id=run_id,
    )
    print()
    print(f"committed       {receipt.appended_records} attestations")
    print(f"run_id          {run_id}")
    print(f"generation      {receipt.generation}")
    print(f"data_hash       {receipt.data_hash}")
    return 0


def _run_extract(args: argparse.Namespace) -> int:
    rules = (
        TermRules.with_comma_splitting(min_chars=args.min_chars, max_words=args.max_words)
        if args.split_commas
        else TermRules(min_chars=args.min_chars, max_words=args.max_words)
    )
    source = ModelCardSource(args.card, source_id=args.source_id, rules=rules)
    result = source.extract()
    _report(source.source_id, source.source_version, result, args.sample)
    return _commit(args, source, result)


def _run_arxiv(args: argparse.Namespace) -> int:
    source = ArxivTaxonomySource(args.html, source_id=args.source_id, max_words=args.max_words)
    result = source.extract()
    _report(source.source_id, source.source_version, result, args.sample)
    return _commit(args, source, result)


def _run_list(args: argparse.Namespace) -> int:
    store = StimulusStore(args.store)
    stimuli = store.stimuli(source_id=args.source_id)
    if args.json:
        json.dump([item.to_dict() for item in stimuli], sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    for stimulus in stimuli:
        sources = ",".join(stimulus.sources)
        print(f"{stimulus.review_status:<10} {stimulus.surface:<40} {sources}")
    print()
    print(f"{len(stimuli)} stimuli")
    return 0


def _run_export(args: argparse.Namespace) -> int:
    store = StimulusStore(args.store)
    document = store.export_dataset(
        args.out,
        name=args.name,
        description=args.description,
        corpus_version=args.corpus_version,
        review_states=tuple(args.review_state or ("accepted",)),
    )
    print(f"wrote {args.out}")
    print(f"stimuli         {document['stimulus_count']}")
    print(f"content_sha256  {document['content_sha256']}")
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    store = StimulusStore(args.store)
    attestations = store.attestations()
    stimuli = store.stimuli()
    sources = sorted({str(record["source_id"]) for record in attestations})
    print(f"path            {store.path}")
    print(f"attestations    {len(attestations)}")
    print(f"stimuli         {len(stimuli)}")
    print(f"sources         {', '.join(sources) if sources else '(none)'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "extract": _run_extract,
        "arxiv": _run_arxiv,
        "list": _run_list,
        "export": _run_export,
        "inspect": _run_inspect,
    }
    try:
        return handlers[args.command](args)
    except StimulusCorpusError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
