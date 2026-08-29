"""The source protocol every stimulus extractor implements.

A source turns some external artifact — a model card, a taxonomy export, a
generator transcript — into candidate surfaces with enough provenance to
justify each one. Sources do not decide what a term *means* and never adjudicate
hazard: they report where a surface came from and how it got there.

Adding a source means implementing :class:`StimulusSource`. Most future sources
are structured (arXiv's category tree, MeSH, LCC) and need parsing rather than
term extraction from prose; the model card is the awkward case, not the typical
one, so nothing here assumes an NLP pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Candidate:
    """One proposed surface, before normalization and admission."""

    surface: str
    #: Where in the source this came from, precise enough to re-find by hand.
    source_ref: str
    #: One of :data:`stimulus_corpus.records.PROJECTION_SOURCES`.
    projection_source: str
    #: Free grouping labels from the source, e.g. ``{"policy_category": "s9"}``.
    #: Any number of them, for colouring a plot, faceting a table, or
    #: stratifying a summary. They assert nothing and cost nothing to be wrong
    #: about — unlike ``split_key``, which is a single committed partition.
    groups: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)
    #: Canonical expression, when the source builds something other than an
    #: atom. ``None`` means "the atom form of ``surface``".
    expression: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class Rejection:
    """One candidate the source declined, and why.

    Rejections are returned rather than discarded so that a dry run can show
    what a filter removed. A silent filter is indistinguishable from a source
    that had nothing to give.
    """

    surface: str
    source_ref: str
    reason: str


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything one extraction pass produced."""

    candidates: tuple[Candidate, ...]
    rejected: tuple[Rejection, ...] = ()
    stats: Mapping[str, Any] = field(default_factory=dict)

    def reason_counts(self) -> dict[str, int]:
        """Return rejection counts by reason, most frequent first."""

        counts: dict[str, int] = {}
        for rejection in self.rejected:
            counts[rejection.reason] = counts.get(rejection.reason, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@runtime_checkable
class StimulusSource(Protocol):
    """Anything that can propose stimuli with provenance."""

    @property
    def source_id(self) -> str:
        """Stable dotted identifier, e.g. ``card.llama-guard-3-1b``."""

    @property
    def source_version(self) -> str:
        """Content identity of the input, so a re-extraction is comparable."""

    def extract(self) -> ExtractionResult:
        """Read the source once and return its candidates and rejections."""


def file_content_version(path: str | Path) -> str:
    """Return ``sha256:<hex>`` for one source file.

    Used as ``source_version`` so that two extraction runs over the same bytes
    are recognisably the same, and a revised card is recognisably not.
    """

    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


def dedupe_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Drop repeats within one extraction pass, keeping the first occurrence.

    Cross-source duplicates are *not* handled here: two sources attesting the
    same surface is meaningful and is folded on read, not collapsed on write.
    """

    seen: set[str] = set()
    kept: list[Candidate] = []
    for candidate in candidates:
        key = candidate.surface.casefold().strip()
        if key in seen:
            continue
        seen.add(key)
        kept.append(candidate)
    return tuple(kept)
