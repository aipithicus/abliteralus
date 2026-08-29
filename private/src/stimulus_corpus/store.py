"""The stimulus corpus store: attestations in, folded stimuli out.

Corpus construction is incremental and multi-source by nature — the model card
today, a taxonomy export next month, matched counterparts after that — so the
authoritative form is an append-only log of attestations rather than a file that
gets regenerated. Each extraction run commits one batch carrying a partition
descriptor, which is what makes "these forty terms came from that card at that
content hash" answerable later without re-deriving anything.

A *stimulus* is the fold of every attestation sharing one derived id. Two
sources proposing ``virology`` do not collide: they produce one stimulus with
two attestations, and the fact that it was reachable from both is retained
rather than flattened.

Frozen battery datasets are generated views over a committed store boundary,
not a parallel source of truth.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonl_engine import (
    CommitReceipt,
    JsonlCorruptionError,
    JsonlFormatError,
    JsonlStore,
    canonical_bytes,
)

from .errors import StimulusCorruptionError, StimulusFormatError
from .records import (
    ATTESTATION_RECORD_TYPE,
    MAX_RECORD_BYTES,
    SCHEMA_ID,
    build_attestation,
)
from .sources.base import Candidate, StimulusSource


@dataclass(frozen=True, slots=True)
class Stimulus:
    """One stimulus, folded from every attestation that proposed it."""

    stimulus_id: str
    surface: str
    expression: Mapping[str, Any]
    split_key: str | None
    groups: Mapping[str, str]
    review_status: str
    sources: tuple[str, ...]
    projection_sources: tuple[str, ...]
    attestations: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the export shape: identity, coordinates, provenance summary."""

        return {
            "stimulus_id": self.stimulus_id,
            "surface": self.surface,
            "split_key": self.split_key,
            "groups": dict(self.groups),
            "review_status": self.review_status,
            "sources": list(self.sources),
            "projection_sources": list(self.projection_sources),
            "attestation_count": len(self.attestations),
        }


def new_run_id() -> str:
    """Mint an extraction-run identifier."""

    return uuid.uuid4().hex


class StimulusStore:
    """Corpus-domain rules composed over one authoritative :class:`JsonlStore`."""

    def __init__(self, path: str | Path, *, lock_timeout: float = 30.0) -> None:
        self.store = JsonlStore(
            path,
            lock_timeout=lock_timeout,
            maximum_record_bytes=MAX_RECORD_BYTES,
            recover_uncommitted=False,
        )

    @property
    def path(self) -> Path:
        """Return the path owned and normalized by the physical store."""

        return self.store.path

    def append_candidates(
        self,
        candidates: Sequence[Candidate],
        *,
        source_id: str,
        source_version: str,
        corpus_version: str,
        run_id: str | None = None,
        review_status: str = "proposed",
    ) -> CommitReceipt:
        """Durably append one extraction pass as a single committed batch."""

        if not candidates:
            raise StimulusFormatError("refusing to commit an empty extraction pass")
        run = run_id or new_run_id()
        records = [
            build_attestation(
                surface=candidate.surface,
                corpus_version=corpus_version,
                source_id=source_id,
                source_version=source_version,
                source_ref=candidate.source_ref,
                projection_source=candidate.projection_source,
                run_id=run,
                split_key=None,
                groups=candidate.groups,
                review_status=review_status,
                extra=dict(candidate.extra),
                expression=candidate.expression,
            )
            for candidate in candidates
        ]
        try:
            return self.store.append(
                records,
                metadata={"schema": SCHEMA_ID},
                partition={
                    "source_id": source_id,
                    "source_version": source_version,
                    "run_id": run,
                },
            )
        except JsonlCorruptionError as error:
            raise StimulusCorruptionError(str(error)) from error
        except JsonlFormatError as error:
            raise StimulusFormatError(str(error)) from error

    def append_from_source(
        self,
        source: StimulusSource,
        *,
        corpus_version: str,
        run_id: str | None = None,
        review_status: str = "proposed",
    ) -> tuple[CommitReceipt, Mapping[str, Any]]:
        """Extract from one source and commit the result in one batch."""

        result = source.extract()
        receipt = self.append_candidates(
            result.candidates,
            source_id=source.source_id,
            source_version=source.source_version,
            corpus_version=corpus_version,
            run_id=run_id,
            review_status=review_status,
        )
        return receipt, dict(result.stats)

    def attestations(self, *, source_id: str | None = None) -> list[dict[str, Any]]:
        """Read attestations, optionally narrowed to one source via JIDX ranges."""

        if source_id is None:
            records = self.store.read_records()
        else:
            records = self.store.read_partition({"source_id": source_id})
        return [
            record for record in records if record.get("record_type") == ATTESTATION_RECORD_TYPE
        ]

    def stimuli(self, *, source_id: str | None = None) -> list[Stimulus]:
        """Fold attestations into stimuli, ordered by surface.

        Where attestations disagree, the most recently recorded one wins for
        ``split_key`` and ``review_status``. Both are read-side resolutions of
        an append-only log, so a later correction simply appends.
        """

        return fold_attestations(self.attestations(source_id=source_id))

    def export_dataset(
        self,
        path: str | Path,
        *,
        name: str,
        description: str,
        corpus_version: str,
        review_states: Iterable[str] = ("accepted",),
    ) -> dict[str, Any]:
        """Write a frozen battery dataset as a view over the current corpus.

        The dataset is disposable: it can always be regenerated from the store.
        Its ``content_sha256`` identifies exactly which stimuli a battery ran
        against, which is what a study spec pins.
        """

        import yaml

        wanted = frozenset(review_states)
        selected = [
            stimulus.to_dict() for stimulus in self.stimuli() if stimulus.review_status in wanted
        ]
        if not selected:
            raise StimulusFormatError(
                f"no stimuli in review states {sorted(wanted)}; nothing to export"
            )
        digest = hashlib.sha256(canonical_bytes(selected)).hexdigest()
        document = {
            "schema_version": 1,
            "name": name,
            "description": description,
            "corpus_version": corpus_version,
            "content_sha256": digest,
            "stimulus_count": len(selected),
            "stimuli": selected,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return document


def fold_attestations(records: Iterable[Mapping[str, Any]]) -> list[Stimulus]:
    """Fold attestation records into stimuli, ordered by surface.

    Where attestations disagree, the most recently recorded one wins for
    ``split_key`` and ``review_status``, and grouping labels merge key by key.
    All three are read-side resolutions of an append-only log, so a correction
    is an append rather than an edit.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["stimulus_id"]), []).append(record)

    folded: list[Stimulus] = []
    for stimulus_id, entries in grouped.items():
        ordered = sorted(entries, key=lambda item: str(item.get("recorded_at", "")))
        latest = ordered[-1]
        split_key = next(
            (record["split_key"] for record in reversed(ordered) if record.get("split_key")),
            None,
        )
        merged: dict[str, str] = {}
        for record in ordered:
            merged.update(
                {str(key): str(value) for key, value in (record.get("groups") or {}).items()}
            )
        folded.append(
            Stimulus(
                stimulus_id=stimulus_id,
                surface=str(latest["surface"]),
                expression=dict(latest["expression"]),
                split_key=split_key,
                groups=merged,
                review_status=str(latest.get("review_status", "proposed")),
                sources=tuple(sorted({str(r["source_id"]) for r in ordered})),
                projection_sources=tuple(sorted({str(r["projection_source"]) for r in ordered})),
                attestations=tuple(ordered),
            )
        )
    folded.sort(key=lambda stimulus: stimulus.surface)
    return folded


def stimuli_across(stores: Iterable[StimulusStore]) -> list[Stimulus]:
    """Fold stimuli across several stores.

    Separate stores buy independent release cycles and separate pinning — a
    battery can name the card corpus at one boundary and the taxonomy at
    another. What they cost is exactly this: a term attested in two stores no
    longer folds on its own, so anything assembling a battery has to read them
    together. Cheap, but it has to be deliberate rather than forgotten.
    """

    records: list[Mapping[str, Any]] = []
    for store in stores:
        records.extend(store.attestations())
    return fold_attestations(records)


def group_values(stimuli: Iterable[Stimulus], name: str) -> dict[str, str | None]:
    """Map ``stimulus_id -> group value`` for one grouping name.

    The shape a plot wants: pass it straight to a colour scale, and every
    stimulus keeps the same colour across every figure because the mapping
    lives in the corpus rather than in whichever notebook drew the chart.
    """

    return {stimulus.stimulus_id: stimulus.groups.get(name) for stimulus in stimuli}
