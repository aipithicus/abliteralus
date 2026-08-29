"""Stimulus records: canonical expression, derived identity, and validation.

A stimulus record is an *attestation*: one source asserting that one surface
form belongs in the corpus, with the provenance to justify it. Two sources
proposing the same surface produce two attestations sharing one derived
``stimulus_id``; folding them into a single stimulus with several attestations
is a read-side concern (see :mod:`stimulus_corpus.store`).

The record carries construction coordinates and provenance only. It does not
carry an expected label, an author-intended framing, or a hazard rating: those
presuppose the measurement, and a stimulus that presupposes its own response
cannot be used to test the hypothesis it was built for.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from jsonl_engine import canonical_bytes

from .errors import StimulusFormatError

SCHEMA_ID = "aipithicus.stimulus-corpus"
SCHEMA_VERSION = 1
ATTESTATION_RECORD_TYPE = "attestation"

#: Bumping this changes every derived identity, so treat it as a corpus break.
EXPRESSION_FORMAT_VERSION = 1

#: Why a surface entered the corpus. Retained so that source proximity stays a
#: testable factor rather than an invisible assumption.
PROJECTION_SOURCES = frozenset(
    {
        "literal",  # appears verbatim in a policy or model-card source
        "ontological",  # taxonomic or disciplinary relation to a seed
        "lexical",  # derivational or morphological relation to a seed
        "commonsense",  # proposed by a generator, then reviewed
        "matched_control",  # deliberately paired counterpart
        "nonce",  # constructed pseudoword, no referent
    }
)

REVIEW_STATES = frozenset({"proposed", "accepted", "rejected"})

MAX_SURFACE_CHARS = 120
MAX_RECORD_BYTES = 64 * 1024

_COLLAPSIBLE = re.compile(r"\s+")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def utc_now() -> str:
    """Return an RFC 3339 timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_surface(text: str) -> str:
    """Fold one raw surface form to the corpus's canonical spelling.

    Normalization is deliberately conservative: NFKC, collapsed internal
    whitespace, stripped ends, and case folded. It does not stem, singularize,
    or strip punctuation, because those change the token sequence the model
    actually sees.
    """

    if not isinstance(text, str):
        raise StimulusFormatError("surface must be a string")
    folded = unicodedata.normalize("NFKC", text).strip()
    folded = _COLLAPSIBLE.sub(" ", folded)
    folded = folded.casefold()
    if not folded:
        raise StimulusFormatError("surface is empty after normalization")
    if len(folded) > MAX_SURFACE_CHARS:
        raise StimulusFormatError(
            f"surface exceeds {MAX_SURFACE_CHARS} characters: {folded[:40]!r}..."
        )
    return folded


def atom_expression(surface: str) -> dict[str, Any]:
    """Return the canonical expression for a single-atom stimulus.

    A bare topic is the degenerate composed stimulus: one atom and no
    operators. Composed stimuli reuse this record shape with a richer
    expression, which is why identity derives from the expression rather than
    from the surface string.
    """

    return {
        "kind": "atom",
        "format_version": EXPRESSION_FORMAT_VERSION,
        "surface": normalize_surface(surface),
    }


def compose_expression(parts: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Return the canonical expression for a composed stimulus.

    ``parts`` is an ordered sequence of ``(role, surface)``. Roles are open
    rather than enumerated, because the grammar is not settled: today's are
    ``modifier``/``head`` for left modification and ``head``/``pp`` for
    prepositional complementation, and framing work will want ``context``,
    ``stance``, and ``relation`` alongside them.

        compose_expression([("modifier", "chemical"), ("head", "weapons")])
        compose_expression([("head", "time"), ("pp", "of voting")])

    A composed expression is not the same object as the atom that renders to
    the same string. ``chemical`` + ``weapons`` was assembled; the atom
    ``chemical weapons`` was attested. Identity derives from the expression, so
    they get different ids — which keeps "observed or constructed?" answerable
    at analysis time instead of requiring a separate flag.
    """

    if not parts:
        raise StimulusFormatError("a composed expression needs at least one part")
    return {
        "kind": "compose",
        "format_version": EXPRESSION_FORMAT_VERSION,
        "parts": [
            {
                "role": require_identifier(role, label="part role"),
                "surface": normalize_surface(surface),
            }
            for role, surface in parts
        ],
    }


def expression_surface(expression: Mapping[str, Any]) -> str:
    """Render one expression to the string a model would be shown."""

    kind = expression.get("kind")
    if kind == "atom":
        return str(expression["surface"])
    if kind == "compose":
        return " ".join(str(part["surface"]) for part in expression["parts"])
    raise StimulusFormatError(f"unknown expression kind: {kind!r}")


def derive_stimulus_id(expression: Mapping[str, Any], *, corpus_version: str) -> str:
    """Derive the stable identity of one stimulus from its expression.

    Identity is a function of the expression and the corpus version only.
    Anything that changes the expression mints a new stimulus; provenance,
    review state, grouping labels, and split assignment do not. Two sources
    proposing the same surface therefore agree on the identity without
    coordinating.
    """

    require_identifier(corpus_version, label="corpus_version")
    payload = {"corpus_version": corpus_version, "expression": dict(expression)}
    digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return f"stim.{corpus_version}.{digest[:16]}"


def require_identifier(value: str, *, label: str) -> str:
    """Validate one lowercase dotted identifier."""

    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise StimulusFormatError(f"{label} must match {_IDENTIFIER.pattern!r}; got {value!r}")
    return value


def build_attestation(
    *,
    surface: str,
    corpus_version: str,
    source_id: str,
    source_version: str,
    source_ref: str,
    projection_source: str,
    run_id: str,
    split_key: str | None = None,
    groups: Mapping[str, str] | None = None,
    review_status: str = "proposed",
    extra: Mapping[str, Any] | None = None,
    expression: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one candidate and return its durable attestation record.

    ``expression`` defaults to the atom form. Pass a composed expression to
    record a stimulus that was assembled rather than attested.

    ``groups`` and ``split_key`` look similar and are not. Groups are free,
    plural display and stratification labels that assert nothing; the split key
    is the single partition that decides what a held-out set means, so a wrong
    group is cosmetic while a wrong split key silently invalidates a transfer
    result.
    """

    if projection_source not in PROJECTION_SOURCES:
        raise StimulusFormatError(
            f"projection_source must be one of {sorted(PROJECTION_SOURCES)}; "
            f"got {projection_source!r}"
        )
    if review_status not in REVIEW_STATES:
        raise StimulusFormatError(
            f"review_status must be one of {sorted(REVIEW_STATES)}; got {review_status!r}"
        )
    require_identifier(source_id, label="source_id")
    if split_key is not None:
        require_identifier(split_key, label="split_key")
    normalized_groups = {str(key): str(value) for key, value in (groups or {}).items()}
    for key in normalized_groups:
        require_identifier(key, label="group name")
    for label, value in (("source_version", source_version), ("run_id", run_id)):
        if not isinstance(value, str) or not value.strip():
            raise StimulusFormatError(f"{label} must be a non-empty string")

    resolved = dict(expression) if expression is not None else atom_expression(surface)
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "record_type": ATTESTATION_RECORD_TYPE,
        "stimulus_id": derive_stimulus_id(resolved, corpus_version=corpus_version),
        "corpus_version": corpus_version,
        "surface": expression_surface(resolved),
        "expression": resolved,
        "source_id": source_id,
        "source_version": source_version,
        "source_ref": str(source_ref),
        "projection_source": projection_source,
        "split_key": split_key,
        "groups": normalized_groups,
        "review_status": review_status,
        "run_id": run_id,
        "recorded_at": utc_now(),
        "extra": dict(extra or {}),
    }
