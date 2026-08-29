"""Literal term extraction from a hazard-taxonomy model card.

This source answers one narrow, checkable question: *which surfaces appear
verbatim in which section of the card?* It does not decide whether a term is
hazardous, and it must not be asked to — a model's opinion about harm has no
business shaping the stimulus set used to measure a model's response to harm.
Everything it emits is ``projection_source="literal"`` with an offset you can
open the file and check.

Extraction from prose is heuristic, so every candidate lands as
``review_status="proposed"`` and every filtered fragment is returned as a
:class:`~stimulus_corpus.sources.base.Rejection` with its reason. Tune the rules
against a dry run before committing anything.

Section codes are recorded as the ``policy_category`` group. Groups are free
labels — colour a point cloud by policy category, facet a table by it, stratify
a summary by it; none of that asserts anything, because the geometry is
whatever it is regardless of the palette.

What the code is *not* is the ``split_key``. Partitioning held-out sets by
policy category would make the readout taxonomy organize the stimulus basis,
and the question is how the model groups things, not how the card does. That
would be a defensible design — leave-category-out transfer — but a different
one, and it should be chosen rather than inherited.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import StimulusFormatError, StimulusSourceError
from ..records import normalize_surface
from .base import (
    Candidate,
    ExtractionResult,
    Rejection,
    dedupe_candidates,
    file_content_version,
)

#: Matches a hazard-category heading, bold label, or table cell:
#: ``### S9: Weapons``, ``**S9 - Weapons**``, ``| S9 | Weapons |``.
DEFAULT_SECTION_PATTERN = re.compile(
    r"(?m)^(?:\#{1,6}\s*|\|\s*|\*{1,2}|[-*]\s+)?(?P<code>S\d{1,2})\b"
    r"[\s:.,|*-]*(?P<title>[^\n|]{0,80})"
)

#: Terms enumerated inside an example or gloss group: ``(ex: nerve gas)``,
#: ``(i.e., race, color, ethnicity)``. Policy prose states the rule; these
#: groups are where it names things, so they are the highest-yield structure in
#: a card of this shape and the only one whose members are reliably terms.
DEFAULT_EXAMPLE_PATTERN = re.compile(r"\((?:ex|i\.e\.?)[:,]?\s*(?P<items>[^)]+)\)", re.I)

#: Enumerated policy sub-clauses: ``(1) chemical weapons``, ``(3) property
#: crimes``. The clause names a class; its trailing example group names its
#: members, so both are worth keeping and they sit at different granularities.
DEFAULT_ENUM_PATTERN = re.compile(r"\(\d+\)\s*(?P<item>[^(,;)]+)")

#: Trailing enumerations introduced by "including": ``including those that
#: enable denial of service attacks, container escapes or privilege escalation
#: exploits``. Sections vary in construction even within one card — some
#: enumerate with numbered clauses, others just append a list — so this is a
#: third structure rather than a fallback.
DEFAULT_INCLUSION_PATTERN = re.compile(
    r"\bincluding\b(?:\s+(?:those\s+)?(?:that|which)\s+\w+)?\s*:?\s*(?P<items>[^.;()]+)",
    re.I,
)

#: Split an example group into members. Handles ``a, b, or c``, ``a and b``,
#: and the Oxford-comma form ``a, b, and c`` — where the conjunction has to be
#: consumed with the comma, or it strands at the head of the next member. The
#: trailing ``\s+`` after the conjunction is load-bearing: without it, ``,
#: organized crime`` would lose its first two letters.
_EXAMPLE_ITEMS = re.compile(r"\s*,\s*(?:(?:and/or|and|or)\s+)?|\s+(?:and/or|and|or)\s+")

#: Ends a hazard section. Without one, the final section runs to end of file and
#: absorbs everything after the taxonomy — evaluation tables, references,
#: citations — which a delimiter pass will happily mine for nonexistent terms.
DEFAULT_SECTION_TERMINATOR = re.compile(r"(?m)^#{1,6}\s")

#: Strong separators only. Commas are excluded by default and opt-in via
#: ``split_commas``: a list like "chemical, biological, radiological, nuclear,
#: high-yield explosive weapons" splits on commas into bare adjectives, which
#: are not terms. Splitting on semicolons and line breaks is safe.
DEFAULT_DELIMITERS = re.compile(r"[;\n]+|(?:^|\s)[-*•]\s+")
COMMA_DELIMITERS = re.compile(r"[;,\n]+|(?:^|\s)[-*•]\s+")

#: Fragments carrying policy prose rather than a term.
DEFAULT_DROP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:should|must|may|will|can|shall)\s+not\b", re.I),
    re.compile(r"\b(?:such as|including|for example|e\.g|i\.e|etc)\b", re.I),
    re.compile(r"\b(?:content|response|responses|model|models|assistant|user|users)\b", re.I),
    re.compile(r"\b(?:that|which|who|whom|whose|where|when|with|without)\b", re.I),
    re.compile(r"\b(?:enabl|encourag|endors|facilitat|assist|provid|creat)\w*\b", re.I),
    re.compile(r"\b(?:example|examples|includ\w*)\b", re.I),
    re.compile(r"<[^>]+>|^\s*<"),  # raw HTML from tables embedded in the card
)

DEFAULT_STOP_SURFACES = frozenset(
    {
        "and",
        "or",
        "the",
        "a",
        "an",
        "other",
        "others",
        "any",
        "all",
        "unsafe",
        "safe",
        "hazard",
        "hazards",
        "category",
        "categories",
        "description",
        "examples",
        "example",
    }
)

#: A term does not begin with a preposition or a conjunction. Splitting a
#: coordination that shares a trailing prepositional phrase — "the time, place,
#: or manner of voting" — leaves fragments like "in the time"; this catches them
#: without knowing anything about the source.
_LEADING_FUNCTION_WORD = re.compile(
    r"^(?:in|on|at|of|for|with|without|by|to|from|as|into|over|under|about|and|or|but)\b",
    re.I,
)

#: Prepositions that can head a complement shared across a coordination.
_TAIL_PREPOSITION = re.compile(r"\b(?:of|for|toward|towards|against|to|on|in|with)\b", re.I)

#: A leading preposition, optionally with its article, on a coordination member.
_LEADING_PP = re.compile(r"^(?:in|on|at|of|for|with|by|to|from)\s+(?:the|a|an)?\s*", re.I)

_MARKDOWN_FURNITURE = re.compile(r"[*_`\[\]]+")
_LEADING_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TermRules:
    """Tunable filters for pulling terms out of policy prose."""

    delimiters: re.Pattern[str] = DEFAULT_DELIMITERS
    drop_patterns: tuple[re.Pattern[str], ...] = DEFAULT_DROP_PATTERNS
    stop_surfaces: frozenset[str] = DEFAULT_STOP_SURFACES
    min_chars: int = 3
    max_words: int = 5
    #: Set to ``None`` to disable structured extraction and fall back to
    #: delimiter splitting alone.
    example_pattern: re.Pattern[str] | None = DEFAULT_EXAMPLE_PATTERN
    enum_pattern: re.Pattern[str] | None = DEFAULT_ENUM_PATTERN
    inclusion_pattern: re.Pattern[str] | None = DEFAULT_INCLUSION_PATTERN

    @classmethod
    def with_comma_splitting(cls, **overrides: Any) -> TermRules:
        """Rules that also split on commas. Expect adjective fragments."""

        return cls(delimiters=COMMA_DELIMITERS, **overrides)


@dataclass(frozen=True, slots=True)
class Section:
    """One hazard-category section located in the card."""

    code: str
    title: str
    start: int
    end: int


class ModelCardSource:
    """Extract literal terms from a hazard-taxonomy model card."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str,
        rules: TermRules | None = None,
        section_pattern: re.Pattern[str] = DEFAULT_SECTION_PATTERN,
        section_terminator: re.Pattern[str] | None = DEFAULT_SECTION_TERMINATOR,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise StimulusSourceError(f"model card not found: {self.path}")
        self._source_id = source_id
        self.rules = rules or TermRules()
        self.section_pattern = section_pattern
        self.section_terminator = section_terminator
        self._text = self.path.read_text(encoding="utf-8")

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_version(self) -> str:
        return file_content_version(self.path)

    def sections(self) -> tuple[Section, ...]:
        """Locate hazard-category sections and their body spans."""

        matches = list(self.section_pattern.finditer(self._text))
        if not matches:
            raise StimulusSourceError(
                f"no hazard-category sections matched in {self.path}. "
                f"Adjust section_pattern (current: {self.section_pattern.pattern!r})"
            )
        found: list[Section] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(self._text)
            if self.section_terminator is not None:
                boundary = self.section_terminator.search(self._text, match.end(), end)
                if boundary is not None:
                    end = boundary.start()
            found.append(
                Section(
                    code=match.group("code").casefold(),
                    title=(match.group("title") or "").strip(" :.-|*"),
                    start=match.end(),
                    end=end,
                )
            )
        return tuple(found)

    def extract(self) -> ExtractionResult:
        """Read the card once and return candidate terms with provenance.

        Three passes at descending precision. Enumerated clauses name classes,
        example groups name their members, and whatever prose is left over is
        split on delimiters. The first two consume the spans they match, so the
        prose pass sees only the residue — which for a policy card is mostly
        the rule statement itself, and is expected to be rejected.
        """

        sections = self.sections()
        candidates: list[Candidate] = []
        rejected: list[Rejection] = []
        by_role: dict[str, int] = {}

        def admit(
            raw: str, offset: int, section: Section, role: str, tail: str | None = None
        ) -> None:
            outcome = _admit(raw, offset, section, role, self.rules, tail)
            if isinstance(outcome, Candidate):
                candidates.append(outcome)
                by_role[role] = by_role.get(role, 0) + 1
            elif isinstance(outcome, Rejection):
                rejected.append(outcome)

        for section in sections:
            working = self._text[section.start : section.end]

            if self.rules.enum_pattern is not None:
                spans: list[tuple[int, int]] = []
                for match in self.rules.enum_pattern.finditer(working):
                    spans.append((match.start(), match.end()))
                    admit(
                        match.group("item"),
                        section.start + match.start("item"),
                        section,
                        "clause",
                    )
                working = _mask(working, spans)

            for pattern, role in (
                (self.rules.example_pattern, "example"),
                (self.rules.inclusion_pattern, "inclusion"),
            ):
                if pattern is None:
                    continue
                spans = []
                for match in pattern.finditer(working):
                    spans.append((match.start(), match.end()))
                    base = section.start + match.start("items")
                    pieces = _fragments(match.group("items"), base, _EXAMPLE_ITEMS)
                    for piece, offset, tail in _distribute_tail(pieces):
                        admit(piece, offset, section, role, tail)
                working = _mask(working, spans)

            for raw, offset in _fragments(working, section.start, self.rules.delimiters):
                admit(raw, offset, section, "prose")

        kept = dedupe_candidates(candidates)
        return ExtractionResult(
            candidates=kept,
            rejected=tuple(rejected),
            stats={
                "path": str(self.path),
                "sections": len(sections),
                "section_codes": [section.code for section in sections],
                "raw_candidates": len(candidates),
                "by_role": by_role,
                "deduped": len(candidates) - len(kept),
                "kept": len(kept),
                "rejected": len(rejected),
            },
        )


def _fragments(body: str, base_offset: int, delimiters: re.Pattern[str]) -> list[tuple[str, int]]:
    """Split a section body, keeping each fragment's offset in the file."""

    pieces: list[tuple[str, int]] = []
    position = 0
    for match in delimiters.finditer(body):
        pieces.append((body[position : match.start()], base_offset + position))
        position = match.end()
    pieces.append((body[position:], base_offset + position))
    return pieces


def _clean(raw: str, offset: int) -> tuple[str, int]:
    """Strip markdown furniture, returning the text and its adjusted offset."""

    bullet = _LEADING_BULLET.match(raw)
    if bullet:
        raw = raw[bullet.end() :]
        offset += bullet.end()
    leading = len(raw) - len(raw.lstrip())
    offset += leading
    text = _MARKDOWN_FURNITURE.sub("", raw.strip())
    return text.strip(" .:,;\"'()"), offset


def _distribute_tail(
    pieces: Sequence[tuple[str, int]],
) -> list[tuple[str, int, str | None]]:
    """Expand a coordination sharing a trailing complement.

    ``the time, place, or manner of voting`` means three things, not two plus a
    fragment: the prepositional phrase is elided from every member but the
    last. Splitting on commas alone yields ``the time`` and ``place``, which
    look like noise and are actually incomplete.

    Distribution is refused unless the shape is unambiguous — the last member
    carries a preposition, the earlier ones are short and carry none — because
    a wrong expansion invents a term the source never asserted.
    """

    plain = [(text, offset, None) for text, offset in pieces]
    if len(pieces) < 2:
        return plain

    cleaned = [(_LEADING_PP.sub("", text).strip(), offset) for text, offset in pieces]
    last_text, last_offset = cleaned[-1]
    match = _TAIL_PREPOSITION.search(last_text)
    if match is None or match.start() == 0:
        return plain

    region = last_text[match.start() :]
    following = _TAIL_PREPOSITION.search(region, 1)
    tail = (region[: following.start()] if following else region).strip()
    stem = last_text[: match.start()].strip()
    earlier = [text for text, _ in cleaned[:-1]]
    if not stem or not tail or not all(earlier):
        return plain
    if any(_TAIL_PREPOSITION.search(text) for text in earlier):
        return plain
    if any(len(text.split()) > 3 for text in earlier):
        return plain

    expanded = [(f"{text} {tail}", offset, tail) for text, offset in cleaned[:-1]]
    expanded.append((f"{stem} {tail}", last_offset, tail))
    return expanded


def _mask(body: str, spans: Sequence[tuple[int, int]]) -> str:
    """Blank out already-consumed spans, preserving offsets and line structure.

    Masking rather than deleting keeps every later offset honest, which is the
    whole point of carrying provenance you can open the file and check.
    """

    if not spans:
        return body
    chars = list(body)
    for start, end in spans:
        for index in range(max(start, 0), min(end, len(chars))):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _admit(
    raw: str,
    offset: int,
    section: Section,
    role: str,
    rules: TermRules,
    tail: str | None = None,
) -> Candidate | Rejection | None:
    """Clean, filter, and normalize one fragment into a candidate or rejection.

    ``tail`` records a prepositional phrase distributed onto this member from
    the end of a coordination. The offset still points at the member, so the
    record says exactly which part was found and which was resolved.
    """

    cleaned, start = _clean(raw, offset)
    if not cleaned:
        return None
    ref = f"{section.code}@{start}-{start + len(cleaned)}"
    reason = _reject_reason(cleaned, rules)
    if reason is not None:
        return Rejection(cleaned, ref, reason)
    try:
        surface = normalize_surface(cleaned)
    except StimulusFormatError:
        return Rejection(cleaned, ref, "unnormalizable")
    return Candidate(
        surface=surface,
        source_ref=ref,
        projection_source="literal",
        groups={"policy_category": section.code},
        extra={
            "section_title": section.title,
            "role": role,
            **({"distributed_tail": tail} if tail else {}),
        },
    )


def _reject_reason(surface: str, rules: TermRules) -> str | None:
    """Return why this fragment is not a term, or ``None`` to keep it."""

    if not surface:
        return "empty"
    if not _HAS_LETTER.search(surface):
        return "no-letters"
    if len(surface) < rules.min_chars:
        return "too-short"
    words = surface.split()
    if len(words) > rules.max_words:
        return "too-many-words"
    if surface.casefold() in rules.stop_surfaces:
        return "stop-surface"
    if _LEADING_FUNCTION_WORD.match(surface):
        return "leading-function-word"
    for pattern in rules.drop_patterns:
        if pattern.search(surface):
            return "policy-language"
    return None
