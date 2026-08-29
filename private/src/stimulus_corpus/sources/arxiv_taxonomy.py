"""Discipline names from the arXiv category taxonomy.

Structured sources need parsing, not extraction. There is no prose to mine
here and no heuristic worth tuning: every category is a curated label with a
code and a place in a hierarchy, so the work is reading the tree and keeping
the coordinates. This is the typical shape for a stimulus source — the model
card, with its policy prose, is the awkward exception.

The stimulus is the category *name* (``Artificial Intelligence``), not its code.
The code becomes ``source_ref`` so a term can be traced back, and the archive
and group become grouping labels, which gives a two-level disciplinary hierarchy
for free.

Parsing is regex over the published HTML rather than a DOM walk. The page is
regular enough that this holds, and the source fails loudly if the structure
moves rather than silently returning nothing.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import StimulusFormatError, StimulusSourceError
from ..records import normalize_surface
from .base import (
    Candidate,
    ExtractionResult,
    Rejection,
    dedupe_candidates,
    file_content_version,
)

#: ``<h4>cs.AI <span>(Artificial Intelligence)</span></h4>``
CATEGORY_PATTERN = re.compile(
    r"<h4>\s*(?P<code>[a-zA-Z-]+(?:\.[a-zA-Z-]+)?)\s*<span>\((?P<name>[^)]+)\)</span>"
)

#: ``id="accordion-head-grp_cs"`` ... ``<button ...>Computer Science</button>``
GROUP_PATTERN = re.compile(
    r'id="accordion-head-grp_(?P<gid>[\w-]+)".*?<button[^>]*>(?P<label>.*?)</button>',
    re.S,
)

_TAGS = re.compile(r"<[^>]+>")
_NON_IDENT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Category:
    """One arXiv category and where it sits in the tree."""

    code: str
    name: str
    archive: str
    group: str


def _slug(text: str) -> str:
    """Fold a label to an identifier usable as a group value."""

    return _NON_IDENT.sub("_", text.strip().casefold()).strip("_")


class ArxivTaxonomySource:
    """Emit arXiv category names as disciplinary stimuli."""

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str = "taxonomy.arxiv",
        max_words: int = 4,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise StimulusSourceError(f"arXiv taxonomy not found: {self.path}")
        self._source_id = source_id
        self.max_words = max_words
        self._text = self.path.read_text(encoding="utf-8")

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_version(self) -> str:
        return file_content_version(self.path)

    def categories(self) -> tuple[Category, ...]:
        """Parse the category tree, assigning each category to its group."""

        groups = [
            (match.start(), _slug(_TAGS.sub("", html.unescape(match.group("label")))))
            for match in GROUP_PATTERN.finditer(self._text)
        ]
        matches = list(CATEGORY_PATTERN.finditer(self._text))
        if not matches:
            raise StimulusSourceError(
                f"no categories matched in {self.path}; the taxonomy page layout "
                f"has probably changed (pattern: {CATEGORY_PATTERN.pattern!r})"
            )

        found: list[Category] = []
        for match in matches:
            group = ""
            for start, label in groups:
                if start <= match.start():
                    group = label
                else:
                    break
            code = match.group("code")
            found.append(
                Category(
                    code=code,
                    name=html.unescape(match.group("name")).strip(),
                    archive=code.split(".")[0],
                    group=group or code.split(".")[0],
                )
            )
        return tuple(found)

    def extract(self) -> ExtractionResult:
        """Emit one candidate per category name."""

        categories = self.categories()
        candidates: list[Candidate] = []
        rejected: list[Rejection] = []

        for category in categories:
            if len(category.name.split()) > self.max_words:
                rejected.append(Rejection(category.name, category.code, "too-many-words"))
                continue
            try:
                surface = normalize_surface(category.name)
            except StimulusFormatError:
                rejected.append(Rejection(category.name, category.code, "unnormalizable"))
                continue
            candidates.append(
                Candidate(
                    surface=surface,
                    source_ref=category.code,
                    projection_source="ontological",
                    groups={
                        "arxiv_group": category.group,
                        "arxiv_archive": _slug(category.archive),
                    },
                    extra={"code": category.code, "name": category.name},
                )
            )

        kept = dedupe_candidates(candidates)
        return ExtractionResult(
            candidates=kept,
            rejected=tuple(rejected),
            stats={
                "path": str(self.path),
                "categories": len(categories),
                "groups": sorted({category.group for category in categories}),
                "archives": len({category.archive for category in categories}),
                "deduped": len(candidates) - len(kept),
                "kept": len(kept),
                "rejected": len(rejected),
            },
        )
