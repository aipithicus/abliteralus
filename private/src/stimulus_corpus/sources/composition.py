"""Recover an atomic basis from attested multi-word terms.

A policy card names classes compositionally — ``chemical weapons``,
``biological weapons``, ``property crimes``, ``cyber crimes``. Treated as flat
strings those are just terms. Decomposed, they are a **modifier** vocabulary and
a **head** vocabulary that recombine, which is what lets probe generation build
stimuli rather than only replay the ones a document happened to contain.

The basis is discovered, not declared: a head counts only when several attested
terms share it. Nothing here consults a word list or asks a model what a term
means, so the vocabulary reflects the source document rather than anyone's
intuitions about it.

This source deliberately does **not** emit the cross product. Within a head
group every combination is already attested, so the interesting compositions are
cross-head — ``chemical crimes``, ``cyber weapons`` — and those range from
meaningful to nonsense. Minting them is a generation-time decision that needs
review, so :meth:`HeadDecomposition.basis` exposes the vocabulary and stops
there.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from ..records import normalize_surface
from .base import Candidate, ExtractionResult, Rejection


class HeadDecomposition:
    """Split attested multi-word terms into modifier and head atoms."""

    def __init__(
        self,
        surfaces: Iterable[str],
        *,
        source_id: str,
        source_version: str,
        min_group: int = 2,
    ) -> None:
        if min_group < 2:
            raise ValueError("min_group must be at least 2; a head needs corroboration")
        self._source_id = source_id
        self._source_version = source_version
        self.min_group = min_group
        self.surfaces = tuple(dict.fromkeys(normalize_surface(item) for item in surfaces))

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_version(self) -> str:
        return self._source_version

    def basis(self) -> dict[str, tuple[str, ...]]:
        """Return ``{head: (modifier, ...)}`` for every corroborated head."""

        grouped: dict[str, list[str]] = defaultdict(list)
        for surface in self.surfaces:
            tokens = surface.split()
            if len(tokens) < 2:
                continue
            grouped[tokens[-1]].append(" ".join(tokens[:-1]))
        return {
            head: tuple(dict.fromkeys(modifiers))
            for head, modifiers in grouped.items()
            if len(set(modifiers)) >= self.min_group
        }

    def extract(self) -> ExtractionResult:
        """Emit the head and modifier atoms implied by the attested terms."""

        basis = self.basis()
        candidates: list[Candidate] = []
        rejected: list[Rejection] = []
        seen: set[str] = set()

        for head, modifiers in sorted(basis.items()):
            if head not in seen and head not in self.surfaces:
                seen.add(head)
                candidates.append(
                    Candidate(
                        surface=head,
                        source_ref=f"head:{head}",
                        projection_source="lexical",
                        groups={"composition_role": "head", "head": head},
                        extra={
                            "role": "head",
                            "attested_modifiers": list(modifiers),
                            "attested_terms": [f"{item} {head}" for item in modifiers],
                        },
                    )
                )
            for modifier in modifiers:
                if modifier in seen or modifier in self.surfaces:
                    continue
                seen.add(modifier)
                candidates.append(
                    Candidate(
                        surface=modifier,
                        source_ref=f"modifier:{modifier} {head}",
                        projection_source="lexical",
                        groups={"composition_role": "modifier", "head": head},
                        extra={
                            "role": "modifier",
                            "attested_head": head,
                            "attested_in": f"{modifier} {head}",
                        },
                    )
                )

        for surface in self.surfaces:
            tokens = surface.split()
            if len(tokens) < 2:
                rejected.append(Rejection(surface, "-", "not-decomposable"))
            elif tokens[-1] not in basis:
                rejected.append(Rejection(surface, "-", "head-not-corroborated"))

        return ExtractionResult(
            candidates=tuple(candidates),
            rejected=tuple(rejected),
            stats={
                "input_surfaces": len(self.surfaces),
                "heads": len(basis),
                "basis": {head: list(mods) for head, mods in sorted(basis.items())},
                "kept": len(candidates),
                "rejected": len(rejected),
            },
        )


def basis_from_stimuli(stimuli: Iterable[Mapping[str, object] | object]) -> tuple[str, ...]:
    """Pull surfaces out of folded stimuli or raw attestation records."""

    surfaces: list[str] = []
    for item in stimuli:
        if isinstance(item, Mapping):
            surfaces.append(str(item["surface"]))
        else:
            surfaces.append(str(getattr(item, "surface")))
    return tuple(surfaces)
