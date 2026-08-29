"""Stimulus sources: model cards now, structured taxonomies next."""

from .base import (
    Candidate,
    ExtractionResult,
    Rejection,
    StimulusSource,
    dedupe_candidates,
    file_content_version,
)
from .arxiv_taxonomy import ArxivTaxonomySource, Category
from .composition import HeadDecomposition, basis_from_stimuli
from .model_card import (
    COMMA_DELIMITERS,
    DEFAULT_DELIMITERS,
    DEFAULT_DROP_PATTERNS,
    DEFAULT_SECTION_PATTERN,
    DEFAULT_STOP_SURFACES,
    ModelCardSource,
    Section,
    TermRules,
)

__all__ = [
    "ArxivTaxonomySource",
    "Category",
    "COMMA_DELIMITERS",
    "Candidate",
    "DEFAULT_DELIMITERS",
    "DEFAULT_DROP_PATTERNS",
    "DEFAULT_SECTION_PATTERN",
    "DEFAULT_STOP_SURFACES",
    "ExtractionResult",
    "HeadDecomposition",
    "ModelCardSource",
    "Rejection",
    "Section",
    "StimulusSource",
    "TermRules",
    "basis_from_stimuli",
    "dedupe_candidates",
    "file_content_version",
]
