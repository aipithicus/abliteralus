"""Append-only stimulus corpora assembled from multiple provenanced sources."""

from .errors import (
    StimulusCorpusError,
    StimulusCorruptionError,
    StimulusFormatError,
    StimulusSourceError,
)
from .records import (
    ATTESTATION_RECORD_TYPE,
    EXPRESSION_FORMAT_VERSION,
    PROJECTION_SOURCES,
    REVIEW_STATES,
    SCHEMA_ID,
    SCHEMA_VERSION,
    atom_expression,
    build_attestation,
    compose_expression,
    derive_stimulus_id,
    expression_surface,
    normalize_surface,
)
from .sources import (
    ArxivTaxonomySource,
    Candidate,
    ExtractionResult,
    HeadDecomposition,
    ModelCardSource,
    StimulusSource,
    TermRules,
    basis_from_stimuli,
)
from .store import (
    Stimulus,
    StimulusStore,
    fold_attestations,
    group_values,
    new_run_id,
    stimuli_across,
)

__version__ = "0.1.0"

__all__ = [
    "ATTESTATION_RECORD_TYPE",
    "ArxivTaxonomySource",
    "Candidate",
    "EXPRESSION_FORMAT_VERSION",
    "ExtractionResult",
    "HeadDecomposition",
    "ModelCardSource",
    "PROJECTION_SOURCES",
    "REVIEW_STATES",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "Stimulus",
    "StimulusCorpusError",
    "StimulusCorruptionError",
    "StimulusFormatError",
    "StimulusSource",
    "StimulusSourceError",
    "StimulusStore",
    "TermRules",
    "atom_expression",
    "basis_from_stimuli",
    "build_attestation",
    "compose_expression",
    "derive_stimulus_id",
    "expression_surface",
    "fold_attestations",
    "group_values",
    "new_run_id",
    "normalize_surface",
    "stimuli_across",
]
