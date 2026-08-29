"""Append-only, integrity-checked JSONL research journals."""

from .errors import JournalCorruptionError, JournalFormatError, ResearchJournalError
from .store import (
    ENTRY_RECORD_TYPE,
    HEADER_RECORD_TYPE,
    SCHEMA_ID,
    SCHEMA_VERSION,
    SHARING_STATES,
    JournalInspection,
    RepairResult,
    ResearchJournal,
)

__version__ = "0.1.0"

__all__ = [
    "ENTRY_RECORD_TYPE",
    "HEADER_RECORD_TYPE",
    "JournalCorruptionError",
    "JournalFormatError",
    "JournalInspection",
    "RepairResult",
    "ResearchJournal",
    "ResearchJournalError",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "SHARING_STATES",
]
