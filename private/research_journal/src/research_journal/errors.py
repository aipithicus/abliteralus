"""Errors raised by the research journal package."""


class ResearchJournalError(RuntimeError):
    """Base class for journal failures."""


class JournalFormatError(ResearchJournalError, ValueError):
    """A record or requested entry does not satisfy the journal contract."""


class JournalCorruptionError(ResearchJournalError):
    """The journal cannot be extended until its invalid suffix is repaired."""
