"""Errors raised by the stimulus corpus package."""


class StimulusCorpusError(RuntimeError):
    """Base class for stimulus corpus failures."""


class StimulusFormatError(StimulusCorpusError, ValueError):
    """A record or candidate does not satisfy the stimulus contract."""


class StimulusSourceError(StimulusCorpusError):
    """A source could not be read or produced no usable candidates."""


class StimulusCorruptionError(StimulusCorpusError):
    """The corpus cannot be extended until its invalid suffix is repaired."""
