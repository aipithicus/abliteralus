"""Failure taxonomy for surgery artifact operations."""


class ArtifactError(RuntimeError):
    """Base class for an artifact operation that could not complete safely."""


class CapsuleFormatError(ArtifactError):
    """A capsule is malformed, incomplete, or fails integrity validation."""


class ExactnessError(ArtifactError):
    """An operation cannot meet the requested exact rehydration contract."""


class UnsupportedCheckpointError(ArtifactError):
    """A checkpoint uses a storage layout unsupported by capsule v1."""


class UnsupportedExportError(ArtifactError):
    """A native capsule cannot be represented by a requested adapter format."""
