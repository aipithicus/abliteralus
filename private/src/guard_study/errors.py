"""Errors exposed by the guard-study package."""


class ContractError(ValueError):
    """A tracked study or dataset contract is invalid."""


class StudyRuntimeError(RuntimeError):
    """A study cannot continue without violating its runtime contract."""
