"""Errors raised at the probe-harness boundary."""


class ProbeHarnessError(RuntimeError):
    """A subject, session, command, or artifact contract could not be satisfied."""

