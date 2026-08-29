"""Airgap sentinels. Injected by private/tools/shadow/sync.py; not part of upstream.

Network-touching imports in the vendored core are rewritten to bind the same names to
these sentinels. Any attempt to use them raises :class:`DisabledError` with the original
dotted name, so a code path that would have reached the network fails loudly and
identifiably instead of silently.
"""

from __future__ import annotations

__all__ = ["DisabledError", "disabled", "disabled_module"]


class DisabledError(RuntimeError):
    """Raised when vendored code reaches for a network facility removed in the lab."""


def disabled(dotted_name: str):
    """Return a callable that raises :class:`DisabledError` naming ``dotted_name``."""

    def _raise(*args, **kwargs):
        raise DisabledError(f"{dotted_name} is disabled in the airgapped lab")

    _raise.__name__ = dotted_name.rsplit(".", 1)[-1]
    _raise.__qualname__ = _raise.__name__
    _raise.__doc__ = f"Airgap sentinel for {dotted_name}."
    return _raise


class _DisabledModule:
    """Attribute access yields sentinels; exception-like names yield :class:`DisabledError`."""

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str):
        if attr.startswith("__"):
            raise AttributeError(attr)
        if attr.endswith(("Error", "Exception", "Warning", "Expired")):
            return DisabledError
        return disabled(f"{self._name}.{attr}")

    def __repr__(self) -> str:
        return f"<disabled module {self._name}>"


def disabled_module(name: str) -> _DisabledModule:
    """Return a module-like sentinel for ``import name`` rewrites."""
    return _DisabledModule(name)
