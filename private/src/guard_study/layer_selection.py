"""Configurable strategies for ranking layers in a causal map.

Ranking by raw causal magnitude finds where a decision has already been made.
Ranking by finite difference finds where it *becomes* made, which is a different
question and usually the interesting one for a saturating effect profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .errors import ContractError

DEFAULT_STRATEGY = "mean_absolute_effect"


@dataclass(frozen=True)
class ParamSpec:
    """One integer parameter a strategy declares, with its floor and default."""

    name: str
    minimum: int
    default: int


@dataclass(frozen=True)
class LayerStrategy:
    name: str
    description: str
    params: tuple[ParamSpec, ...]
    score: Callable[..., Sequence[float]]


def _mean_absolute_effect(effects: Sequence[float]) -> list[float]:
    """Score each layer by its own causal magnitude."""

    return [float(value) for value in effects]


def _differential_effect(effects: Sequence[float], *, order: int) -> list[float]:
    """Score each layer by the n-th finite difference, treating effect[L < 0] as zero.

    Order 1 scores a layer by how much causal power appears at it rather than how
    much has accumulated by it, which locates a seam instead of the plateau after
    one. Higher orders difference the measurement noise as many times as the
    signal, so they need a logit floor well below the step being resolved.
    """

    current = [float(value) for value in effects]
    for _ in range(order):
        current = [
            current[index] - (current[index - 1] if index > 0 else 0.0)
            for index in range(len(current))
        ]
    return current


_STRATEGIES: dict[str, LayerStrategy] = {
    strategy.name: strategy
    for strategy in (
        LayerStrategy(
            name="mean_absolute_effect",
            description="rank by mean absolute patch effect",
            params=(),
            score=_mean_absolute_effect,
        ),
        LayerStrategy(
            name="differential_effect",
            description="rank by the n-th finite difference of the patch effect",
            params=(ParamSpec(name="order", minimum=1, default=1),),
            score=_differential_effect,
        ),
    )
}


def strategy_names() -> tuple[str, ...]:
    return tuple(sorted(_STRATEGIES))


def get_strategy(name: str) -> LayerStrategy:
    strategy = _STRATEGIES.get(name)
    if strategy is None:
        raise ContractError(
            f"unknown layer-selection strategy {name!r}; "
            f"expected one of {', '.join(strategy_names())}"
        )
    return strategy


@dataclass(frozen=True)
class LayerSelection:
    """A validated layer-selection configuration and the ranking it induces."""

    strategy: str
    top_k: int
    params: Mapping[str, int]

    def rank(self, effects: Sequence[float]) -> list[tuple[int, float]]:
        """Score every layer and order it best-first, breaking ties by depth."""

        scores = get_strategy(self.strategy).score(effects, **dict(self.params))
        if len(scores) != len(effects):
            raise ContractError(f"strategy {self.strategy!r} returned a mis-sized score vector")
        ranked = [(index, float(value)) for index, value in enumerate(scores)]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked

    def top_layers(self, effects: Sequence[float]) -> tuple[int, ...]:
        ranked = self.rank(effects)
        return tuple(layer for layer, _score in ranked[: min(self.top_k, len(ranked))])

    def summary(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "top_k": self.top_k, "params": dict(self.params)}


def parse_layer_selection(
    raw: Mapping[str, Any],
    *,
    top_k: int,
    label: str,
) -> LayerSelection:
    """Validate a strategy name and its parameters against what that strategy declares."""

    name = raw.get("strategy", DEFAULT_STRATEGY)
    if not isinstance(name, str):
        raise ContractError(f"{label}.strategy must be a string")
    strategy = get_strategy(name)
    declared = {spec.name: spec for spec in strategy.params}

    given = raw.get("strategy_params", {})
    if not isinstance(given, dict):
        raise ContractError(f"{label}.strategy_params must be a mapping")
    unknown = sorted(set(given) - set(declared))
    if unknown:
        accepted = ", ".join(sorted(declared)) if declared else "no parameters"
        raise ContractError(
            f"{label}.strategy_params has unknown keys for strategy {name!r}: "
            f"{', '.join(unknown)}; {name!r} accepts {accepted}"
        )

    params: dict[str, int] = {}
    for key, spec in sorted(declared.items()):
        value = given.get(key, spec.default)
        if isinstance(value, bool) or not isinstance(value, int) or value < spec.minimum:
            raise ContractError(
                f"{label}.strategy_params.{key} must be an integer >= {spec.minimum}"
            )
        params[key] = value
    return LayerSelection(strategy=name, top_k=top_k, params=MappingProxyType(params))
