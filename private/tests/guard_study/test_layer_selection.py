from __future__ import annotations

import pytest
from guard_study.errors import ContractError
from guard_study.layer_selection import (
    DEFAULT_STRATEGY,
    LayerSelection,
    parse_layer_selection,
    strategy_names,
)

# The measured Llama-Guard-3-1B profile: flat, a seam at 8-9, then a saturated plateau.
_MEASURED = [
    0.094,
    0.031,
    0.031,
    0.047,
    0.062,
    0.031,
    0.016,
    0.094,
    1.844,
    7.688,
    8.047,
    8.344,
    8.516,
    8.609,
    8.609,
    8.656,
]


def _selection(strategy: str, top_k: int, **params) -> LayerSelection:
    return parse_layer_selection(
        {"strategy": strategy, "strategy_params": params},
        top_k=top_k,
        label="study.causal_mapping",
    )


def test_magnitude_ranking_selects_the_saturated_plateau():
    selection = _selection("mean_absolute_effect", 3)

    assert selection.top_layers(_MEASURED) == (15, 13, 14)


def test_first_order_difference_selects_the_seam_instead_of_the_plateau():
    selection = _selection("differential_effect", 3, order=1)

    assert selection.top_layers(_MEASURED) == (9, 8, 10)


def test_differential_order_defaults_to_one_and_treats_earlier_layers_as_zero():
    selection = _selection("differential_effect", 1)

    assert dict(selection.params) == {"order": 1}
    assert selection.rank([2.0, 5.0])[0] == (1, 3.0)
    assert selection.rank([2.0, 5.0])[1] == (0, 2.0)


def test_absent_strategy_block_keeps_the_established_default():
    selection = parse_layer_selection({}, top_k=2, label="study.causal_mapping")

    assert selection.strategy == DEFAULT_STRATEGY
    assert dict(selection.params) == {}
    assert selection.top_layers(_MEASURED) == (15, 13)


def test_unknown_strategy_names_the_registered_alternatives():
    with pytest.raises(ContractError, match="unknown layer-selection strategy"):
        _selection("steepest_descent", 1)

    assert "differential_effect" in strategy_names()


def test_parameters_are_rejected_when_the_strategy_does_not_declare_them():
    with pytest.raises(ContractError, match="accepts no parameters"):
        _selection("mean_absolute_effect", 1, order=2)

    with pytest.raises(ContractError, match="unknown keys"):
        _selection("differential_effect", 1, oder=2)


def test_parameter_values_are_bounded_at_load_time():
    with pytest.raises(ContractError, match="order must be an integer >= 1"):
        _selection("differential_effect", 1, order=0)

    with pytest.raises(ContractError, match="order must be an integer >= 1"):
        _selection("differential_effect", 1, order=True)
