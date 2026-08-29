from __future__ import annotations

import pytest

from probe_harness.cli import _layer_list, _parse_inspect_command


def test_parse_inspect_command_supports_trial_and_layers() -> None:
    target, layers, positions = _parse_inspect_command(
        ":inspect trial-abc --layers 14,12,14 --positions all"
    )

    assert target == "trial-abc"
    assert layers == (12, 14)
    assert positions == "all"


def test_layer_list_rejects_negative_layers() -> None:
    with pytest.raises(Exception, match="non-negative"):
        _layer_list("-1,2")
