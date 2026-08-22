"""Contracts for every documented lazy package export."""

from __future__ import annotations

import subprocess
import sys

import pytest

import abliteralus
import abliteralus.analysis as analysis


@pytest.mark.parametrize(
    "name",
    [
        "AbliterationPipeline",
        "InformedAbliterationPipeline",
    ],
)
def test_documented_lazy_export_resolves(name):
    assert getattr(abliteralus, name) is not None


def test_unknown_lazy_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'not_an_export'"):
        getattr(abliteralus, "not_an_export")


def test_analysis_export_map_covers_the_documented_surface():
    assert set(analysis._LAZY_IMPORTS) == set(analysis.__all__)
    assert analysis.CrossLayerAlignmentAnalyzer.__name__ == "CrossLayerAlignmentAnalyzer"
    assert "CrossLayerAlignmentAnalyzer" in dir(analysis)


def test_unknown_analysis_export_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute 'not_an_export'"):
        getattr(analysis, "not_an_export")


def test_importing_analysis_package_does_not_eagerly_import_torch():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import abliteralus.analysis as analysis; "
                "assert 'torch' not in sys.modules; "
                "assert not any(name.startswith('abliteralus.analysis.') "
                "for name in sys.modules); "
                "assert analysis.__all__"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
