from __future__ import annotations

import pytest
import torch
from guard_study.errors import ContractError
from guard_study.precision import (
    DEFAULT_PRECISION,
    PrecisionSpec,
    parse_precision,
    pinned_matmul_precision,
    resolution_floor,
    torch_dtype,
)


def test_resolution_floor_tracks_the_operand_magnitude_not_the_result():
    # A guard margin near 13 logits quantizes on this grid, whatever the
    # difference being resolved happens to be.
    assert resolution_floor(torch.bfloat16, 13.0) == pytest.approx(0.0625)
    assert resolution_floor(torch.float16, 13.0) == pytest.approx(0.0078125)
    assert resolution_floor(torch.float32, 13.0) < 1e-5


def test_bfloat16_differences_of_large_margins_land_on_the_ulp_grid():
    quantum = resolution_floor(torch.bfloat16, 13.0)
    rounded = float(
        torch.tensor(13.0, dtype=torch.bfloat16) - torch.tensor(12.95, dtype=torch.bfloat16)
    )

    # The difference is forced onto a multiple of the quantum, so a true effect of
    # 0.05 is reported as one whole quantum and anything smaller reports as zero.
    assert rounded == pytest.approx(quantum, abs=1e-9)
    assert quantum == pytest.approx(0.0625)
    assert float(torch.tensor(13.0) - torch.tensor(12.95)) == pytest.approx(0.05, abs=1e-6)


def test_absent_precision_block_inherits_compute_and_reads_out_in_float32():
    spec = parse_precision(None, label="study.precision")

    assert spec == DEFAULT_PRECISION
    assert spec.resolve_compute("bfloat16") == "bfloat16"
    assert spec.readout_dtype is torch.float32
    assert spec.allow_tf32 is False


def test_declared_compute_overrides_the_inherited_surgery_dtype():
    spec = parse_precision({"compute": "float32"}, label="study.precision")

    assert spec.resolve_compute("bfloat16") == "float32"
    assert spec.summary(inherited="bfloat16")["compute"] == "float32"
    assert spec.summary(inherited="bfloat16")["compute_declared"] == "float32"


def test_precision_block_rejects_unknown_keys_and_values():
    with pytest.raises(ContractError, match="unknown keys"):
        parse_precision({"redout": "float32"}, label="study.precision")
    with pytest.raises(ContractError, match="compute must be one of"):
        parse_precision({"compute": "int8"}, label="study.precision")
    with pytest.raises(ContractError, match="readout must be one of"):
        parse_precision({"readout": "bfloat16"}, label="study.precision")
    with pytest.raises(ContractError, match="allow_tf32 must be true or false"):
        parse_precision({"allow_tf32": "no"}, label="study.precision")


def test_unsupported_dtype_names_are_rejected():
    with pytest.raises(ContractError, match="unsupported dtype"):
        torch_dtype("float8")


def test_tf32_is_pinned_for_the_run_and_restored_afterwards():
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    spec = PrecisionSpec(compute="inherit", readout="float32", allow_tf32=False)

    with pinned_matmul_precision(spec) as state:
        assert state["allow_tf32"] is False
        assert state["float32_matmul_precision"] == "highest"

    assert torch.backends.cuda.matmul.allow_tf32 == previous[0]
    assert torch.get_float32_matmul_precision() == previous[1]


def test_pinning_reports_tf32_when_a_run_asks_for_it():
    spec = PrecisionSpec(compute="float32", readout="float32", allow_tf32=True)

    with pinned_matmul_precision(spec) as state:
        assert state["allow_tf32"] is True
        # TF32 keeps an 11-bit significand, the same width as float16, so a run
        # that declared float32 must record that it did not get float32 precision.
        assert state["float32_matmul_precision"] == "high"
