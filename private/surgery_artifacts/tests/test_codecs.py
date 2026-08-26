from __future__ import annotations

import torch
from abliteralus.analysis.numerical_contracts import project_weight_against_direction
from surgery_artifacts.codecs import (
    apply_operation,
    encode_projection_hint,
    encode_tensor_change,
)


def test_low_rank_codec_replays_bit_exactly() -> None:
    generator = torch.Generator().manual_seed(7)
    before = torch.randn((16, 12), generator=generator, dtype=torch.float32)
    direction = torch.randn((12,), generator=generator, dtype=torch.float32)
    after = project_weight_against_direction(
        before, direction, norm_preserve=False, regularization=0.2
    ).weight

    operation = encode_projection_hint(
        "model.layers.0.q_proj.weight",
        before,
        {
            "direction": direction,
            "norm_preserve": False,
            "regularization": 0.2,
            "projection_row_fraction": 1.0,
            "max_norm_ratio": 1.1,
        },
    )

    assert operation is not None
    assert operation.kind == "project_direction"
    assert operation.metadata["peft_compatible"] is True
    replayed = apply_operation(operation.semantic_record(), before, operation.tensors)
    assert torch.equal(replayed, after)


def test_sparse_row_codec_is_selected_when_smaller() -> None:
    before = torch.zeros((16, 16), dtype=torch.float16)
    after = before.clone()
    after[3] = torch.arange(16, dtype=torch.float16)

    operation = encode_tensor_change("model.layers.0.mlp.weight", before, after)

    assert operation.kind == "add_sparse_rows"
    assert torch.equal(
        apply_operation(operation.semantic_record(), before, operation.tensors),
        after,
    )


def test_norm_preserving_projection_is_exact_native_but_not_peft() -> None:
    generator = torch.Generator().manual_seed(19)
    before = torch.randn((10, 8), generator=generator, dtype=torch.float16)
    direction = torch.randn((8,), generator=generator, dtype=torch.float16)
    after = project_weight_against_direction(
        before, direction, norm_preserve=True, regularization=0.05
    ).weight

    operation = encode_projection_hint(
        "model.layers.0.o_proj.weight",
        before,
        {
            "direction": direction,
            "norm_preserve": True,
            "regularization": 0.05,
            "projection_row_fraction": 1.0,
            "max_norm_ratio": 1.1,
        },
    )

    assert operation is not None
    assert operation.kind == "project_direction"
    assert operation.metadata["peft_compatible"] is False
    assert torch.equal(
        apply_operation(operation.semantic_record(), before, operation.tensors),
        after,
    )


def test_non_floating_change_uses_exact_replacement() -> None:
    before = torch.tensor([1, 2, 3], dtype=torch.int64)
    after = torch.tensor([1, 8, 3], dtype=torch.int64)

    operation = encode_tensor_change("counter", before, after)

    assert operation.kind == "replace_tensor"
    assert torch.equal(
        apply_operation(operation.semantic_record(), before, operation.tensors),
        after,
    )
