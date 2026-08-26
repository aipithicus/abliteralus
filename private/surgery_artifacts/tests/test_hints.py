from __future__ import annotations

import torch
from abliteralus.abliterate import AbliterationPipeline
from abliteralus.artifact_contracts import capture_mutation_hints
from surgery_artifacts.hints import ProjectionHintRecorder
from torch import nn


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.q_proj = nn.Linear(4, 4, bias=False)


def test_projection_hint_recorder_uses_stable_state_dict_name() -> None:
    model = TinyModel()
    recorder = ProjectionHintRecorder()
    recorder.bind_model(model)

    recorder.record_projection(
        model.block.q_proj,
        "weight",
        torch.ones(4),
        norm_preserve=False,
        regularization=0.1,
        projection_row_fraction=0.5,
        max_norm_ratio=1.1,
    )

    (hint,) = recorder.snapshot()
    assert hint.parameter == "block.q_proj.weight"
    assert hint.direction.device.type == "cpu"
    assert hint.projection_row_fraction == 0.5


def test_pipeline_projection_emits_hint_after_commit() -> None:
    model = TinyModel()
    recorder = ProjectionHintRecorder()
    recorder.bind_model(model)
    direction = torch.arange(1, 5, dtype=torch.float32)

    with capture_mutation_hints(recorder):
        count = AbliterationPipeline._project_out_advanced(
            model.block,
            direction,
            ["q_proj"],
            norm_preserve=True,
            regularization=0.05,
        )

    assert count == 1
    (hint,) = recorder.snapshot()
    assert hint.parameter == "block.q_proj.weight"
    assert hint.norm_preserve is True
