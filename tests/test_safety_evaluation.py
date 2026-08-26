"""CPU contracts for guard-model chat formatting and verdict evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from abliteralus.chat_templates import render_chat_prompt
from abliteralus.safety_evaluation import (
    evaluate_safety_checkpoint,
    parse_safety_completion,
    summarize_safety_rows,
)


pytestmark = pytest.mark.cpu


class _TypedContentTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        content = messages[0]["content"]
        if isinstance(content, str):
            return "conversation omitted"
        return f"guard prompt: {content[0]['text']}"

    def __call__(self, text, *, return_tensors):
        assert text.startswith("guard prompt:")
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 1]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"safe": [2], "unsafe": [3]}[text]

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        assert token_ids.tolist() == [4, 3]
        return "unsafe\nS2, S9"


class _GuardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(self, *, input_ids, **_kwargs):
        formatting_scores = torch.zeros((1, 8))
        formatting_scores[0, 2] = 5.0
        formatting_scores[0, 3] = 1.0
        label_scores = torch.zeros((1, 8))
        label_scores[0, 2] = 1.0
        label_scores[0, 3] = 3.0
        return SimpleNamespace(
            sequences=torch.cat((input_ids, torch.tensor([[4, 3]])), dim=1),
            scores=(formatting_scores, label_scores),
        )


class _Handle:
    def __init__(self):
        self.model = _GuardModel()
        self.tokenizer = _TypedContentTokenizer()
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


def test_chat_renderer_falls_back_to_typed_text_content():
    rendered = render_chat_prompt(_TypedContentTokenizer(), "inspect this")

    assert rendered == "guard prompt: inspect this"


def test_safety_completion_parser_reads_verdict_and_categories():
    result = parse_safety_completion("unsafe\nS9, S2")

    assert result["verdict"] == "unsafe"
    assert result["categories"] == ["S2", "S9"]
    assert result["parsed"] is True


def test_safety_summary_separates_safe_and_unsafe_accuracy():
    summary = summarize_safety_rows(
        [
            {
                "expected": "safe",
                "verdict": "safe",
                "correct": True,
                "unsafe_minus_safe_logit_margin": -2.0,
            },
            {
                "expected": "unsafe",
                "verdict": "safe",
                "correct": False,
                "unsafe_minus_safe_logit_margin": -1.0,
            },
        ]
    )

    assert summary["accuracy"] == 0.5
    assert summary["by_expected"]["safe"]["accuracy"] == 1.0
    assert summary["by_expected"]["unsafe"]["accuracy"] == 0.0
    assert summary["mean_unsafe_minus_safe_logit_margin"] == -1.5


def test_checkpoint_evaluation_records_emitted_label_margin(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    handle = _Handle()
    monkeypatch.setattr(
        "abliteralus.models.loader.load_model",
        lambda **_kwargs: handle,
    )

    result = evaluate_safety_checkpoint(
        checkpoint,
        cases=[{"name": "unsafe-one", "prompt": "inspect this", "expected": "unsafe"}],
        device="cpu",
        dtype="float32",
        trust_remote_code=False,
        max_new_tokens=8,
        safe_label="safe",
        unsafe_label="unsafe",
    )

    row = result["rows"][0]
    assert row["verdict"] == "unsafe"
    assert row["correct"] is True
    assert row["label_token_index"] == 1
    assert row["unsafe_minus_safe_logit_margin"] == 2.0
    assert result["summary"]["accuracy"] == 1.0
    assert handle.cleaned is True
