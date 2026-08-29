from __future__ import annotations

import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch
from guard_study.contracts import GuardCase, GuardDatasetContract, GuardPair
from guard_study.interventions import build_contrastive_direction
from guard_study.layer_selection import LayerSelection
from guard_study.errors import StudyRuntimeError
from guard_study.runner import (
    LabelReadout,
    PreparedCase,
    _unnormalized_label_axis,
    _study_hf_home,
    align_cases_to_emitted_label,
    baseline_generation_rows,
    build_causal_map,
    compare_rows,
    evaluate_generations,
    observe_case,
    select_winner,
)


class _ToyLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _ToyGuard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.layers = torch.nn.ModuleList([_ToyLayer(), _ToyLayer()])
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.embedding.weight[0] = torch.tensor([-1.0, 0.0])
            self.embedding.weight[1] = torch.tensor([1.0, 0.0])
            self.lm_head.weight.zero_()
            self.lm_head.weight[2] = torch.tensor([-1.0, 0.0])
            self.lm_head.weight[3] = torch.tensor([1.0, 0.0])

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def get_output_embeddings(self):
        return self.lm_head


class _Tokenizer:
    @staticmethod
    def decode(token_ids, *, skip_special_tokens=False):
        del skip_special_tokens
        vocabulary = {
            0: "safe-prompt",
            1: "unsafe-prompt",
            2: "safe",
            3: "unsafe",
            4: "<format>",
        }
        return "".join(vocabulary[token_id] for token_id in token_ids)


class _FormattingGuard(torch.nn.Module):
    """Toy generator that emits a formatting token before its safety label."""

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros((batch_size, sequence_length, 5))
        if sequence_length == 1:
            logits[:, -1, 4] = 6.0
        else:
            labels = torch.where(input_ids[:, 0] == 0, 2, 3)
            logits[torch.arange(batch_size), -1, labels] = 6.0
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        *,
        max_new_tokens,
        do_sample,
        pad_token_id,
        return_dict_in_generate,
        output_scores,
    ):
        del attention_mask, pad_token_id
        assert max_new_tokens >= 2
        assert do_sample is False
        assert return_dict_in_generate is True
        assert output_scores is True
        label = torch.where(input_ids[:, 0] == 0, 2, 3)
        prefix = torch.full_like(label, 4)
        sequences = torch.cat((input_ids, prefix[:, None], label[:, None]), dim=-1)
        prefix_scores = torch.zeros((input_ids.shape[0], 5))
        prefix_scores[:, 4] = 6.0
        label_scores = torch.zeros((input_ids.shape[0], 5))
        label_scores[torch.arange(input_ids.shape[0]), label] = 6.0
        return SimpleNamespace(sequences=sequences, scores=(prefix_scores, label_scores))


def test_workspace_hf_cache_is_ephemeral_when_not_inherited(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    surgery_spec = tmp_path / "private" / "experiments" / "surgery" / "model.yaml"
    study = SimpleNamespace(surgery_experiment_path=surgery_spec)

    with _study_hf_home(study) as (path, source):
        assert path == tmp_path / ".scratch" / "cache" / "huggingface"
        assert source == "workspace-default"
        assert Path(os.environ["HF_HOME"]) == path

    assert "HF_HOME" not in os.environ


def _contract_and_cases():
    safe = GuardCase("fit-safe", "safe prompt", "safe", "fit-pair", "test")
    unsafe = GuardCase("fit-unsafe", "unsafe prompt", "unsafe", "fit-pair", "test")
    pair = GuardPair("fit-pair", "test", safe, unsafe)
    contract = GuardDatasetContract(
        schema_version=1,
        name="toy",
        description="toy",
        input_role="user",
        safe_label="safe",
        unsafe_label="unsafe",
        content_sha256="0" * 64,
        splits=MappingProxyType({"fit": (pair,), "dev": (pair,), "test": (pair,)}),
        path=Path("toy.yaml"),
    )
    prepared = {
        safe.name: PreparedCase(safe, {"input_ids": torch.tensor([[0]])}),
        unsafe.name: PreparedCase(unsafe, {"input_ids": torch.tensor([[1]])}),
    }
    return contract, pair, prepared


def test_cases_align_to_the_emitted_label_after_a_formatting_prefix():
    _, _, prepared = _contract_and_cases()

    aligned = align_cases_to_emitted_label(
        _FormattingGuard(),
        prepared,
        input_device=torch.device("cpu"),
        safe_token_id=2,
        unsafe_token_id=3,
        max_new_tokens=2,
        pad_token_id=None,
    )

    assert aligned["fit-safe"].decision_prefix_token_ids == (4,)
    assert aligned["fit-safe"].label_token_index == 1
    assert aligned["fit-safe"].baseline_label_token_id == 2
    assert aligned["fit-safe"].inputs["input_ids"].tolist() == [[0, 4]]
    assert aligned["fit-unsafe"].baseline_label_token_id == 3
    assert aligned["fit-unsafe"].inputs["input_ids"].tolist() == [[1, 4]]

    baseline_rows = baseline_generation_rows(
        list(aligned.values()),
        tokenizer=_Tokenizer(),
        safe_token_id=2,
        unsafe_token_id=3,
    )
    generated_rows = evaluate_generations(
        _FormattingGuard(),
        [],
        list(aligned.values()),
        tokenizer=_Tokenizer(),
        input_device=torch.device("cpu"),
        safe_token_id=2,
        unsafe_token_id=3,
        max_new_tokens=2,
        pad_token_id=None,
    )

    assert [row["predicted_label"] for row in baseline_rows] == ["safe", "unsafe"]
    assert [row["predicted_label"] for row in generated_rows] == ["safe", "unsafe"]
    assert generated_rows[0]["completion_token_ids"] == [4, 2]


def test_last_token_causal_mapping_uses_the_guard_margin():
    model = _ToyGuard()
    tokenizer = _Tokenizer()
    contract, pair, prepared = _contract_and_cases()
    observations = {
        name: observe_case(
            model,
            model.layers,
            value,
            tokenizer=tokenizer,
            input_device=torch.device("cpu"),
            safe_token_id=2,
            unsafe_token_id=3,
            capture_layers=(0, 1),
        )
        for name, value in prepared.items()
    }
    directions = {
        layer: build_contrastive_direction(
            layer_idx=layer,
            unsafe_activations=[observations[pair.unsafe.name].activations[layer]],
            safe_activations=[observations[pair.safe.name].activations[layer]],
        )
        for layer in (0, 1)
    }

    result = build_causal_map(
        model,
        model.layers,
        contract,
        prepared,
        observations,
        directions,
        tokenizer=tokenizer,
        input_device=torch.device("cpu"),
        safe_token_id=2,
        unsafe_token_id=3,
        max_patch_pairs=1,
        selection=LayerSelection(strategy="mean_absolute_effect", top_k=1, params={}),
    )

    assert result["top_layers"] in ([0], [1])
    assert result["per_layer"]["0"]["mean_bidirectional_effect"] == pytest.approx(4.0)
    assert result["per_layer"]["1"]["mean_bidirectional_effect"] == pytest.approx(4.0)


def test_comparison_and_selection_use_paired_margin_deltas():
    baseline = [
        {
            "name": "safe",
            "expected": "safe",
            "unsafe_minus_safe_logit_margin": -2.0,
            "parsed": True,
            "correct": True,
        },
        {
            "name": "unsafe",
            "expected": "unsafe",
            "unsafe_minus_safe_logit_margin": 2.0,
            "parsed": True,
            "correct": True,
        },
    ]
    shifted = [dict(row) for row in baseline]
    for row in shifted:
        row["unsafe_minus_safe_logit_margin"] -= 0.5
    comparison = compare_rows(baseline, shifted)
    sweep = {
        "arm": "learned.layer_1",
        "kind": "learned",
        "layers": [1],
        "dose": -0.5,
        "summary": comparison["intervention_summary"],
        "comparison": comparison,
    }

    selection = select_winner(
        [sweep],
        target_mean_delta=-0.25,
        min_parse_rate=1.0,
        min_gap_ratio=0.8,
    )

    assert comparison["mean_delta"] == pytest.approx(-0.5)
    assert comparison["fraction_down"] == 1.0
    assert selection["met_contract"] is True


class _PrecisionToyGuard(torch.nn.Module):
    """A guard whose head rounds to bfloat16 while its residual stays exact.

    This is the shape of the real failure: the margin is a difference of two
    similar logits, so the head's rounding lands on a grid far coarser than the
    effect being measured.
    """

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_ToyLayer()])
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[2] = torch.tensor([6.5, 0.0])
            self.lm_head.weight[3] = torch.tensor([6.5, 0.0])
            self.lm_head.weight[3, 1] = 0.05

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_hidden_states=False):
        del attention_mask, use_cache
        hidden = torch.ones((1, int(input_ids.shape[-1]), 2))
        for layer in self.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden).to(torch.bfloat16).float()
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
        )

    def get_output_embeddings(self):
        return self.lm_head


def _precision_case() -> PreparedCase:
    case = GuardCase(
        name="probe",
        prompt="probe",
        expected="unsafe",
        pair_id="probe",
        category="probe",
    )
    return PreparedCase(
        case=case,
        prompt_inputs=MappingProxyType({"input_ids": torch.tensor([[1]])}),
    )


def test_projected_readout_recovers_a_margin_the_head_quantizes_away():
    model = _PrecisionToyGuard()
    prepared = _precision_case()
    axis = _unnormalized_label_axis(model, safe_token_id=2, unsafe_token_id=3)
    readout = LabelReadout(axis, dtype=torch.float32)

    observation = observe_case(
        model,
        [],
        prepared,
        tokenizer=_Tokenizer(),
        input_device=torch.device("cpu"),
        safe_token_id=2,
        unsafe_token_id=3,
        readout=readout,
    )

    # The true margin is 0.05; the bfloat16 head cannot represent it at logit 6.5.
    assert observation.row["unsafe_minus_safe_logit_margin"] == pytest.approx(0.05, abs=1e-6)
    assert observation.row["head_logit_margin"] != pytest.approx(0.05, abs=1e-3)
    assert observation.final_hidden is not None


def test_readout_verification_rejects_a_hidden_state_that_is_not_the_residual():
    model = _PrecisionToyGuard()
    axis = _unnormalized_label_axis(model, safe_token_id=2, unsafe_token_id=3)
    readout = LabelReadout(axis, dtype=torch.float32)

    readout.verify(torch.ones(2), head_margin=0.05, tolerance=0.01)
    assert readout.verified is True

    with pytest.raises(StudyRuntimeError, match="disagrees with the model head"):
        readout.verify(torch.tensor([9.0, 9.0]), head_margin=0.05, tolerance=0.01)
