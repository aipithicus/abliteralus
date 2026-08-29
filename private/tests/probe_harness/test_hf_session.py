from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch
from torch import nn

from probe_harness.cli import ProbeController
from probe_harness.direction_bundle import DirectionBundle
from probe_harness.hf_subject import HFSubject, _resolve_checkpoint
from probe_harness.session_store import SessionStore
from guard_study.contracts import GuardCase
from guard_study.runner import PreparedCase, observe_case


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(
        self,
        conversation,
        *,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        content = conversation[0]["content"]
        if isinstance(content, list):
            return content[0]["text"]
        return content

    def __call__(self, text, *, return_tensors):
        assert return_tensors == "pt"
        if "sequence" in text.casefold():
            tokens = [4, 5]
        else:
            tokens = [5 if "unsafe" in text.casefold() else 4]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.int64),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.int64),
        }

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"safe": [1], "unsafe": [2]}[text]

    def decode(self, token_ids, *, skip_special_tokens=False):
        values = {0: "", 1: "safe", 2: "unsafe", 3: "S2", 4: "safe-prompt", 5: "unsafe-prompt"}
        return "\n".join(values.get(int(token), f"<{token}>") for token in token_ids).strip()


class FakeBlock(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, hidden):
        return (hidden + self.offset,)


class FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeBlock(0.05), FakeBlock(0.05)])


class FakeGuardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = FakeCore()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        sign = torch.where(input_ids == 5, 1.0, -1.0)
        hidden = torch.stack((sign, sign * 2.0, torch.ones_like(sign)), dim=-1)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        logits[..., 1] = -3.0 * hidden[..., 0]
        logits[..., 2] = 3.0 * hidden[..., 0]
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
        del max_new_tokens, do_sample, pad_token_id, return_dict_in_generate, output_scores
        output = self.forward(input_ids, attention_mask=attention_mask)
        label_scores = output.logits[:, -1, :]
        label = torch.argmax(label_scores, dim=-1, keepdim=True)
        if int(label[0, 0]) == 2:
            category = torch.tensor([[3]], dtype=input_ids.dtype)
            category_scores = torch.zeros_like(label_scores)
            category_scores[:, 3] = 5.0
            completion = torch.cat((label, category), dim=-1)
            scores = (label_scores, category_scores)
        else:
            completion = label
            scores = (label_scores,)
        return SimpleNamespace(
            sequences=torch.cat((input_ids, completion), dim=-1),
            scores=scores,
        )


def _subject(*, checkpoint_path: str | None = None) -> HFSubject:
    return HFSubject(
        model=FakeGuardModel(),
        tokenizer=FakeTokenizer(),
        identity={
            "kind": "test-causal-lm",
            "name": "fake-guard",
            "source": "fixture",
            "revision": "one",
            "checkpoint_path": checkpoint_path or str(Path(__file__).resolve()),
            "device": "cpu",
            "dtype": "float32",
            "hidden_size": 3,
        },
        max_seq_length=32,
        max_new_tokens=4,
        safe_label="safe",
        unsafe_label="unsafe",
    )


def test_hf_subject_generates_and_exact_token_inspection_captures_layers() -> None:
    subject = _subject()

    trial = subject.generate("unsafe request")
    observation, tensors = subject.inspect(trial, capture_layers=[0, 1])

    assert trial["completion_text"] == "unsafe\nS2"
    assert trial["emitted_label"] == "unsafe"
    assert trial["emitted_category"] == "S2"
    assert trial["generated_unsafe_minus_safe_logit_margin"] > 0
    assert observation["predicted_label"] == "unsafe"
    assert observation["generated_margin_alignment_error"] == 0.0
    assert set(tensors) == {"layer.0.decision", "layer.1.decision"}
    assert tensors["layer.1.decision"].shape == (3,)
    assert all(not layer._forward_hooks for layer in subject.layers)


def test_hf_subject_all_position_capture_stays_token_aligned() -> None:
    subject = _subject()

    trial = subject.generate("sequence unsafe request")
    observation, tensors = subject.inspect(
        trial,
        capture_layers=[1],
        positions="all",
    )

    assert tensors["layer.1.all"].shape == (2, 3)
    assert [value["token_id"] for value in observation["token_positions"]] == [4, 5]
    assert [value["position"] for value in observation["token_positions"]] == [0, 1]
    assert observation["positions"] == "all"
    assert observation["logits"]["unsafe"]["logit"] > observation["logits"]["safe"]["logit"]


def test_decision_observation_agrees_with_guard_study_observe_case() -> None:
    subject = _subject()
    trial = subject.generate("unsafe request")
    observed, tensors = subject.inspect(trial, capture_layers=[1], positions="decision")
    inputs = {
        name: torch.tensor(value, dtype=torch.int64)
        for name, value in trial["prompt_inputs"].items()
    }
    prepared = PreparedCase(
        case=GuardCase(
            name="fixture",
            prompt="unsafe request",
            expected="unsafe",
            pair_id="fixture-pair",
            category="fixture",
        ),
        prompt_inputs=inputs,
        decision_inputs=inputs,
        baseline_label_token_id=2,
        baseline_generated_margin=trial["generated_unsafe_minus_safe_logit_margin"],
    )

    reference = observe_case(
        subject.model,
        subject.layers,
        prepared,
        tokenizer=subject.tokenizer,
        input_device=subject.input_device,
        safe_token_id=subject.safe_token_id,
        unsafe_token_id=subject.unsafe_token_id,
        capture_layers=[1],
    )

    assert observed["unsafe_minus_safe_logit_margin"] == pytest.approx(
        reference.row["unsafe_minus_safe_logit_margin"]
    )
    assert torch.equal(tensors["layer.1.decision"], reference.activations[1])


def test_session_resume_annotations_replay_and_observation_artifact(tmp_path) -> None:
    subject = _subject()
    session = SessionStore.create(
        tmp_path / "sessions",
        subject=subject.identity,
        settings={"history_mode": "independent", "schema_version": 1},
    )
    controller = ProbeController(subject, session)

    original = controller.submit("unsafe request")
    controller.annotate_current(note="boundary case")
    controller.annotate_current(tag="interesting")
    controller.annotate_current(marked=True)
    observation = controller.inspect(original.trial_id, layers=[1])
    artifact = controller.save_activations(observation["observation_id"])
    replay = controller.replay(original.trial_id)

    assert replay.payload["replay_of"] == original.trial_id
    assert replay.payload["parent_trial_id"] == original.trial_id
    assert replay.payload["prompt_inputs"] == original.payload["prompt_inputs"]
    assert (session.directory / artifact["path"]).is_file()

    resumed = SessionStore.resume(
        session.directory,
        expected_subject_digest=subject.identity["digest"],
    )
    state = resumed.state()
    annotated = state.resolve_trial(original.trial_id)
    assert annotated.notes == ["boundary case"]
    assert annotated.tags == {"interesting"}
    assert annotated.marked is True
    assert annotated.observations[0]["observation_id"] == observation["observation_id"]
    assert annotated.observations[0]["activation_artifact"] == artifact
    assert len(state.trials) == 2
    assert (session.directory / "events.jsonl.transactions.jsonl").is_file()


def test_activation_artifacts_are_reproducibly_content_addressed(tmp_path) -> None:
    subject = _subject()
    session = SessionStore.create(
        tmp_path / "sessions",
        subject=subject.identity,
        settings={"history_mode": "independent", "schema_version": 1},
    )
    controller = ProbeController(subject, session)
    trial = controller.submit("unsafe request")

    first = controller.inspect(trial.trial_id, layers=[1])
    first_artifact = controller.save_activations(first["observation_id"])
    second = controller.inspect(trial.trial_id, layers=[1])
    second_artifact = controller.save_activations(second["observation_id"])

    assert first_artifact["sha256"] == second_artifact["sha256"]
    assert first_artifact["path"] == second_artifact["path"]


def test_checkpoint_resolution_uses_explicit_repository_hub_cache(monkeypatch, tmp_path) -> None:
    captured = {}
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(snapshot)

    import huggingface_hub

    monkeypatch.setenv("HF_HOME", str(tmp_path / "ambient-must-not-be-used"))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved = _resolve_checkpoint(
        source="example/subject",
        revision="pinned",
        repository=tmp_path / "repository",
        cache_root=tmp_path / "repository" / ".scratch" / "cache" / "huggingface",
        allow_download=False,
    )

    assert resolved == snapshot.resolve()
    assert captured["cache_dir"] == str(
        tmp_path / "repository" / ".scratch" / "cache" / "huggingface" / "hub"
    )
    assert captured["local_files_only"] is True
    assert captured["token"] is False


def test_inspection_removes_hooks_when_the_model_forward_fails() -> None:
    subject = _subject()
    trial = subject.generate("unsafe request")

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("planted failure")

    subject.model.forward = fail_forward

    with pytest.raises(RuntimeError, match="planted failure"):
        subject.inspect(trial, capture_layers=[0, 1])
    assert all(not layer._forward_hooks for layer in subject.layers)


def _write_direction_run(
    root: Path,
    *,
    checkpoint_path: str,
    dataset_digest: str = "a" * 64,
    metric: str = "unsafe_minus_safe_logit_margin",
    hidden_size: int = 3,
) -> Path:
    run = root / "guard-run"
    run.mkdir()
    (run / "run-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "weights_mutated": False,
                "checkpoint": checkpoint_path,
                "dataset": {"content_sha256": dataset_digest},
            }
        ),
        encoding="utf-8",
    )
    (run / "dataset-contract.json").write_text(
        json.dumps({"schema_version": 1, "content_sha256": dataset_digest}),
        encoding="utf-8",
    )
    per_layer = {
        str(layer): {
            "projection_std": 2.0,
            "safe_projection_mean": -1.0,
            "unsafe_projection_mean": 1.0,
            "projection_gap": 2.0,
        }
        for layer in range(2)
    }
    (run / "causal-map.json").write_text(
        json.dumps({"metric": metric, "per_layer": per_layer}),
        encoding="utf-8",
    )
    vectors = {
        "layer.0.direction": torch.nn.functional.pad(
            torch.tensor([1.0]),
            (0, hidden_size - 1),
        ),
        "layer.1.direction": torch.nn.functional.pad(
            torch.tensor([0.0, 1.0]),
            (0, hidden_size - 2),
        ),
        "control.label_axis.direction": torch.ones(hidden_size),
    }
    save_file(
        vectors,
        str(run / "directions.safetensors"),
        metadata={"dataset_sha256": dataset_digest, "metric": metric},
    )
    return run


def test_direction_bundle_load_projection_and_session_provenance(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    subject = _subject(checkpoint_path=str(checkpoint))
    run = _write_direction_run(tmp_path, checkpoint_path=str(checkpoint))
    bundle = DirectionBundle.load(run, subject_identity=subject.identity)

    assert bundle.resolve("learned.layer_1").layer_index == 1
    direct = bundle.project("learned.layer_1", torch.tensor([1.0, 4.0, 0.0]))
    assert direct["raw_projection_last"] == pytest.approx(4.0)
    assert direct["profile"][0]["midpoint_standardized_projection"] == pytest.approx(2.0)

    session = SessionStore.create(
        tmp_path / "sessions",
        subject=subject.identity,
        settings={"history_mode": "independent", "schema_version": 1},
    )
    controller = ProbeController(subject, session)
    trial = controller.submit("sequence unsafe request")
    inspected = controller.inspect(trial.trial_id, layers=[1], positions="all")
    loaded = controller.load_directions(run)
    projected = controller.project("learned.layer_1")

    assert loaded["digest"] == bundle.identity["digest"]
    assert projected["evidence_kind"] == "geometric-estimate"
    assert projected["parent_observation_id"] == inspected["observation_id"]
    assert [value["token_id"] for value in projected["projection"]["profile"]] == [4, 5]
    state = session.state()
    assert state.direction_bundles[-1]["digest"] == bundle.identity["digest"]
    assert len(state.observations) == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("checkpoint", "checkpoint identity"),
        ("dataset", "dataset digest"),
        ("metric", "metric must be"),
        ("hidden", "hidden size"),
    ],
)
def test_direction_bundle_refuses_subject_and_provenance_mismatches(
    tmp_path,
    mutation,
    message,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    subject = _subject(checkpoint_path=str(checkpoint))
    run = _write_direction_run(tmp_path, checkpoint_path=str(checkpoint))

    if mutation == "checkpoint":
        subject.identity["checkpoint_path"] = str(tmp_path / "different-checkpoint")
    elif mutation == "dataset":
        manifest = json.loads((run / "run-manifest.json").read_text(encoding="utf-8"))
        manifest["dataset"]["content_sha256"] = "b" * 64
        (run / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "metric":
        causal = json.loads((run / "causal-map.json").read_text(encoding="utf-8"))
        causal["metric"] = "other"
        (run / "causal-map.json").write_text(json.dumps(causal), encoding="utf-8")
    else:
        subject.identity["hidden_size"] = 4

    with pytest.raises(Exception, match=message):
        DirectionBundle.load(run, subject_identity=subject.identity)
