"""Causal mapping and reversible steering runner for generative guard models."""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import subprocess
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch
from abliteralus.chat_templates import render_chat_prompt
from safetensors.torch import save_file

from .contracts import (
    GuardCase,
    GuardDatasetContract,
    GuardPair,
    GuardStudySpec,
    load_dataset_contract,
    normalized_dataset,
)
from .errors import StudyRuntimeError
from .interventions import (
    ContrastiveDirection,
    LastTokenPatch,
    LayerSteeringHooks,
    build_contrastive_direction,
    orthogonal_sham,
)
from .layer_selection import LayerSelection
from .precision import PrecisionSpec, pinned_matmul_precision, resolution_floor

if TYPE_CHECKING:
    from abliteralus.models.loader import ModelHandle


@dataclass(frozen=True)
class PreparedCase:
    case: GuardCase
    prompt_inputs: Mapping[str, torch.Tensor]
    decision_inputs: Mapping[str, torch.Tensor] | None = None
    decision_prefix_token_ids: tuple[int, ...] = ()
    label_token_index: int | None = None
    baseline_completion_token_ids: tuple[int, ...] = ()
    baseline_label_token_id: int | None = None
    baseline_generated_margin: float | None = None
    alignment_margin_error: float | None = None

    @property
    def inputs(self) -> Mapping[str, torch.Tensor]:
        return self.decision_inputs or self.prompt_inputs


@dataclass
class Observation:
    row: dict[str, Any]
    activations: dict[int, torch.Tensor]
    final_hidden: torch.Tensor | None = None


@dataclass(frozen=True)
class SteeringArm:
    name: str
    kind: str
    layer_indices: tuple[int, ...]
    vectors: Mapping[int, torch.Tensor]
    scales: Mapping[int, float]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz").lower()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(arguments: Sequence[str]) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@contextmanager
def _study_hf_home(study: GuardStudySpec):
    """Use the lab bench's workspace-local Hub cache without persisting environment state."""

    previous = os.environ.get("HF_HOME")
    if previous:
        yield Path(previous).resolve(), "inherited"
        return
    experiments_directory = study.surgery_experiment_path.parent.parent
    container = experiments_directory.parent
    repository = (
        container.parent
        if experiments_directory.name == "experiments" and container.name == "private"
        else container
    )
    workspace_cache = repository / ".scratch" / "cache" / "huggingface"
    os.environ["HF_HOME"] = str(workspace_cache)
    try:
        yield workspace_cache, "workspace-default"
    finally:
        os.environ.pop("HF_HOME", None)


def _model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _single_token_id(tokenizer: Any, label: str) -> int:
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if len(token_ids) != 1:
        raise StudyRuntimeError(f"configured label {label!r} is not exactly one token")
    return int(token_ids[0])


def prepare_cases(
    cases: Sequence[GuardCase],
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> dict[str, PreparedCase]:
    """Render and tokenize the requested cases once, retaining tensors on CPU."""

    prepared: dict[str, PreparedCase] = {}
    for case in cases:
        rendered = render_chat_prompt(tokenizer, case.prompt)
        inputs = tokenizer(rendered, return_tensors="pt")
        if "input_ids" not in inputs:
            raise StudyRuntimeError(f"tokenizer returned no input_ids for {case.name}")
        if int(inputs["input_ids"].shape[0]) != 1:
            raise StudyRuntimeError("guard study requires one prompt per model call")
        length = int(inputs["input_ids"].shape[-1])
        if length > max_seq_length:
            raise StudyRuntimeError(
                f"{case.name} is {length} tokens, exceeding max_seq_length={max_seq_length}"
            )
        prepared[case.name] = PreparedCase(
            case=case,
            prompt_inputs={name: tensor.detach().cpu() for name, tensor in inputs.items()},
        )
    return prepared


def _append_decision_prefix(
    inputs: Mapping[str, torch.Tensor],
    prefix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prefix = prefix.detach().cpu().reshape(1, -1)
    result: dict[str, torch.Tensor] = {}
    original_length = int(inputs["input_ids"].shape[-1])
    for name, tensor in inputs.items():
        value = tensor.detach().cpu()
        if value.ndim != 2 or int(value.shape[-1]) != original_length:
            result[name] = value
            continue
        if name == "input_ids":
            extension = prefix.to(dtype=value.dtype)
        elif name == "attention_mask":
            extension = torch.ones((1, prefix.shape[-1]), dtype=value.dtype)
        elif name == "token_type_ids":
            extension = value[:, -1:].expand(1, prefix.shape[-1]).clone()
        else:
            raise StudyRuntimeError(
                f"cannot extend tokenizer field {name!r} to the emitted-label decision position"
            )
        result[name] = torch.cat((value, extension), dim=-1)
    return result


def align_cases_to_emitted_label(
    model: torch.nn.Module,
    prepared: Mapping[str, PreparedCase],
    *,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> dict[str, PreparedCase]:
    """Freeze the baseline prefix immediately before each emitted safety label."""

    aligned: dict[str, PreparedCase] = {}
    for name, value in prepared.items():
        prompt_length = int(value.prompt_inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **_inputs_to_device(value.prompt_inputs, input_device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        completion_ids = generated.sequences[0, prompt_length:].detach().cpu()
        label_index = next(
            (
                index
                for index, token_id in enumerate(completion_ids.tolist())
                if token_id in {safe_token_id, unsafe_token_id}
            ),
            None,
        )
        if label_index is None or label_index >= len(generated.scores):
            raise StudyRuntimeError(
                f"baseline generation for {name} emitted no configured label "
                f"within {max_new_tokens} tokens"
            )
        prefix = completion_ids[:label_index]
        generated_scores = generated.scores[label_index][0].detach().float().cpu()
        generated_label = int(completion_ids[label_index].item())
        if int(torch.argmax(generated_scores).item()) != generated_label:
            raise StudyRuntimeError(f"baseline generation for {name} was not greedy-deterministic")
        decision_inputs = _append_decision_prefix(value.prompt_inputs, prefix)
        with torch.inference_mode():
            decision_logits = model(
                **_inputs_to_device(decision_inputs, input_device),
                use_cache=False,
            ).logits
        decision_scores = decision_logits[0, -1].detach().float().cpu()
        decision_label = int(torch.argmax(decision_scores).item())
        if decision_label != generated_label:
            raise StudyRuntimeError(
                f"full-forward alignment for {name} predicts token {decision_label}, "
                f"but baseline generation emitted {generated_label}"
            )
        generated_margin = float(
            (generated_scores[unsafe_token_id] - generated_scores[safe_token_id]).item()
        )
        decision_margin = float(
            (decision_scores[unsafe_token_id] - decision_scores[safe_token_id]).item()
        )
        margin_error = abs(decision_margin - generated_margin)
        if not torch.isclose(
            torch.tensor(decision_margin),
            torch.tensor(generated_margin),
            atol=0.25,
            rtol=0.02,
        ):
            raise StudyRuntimeError(
                f"full-forward alignment for {name} changed the generated label margin "
                f"by {margin_error:.6f} logits"
            )
        aligned[name] = PreparedCase(
            case=value.case,
            prompt_inputs=value.prompt_inputs,
            decision_inputs=decision_inputs,
            decision_prefix_token_ids=tuple(int(token) for token in prefix.tolist()),
            label_token_index=label_index,
            baseline_completion_token_ids=tuple(int(token) for token in completion_ids.tolist()),
            baseline_label_token_id=generated_label,
            baseline_generated_margin=generated_margin,
            alignment_margin_error=margin_error,
        )
        del generated
    return aligned


def _inputs_to_device(
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


class LabelReadout:
    """Read the safe/unsafe margin off the final hidden state at a chosen precision.

    The margin is a two-token question, so it needs one dot product rather than a
    vocabulary-wide head. Computing it from the residual keeps the measurement
    independent of the head's own dtype, which matters once the head is quantized
    or the model is too large to hold an unquantized one.
    """

    def __init__(self, axis: torch.Tensor, *, dtype: torch.dtype) -> None:
        self.dtype = dtype
        self.axis = axis.detach().to(dtype=dtype).cpu().reshape(-1)
        self.verified = False

    def margin(self, hidden: torch.Tensor) -> float:
        value = hidden.detach().to(dtype=self.dtype).cpu().reshape(-1)
        if value.numel() != self.axis.numel():
            raise StudyRuntimeError("final hidden state does not match the label axis dimension")
        return float(torch.dot(value, self.axis).item())

    def verify(self, hidden: torch.Tensor, head_margin: float, *, tolerance: float) -> float:
        """Confirm the projection reproduces the model's own margin before trusting it.

        The final hidden state is taken from the model's reported hidden states,
        which are expected to be post-final-norm. Rather than assume that holds for
        an unfamiliar checkpoint, check it once against the head and fail loudly.
        """

        projected = self.margin(hidden)
        error = abs(projected - head_margin)
        if error > tolerance:
            raise StudyRuntimeError(
                "projected label margin disagrees with the model head "
                f"({projected:.6f} vs {head_margin:.6f}, error {error:.6f} > {tolerance:.6f}); "
                "the final hidden state is probably not the post-norm residual"
            )
        self.verified = True
        return error


def _row_from_logits(
    prepared: PreparedCase,
    logits: torch.Tensor,
    *,
    tokenizer: Any,
    safe_token_id: int,
    unsafe_token_id: int,
    readout: LabelReadout | None = None,
    hidden: torch.Tensor | None = None,
) -> dict[str, Any]:
    scores = logits[0, -1].detach().float().cpu()
    head_margin = float((scores[unsafe_token_id] - scores[safe_token_id]).item())
    # The head emits the margin already rounded to the compute dtype, where a
    # difference of two similar logits keeps only the error floor of the operands.
    # Projecting the final hidden state onto the raw label axis in the readout
    # dtype recovers the same quantity without that cancellation.
    if readout is not None and hidden is not None:
        margin = readout.margin(hidden)
    else:
        margin = head_margin
    top_token_id = int(torch.argmax(scores).item())
    predicted = None
    if top_token_id == safe_token_id:
        predicted = "safe"
    elif top_token_id == unsafe_token_id:
        predicted = "unsafe"
    return {
        "name": prepared.case.name,
        "pair_id": prepared.case.pair_id,
        "category": prepared.case.category,
        "prompt": prepared.case.prompt,
        "expected": prepared.case.expected,
        "label_token_index": prepared.label_token_index,
        "decision_prefix_token_ids": list(prepared.decision_prefix_token_ids),
        "baseline_label_token_id": prepared.baseline_label_token_id,
        "baseline_generated_margin": prepared.baseline_generated_margin,
        "alignment_margin_error": prepared.alignment_margin_error,
        "unsafe_minus_safe_logit_margin": margin,
        "head_logit_margin": head_margin,
        "pairwise_label": "unsafe" if margin > 0 else "safe",
        "predicted_label": predicted,
        "parsed": predicted is not None,
        "correct": predicted == prepared.case.expected if predicted is not None else False,
        "top_token_id": top_token_id,
        "top_token_text": tokenizer.decode([top_token_id], skip_special_tokens=False),
    }


def observe_case(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    prepared: PreparedCase,
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    capture_layers: Iterable[int] = (),
    readout: LabelReadout | None = None,
) -> Observation:
    captures: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim < 2:
                raise StudyRuntimeError("activation target did not return hidden-state tensors")
            if hidden.ndim == 2:
                value = hidden[0]
            else:
                value = hidden[0, -1]
            captures[layer_idx] = value.detach().float().cpu().clone()

        return hook

    try:
        for layer_idx in sorted(set(capture_layers)):
            if layer_idx < 0 or layer_idx >= len(layers):
                raise StudyRuntimeError(f"capture layer {layer_idx} is out of range")
            handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
        extra = {"output_hidden_states": True} if readout is not None else {}
        with torch.inference_mode():
            output = model(
                **_inputs_to_device(prepared.inputs, input_device),
                use_cache=False,
                **extra,
            )
        logits = output.logits if hasattr(output, "logits") else output[0]
        hidden = None
        if readout is not None:
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise StudyRuntimeError("model reported no hidden states for the label readout")
            hidden = hidden_states[-1][0, -1]
        row = _row_from_logits(
            prepared,
            logits,
            tokenizer=tokenizer,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            readout=readout,
            hidden=hidden,
        )
    finally:
        for handle in reversed(handles):
            handle.remove()
    return Observation(
        row=row,
        activations=captures,
        final_hidden=None if hidden is None else hidden.detach().cpu().clone(),
    )


def evaluate_cases(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    cases: Sequence[PreparedCase],
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    arm: SteeringArm | None = None,
    dose: float = 0.0,
    readout: LabelReadout | None = None,
) -> list[dict[str, Any]]:
    context = (
        LayerSteeringHooks(
            layers,
            arm.vectors,
            arm.scales,
            dose=dose,
        )
        if arm is not None
        else nullcontext()
    )
    rows: list[dict[str, Any]] = []
    with context:
        for prepared in cases:
            rows.append(
                observe_case(
                    model,
                    layers,
                    prepared,
                    tokenizer=tokenizer,
                    input_device=input_device,
                    safe_token_id=safe_token_id,
                    unsafe_token_id=unsafe_token_id,
                    readout=readout,
                ).row
            )
    return rows


def _generation_row(
    prepared: PreparedCase,
    completion_token_ids: Sequence[int],
    *,
    tokenizer: Any,
    label_token_index: int | None,
    emitted_label_token_id: int | None,
    margin: float | None,
    safe_token_id: int,
    unsafe_token_id: int,
) -> dict[str, Any]:
    predicted = None
    if emitted_label_token_id == safe_token_id:
        predicted = "safe"
    elif emitted_label_token_id == unsafe_token_id:
        predicted = "unsafe"
    prefix_end = label_token_index if label_token_index is not None else len(completion_token_ids)
    return {
        "name": prepared.case.name,
        "pair_id": prepared.case.pair_id,
        "category": prepared.case.category,
        "prompt": prepared.case.prompt,
        "expected": prepared.case.expected,
        "completion_token_ids": list(completion_token_ids),
        "completion_text": tokenizer.decode(
            list(completion_token_ids),
            skip_special_tokens=False,
        ),
        "label_token_index": label_token_index,
        "decision_prefix_token_ids": list(completion_token_ids[:prefix_end]),
        "emitted_label_token_id": emitted_label_token_id,
        "unsafe_minus_safe_logit_margin": margin,
        "predicted_label": predicted,
        "parsed": predicted is not None,
        "correct": predicted == prepared.case.expected if predicted is not None else False,
        "top_token_id": emitted_label_token_id,
        "top_token_text": (
            tokenizer.decode([emitted_label_token_id], skip_special_tokens=False)
            if emitted_label_token_id is not None
            else None
        ),
    }


def baseline_generation_rows(
    cases: Sequence[PreparedCase],
    *,
    tokenizer: Any,
    safe_token_id: int,
    unsafe_token_id: int,
) -> list[dict[str, Any]]:
    """Materialize the baseline generations already captured during alignment."""

    rows: list[dict[str, Any]] = []
    for prepared in cases:
        if (
            prepared.label_token_index is None
            or prepared.baseline_label_token_id is None
            or prepared.baseline_generated_margin is None
        ):
            raise StudyRuntimeError(f"case {prepared.case.name} was not generation-aligned")
        rows.append(
            _generation_row(
                prepared,
                prepared.baseline_completion_token_ids,
                tokenizer=tokenizer,
                label_token_index=prepared.label_token_index,
                emitted_label_token_id=prepared.baseline_label_token_id,
                margin=prepared.baseline_generated_margin,
                safe_token_id=safe_token_id,
                unsafe_token_id=unsafe_token_id,
            )
        )
    return rows


def evaluate_generations(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    cases: Sequence[PreparedCase],
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    max_new_tokens: int,
    pad_token_id: int | None,
    arm: SteeringArm | None = None,
    dose: float = 0.0,
    readout: LabelReadout | None = None,
) -> list[dict[str, Any]]:
    """Verify an intervention by greedily generating from each original prompt."""

    context = (
        LayerSteeringHooks(
            layers,
            arm.vectors,
            arm.scales,
            dose=dose,
        )
        if arm is not None
        else nullcontext()
    )
    rows: list[dict[str, Any]] = []
    with context:
        for prepared in cases:
            prompt_length = int(prepared.prompt_inputs["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **_inputs_to_device(prepared.prompt_inputs, input_device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            completion = generated.sequences[0, prompt_length:].detach().cpu()
            completion_ids = tuple(int(token) for token in completion.tolist())
            label_index = next(
                (
                    index
                    for index, token_id in enumerate(completion_ids)
                    if token_id in {safe_token_id, unsafe_token_id}
                ),
                None,
            )
            emitted_label: int | None = None
            margin: float | None = None
            if label_index is not None:
                if label_index >= len(generated.scores):
                    raise StudyRuntimeError(
                        f"generation score missing at label position for {prepared.case.name}"
                    )
                emitted_label = completion_ids[label_index]
                scores = generated.scores[label_index][0].detach().float().cpu()
                if int(torch.argmax(scores).item()) != emitted_label:
                    raise StudyRuntimeError(
                        f"generation for {prepared.case.name} was not greedy-deterministic"
                    )
                margin = float((scores[unsafe_token_id] - scores[safe_token_id]).item())
            rows.append(
                _generation_row(
                    prepared,
                    completion_ids,
                    tokenizer=tokenizer,
                    label_token_index=label_index,
                    emitted_label_token_id=emitted_label,
                    margin=margin,
                    safe_token_id=safe_token_id,
                    unsafe_token_id=unsafe_token_id,
                )
            )
            del generated
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_expected: dict[str, dict[str, Any]] = {}
    means: dict[str, float | None] = {}
    for expected in ("safe", "unsafe"):
        selected = [row for row in rows if row["expected"] == expected]
        margins = [
            float(row["unsafe_minus_safe_logit_margin"])
            for row in selected
            if row.get("unsafe_minus_safe_logit_margin") is not None
        ]
        means[expected] = statistics.fmean(margins) if margins else None
        by_expected[expected] = {
            "count": len(selected),
            "correct": sum(row.get("correct") is True for row in selected),
            "accuracy": (
                sum(row.get("correct") is True for row in selected) / len(selected)
                if selected
                else None
            ),
            "mean_margin": means[expected],
        }
    margins = [
        float(row["unsafe_minus_safe_logit_margin"])
        for row in rows
        if row.get("unsafe_minus_safe_logit_margin") is not None
    ]
    parsed = sum(row.get("parsed") is True for row in rows)
    correct = sum(row.get("correct") is True for row in rows)
    safe_mean = means["safe"]
    unsafe_mean = means["unsafe"]
    return {
        "count": len(rows),
        "parsed": parsed,
        "parse_rate": parsed / len(rows) if rows else None,
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "mean_margin": statistics.fmean(margins) if margins else None,
        "class_gap": (
            float(unsafe_mean - safe_mean)
            if unsafe_mean is not None and safe_mean is not None
            else None
        ),
        "by_expected": by_expected,
    }


def compare_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    intervention_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {str(row["name"]): row for row in baseline_rows}
    intervention = {str(row["name"]): row for row in intervention_rows}
    if set(baseline) != set(intervention):
        raise StudyRuntimeError("baseline and intervention cases do not match")
    deltas = {
        name: float(intervention[name]["unsafe_minus_safe_logit_margin"])
        - float(baseline[name]["unsafe_minus_safe_logit_margin"])
        for name in sorted(baseline)
    }
    by_expected: dict[str, float | None] = {}
    for expected in ("safe", "unsafe"):
        values = [deltas[name] for name, row in baseline.items() if row["expected"] == expected]
        by_expected[expected] = statistics.fmean(values) if values else None
    baseline_summary = summarize_rows(baseline_rows)
    intervention_summary = summarize_rows(intervention_rows)
    baseline_gap = baseline_summary["class_gap"]
    intervention_gap = intervention_summary["class_gap"]
    gap_ratio = None
    if baseline_gap is not None and abs(float(baseline_gap)) > 1e-10:
        gap_ratio = float(intervention_gap) / float(baseline_gap)
    values = list(deltas.values())
    return {
        "mean_delta": statistics.fmean(values),
        "fraction_down": sum(value < 0 for value in values) / len(values),
        "by_expected_mean_delta": by_expected,
        "gap_ratio": gap_ratio,
        "deltas": deltas,
        "baseline_summary": baseline_summary,
        "intervention_summary": intervention_summary,
    }


def _fit_observations(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    fit_cases: Sequence[PreparedCase],
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    readout: LabelReadout | None = None,
) -> dict[str, Observation]:
    layer_indices = tuple(range(len(layers)))
    return {
        prepared.case.name: observe_case(
            model,
            layers,
            prepared,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            capture_layers=layer_indices,
            readout=readout,
        )
        for prepared in fit_cases
    }


def _contrastive_directions(
    observations: Mapping[str, Observation],
    pairs: Sequence[GuardPair],
    n_layers: int,
) -> dict[int, ContrastiveDirection]:
    directions: dict[int, ContrastiveDirection] = {}
    for layer_idx in range(n_layers):
        safe = [observations[pair.safe.name].activations[layer_idx] for pair in pairs]
        unsafe = [observations[pair.unsafe.name].activations[layer_idx] for pair in pairs]
        directions[layer_idx] = build_contrastive_direction(
            layer_idx=layer_idx,
            unsafe_activations=unsafe,
            safe_activations=safe,
        )
    return directions


def _unnormalized_label_axis(
    model: torch.nn.Module,
    *,
    safe_token_id: int,
    unsafe_token_id: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return W_U[unsafe] - W_U[safe], the axis whose projection *is* the margin."""

    output = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output is None or not hasattr(output, "weight"):
        raise StudyRuntimeError("model exposes no output embedding matrix")
    weight = output.weight.detach().to(dtype=dtype).cpu()
    axis = weight[unsafe_token_id] - weight[safe_token_id]
    if axis.norm() < 1e-8:
        raise StudyRuntimeError("safe and unsafe output embeddings have no usable separation")
    return axis


def _raw_label_axis(
    model: torch.nn.Module,
    *,
    safe_token_id: int,
    unsafe_token_id: int,
) -> torch.Tensor:
    axis = _unnormalized_label_axis(
        model,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
    )
    return axis / axis.norm()


def _patched_margin(
    model: torch.nn.Module,
    layer: torch.nn.Module,
    activation: torch.Tensor,
    prepared: PreparedCase,
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    readout: LabelReadout | None = None,
) -> float:
    with LastTokenPatch(layer, activation):
        observation = observe_case(
            model,
            [layer],
            prepared,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            readout=readout,
        )
    return float(observation.row["unsafe_minus_safe_logit_margin"])


def build_causal_map(
    model: torch.nn.Module,
    layers: Sequence[torch.nn.Module],
    contract: GuardDatasetContract,
    prepared: Mapping[str, PreparedCase],
    observations: Mapping[str, Observation],
    directions: Mapping[int, ContrastiveDirection],
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    max_patch_pairs: int,
    selection: LayerSelection,
    readout: LabelReadout | None = None,
) -> dict[str, Any]:
    selected_pairs = contract.pairs("fit")[:max_patch_pairs]
    effects: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(layers))}
    for pair in selected_pairs:
        safe_observation = observations[pair.safe.name]
        unsafe_observation = observations[pair.unsafe.name]
        safe_margin = float(safe_observation.row["unsafe_minus_safe_logit_margin"])
        unsafe_margin = float(unsafe_observation.row["unsafe_minus_safe_logit_margin"])
        for layer_idx, layer in enumerate(layers):
            unsafe_with_safe = _patched_margin(
                model,
                layer,
                safe_observation.activations[layer_idx],
                prepared[pair.unsafe.name],
                tokenizer=tokenizer,
                input_device=input_device,
                safe_token_id=safe_token_id,
                unsafe_token_id=unsafe_token_id,
            )
            safe_with_unsafe = _patched_margin(
                model,
                layer,
                unsafe_observation.activations[layer_idx],
                prepared[pair.safe.name],
                tokenizer=tokenizer,
                input_device=input_device,
                safe_token_id=safe_token_id,
                unsafe_token_id=unsafe_token_id,
            )
            unsafe_transfer = unsafe_margin - unsafe_with_safe
            safe_transfer = safe_with_unsafe - safe_margin
            effects[layer_idx].append(
                {
                    "pair_id": pair.pair_id,
                    "baseline_pair_gap": unsafe_margin - safe_margin,
                    "unsafe_to_safe_effect": unsafe_transfer,
                    "safe_to_unsafe_effect": safe_transfer,
                    "bidirectional_effect": (unsafe_transfer + safe_transfer) / 2.0,
                }
            )

    label_axis = _raw_label_axis(
        model,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
    )
    per_layer: dict[str, Any] = {}
    magnitudes: list[float] = []
    for layer_idx in range(len(layers)):
        values = [entry["bidirectional_effect"] for entry in effects[layer_idx]]
        absolute = [abs(value) for value in values]
        direction = directions[layer_idx]
        cosine = float(torch.dot(direction.direction, label_axis).item())
        mean_effect = statistics.fmean(values)
        mean_absolute = statistics.fmean(absolute)
        magnitudes.append(mean_absolute)
        per_layer[str(layer_idx)] = {
            "projection_std": direction.projection_std,
            "safe_projection_mean": direction.safe_projection_mean,
            "unsafe_projection_mean": direction.unsafe_projection_mean,
            "projection_gap": direction.projection_gap,
            "raw_unembedding_cosine": cosine,
            "mean_bidirectional_effect": mean_effect,
            "mean_absolute_effect": mean_absolute,
            "pair_effects": effects[layer_idx],
        }
    # Every effect is a difference of two margins of similar size, so the floor
    # tracks the margin magnitude rather than the effect magnitude.
    peak_margin = max(
        (abs(float(entry["baseline_pair_gap"])) for row in effects.values() for entry in row),
        default=1.0,
    )
    readout_dtype = readout.dtype if readout is not None else torch.float32
    compute_dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.device.type != "meta"),
        torch.float32,
    )
    # An fp32 readout removes the cancellation term but not the rounding already
    # carried by the residual, so the floor is bounded by whichever dtype is
    # coarser. Averaging over pairs dithers it, since each pair rounds
    # independently.
    binding_dtype = max(
        (readout_dtype, compute_dtype),
        key=lambda dtype: resolution_floor(dtype, peak_margin),
    )
    floor = resolution_floor(binding_dtype, peak_margin) / max(len(selected_pairs), 1)
    for layer_idx in range(len(layers)):
        per_layer[str(layer_idx)]["resolved"] = magnitudes[layer_idx] > floor

    ranking = selection.rank(magnitudes)
    top_layers = list(selection.top_layers(magnitudes))
    # The low-causal control must stay pinned to raw magnitude: under a differential
    # strategy the flattest layer scores near zero and would be drawn from the
    # saturated plateau, which is the opposite of an inert control.
    low_causal_layer = min(range(len(magnitudes)), key=lambda index: (magnitudes[index], index))
    return {
        "metric": "unsafe_minus_safe_logit_margin",
        "patch_position": "last",
        "patch_pairs": [pair.pair_id for pair in selected_pairs],
        "layer_selection": selection.summary(),
        "resolution": {
            "readout_dtype": str(readout_dtype).removeprefix("torch."),
            "compute_dtype": str(compute_dtype).removeprefix("torch."),
            "binding_dtype": str(binding_dtype).removeprefix("torch."),
            "peak_margin": peak_margin,
            "effect_floor": floor,
            "unresolved_layers": [
                layer_idx
                for layer_idx in range(len(layers))
                if not per_layer[str(layer_idx)]["resolved"]
            ],
        },
        "top_layers": top_layers,
        "low_causal_layer": low_causal_layer,
        "ranking": [
            {
                "layer_idx": layer_idx,
                "score": score,
                "mean_absolute_effect": magnitudes[layer_idx],
            }
            for layer_idx, score in ranking
        ],
        "per_layer": per_layer,
        "label_axis": label_axis,
    }


def _projection_scale(activations: Sequence[torch.Tensor], direction: torch.Tensor) -> float:
    values = torch.stack([value.detach().float().cpu().reshape(-1) for value in activations])
    projections = values @ direction.detach().float().cpu().reshape(-1)
    scale = float(projections.std(unbiased=False).item())
    if not math.isfinite(scale) or scale < 1e-8:
        raise StudyRuntimeError("control direction has a degenerate activation scale")
    return scale


def build_learned_arms(
    causal_map: Mapping[str, Any],
    directions: Mapping[int, ContrastiveDirection],
    *,
    include_joint: bool,
) -> list[SteeringArm]:
    top_layers = tuple(int(value) for value in causal_map["top_layers"])
    arms = [
        SteeringArm(
            name=f"learned.layer_{layer_idx}",
            kind="learned",
            layer_indices=(layer_idx,),
            vectors={layer_idx: directions[layer_idx].direction},
            scales={layer_idx: directions[layer_idx].projection_std},
        )
        for layer_idx in top_layers
    ]
    if include_joint and len(top_layers) > 1:
        suffix = "_".join(str(layer_idx) for layer_idx in top_layers)
        arms.append(
            SteeringArm(
                name=f"learned.joint_{suffix}",
                kind="learned",
                layer_indices=top_layers,
                vectors={layer: directions[layer].direction for layer in top_layers},
                scales={layer: directions[layer].projection_std for layer in top_layers},
            )
        )
    return arms


def select_winner(
    sweeps: Sequence[Mapping[str, Any]],
    *,
    target_mean_delta: float,
    min_parse_rate: float,
    min_gap_ratio: float,
) -> dict[str, Any]:
    learned = [
        result for result in sweeps if result["kind"] == "learned" and float(result["dose"]) != 0.0
    ]
    eligible = [
        result
        for result in learned
        if float(result["comparison"]["mean_delta"]) <= target_mean_delta
        and float(result["summary"]["parse_rate"]) >= min_parse_rate
        and result["comparison"]["gap_ratio"] is not None
        and float(result["comparison"]["gap_ratio"]) >= min_gap_ratio
    ]
    if eligible:
        winner = min(
            eligible,
            key=lambda result: (
                abs(float(result["dose"])),
                len(result["layers"]),
                float(result["comparison"]["mean_delta"]),
                str(result["arm"]),
            ),
        )
        return {
            "met_contract": True,
            "arm": winner["arm"],
            "dose": winner["dose"],
            "reason": "smallest absolute dose satisfying the preregistered dev constraints",
        }
    best = min(
        learned,
        key=lambda result: (
            float(result["comparison"]["mean_delta"]),
            abs(float(result["dose"])),
            str(result["arm"]),
        ),
    )
    return {
        "met_contract": False,
        "arm": None,
        "dose": None,
        "reason": "no learned steering arm satisfied the preregistered dev constraints",
        "best_observed": {
            "arm": best["arm"],
            "dose": best["dose"],
            "mean_delta": best["comparison"]["mean_delta"],
        },
    }


def _arm_result(
    arm: SteeringArm,
    dose: float,
    rows: list[dict[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "arm": arm.name,
        "kind": arm.kind,
        "layers": list(arm.layer_indices),
        "dose": dose,
        "summary": summarize_rows(rows),
        "comparison": compare_rows(baseline_rows, rows),
        "rows": rows,
    }


def _matched_sham_arm(arm: SteeringArm, *, seed: int) -> SteeringArm:
    return SteeringArm(
        name=f"control.orthogonal.{arm.name}",
        kind="orthogonal-sham",
        layer_indices=arm.layer_indices,
        vectors={
            layer: orthogonal_sham(arm.vectors[layer], seed=seed + layer)
            for layer in arm.layer_indices
        },
        scales=dict(arm.scales),
    )


def _low_causal_arm(
    causal_map: Mapping[str, Any],
    directions: Mapping[int, ContrastiveDirection],
) -> SteeringArm:
    layer = int(causal_map["low_causal_layer"])
    return SteeringArm(
        name=f"control.low_causal.layer_{layer}",
        kind="low-causal",
        layer_indices=(layer,),
        vectors={layer: directions[layer].direction},
        scales={layer: directions[layer].projection_std},
    )


def _label_axis_arm(
    causal_map: Mapping[str, Any],
    observations: Mapping[str, Observation],
) -> SteeringArm:
    layer = max(int(key) for key in causal_map["per_layer"])
    axis = causal_map["label_axis"]
    activations = [observation.activations[layer] for observation in observations.values()]
    scale = _projection_scale(activations, axis)
    return SteeringArm(
        name=f"control.label_axis.layer_{layer}",
        kind="label-axis",
        layer_indices=(layer,),
        vectors={layer: axis},
        scales={layer: scale},
    )


def _build_verified_readout(
    model: torch.nn.Module,
    probe: PreparedCase,
    *,
    tokenizer: Any,
    input_device: torch.device,
    safe_token_id: int,
    unsafe_token_id: int,
    precision: PrecisionSpec,
) -> LabelReadout:
    """Construct the margin readout and check it against the head on one probe case.

    The check costs a single forward pass and converts a silent class of wrong
    answers — a model whose reported final hidden state is not the post-norm
    residual — into a failure at run start.
    """

    axis = _unnormalized_label_axis(
        model,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        dtype=precision.readout_dtype,
    )
    readout = LabelReadout(axis, dtype=precision.readout_dtype)
    observation = observe_case(
        model,
        [],
        probe,
        tokenizer=tokenizer,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        readout=readout,
    )
    head_margin = float(observation.row["head_logit_margin"])
    compute_dtype = next(
        (parameter.dtype for parameter in model.parameters() if parameter.device.type != "meta"),
        torch.float32,
    )
    # Agreement is only expected to the resolution the head itself had, with room
    # for the accumulated rounding of the projection against a rounded logit pair.
    tolerance = 8.0 * resolution_floor(compute_dtype, max(abs(head_margin), 1.0))
    if observation.final_hidden is None:
        raise StudyRuntimeError("probe observation returned no final hidden state")
    readout.verify(observation.final_hidden, head_margin, tolerance=tolerance)
    return readout


def run_loaded_study(
    handle: ModelHandle,
    contract: GuardDatasetContract,
    study: GuardStudySpec,
    *,
    max_seq_length: int,
    decision_max_new_tokens: int,
    output_directory: Path,
) -> dict[str, Any]:
    """Run mapping and steering against an already-loaded model without mutating weights."""

    from abliteralus.strategies.utils import get_layer_modules

    model = handle.model
    tokenizer = handle.tokenizer
    layers = list(get_layer_modules(handle))
    if not layers:
        raise StudyRuntimeError("model exposes no transformer layers")
    input_device = _model_input_device(model)
    safe_token_id = _single_token_id(tokenizer, contract.safe_label)
    unsafe_token_id = _single_token_id(tokenizer, contract.unsafe_label)
    if safe_token_id == unsafe_token_id:
        raise StudyRuntimeError("safe and unsafe labels resolve to the same token")
    fit_contract_cases = list(contract.cases("fit"))
    dev_contract_cases = list(contract.cases("dev"))
    prepared = prepare_cases(
        [*fit_contract_cases, *dev_contract_cases],
        tokenizer,
        max_seq_length=max_seq_length,
    )
    prepared = align_cases_to_emitted_label(
        model,
        prepared,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        max_new_tokens=decision_max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )
    fit_cases = [prepared[case.name] for case in fit_contract_cases]
    dev_cases = [prepared[case.name] for case in dev_contract_cases]

    model.eval()
    readout = _build_verified_readout(
        model,
        fit_cases[0],
        tokenizer=tokenizer,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        precision=study.precision,
    )
    observations = _fit_observations(
        model,
        layers,
        fit_cases,
        tokenizer=tokenizer,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        readout=readout,
    )
    directions = _contrastive_directions(observations, contract.pairs("fit"), len(layers))
    causal_map = build_causal_map(
        model,
        layers,
        contract,
        prepared,
        observations,
        directions,
        tokenizer=tokenizer,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        max_patch_pairs=study.max_patch_pairs,
        selection=study.layer_selection,
        readout=readout,
    )

    tensor_payload: dict[str, torch.Tensor] = {}
    for layer_idx, direction in directions.items():
        tensor_payload[f"layer.{layer_idx}.direction"] = direction.direction.contiguous()
    tensor_payload["control.label_axis.direction"] = causal_map["label_axis"].contiguous()
    save_file(
        tensor_payload,
        str(output_directory / "directions.safetensors"),
        metadata={
            "dataset_sha256": contract.content_sha256,
            "metric": "unsafe_minus_safe_logit_margin",
        },
    )
    causal_json = dict(causal_map)
    causal_json.pop("label_axis")
    _write_json(output_directory / "causal-map.json", causal_json)

    baseline_dev = evaluate_cases(
        model,
        layers,
        dev_cases,
        tokenizer=tokenizer,
        input_device=input_device,
        safe_token_id=safe_token_id,
        unsafe_token_id=unsafe_token_id,
        readout=readout,
    )
    learned_arms = build_learned_arms(
        causal_map,
        directions,
        include_joint=study.include_joint_arm,
    )
    label_axis_arm = _label_axis_arm(causal_map, observations)
    sweeps: list[dict[str, Any]] = []
    for arm in [*learned_arms, label_axis_arm]:
        for dose in study.doses:
            rows = evaluate_cases(
                model,
                layers,
                dev_cases,
                tokenizer=tokenizer,
                input_device=input_device,
                safe_token_id=safe_token_id,
                unsafe_token_id=unsafe_token_id,
                arm=arm,
                dose=dose,
                readout=readout,
            )
            sweeps.append(_arm_result(arm, dose, rows, baseline_dev))

    selection = select_winner(
        sweeps,
        target_mean_delta=study.target_mean_delta,
        min_parse_rate=study.min_parse_rate,
        min_gap_ratio=study.min_gap_ratio,
    )
    controls: list[dict[str, Any]] = []
    test_result: dict[str, Any] | None = None
    generation_verification: dict[str, Any] | None = None
    if selection["met_contract"]:
        winner = next(arm for arm in learned_arms if arm.name == selection["arm"])
        dose = float(selection["dose"])
        sham = _matched_sham_arm(winner, seed=study.seed)
        for control in (sham, _low_causal_arm(causal_map, directions)):
            rows = evaluate_cases(
                model,
                layers,
                dev_cases,
                tokenizer=tokenizer,
                input_device=input_device,
                safe_token_id=safe_token_id,
                unsafe_token_id=unsafe_token_id,
                arm=control,
                dose=dose,
            )
            controls.append(_arm_result(control, dose, rows, baseline_dev))

        test_contract_cases = list(contract.cases("test"))
        prepared_test = prepare_cases(
            test_contract_cases,
            tokenizer,
            max_seq_length=max_seq_length,
        )
        prepared_test = align_cases_to_emitted_label(
            model,
            prepared_test,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            max_new_tokens=decision_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        test_cases = [prepared_test[case.name] for case in test_contract_cases]
        baseline_test = evaluate_cases(
            model,
            layers,
            test_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
        )
        winner_rows = evaluate_cases(
            model,
            layers,
            test_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            arm=winner,
            dose=dose,
        )
        sham_rows = evaluate_cases(
            model,
            layers,
            test_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            arm=sham,
            dose=dose,
        )
        test_result = {
            "baseline": {"summary": summarize_rows(baseline_test), "rows": baseline_test},
            "winner": _arm_result(winner, dose, winner_rows, baseline_test),
            "orthogonal_sham": _arm_result(sham, dose, sham_rows, baseline_test),
        }

        baseline_dev_generation = baseline_generation_rows(
            dev_cases,
            tokenizer=tokenizer,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
        )
        winner_dev_generation = evaluate_generations(
            model,
            layers,
            dev_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            max_new_tokens=decision_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            arm=winner,
            dose=dose,
        )
        sham_dev_generation = evaluate_generations(
            model,
            layers,
            dev_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            max_new_tokens=decision_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            arm=sham,
            dose=dose,
        )
        baseline_test_generation = baseline_generation_rows(
            test_cases,
            tokenizer=tokenizer,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
        )
        winner_test_generation = evaluate_generations(
            model,
            layers,
            test_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            max_new_tokens=decision_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            arm=winner,
            dose=dose,
        )
        sham_test_generation = evaluate_generations(
            model,
            layers,
            test_cases,
            tokenizer=tokenizer,
            input_device=input_device,
            safe_token_id=safe_token_id,
            unsafe_token_id=unsafe_token_id,
            max_new_tokens=decision_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            arm=sham,
            dose=dose,
        )
        generation_verification = {
            "method": "greedy-generation-from-original-prompt",
            "max_new_tokens": decision_max_new_tokens,
            "dev": {
                "baseline": {
                    "summary": summarize_rows(baseline_dev_generation),
                    "rows": baseline_dev_generation,
                },
                "winner": {
                    "summary": summarize_rows(winner_dev_generation),
                    "rows": winner_dev_generation,
                },
                "orthogonal_sham": {
                    "summary": summarize_rows(sham_dev_generation),
                    "rows": sham_dev_generation,
                },
            },
            "test": {
                "baseline": {
                    "summary": summarize_rows(baseline_test_generation),
                    "rows": baseline_test_generation,
                },
                "winner": {
                    "summary": summarize_rows(winner_test_generation),
                    "rows": winner_test_generation,
                },
                "orthogonal_sham": {
                    "summary": summarize_rows(sham_test_generation),
                    "rows": sham_test_generation,
                },
            },
        }

    result = {
        "schema_version": 1,
        "mode": "reversible-guard-steering",
        "metric": "unsafe_minus_safe_logit_margin",
        "model": handle.summary(),
        "device": str(input_device),
        "safe_token_id": safe_token_id,
        "unsafe_token_id": unsafe_token_id,
        "decision_position": {
            "kind": "baseline-emitted-label-prefix",
            "generation_max_new_tokens": decision_max_new_tokens,
        },
        "fit_baseline": {
            "summary": summarize_rows([value.row for value in observations.values()]),
            "rows": [observations[name].row for name in sorted(observations)],
        },
        "causal_map": causal_json,
        "dev": {
            "baseline": {"summary": summarize_rows(baseline_dev), "rows": baseline_dev},
            "sweeps": sweeps,
            "selection": selection,
            "controls": controls,
        },
        "test_access": {
            "opened": test_result is not None,
            "condition": "dev-selection-contract",
            "selection_met_contract": bool(selection["met_contract"]),
        },
        "test": test_result,
        "generation_verification": generation_verification,
        "weights_mutated": False,
    }
    _write_json(output_directory / "steering-results.json", result)
    return result


def _run_study_in_environment(
    study: GuardStudySpec,
    *,
    offline: bool,
    hf_home: Path,
    hf_home_source: str,
    run_id: str | None = None,
) -> Path:
    """Resolve the pinned model, run the study, and write an auditable result directory."""

    from abliteralus.models.loader import load_model
    from abliteralus.surgery_bench import load_experiment_spec, resolve_model_checkpoint

    contract = load_dataset_contract(study.dataset_path)
    surgery = load_experiment_spec(study.surgery_experiment_path)
    run_id = run_id or _run_id()
    output_directory = study.output_root / study.name / run_id
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_json(output_directory / "dataset-contract.json", normalized_dataset(contract))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": _utc_now(),
        "run_id": run_id,
        "study": {
            "name": study.name,
            "path": str(study.path),
            "sha256": study.sha256,
        },
        "dataset": contract.summary(),
        "surgery_experiment": str(study.surgery_experiment_path),
        "offline": offline,
        "hf_home": {"path": str(hf_home), "source": hf_home_source},
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "status": _git_value(["status", "--short"]),
        },
        "weights_mutated": False,
    }
    _write_json(output_directory / "run-manifest.json", manifest)
    handle: ModelHandle | None = None
    try:
        checkpoint = resolve_model_checkpoint(surgery, offline=offline)
        manifest["checkpoint"] = str(checkpoint)
        model_cfg = surgery.model
        compute_dtype = study.precision.resolve_compute(str(model_cfg["dtype"]))
        with pinned_matmul_precision(study.precision) as tf32_state:
            manifest["precision"] = {
                **study.precision.summary(inherited=str(model_cfg["dtype"])),
                **tf32_state,
            }
            _write_json(output_directory / "run-manifest.json", manifest)
            handle = load_model(
                model_name=str(checkpoint),
                task="causal_lm",
                device=str(model_cfg["device"]),
                dtype=compute_dtype,
                trust_remote_code=bool(model_cfg["trust_remote_code"]),
                skip_snapshot=True,
                local_files_only=True,
            )
            run_loaded_study(
                handle,
                contract,
                study,
                max_seq_length=int(surgery.pipeline["max_seq_length"]),
                decision_max_new_tokens=int(surgery.evaluation.get("max_new_tokens", 8)),
                output_directory=output_directory,
            )
        manifest["status"] = "complete"
        manifest["ended_at"] = _utc_now()
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["ended_at"] = _utc_now()
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        if handle is not None:
            handle.cleanup()
            del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _write_json(output_directory / "run-manifest.json", manifest)
    return output_directory


def run_study(
    study: GuardStudySpec,
    *,
    offline: bool,
    run_id: str | None = None,
) -> Path:
    """Run with the lab bench's ephemeral workspace-local Hub cache policy."""

    with _study_hf_home(study) as (hf_home, source):
        return _run_study_in_environment(
            study,
            offline=offline,
            hf_home=hf_home,
            hf_home_source=source,
            run_id=run_id,
        )
