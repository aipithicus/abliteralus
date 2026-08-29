"""Resident Hugging Face subject used for generation and exact-token inspection."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from abliteralus.chat_templates import render_chat_prompt
from abliteralus.models.loader import load_model
from abliteralus.surgery_bench import load_experiment_spec

from .contracts import json_sha256
from .errors import ProbeHarnessError
from .observations import (
    activation_summary,
    capture_activation,
    compact_logit_summary,
    token_position_profile,
    validate_position_mode,
)


_CATEGORY_CODE = re.compile(r"\bS\d+\b")
_TORCH_DTYPES = {
    "bool": torch.bool,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, repository: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repository / candidate
    return candidate.resolve()


def _resolve_checkpoint(
    *,
    source: str,
    revision: str,
    repository: Path,
    cache_root: Path,
    allow_download: bool,
) -> Path:
    candidate = Path(source)
    local_candidate = candidate if candidate.is_absolute() else repository / candidate
    if local_candidate.exists():
        if not local_candidate.is_dir():
            raise ProbeHarnessError(f"HF checkpoint is not a directory: {local_candidate}")
        return local_candidate.resolve()
    if candidate.is_absolute() or source.startswith("."):
        raise ProbeHarnessError(f"local HF checkpoint does not exist: {local_candidate}")

    from huggingface_hub import snapshot_download

    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    try:
        resolved = snapshot_download(
            repo_id=source,
            revision=revision,
            cache_dir=str(hub_cache),
            local_files_only=not allow_download,
            token=False,
            ignore_patterns=[
                "*.gguf",
                "*.onnx",
                "*.h5",
                "*.msgpack",
                "*.ot",
                "onnx/*",
                "original/*",
            ],
        )
    except Exception as error:
        mode = "download-enabled" if allow_download else "offline"
        raise ProbeHarnessError(
            f"could not resolve pinned HF checkpoint {source}@{revision} in {mode} mode: {error}"
        ) from error
    return Path(resolved).resolve()


def _verify_checkpoint(
    checkpoint: Path,
    expected: Mapping[str, str],
) -> dict[str, dict[str, object]]:
    verified: dict[str, dict[str, object]] = {}
    for relative, expected_digest in sorted(expected.items()):
        candidate = checkpoint.joinpath(*Path(relative).parts)
        if not candidate.is_file():
            raise ProbeHarnessError(f"pinned checkpoint file is missing: {relative}")
        actual = _sha256_file(candidate)
        if actual != expected_digest:
            raise ProbeHarnessError(
                f"pinned checkpoint file hash mismatch: {relative}; "
                f"expected {expected_digest}, got {actual}"
            )
        verified[relative] = {"sha256": actual, "bytes": candidate.stat().st_size}
    return verified


def _model_input_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cpu")


def _find_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    for attr_path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "model.decoder.layers",
        "encoder.layer",
    ):
        value: Any = model
        try:
            for part in attr_path.split("."):
                value = getattr(value, part)
            layers = list(value)
        except (AttributeError, TypeError):
            continue
        if layers and all(isinstance(layer, torch.nn.Module) for layer in layers):
            return layers
    raise ProbeHarnessError("could not locate transformer layer modules on the HF subject")


def _single_token_id(tokenizer: Any, label: str) -> int:
    values = tokenizer.encode(label, add_special_tokens=False)
    if len(values) != 1:
        raise ProbeHarnessError(f"configured label {label!r} is not exactly one token")
    return int(values[0])


def _decode(tokenizer: Any, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)


def _tensor_dtype_name(value: torch.Tensor) -> str:
    text = str(value.dtype).removeprefix("torch.")
    if text not in _TORCH_DTYPES:
        raise ProbeHarnessError(f"unsupported tokenizer tensor dtype: {value.dtype}")
    return text


def _serialize_inputs(inputs: Mapping[str, torch.Tensor]) -> tuple[dict[str, Any], dict[str, str]]:
    payload: dict[str, Any] = {}
    dtypes: dict[str, str] = {}
    for name, tensor in inputs.items():
        value = tensor.detach().cpu()
        if value.ndim != 2 or int(value.shape[0]) != 1:
            raise ProbeHarnessError(f"tokenizer field {name!r} is not a single 2D batch")
        payload[name] = value.tolist()
        dtypes[name] = _tensor_dtype_name(value)
    if "input_ids" not in payload:
        raise ProbeHarnessError("tokenizer returned no input_ids")
    return payload, dtypes


def _deserialize_inputs(
    payload: Mapping[str, Any],
    dtypes: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    if set(payload) != set(dtypes):
        raise ProbeHarnessError("stored tokenizer fields and dtypes do not agree")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(payload):
        dtype_name = dtypes[name]
        if dtype_name not in _TORCH_DTYPES:
            raise ProbeHarnessError(f"stored tokenizer dtype is unsupported: {dtype_name}")
        tensor = torch.tensor(payload[name], dtype=_TORCH_DTYPES[dtype_name])
        if tensor.ndim != 2 or int(tensor.shape[0]) != 1:
            raise ProbeHarnessError(f"stored tokenizer field {name!r} is not a single 2D batch")
        result[name] = tensor
    if "input_ids" not in result:
        raise ProbeHarnessError("stored trial has no input_ids")
    return result


def _inputs_to_device(
    inputs: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _append_prefix(
    inputs: Mapping[str, torch.Tensor],
    prefix_token_ids: Sequence[int],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    input_ids = inputs["input_ids"]
    original_length = int(input_ids.shape[-1])
    prefix = torch.tensor([list(prefix_token_ids)], dtype=input_ids.dtype)
    for name, tensor in inputs.items():
        value = tensor.detach().cpu()
        if value.ndim != 2 or int(value.shape[-1]) != original_length:
            result[name] = value
            continue
        if name == "input_ids":
            extension = prefix
        elif name == "attention_mask":
            extension = torch.ones((1, prefix.shape[-1]), dtype=value.dtype)
        elif name == "token_type_ids":
            extension = value[:, -1:].expand(1, prefix.shape[-1]).clone()
        else:
            raise ProbeHarnessError(
                f"cannot extend tokenizer field {name!r} to the emitted-label decision position"
            )
        result[name] = torch.cat((value, extension), dim=-1)
    return result


class HFSubject:
    """One resident model/tokenizer pair with deterministic Guard probing methods."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        identity: Mapping[str, Any],
        max_seq_length: int,
        max_new_tokens: int,
        safe_label: str,
        unsafe_label: str,
        handle: Any | None = None,
    ) -> None:
        if max_seq_length <= 0 or max_new_tokens <= 0:
            raise ProbeHarnessError("subject token limits must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.handle = handle
        self.model.eval()
        self.layers = _find_layers(model)
        self.input_device = _model_input_device(model)
        self.max_seq_length = int(max_seq_length)
        self.max_new_tokens = int(max_new_tokens)
        self.safe_label = safe_label
        self.unsafe_label = unsafe_label
        self.safe_token_id = _single_token_id(tokenizer, safe_label)
        self.unsafe_token_id = _single_token_id(tokenizer, unsafe_label)
        if self.safe_token_id == self.unsafe_token_id:
            raise ProbeHarnessError("safe and unsafe labels resolve to the same token")
        value = dict(identity)
        value.update(
            {
                "safe_label": safe_label,
                "unsafe_label": unsafe_label,
                "safe_token_id": self.safe_token_id,
                "unsafe_token_id": self.unsafe_token_id,
                "max_seq_length": self.max_seq_length,
                "default_max_new_tokens": self.max_new_tokens,
                "num_layers": len(self.layers),
            }
        )
        identity_basis = dict(value)
        identity_basis.pop("checkpoint_path", None)
        identity_basis.pop("digest", None)
        value["digest"] = json_sha256(identity_basis)
        self.identity = value

    @classmethod
    def load_from_experiment(
        cls,
        experiment_path: str | Path,
        *,
        repository: str | Path,
        cache_root: str | Path,
        allow_download: bool = False,
        max_new_tokens: int | None = None,
    ) -> "HFSubject":
        repository_path = Path(repository).resolve()
        experiment = _resolve_path(experiment_path, repository=repository_path)
        if not experiment.is_file():
            raise ProbeHarnessError(f"surgery experiment does not exist: {experiment}")
        spec = load_experiment_spec(experiment)
        model_config = spec.model
        checkpoint = _resolve_checkpoint(
            source=str(model_config["source"]),
            revision=str(model_config["revision"]),
            repository=repository_path,
            cache_root=_resolve_path(cache_root, repository=repository_path),
            allow_download=allow_download,
        )
        verified = _verify_checkpoint(checkpoint, model_config.get("expected_sha256", {}))
        try:
            handle = load_model(
                model_name=str(checkpoint),
                task="causal_lm",
                device=str(model_config["device"]),
                dtype=str(model_config["dtype"]),
                trust_remote_code=bool(model_config["trust_remote_code"]),
                skip_snapshot=True,
                local_files_only=True,
            )
        except Exception as error:
            raise ProbeHarnessError(f"could not load HF subject from {checkpoint}: {error}") from error

        evaluation = spec.evaluation
        selected_max_tokens = (
            int(max_new_tokens)
            if max_new_tokens is not None
            else int(evaluation.get("max_new_tokens", 8))
        )
        identity = {
            "kind": "huggingface-causal-lm",
            "name": spec.name,
            "experiment_path": str(experiment),
            "experiment_sha256": _sha256_file(experiment),
            "source": str(model_config["source"]),
            "revision": str(model_config["revision"]),
            "checkpoint_path": str(checkpoint),
            "expected_sha256": dict(model_config.get("expected_sha256", {})),
            "verified_files": verified,
            "device": str(model_config["device"]),
            "dtype": str(model_config["dtype"]),
            "trust_remote_code": bool(model_config["trust_remote_code"]),
            "architecture": str(handle.architecture),
            "hidden_size": int(handle.hidden_size),
        }
        return cls(
            model=handle.model,
            tokenizer=handle.tokenizer,
            handle=handle,
            identity=identity,
            max_seq_length=int(spec.pipeline["max_seq_length"]),
            max_new_tokens=selected_max_tokens,
            safe_label=str(evaluation.get("safe_label", "safe")),
            unsafe_label=str(evaluation.get("unsafe_label", "unsafe")),
        )

    def close(self) -> None:
        if self.handle is not None:
            self.handle.cleanup()

    def prepare_prompt(self, prompt: str) -> tuple[str, dict[str, torch.Tensor]]:
        if not prompt.strip():
            raise ProbeHarnessError("prompt must not be empty")
        rendered = render_chat_prompt(self.tokenizer, prompt)
        inputs = dict(self.tokenizer(rendered, return_tensors="pt"))
        if "input_ids" not in inputs:
            raise ProbeHarnessError("tokenizer returned no input_ids")
        length = int(inputs["input_ids"].shape[-1])
        if length > self.max_seq_length:
            raise ProbeHarnessError(
                f"prompt is {length} tokens, exceeding max_seq_length={self.max_seq_length}"
            )
        return rendered, {name: value.detach().cpu() for name, value in inputs.items()}

    def generate(
        self,
        prompt: str,
        *,
        stored_inputs: Mapping[str, Any] | None = None,
        stored_input_dtypes: Mapping[str, str] | None = None,
        stored_rendered_prompt: str | None = None,
    ) -> dict[str, Any]:
        if stored_inputs is None:
            rendered, prompt_inputs = self.prepare_prompt(prompt)
        else:
            if stored_input_dtypes is None or stored_rendered_prompt is None:
                raise ProbeHarnessError("exact replay requires stored dtypes and rendered prompt")
            rendered = stored_rendered_prompt
            prompt_inputs = _deserialize_inputs(stored_inputs, stored_input_dtypes)
        serialized_inputs, input_dtypes = _serialize_inputs(prompt_inputs)
        prompt_length = int(prompt_inputs["input_ids"].shape[-1])
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        with torch.inference_mode():
            generated = self.model.generate(
                **_inputs_to_device(prompt_inputs, self.input_device),
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=int(pad_token_id),
                return_dict_in_generate=True,
                output_scores=True,
            )
        completion = generated.sequences[0, prompt_length:].detach().cpu()
        completion_ids = tuple(int(token) for token in completion.tolist())
        label_index = next(
            (
                index
                for index, token_id in enumerate(completion_ids)
                if token_id in {self.safe_token_id, self.unsafe_token_id}
            ),
            None,
        )
        emitted_label = None
        emitted_label_token_id = None
        generated_margin = None
        decision_prefix: tuple[int, ...] = ()
        if label_index is not None:
            emitted_label_token_id = completion_ids[label_index]
            emitted_label = (
                self.safe_label
                if emitted_label_token_id == self.safe_token_id
                else self.unsafe_label
            )
            decision_prefix = completion_ids[:label_index]
            if label_index >= len(generated.scores):
                raise ProbeHarnessError("generation returned no score for its emitted label")
            scores = generated.scores[label_index][0].detach().float().cpu()
            if int(torch.argmax(scores).item()) != emitted_label_token_id:
                raise ProbeHarnessError("HF subject generation was not greedy-deterministic")
            generated_margin = float(
                (scores[self.unsafe_token_id] - scores[self.safe_token_id]).item()
            )

        category = None
        if label_index is not None and label_index + 1 < len(completion_ids):
            category_text = _decode(
                self.tokenizer,
                completion_ids[label_index + 1 :],
                skip_special_tokens=True,
            )
            match = _CATEGORY_CODE.search(category_text)
            category = match.group(0) if match else None

        return {
            "prompt": prompt,
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": f"sha256:{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}",
            "prompt_inputs": serialized_inputs,
            "prompt_input_dtypes": input_dtypes,
            "prompt_token_count": prompt_length,
            "completion_token_ids": list(completion_ids),
            "completion_text": _decode(
                self.tokenizer,
                completion_ids,
                skip_special_tokens=True,
            ).strip(),
            "label_token_index": label_index,
            "decision_prefix_token_ids": list(decision_prefix),
            "emitted_label": emitted_label,
            "emitted_label_token_id": emitted_label_token_id,
            "emitted_category": category,
            "generated_unsafe_minus_safe_logit_margin": generated_margin,
            "generation": {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "pad_token_id": int(pad_token_id),
            },
        }

    def inspect(
        self,
        trial: Mapping[str, Any],
        *,
        capture_layers: Sequence[int],
        positions: str = "decision",
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        position_mode = validate_position_mode(positions)
        prompt_inputs = _deserialize_inputs(
            trial["prompt_inputs"],
            trial["prompt_input_dtypes"],
        )
        prefix = tuple(int(token) for token in trial.get("decision_prefix_token_ids", []))
        decision_inputs = _append_prefix(prompt_inputs, prefix)
        decision_token_count = int(decision_inputs["input_ids"].shape[-1])
        selected_layers = sorted(set(int(layer) for layer in capture_layers))
        for layer in selected_layers:
            if layer < 0 or layer >= len(self.layers):
                raise ProbeHarnessError(
                    f"capture layer {layer} is out of range for {len(self.layers)} layers"
                )

        captures: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[layer_idx] = capture_activation(
                    hidden,
                    positions=position_mode,
                    expected_positions=decision_token_count,
                )

            return hook

        try:
            for layer_idx in selected_layers:
                handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))
            with torch.inference_mode():
                output = self.model(
                    **_inputs_to_device(decision_inputs, self.input_device),
                    use_cache=False,
                )
            logits = output.logits if hasattr(output, "logits") else output[0]
            scores = logits[0, -1].detach().float().cpu()
        finally:
            for handle in reversed(handles):
                handle.remove()

        margin = float((scores[self.unsafe_token_id] - scores[self.safe_token_id]).item())
        top_token_id = int(torch.argmax(scores).item())
        predicted_label = None
        if top_token_id == self.safe_token_id:
            predicted_label = self.safe_label
        elif top_token_id == self.unsafe_token_id:
            predicted_label = self.unsafe_label

        generated_label_token_id = trial.get("emitted_label_token_id")
        if generated_label_token_id is not None and top_token_id != int(generated_label_token_id):
            raise ProbeHarnessError(
                "exact-token inspection did not reproduce the baseline emitted label"
            )
        generated_margin = trial.get("generated_unsafe_minus_safe_logit_margin")
        alignment_error = None
        if generated_margin is not None:
            generated_margin = float(generated_margin)
            alignment_error = abs(margin - generated_margin)
            if not torch.isclose(
                torch.tensor(margin),
                torch.tensor(generated_margin),
                atol=0.25,
                rtol=0.02,
            ):
                raise ProbeHarnessError(
                    "exact-token inspection changed the generated label margin "
                    f"by {alignment_error:.6f} logits"
                )

        summaries: dict[str, dict[str, object]] = {}
        tensor_payload: dict[str, torch.Tensor] = {}
        for layer_idx, activation in sorted(captures.items()):
            key = f"layer.{layer_idx}.{position_mode}"
            tensor_payload[key] = activation
            summaries[str(layer_idx)] = activation_summary(activation)

        decision_input_ids = decision_inputs["input_ids"].tolist()
        flat_decision_input_ids = [int(token) for token in decision_input_ids[0]]
        result = {
            "evidence_kind": "observational",
            "operation": "exact-token-inspection",
            "capture_scope": f"emitted-label-{position_mode}",
            "positions": position_mode,
            "capture_layers": selected_layers,
            "decision_input_token_count": decision_token_count,
            "decision_input_ids_sha256": json_sha256(decision_input_ids),
            "token_positions": token_position_profile(
                self.tokenizer,
                flat_decision_input_ids,
                prompt_token_count=int(trial["prompt_token_count"]),
                positions=position_mode,
            ),
            "unsafe_minus_safe_logit_margin": margin,
            "pairwise_label": self.unsafe_label if margin > 0 else self.safe_label,
            "predicted_label": predicted_label,
            "top_token_id": top_token_id,
            "top_token_text": _decode(
                self.tokenizer,
                [top_token_id],
                skip_special_tokens=False,
            ),
            "generated_margin_alignment_error": alignment_error,
            "logits": compact_logit_summary(
                self.tokenizer,
                scores,
                safe_token_id=self.safe_token_id,
                unsafe_token_id=self.unsafe_token_id,
            ),
            "activation_summaries": summaries,
        }
        return result, tensor_payload
