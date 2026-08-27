"""Reproducible local and remote surgery experiment bench.

Surgery is intentionally performed on a Transformers/Safetensors checkpoint.
GGUF is a post-surgery deployment and A/B inference artifact, not an in-place
surgery format.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from abliteralus.artifact_contracts import (
    ArtifactStage,
    ArtifactStageContext,
    MutationHintSink,
)


SCHEMA_VERSION = 1
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,79}")
_HF_REPO = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_REVISION = re.compile(r"[A-Za-z0-9._/-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_QUANTIZATION = re.compile(r"[A-Z0-9_]+")
_ALLOWED_TOP_LEVEL = {
    "schema_version",
    "name",
    "model",
    "pipeline",
    "prompts",
    "output",
    "gguf",
    "artifact",
    "evaluation",
}
_ALLOWED_MODEL = {
    "source",
    "revision",
    "device",
    "dtype",
    "trust_remote_code",
    "local_files_only",
    "upstream_source",
    "upstream_revision",
    "expected_sha256",
}
_ALLOWED_PIPELINE = {
    "method",
    "n_directions",
    "direction_method",
    "norm_preserve",
    "regularization",
    "refinement_passes",
    "project_biases",
    "use_chat_template",
    "use_whitened_svd",
    "true_iterative_refinement",
    "quantization",
    "gpu_memory_utilization",
    "large_model_mode",
    "max_seq_length",
    "verify_sample_size",
    "refusal_max_tokens",
    "layer_selection",
    "min_layer_fraction",
    "max_layer_fraction",
    "projection_target",
    "projection_row_fraction",
    "skip_standard_verify",
}
_ALLOWED_PROMPTS = {"harmful", "harmless", "jailbreak"}
_ALLOWED_OUTPUT = {"root"}
_ALLOWED_GGUF = {
    "enabled",
    "compare_baseline",
    "outtype",
    "quantization",
    "keep_intermediate",
    "converter",
    "quantizer",
    "llama_cli",
    "gpu_layers",
    "max_tokens",
    "smoke_prompts",
}
_ALLOWED_ARTIFACT = {"mode", "require_exact", "fallback", "exports", "max_low_rank"}
_ALLOWED_EVALUATION = {
    "enabled",
    "mode",
    "safe_label",
    "unsafe_label",
    "max_new_tokens",
    "cases",
}
_ALLOWED_EVALUATION_CASE = {"name", "prompt", "expected"}


class BenchConfigError(ValueError):
    """Raised when an experiment specification is unsafe or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchConfigError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise BenchConfigError(f"unknown {label} keys: {', '.join(unknown)}")


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchConfigError(f"{label} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise BenchConfigError(f"{label} may not contain control characters")
    return value.strip()


def _require_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchConfigError(f"{label} must be true or false")
    return value


def _require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchConfigError(f"{label} must be a non-negative integer")
    return value


def _require_positive_int(value: object, label: str) -> int:
    value = _require_non_negative_int(value, label)
    if value == 0:
        raise BenchConfigError(f"{label} must be a positive integer")
    return value


def _require_fraction(
    value: object,
    label: str,
    *,
    allow_zero: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchConfigError(f"{label} must be a number")
    number = float(value)
    lower_bound_ok = number >= 0.0 if allow_zero else number > 0.0
    interval = "[0, 1]" if allow_zero else "(0, 1]"
    if not lower_bound_ok or number > 1.0:
        raise BenchConfigError(f"{label} must be in {interval}")
    return number


def _validate_pipeline(raw: dict[str, Any]) -> dict[str, Any]:
    pipeline = dict(raw)
    pipeline["method"] = _require_text(pipeline.get("method", "basic"), "pipeline.method")

    for key in ("direction_method", "layer_selection"):
        if key in pipeline:
            pipeline[key] = _require_text(pipeline[key], f"pipeline.{key}")
    for key in (
        "norm_preserve",
        "project_biases",
        "use_chat_template",
        "use_whitened_svd",
        "true_iterative_refinement",
        "large_model_mode",
        "skip_standard_verify",
    ):
        if key in pipeline:
            pipeline[key] = _require_bool(pipeline[key], f"pipeline.{key}")
    for key in ("n_directions", "max_seq_length", "verify_sample_size", "refusal_max_tokens"):
        if key in pipeline:
            pipeline[key] = _require_positive_int(pipeline[key], f"pipeline.{key}")
    if "refinement_passes" in pipeline:
        pipeline["refinement_passes"] = _require_non_negative_int(
            pipeline["refinement_passes"], "pipeline.refinement_passes"
        )
    for key in ("regularization", "min_layer_fraction", "max_layer_fraction"):
        if key in pipeline:
            pipeline[key] = _require_fraction(pipeline[key], f"pipeline.{key}")
    for key in ("gpu_memory_utilization", "projection_row_fraction"):
        if key in pipeline:
            pipeline[key] = _require_fraction(pipeline[key], f"pipeline.{key}", allow_zero=False)
    if (
        "min_layer_fraction" in pipeline
        and "max_layer_fraction" in pipeline
        and pipeline["min_layer_fraction"] > pipeline["max_layer_fraction"]
    ):
        raise BenchConfigError(
            "pipeline.min_layer_fraction may not exceed pipeline.max_layer_fraction"
        )
    if pipeline.get("projection_target") is not None:
        projection_target = _require_text(
            pipeline["projection_target"], "pipeline.projection_target"
        )
        if projection_target not in {"all", "attention", "ffn", "output"}:
            raise BenchConfigError(
                "pipeline.projection_target must be all, attention, ffn, or output"
            )
        pipeline["projection_target"] = projection_target
    if pipeline.get("quantization") not in (None, "4bit", "8bit"):
        raise BenchConfigError("pipeline.quantization must be null, '4bit', or '8bit'")
    return pipeline


def _validate_source(source: str) -> None:
    if source.lower().endswith(".gguf"):
        raise BenchConfigError(
            "model.source may not be a GGUF file: operate on the HF/Safetensors "
            "checkpoint, then enable the GGUF postprocess stage"
        )
    path = Path(source).expanduser()
    if path.exists() and not path.is_dir():
        raise BenchConfigError("a local model.source must be a checkpoint directory")
    if not path.exists() and _HF_REPO.fullmatch(source) is None:
        raise BenchConfigError(
            "model.source must be an existing checkpoint directory or OWNER/MODEL"
        )


def _validate_revision(value: object, label: str) -> str:
    revision = _require_text(value, label)
    if _REVISION.fullmatch(revision) is None or revision.startswith("-"):
        raise BenchConfigError(f"{label} contains unsupported characters")
    return revision


def _validate_expected_sha256(value: object) -> dict[str, str]:
    raw = _require_mapping(value, "model.expected_sha256")
    expected: dict[str, str] = {}
    for key, value in raw.items():
        relative_text = _require_text(key, "model.expected_sha256 path")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or relative.drive
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise BenchConfigError(
                "model.expected_sha256 paths must be normalized checkpoint-relative paths"
            )
        normalized = relative.as_posix()
        if normalized in expected:
            raise BenchConfigError("model.expected_sha256 paths must be unique")
        digest = _require_text(value, f"model.expected_sha256[{normalized}]").lower()
        if _SHA256.fullmatch(digest) is None:
            raise BenchConfigError(
                f"model.expected_sha256[{normalized}] must be a 64-character SHA-256"
            )
        expected[normalized] = digest
    return dict(sorted(expected.items()))


@dataclass(frozen=True)
class SurgeryExperimentSpec:
    """Validated experiment contract shared by the local and Lightning runners."""

    name: str
    model: dict[str, Any]
    pipeline: dict[str, Any]
    prompts: dict[str, int]
    output_root: str
    gguf: dict[str, Any]
    artifact: dict[str, Any]
    evaluation: dict[str, Any]
    source_path: Path

    @property
    def gguf_enabled(self) -> bool:
        return bool(self.gguf["enabled"])

    @property
    def capsule_enabled(self) -> bool:
        return self.artifact["mode"] == "capsule"

    @property
    def evaluation_enabled(self) -> bool:
        return bool(self.evaluation["enabled"])


def load_experiment_spec(path: str | Path) -> SurgeryExperimentSpec:
    """Load and strictly validate one YAML surgery experiment."""
    source_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw, "experiment")
    _reject_unknown(raw, _ALLOWED_TOP_LEVEL, "experiment")

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise BenchConfigError(f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}")

    name = _require_text(raw.get("name"), "name")
    if _NAME.fullmatch(name) is None:
        raise BenchConfigError(
            "name must start with a lowercase letter or digit and contain only "
            "lowercase letters, digits, '.', '_', or '-'"
        )

    model = _require_mapping(raw.get("model"), "model")
    _reject_unknown(model, _ALLOWED_MODEL, "model")
    source = _require_text(model.get("source"), "model.source")
    _validate_source(source)
    revision = model.get("revision")
    if revision is not None:
        revision = _validate_revision(revision, "model.revision")
    upstream_source = model.get("upstream_source")
    upstream_revision = model.get("upstream_revision")
    if (upstream_source is None) != (upstream_revision is None):
        raise BenchConfigError(
            "model.upstream_source and model.upstream_revision must be specified together"
        )
    if upstream_source is not None:
        upstream_source = _require_text(upstream_source, "model.upstream_source")
        if _HF_REPO.fullmatch(upstream_source) is None:
            raise BenchConfigError("model.upstream_source must be OWNER/MODEL")
        upstream_revision = _validate_revision(upstream_revision, "model.upstream_revision")
    expected_sha256 = _validate_expected_sha256(model.get("expected_sha256", {}))
    device = _require_text(model.get("device", "auto"), "model.device")
    if re.fullmatch(r"(?:auto|cpu|mps|cuda(?::[0-9]+)?)", device) is None:
        raise BenchConfigError(
            "model.device must be auto, cpu, mps, cuda, or an indexed CUDA device"
        )
    dtype = _require_text(model.get("dtype", "float16"), "model.dtype")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise BenchConfigError("model.dtype must be float16, bfloat16, or float32")
    model = {
        "source": source,
        "revision": revision,
        "device": device,
        "dtype": dtype,
        "trust_remote_code": _require_bool(
            model.get("trust_remote_code", False), "model.trust_remote_code"
        ),
        "local_files_only": _require_bool(
            model.get("local_files_only", False), "model.local_files_only"
        ),
    }
    if upstream_source is not None:
        model["upstream_source"] = upstream_source
        model["upstream_revision"] = upstream_revision
    if expected_sha256:
        model["expected_sha256"] = expected_sha256

    pipeline = _require_mapping(raw.get("pipeline", {}), "pipeline")
    _reject_unknown(pipeline, _ALLOWED_PIPELINE, "pipeline")
    pipeline = _validate_pipeline(pipeline)

    prompts_raw = _require_mapping(raw.get("prompts", {}), "prompts")
    _reject_unknown(prompts_raw, _ALLOWED_PROMPTS, "prompts")
    prompts = {
        "harmful": _require_non_negative_int(prompts_raw.get("harmful", 8), "prompts.harmful"),
        "harmless": _require_non_negative_int(prompts_raw.get("harmless", 8), "prompts.harmless"),
        "jailbreak": _require_non_negative_int(
            prompts_raw.get("jailbreak", 0), "prompts.jailbreak"
        ),
    }
    if prompts["harmful"] == 0 or prompts["harmless"] == 0:
        raise BenchConfigError("prompts.harmful and prompts.harmless must both be positive")

    output = _require_mapping(raw.get("output", {}), "output")
    _reject_unknown(output, _ALLOWED_OUTPUT, "output")
    output_root = _require_text(output.get("root", "outputs/surgery"), "output.root")

    gguf_raw = _require_mapping(raw.get("gguf", {}), "gguf")
    _reject_unknown(gguf_raw, _ALLOWED_GGUF, "gguf")
    smoke_prompts = gguf_raw.get("smoke_prompts", [])
    if not isinstance(smoke_prompts, list):
        raise BenchConfigError("gguf.smoke_prompts must be a list of non-empty strings")
    smoke_prompts = [
        _require_text(prompt, f"gguf.smoke_prompts[{index}]")
        for index, prompt in enumerate(smoke_prompts)
    ]
    gguf = {
        "enabled": _require_bool(gguf_raw.get("enabled", False), "gguf.enabled"),
        "compare_baseline": _require_bool(
            gguf_raw.get("compare_baseline", True), "gguf.compare_baseline"
        ),
        "outtype": _require_text(gguf_raw.get("outtype", "f16"), "gguf.outtype").lower(),
        "quantization": _require_text(
            gguf_raw.get("quantization", "Q4_K_M"), "gguf.quantization"
        ).upper(),
        "keep_intermediate": _require_bool(
            gguf_raw.get("keep_intermediate", False), "gguf.keep_intermediate"
        ),
        "converter": gguf_raw.get("converter"),
        "quantizer": gguf_raw.get("quantizer"),
        "llama_cli": gguf_raw.get("llama_cli"),
        "gpu_layers": _require_non_negative_int(gguf_raw.get("gpu_layers", 999), "gguf.gpu_layers"),
        "max_tokens": _require_positive_int(gguf_raw.get("max_tokens", 48), "gguf.max_tokens"),
        "smoke_prompts": smoke_prompts,
    }
    if gguf["outtype"] not in {"f16", "bf16", "f32", "q8_0"}:
        raise BenchConfigError("gguf.outtype must be f16, bf16, f32, or q8_0")
    if _QUANTIZATION.fullmatch(gguf["quantization"]) is None:
        raise BenchConfigError("gguf.quantization contains unsupported characters")
    for key in ("converter", "quantizer", "llama_cli"):
        if gguf[key] is not None:
            gguf[key] = _require_text(gguf[key], f"gguf.{key}")

    artifact_raw = _require_mapping(raw.get("artifact", {}), "artifact")
    _reject_unknown(artifact_raw, _ALLOWED_ARTIFACT, "artifact")
    artifact_mode = _require_text(artifact_raw.get("mode", "full-checkpoint"), "artifact.mode")
    if artifact_mode not in {"full-checkpoint", "capsule"}:
        raise BenchConfigError("artifact.mode must be full-checkpoint or capsule")
    artifact_fallback = _require_text(
        artifact_raw.get("fallback", "full-checkpoint"), "artifact.fallback"
    )
    if artifact_fallback not in {"full-checkpoint", "error"}:
        raise BenchConfigError("artifact.fallback must be full-checkpoint or error")
    exports = artifact_raw.get("exports", [])
    if not isinstance(exports, list):
        raise BenchConfigError("artifact.exports must be a list")
    exports = [
        _require_text(value, f"artifact.exports[{index}]") for index, value in enumerate(exports)
    ]
    if len(set(exports)) != len(exports):
        raise BenchConfigError("artifact.exports must not contain duplicates")
    unsupported_exports = sorted(set(exports) - {"peft", "llama_cpp"})
    if unsupported_exports:
        raise BenchConfigError(
            "artifact.exports contains unsupported values: " + ", ".join(unsupported_exports)
        )
    if exports and artifact_mode != "capsule":
        raise BenchConfigError("artifact.exports requires artifact.mode: capsule")
    max_low_rank = _require_positive_int(
        artifact_raw.get("max_low_rank", 32), "artifact.max_low_rank"
    )
    if max_low_rank > 256:
        raise BenchConfigError("artifact.max_low_rank may not exceed 256")
    artifact = {
        "mode": artifact_mode,
        "require_exact": _require_bool(
            artifact_raw.get("require_exact", True), "artifact.require_exact"
        ),
        "fallback": artifact_fallback,
        "exports": exports,
        "max_low_rank": max_low_rank,
    }

    evaluation_raw = _require_mapping(raw.get("evaluation", {}), "evaluation")
    _reject_unknown(evaluation_raw, _ALLOWED_EVALUATION, "evaluation")
    evaluation_enabled = _require_bool(evaluation_raw.get("enabled", False), "evaluation.enabled")
    evaluation_mode = _require_text(evaluation_raw.get("mode", "safety-label"), "evaluation.mode")
    if evaluation_mode != "safety-label":
        raise BenchConfigError("evaluation.mode must be safety-label")
    safe_label = _require_text(evaluation_raw.get("safe_label", "safe"), "evaluation.safe_label")
    unsafe_label = _require_text(
        evaluation_raw.get("unsafe_label", "unsafe"), "evaluation.unsafe_label"
    )
    if safe_label.casefold() == unsafe_label.casefold():
        raise BenchConfigError("evaluation safe and unsafe labels must differ")
    cases_raw = evaluation_raw.get("cases", [])
    if not isinstance(cases_raw, list):
        raise BenchConfigError("evaluation.cases must be a list")
    cases: list[dict[str, str]] = []
    for index, value in enumerate(cases_raw):
        case = _require_mapping(value, f"evaluation.cases[{index}]")
        _reject_unknown(case, _ALLOWED_EVALUATION_CASE, f"evaluation.cases[{index}]")
        case_name = _require_text(case.get("name"), f"evaluation.cases[{index}].name")
        if _NAME.fullmatch(case_name) is None:
            raise BenchConfigError(
                f"evaluation.cases[{index}].name contains unsupported characters"
            )
        expected = _require_text(
            case.get("expected"), f"evaluation.cases[{index}].expected"
        ).casefold()
        if expected not in {"safe", "unsafe"}:
            raise BenchConfigError(f"evaluation.cases[{index}].expected must be safe or unsafe")
        cases.append(
            {
                "name": case_name,
                "prompt": _require_text(case.get("prompt"), f"evaluation.cases[{index}].prompt"),
                "expected": expected,
            }
        )
    case_names = [case["name"] for case in cases]
    if len(set(case_names)) != len(case_names):
        raise BenchConfigError("evaluation case names must be unique")
    if evaluation_enabled and not cases:
        raise BenchConfigError("enabled evaluation requires at least one case")
    evaluation = {
        "enabled": evaluation_enabled,
        "mode": evaluation_mode,
        "safe_label": safe_label,
        "unsafe_label": unsafe_label,
        "max_new_tokens": _require_positive_int(
            evaluation_raw.get("max_new_tokens", 16), "evaluation.max_new_tokens"
        ),
        "cases": cases,
    }
    if pipeline.get("skip_standard_verify", False) and not evaluation_enabled:
        raise BenchConfigError(
            "pipeline.skip_standard_verify requires an enabled external evaluation"
        )

    return SurgeryExperimentSpec(
        name=name,
        model=model,
        pipeline=pipeline,
        prompts=prompts,
        output_root=output_root,
        gguf=gguf,
        artifact=artifact,
        evaluation=evaluation,
        source_path=source_path,
    )


def _which_or_path(value: str | None) -> Path | None:
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    resolved = shutil.which(value)
    return Path(resolved).resolve() if resolved else None


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _llama_root_candidates(executable: Path | None) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.environ.get("LLAMA_CPP_ROOT") or os.environ.get("ABLITERALUS_LLAMA_CPP_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd() / "llama.cpp", Path.cwd().parent / "llama.cpp"])
    if executable:
        for parent in executable.parents:
            candidates.extend([parent, parent / "llama.cpp"])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(str(candidate))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


@dataclass(frozen=True)
class GGUFTools:
    converter: Path | None
    quantizer: Path | None
    llama_cli: Path | None
    llama_cpp_root: Path | None


def discover_gguf_tools(spec: SurgeryExperimentSpec) -> GGUFTools:
    """Resolve the converter and llama.cpp executables without invoking them."""
    quantizer = _which_or_path(
        spec.gguf["quantizer"] or os.environ.get("ABLITERALUS_LLAMA_QUANTIZE") or "llama-quantize"
    )
    llama_cli = _which_or_path(
        spec.gguf["llama_cli"] or os.environ.get("ABLITERALUS_LLAMA_CLI") or "llama-cli"
    )
    explicit_converter = _which_or_path(
        spec.gguf["converter"] or os.environ.get("ABLITERALUS_LLAMA_CPP_CONVERTER")
    )
    root = None
    converter = explicit_converter
    for candidate in _llama_root_candidates(quantizer or llama_cli):
        script = candidate / "convert_hf_to_gguf.py"
        if script.is_file():
            root = candidate.resolve()
            if converter is None:
                converter = script.resolve()
            break
    if root is None and converter is not None:
        root = converter.parent
    return GGUFTools(
        converter=converter,
        quantizer=quantizer,
        llama_cli=llama_cli,
        llama_cpp_root=root,
    )


def _subprocess_env(tools: GGUFTools | None = None) -> dict[str, str]:
    env = os.environ.copy()
    path_entries: list[str] = []
    if os.name == "nt":
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.origin:
            torch_lib = Path(torch_spec.origin).parent / "lib"
            if torch_lib.is_dir():
                path_entries.append(str(torch_lib))
    if path_entries:
        env["PATH"] = os.pathsep.join([*path_entries, env.get("PATH", "")])
    if tools and tools.llama_cpp_root:
        gguf_py = tools.llama_cpp_root / "gguf-py"
        if gguf_py.is_dir():
            env["PYTHONPATH"] = os.pathsep.join([str(gguf_py), env.get("PYTHONPATH", "")]).rstrip(
                os.pathsep
            )
    return env


def _probe_executable(path: Path | None, tools: GGUFTools) -> dict[str, Any]:
    if path is None:
        return {"found": False, "usable": False, "path": None}
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_subprocess_env(tools),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "found": True,
            "usable": False,
            "path": str(path),
            "error": str(error),
        }
    output = (completed.stdout + completed.stderr).strip()
    # llama-quantize currently has no version flag. It exits 1 after printing
    # its complete usage/type table, which still proves that the executable and
    # its dynamic libraries loaded successfully.
    usage_probe = completed.returncode == 1 and output.lower().startswith("usage:")
    display = output.splitlines()[0] if usage_probe else output[:2000]
    return {
        "found": True,
        "usable": completed.returncode == 0 or usage_probe,
        "path": str(path),
        "exit_code": completed.returncode,
        "version": display,
        "version_truncated": len(display) != len(output),
    }


def preflight_experiment(
    spec: SurgeryExperimentSpec,
    *,
    output_root: str | Path | None = None,
    skip_gguf: bool = False,
) -> dict[str, Any]:
    """Return an executable readiness report without downloading a model."""
    required_packages = ["torch", "transformers", "huggingface_hub", "safetensors"]
    packages = {name: _module_available(name) for name in required_packages}
    if spec.pipeline.get("quantization") in {"4bit", "8bit"}:
        packages["bitsandbytes"] = importlib.util.find_spec("bitsandbytes") is not None

    cuda: dict[str, Any] = {"required": spec.model["device"].startswith("cuda")}
    try:
        import torch

        cuda.update(
            {
                "available": torch.cuda.is_available(),
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "total_vram_bytes": (
                    torch.cuda.get_device_properties(0).total_memory
                    if torch.cuda.is_available()
                    else None
                ),
                "bf16_supported": (
                    torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None
                ),
            }
        )
    except Exception as error:  # pragma: no cover - import failures are environment-specific
        cuda.update({"available": False, "error": str(error)})

    destination = Path(output_root or spec.output_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(destination)

    tools = discover_gguf_tools(spec)
    gguf_required = spec.gguf_enabled and not skip_gguf
    if gguf_required:
        packages.update(
            {
                "google.protobuf": _module_available("google.protobuf"),
                "sentencepiece": _module_available("sentencepiece"),
            }
        )
    converter = {
        "found": tools.converter is not None,
        "path": str(tools.converter) if tools.converter else None,
        "gguf_python_path": bool(
            tools.llama_cpp_root and (tools.llama_cpp_root / "gguf-py").is_dir()
        ),
    }
    quantizer = _probe_executable(tools.quantizer, tools) if gguf_required else None
    llama_cli = _probe_executable(tools.llama_cli, tools) if gguf_required else None

    failures: list[str] = []
    failures.extend(
        f"missing Python package: {name}" for name, found in packages.items() if not found
    )
    if cuda["required"] and not cuda.get("available"):
        failures.append("model.device requires CUDA, but torch.cuda.is_available() is false")
    if (
        cuda["required"]
        and spec.model["dtype"] == "bfloat16"
        and cuda.get("available")
        and not cuda.get("bf16_supported")
    ):
        failures.append("model.dtype requires CUDA BF16 support, but the selected GPU lacks it")
    if gguf_required:
        if not converter["found"]:
            failures.append("GGUF converter not found")
        elif not converter["gguf_python_path"] and importlib.util.find_spec("gguf") is None:
            failures.append("GGUF Python package is unavailable to the converter")
        if not quantizer or not quantizer["usable"]:
            failures.append("llama-quantize was not found or could not start")
        if not llama_cli or not llama_cli["usable"]:
            failures.append("llama-cli was not found or could not start")

    warnings: list[str] = []
    if spec.pipeline.get("quantization") in {"4bit", "8bit"}:
        warnings.append(
            "packed destructive surgery remains an experimental path; compare it to the "
            "one-window floating oracle before making fidelity claims"
        )
    if spec.model["revision"] in (None, "main"):
        warnings.append(
            "model revision is not immutable; the resolved snapshot SHA will be recorded"
        )

    return {
        "ready": not failures,
        "experiment": spec.name,
        "model_source": spec.model["source"],
        "model_revision": spec.model["revision"],
        "model_provenance": {
            "upstream_source": spec.model.get("upstream_source"),
            "upstream_revision": spec.model.get("upstream_revision"),
            "expected_sha256": spec.model.get("expected_sha256", {}),
        },
        "packages": packages,
        "cuda": cuda,
        "disk": {"path": str(destination), "free_bytes": disk.free},
        "gguf": {
            "required": gguf_required,
            "converter": converter,
            "quantizer": quantizer,
            "llama_cli": llama_cli,
        },
        "warnings": warnings,
        "failures": failures,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mark_running_stage_failed(manifest: dict[str, Any], error: BaseException) -> None:
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        return
    for stage in reversed(list(stages.values())):
        if isinstance(stage, dict) and stage.get("status") == "running":
            stage.update(
                {
                    "status": "failed",
                    "ended_at": _utc_now(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            return


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_checkpoint_files(
    checkpoint: Path, expected_sha256: dict[str, str]
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for relative, expected in expected_sha256.items():
        candidate = checkpoint.joinpath(*Path(relative).parts)
        if not candidate.is_file():
            raise RuntimeError(f"pinned checkpoint file is missing: {relative}")
        actual = _sha256(candidate)
        if actual != expected:
            raise RuntimeError(
                f"pinned checkpoint file hash mismatch: {relative}; "
                f"expected {expected}, got {actual}"
            )
        verified[relative] = {
            "sha256": actual,
            "bytes": candidate.stat().st_size,
        }
    return verified


def _git_value(arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _resolved_snapshot_revision(path: Path) -> str | None:
    if path.parent.name == "snapshots":
        return path.name
    return None


def resolve_model_checkpoint(spec: SurgeryExperimentSpec, *, offline: bool = False) -> Path:
    """Resolve a local checkpoint or explicitly download the pinned HF snapshot."""
    source = Path(spec.model["source"]).expanduser()
    if source.exists():
        return source.resolve()

    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=spec.model["source"],
        revision=spec.model["revision"],
        local_files_only=offline or spec.model["local_files_only"],
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
    return Path(resolved).resolve()


def _run_logged_command(
    command: Sequence[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        try:
            with process.stdout:
                for line in process.stdout:
                    encoding = sys.stdout.encoding or "utf-8"
                    console_line = line.encode(encoding, errors="backslashreplace").decode(encoding)
                    print(console_line, end="", flush=True)
                    log.write(line)
                    log.flush()
            exit_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    if exit_code != 0:
        raise RuntimeError(f"command failed with exit code {exit_code}: {command[0]}")


def _convert_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    directory: Path,
    spec: SurgeryExperimentSpec,
    tools: GGUFTools,
) -> Path:
    if not tools.converter or not tools.quantizer:
        raise RuntimeError("GGUF tools disappeared after preflight")
    intermediate = directory / f"{label}.{spec.gguf['outtype']}.gguf"
    quantized = directory / f"{label}.{spec.gguf['quantization']}.gguf"
    # A failed conversion/quantization may leave a partial artifact. These are
    # generated files inside the selected run and are safe to replace on an
    # explicit postprocess retry.
    for generated in (intermediate, quantized):
        if generated.exists():
            generated.unlink()
    env = _subprocess_env(tools)
    _run_logged_command(
        [
            sys.executable,
            str(tools.converter),
            str(checkpoint),
            "--outfile",
            str(intermediate),
            "--outtype",
            spec.gguf["outtype"],
        ],
        log_path=directory / f"{label}.convert.log",
        env=env,
    )
    _run_logged_command(
        [
            str(tools.quantizer),
            str(intermediate),
            str(quantized),
            spec.gguf["quantization"],
        ],
        log_path=directory / f"{label}.quantize.log",
        env=env,
    )
    if not spec.gguf["keep_intermediate"]:
        intermediate.unlink()
    return quantized


def _extract_llama_completion(stdout: str, prompt: str) -> tuple[str, str]:
    """Separate generated text from llama-cli's conversation-mode framing."""
    normalized = stdout.replace("\r\n", "\n")
    marker = f"\n> {prompt}\n"
    marker_at = normalized.rfind(marker)
    if marker_at < 0 and normalized.startswith(f"> {prompt}\n"):
        marker_at = 0
        marker = f"> {prompt}\n"
    if marker_at < 0:
        return stdout.strip(), "raw-fallback"

    completion = normalized[marker_at + len(marker) :].lstrip("\n")
    end_positions = [
        position
        for sentinel in ("\n\n[ Prompt:", "\n\nExiting...")
        if (position := completion.find(sentinel)) >= 0
    ]
    if end_positions:
        completion = completion[: min(end_positions)]
    return completion.strip(), "conversation-marker"


def _gguf_evaluation_cases(spec: SurgeryExperimentSpec) -> list[dict[str, str | None]]:
    if spec.evaluation_enabled:
        return [dict(case) for case in spec.evaluation["cases"]]
    return [
        {"name": None, "prompt": prompt, "expected": None} for prompt in spec.gguf["smoke_prompts"]
    ]


def _gguf_evaluation_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows or not any("verdict" in row for row in rows):
        return None
    from abliteralus.safety_evaluation import summarize_safety_rows

    return summarize_safety_rows(rows)


def _llama_completion(
    model: Path,
    prompt: str,
    *,
    spec: SurgeryExperimentSpec,
    tools: GGUFTools,
    case_name: str | None = None,
    expected: str | None = None,
) -> dict[str, Any]:
    if not tools.llama_cli:
        raise RuntimeError("llama-cli disappeared after preflight")
    command = [
        str(tools.llama_cli),
        "--model",
        str(model),
        "--conversation",
        "--single-turn",
        "--prompt",
        prompt,
        "--predict",
        str(spec.gguf["max_tokens"]),
        "--seed",
        "1",
        "--temp",
        "0",
        "--gpu-layers",
        str(spec.gguf["gpu_layers"]),
        "--no-display-prompt",
        "--no-show-timings",
        "--no-warmup",
        "--simple-io",
        "--log-disable",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subprocess_env(tools),
        check=False,
        timeout=300,
    )
    completion, parse_mode = _extract_llama_completion(result.stdout, prompt)
    refused = None
    refusal_reason = None
    safety_result: dict[str, Any] = {}
    if result.returncode == 0:
        if spec.evaluation_enabled:
            from abliteralus.safety_evaluation import parse_safety_completion

            safety_result = parse_safety_completion(
                completion,
                safe_label=spec.evaluation["safe_label"],
                unsafe_label=spec.evaluation["unsafe_label"],
            )
            safety_result["expected"] = expected
            safety_result["correct"] = (
                safety_result["verdict"] == expected
                if expected is not None and safety_result["verdict"] is not None
                else None
            )
        else:
            from abliteralus.evaluation.advanced_metrics import _is_refusal_detailed

            refused, refusal_reason = _is_refusal_detailed(completion, mode="combined")
    return {
        "name": case_name,
        "prompt": prompt,
        "completion": completion,
        "completion_parse": parse_mode,
        "stdout": result.stdout.strip(),
        "refused": refused,
        "refusal_reason": refusal_reason,
        "exit_code": result.returncode,
        "stderr": result.stderr.strip(),
        **safety_result,
    }


def _run_gguf_stage(
    baseline_checkpoint: Path,
    surgery_checkpoint: Path,
    *,
    run_dir: Path,
    spec: SurgeryExperimentSpec,
) -> dict[str, Any]:
    tools = discover_gguf_tools(spec)
    directory = run_dir / "gguf"
    directory.mkdir(parents=True, exist_ok=True)
    models: dict[str, Path] = {}
    if spec.gguf["compare_baseline"]:
        models["baseline"] = _convert_checkpoint(
            baseline_checkpoint,
            label="baseline",
            directory=directory,
            spec=spec,
            tools=tools,
        )
    models["surgery"] = _convert_checkpoint(
        surgery_checkpoint,
        label="surgery",
        directory=directory,
        spec=spec,
        tools=tools,
    )

    cases = _gguf_evaluation_cases(spec)
    evaluations: dict[str, list[dict[str, Any]]] = {}
    for label, model in models.items():
        evaluations[label] = [
            _llama_completion(
                model,
                str(case["prompt"]),
                spec=spec,
                tools=tools,
                case_name=case["name"],
                expected=case["expected"],
            )
            for case in cases
        ]
    summaries = {
        label: summary
        for label, rows in evaluations.items()
        if (summary := _gguf_evaluation_summary(rows)) is not None
    }
    result = {
        "models": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for label, path in models.items()
        },
        "evaluations": evaluations,
        "evaluation_summaries": summaries,
    }
    _write_json(directory / "smoke-results.json", result)
    return result


def _pipeline_arguments(
    spec: SurgeryExperimentSpec,
    *,
    checkpoint: Path,
    output_dir: Path,
    on_log: Callable[[str], None],
    mutation_sink: MutationHintSink | None = None,
) -> dict[str, Any]:
    from abliteralus.prompts import BUILTIN_HARMFUL, BUILTIN_HARMLESS

    arguments = {
        "model_name": str(checkpoint),
        "output_dir": str(output_dir),
        "device": spec.model["device"],
        "dtype": spec.model["dtype"],
        "trust_remote_code": spec.model["trust_remote_code"],
        "harmful_prompts": list(BUILTIN_HARMFUL[: spec.prompts["harmful"]]),
        "harmless_prompts": list(BUILTIN_HARMLESS[: spec.prompts["harmless"]]),
        "on_log": on_log,
        **spec.pipeline,
    }
    if mutation_sink is not None:
        arguments["mutation_sink"] = mutation_sink
    if spec.prompts["jailbreak"]:
        arguments["jailbreak_prompts"] = list(BUILTIN_HARMFUL[: spec.prompts["jailbreak"]])
    return arguments


def _release_accelerator_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - best-effort cleanup after a completed stage
        pass


def _evaluate_safety_checkpoints(
    baseline_checkpoint: Path,
    surgery_checkpoint: Path,
    *,
    spec: SurgeryExperimentSpec,
    evaluation_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if evaluation_runner is None:
        from abliteralus.safety_evaluation import evaluate_safety_ab

        evaluation_runner = evaluate_safety_ab
    return dict(
        evaluation_runner(
            baseline_checkpoint,
            surgery_checkpoint,
            cases=spec.evaluation["cases"],
            device=spec.model["device"],
            dtype=spec.model["dtype"],
            trust_remote_code=spec.model["trust_remote_code"],
            max_new_tokens=spec.evaluation["max_new_tokens"],
            safe_label=spec.evaluation["safe_label"],
            unsafe_label=spec.evaluation["unsafe_label"],
        )
    )


def _safety_evaluation_summaries(
    evaluation_result: dict[str, Any],
) -> dict[str, Any]:
    model_results = _require_mapping(evaluation_result.get("models"), "evaluation result models")
    return {
        label: _require_mapping(result, f"evaluation result {label}").get("summary")
        for label, result in model_results.items()
    }


def run_experiment(
    spec: SurgeryExperimentSpec,
    *,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    skip_gguf: bool = False,
    offline: bool = False,
    pipeline_factory: Callable[..., Any] | None = None,
    artifact_stage: ArtifactStage | None = None,
    mutation_sink: MutationHintSink | None = None,
    evaluation_runner: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    """Execute the full experiment and return its immutable run directory."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    if _NAME.fullmatch(run_id) is None:
        raise BenchConfigError("run_id contains unsupported characters")
    root = Path(output_root or spec.output_root).expanduser().resolve()
    report = preflight_experiment(spec, output_root=root, skip_gguf=skip_gguf)
    if not report["ready"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["failures"]))

    run_dir = root / spec.name / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Reserve the immutable leaf atomically; a check followed by mkdir can
        # race another controller using the same explicit or second-resolution ID.
        run_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"run directory already exists: {run_dir}") from error

    manifest_path = run_dir / "run-manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": spec.name,
        "run_id": run_id,
        "status": "running",
        "started_at": _utc_now(),
        "spec_path": str(spec.source_path),
        "spec_sha256": _sha256(spec.source_path),
        "git": {
            "commit": _git_value(["rev-parse", "HEAD"]),
            "status": _git_value(["status", "--short"]),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "preflight": report,
        "stages": {},
    }
    _write_json(manifest_path, manifest)

    surgery_log = run_dir / "surgery.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with surgery_log.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    try:
        manifest["stages"]["resolve_model"] = {"status": "running", "started_at": _utc_now()}
        _write_json(manifest_path, manifest)
        checkpoint = resolve_model_checkpoint(spec, offline=offline)
        verified_files = _verify_checkpoint_files(checkpoint, spec.model.get("expected_sha256", {}))
        manifest["model"] = {
            "requested_source": spec.model["source"],
            "requested_revision": spec.model["revision"],
            "checkpoint": str(checkpoint),
            "resolved_revision": _resolved_snapshot_revision(checkpoint),
            "upstream_source": spec.model.get("upstream_source"),
            "upstream_revision": spec.model.get("upstream_revision"),
            "verified_files": verified_files,
        }
        manifest["stages"]["resolve_model"].update({"status": "complete", "ended_at": _utc_now()})

        manifest["stages"]["surgery"] = {"status": "running", "started_at": _utc_now()}
        _write_json(manifest_path, manifest)
        if pipeline_factory is None:
            from abliteralus.abliterate import AbliterationPipeline

            pipeline_factory = AbliterationPipeline
        checkpoint_dir = run_dir / "hf"
        pipeline = pipeline_factory(
            **_pipeline_arguments(
                spec,
                checkpoint=checkpoint,
                output_dir=checkpoint_dir,
                on_log=log,
                mutation_sink=mutation_sink,
            )
        )
        saved_checkpoint = Path(pipeline.run()).resolve()
        manifest["stages"]["surgery"].update(
            {
                "status": "complete",
                "ended_at": _utc_now(),
                "checkpoint": str(saved_checkpoint),
            }
        )
        del pipeline
        _release_accelerator_cache()

        if spec.evaluation_enabled:
            manifest["stages"]["evaluation"] = {
                "status": "running",
                "started_at": _utc_now(),
                "mode": spec.evaluation["mode"],
            }
            _write_json(manifest_path, manifest)
            evaluation_result = _evaluate_safety_checkpoints(
                checkpoint,
                saved_checkpoint,
                spec=spec,
                evaluation_runner=evaluation_runner,
            )
            evaluation_path = run_dir / "evaluation" / "hf-results.json"
            _write_json(evaluation_path, evaluation_result)
            summaries = _safety_evaluation_summaries(evaluation_result)
            manifest["stages"]["evaluation"].update(
                {
                    "status": "complete",
                    "ended_at": _utc_now(),
                    "results": str(evaluation_path),
                    "results_sha256": _sha256(evaluation_path),
                    "summaries": summaries,
                }
            )

        if spec.gguf_enabled and not skip_gguf:
            manifest["stages"]["gguf"] = {"status": "running", "started_at": _utc_now()}
            _write_json(manifest_path, manifest)
            gguf_result = _run_gguf_stage(
                checkpoint,
                saved_checkpoint,
                run_dir=run_dir,
                spec=spec,
            )
            manifest["stages"]["gguf"].update(
                {"status": "complete", "ended_at": _utc_now(), **gguf_result}
            )
        elif spec.gguf_enabled:
            manifest["stages"]["gguf"] = {"status": "skipped", "reason": "--skip-gguf"}

        if spec.capsule_enabled:
            manifest["stages"]["artifact"] = {
                "status": "running",
                "started_at": _utc_now(),
                "mode": "capsule",
            }
            _write_json(manifest_path, manifest)
            if artifact_stage is None:
                if spec.artifact["fallback"] == "error":
                    raise RuntimeError(
                        "capsule mode requires an artifact backend; run through lab-bench or "
                        "surgery_artifacts.integration"
                    )
                artifact_result: dict[str, Any] = {
                    "status": "fallback",
                    "reason": "artifact backend unavailable",
                    "checkpoint": str(saved_checkpoint),
                }
            else:
                artifact_result = dict(
                    artifact_stage(
                        ArtifactStageContext(
                            spec=spec,
                            run_dir=run_dir,
                            base_checkpoint=checkpoint,
                            target_checkpoint=saved_checkpoint,
                            manifest_path=manifest_path,
                            mutation_hints=(
                                mutation_sink.snapshot() if mutation_sink is not None else ()
                            ),
                        )
                    )
                )
            artifact_status = artifact_result.pop("status", "complete")
            manifest["stages"]["artifact"].update(
                {
                    "status": artifact_status,
                    "ended_at": _utc_now(),
                    **artifact_result,
                }
            )
        else:
            manifest["stages"]["artifact"] = {
                "status": "complete",
                "mode": "full-checkpoint",
                "checkpoint": str(saved_checkpoint),
            }

        manifest.update({"status": "complete", "ended_at": _utc_now()})
        _write_json(manifest_path, manifest)
        return run_dir
    except BaseException as error:
        _mark_running_stage_failed(manifest, error)
        manifest.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        (run_dir / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        _write_json(manifest_path, manifest)
        raise


def postprocess_experiment(
    spec: SurgeryExperimentSpec,
    *,
    run_dir: str | Path,
) -> Path:
    """Resume only the GGUF stage of an existing surgery run."""
    if not spec.gguf_enabled:
        raise BenchConfigError("the selected experiment does not enable GGUF postprocessing")
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != spec.name:
        raise BenchConfigError("run manifest experiment does not match the selected config")
    if manifest.get("spec_sha256") != _sha256(spec.source_path):
        raise BenchConfigError("run manifest config hash does not match the selected config")

    model = _require_mapping(manifest.get("model"), "run manifest model")
    surgery_stage = _require_mapping(
        _require_mapping(manifest.get("stages"), "run manifest stages").get("surgery"),
        "run manifest surgery stage",
    )
    baseline_checkpoint = Path(_require_text(model.get("checkpoint"), "baseline checkpoint"))
    surgery_checkpoint = Path(_require_text(surgery_stage.get("checkpoint"), "surgery checkpoint"))
    if not baseline_checkpoint.is_dir() or not surgery_checkpoint.is_dir():
        raise FileNotFoundError("baseline or surgery checkpoint is no longer available")

    report = preflight_experiment(spec, output_root=run_dir.parent, skip_gguf=False)
    if not report["ready"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["failures"]))

    previous = {
        "at": _utc_now(),
        "status": manifest.get("status"),
        "error": manifest.get("error"),
    }
    manifest.setdefault("recoveries", []).append(previous)
    manifest.pop("error", None)
    manifest["status"] = "running"
    manifest["preflight"] = report
    manifest["stages"]["gguf"] = {
        "status": "running",
        "started_at": _utc_now(),
        "resumed": True,
    }
    _write_json(manifest_path, manifest)
    try:
        result = _run_gguf_stage(
            baseline_checkpoint,
            surgery_checkpoint,
            run_dir=run_dir,
            spec=spec,
        )
        manifest["stages"]["gguf"].update({"status": "complete", "ended_at": _utc_now(), **result})
        manifest.update({"status": "complete", "ended_at": _utc_now()})
        _write_json(manifest_path, manifest)
        return run_dir
    except BaseException as error:
        _mark_running_stage_failed(manifest, error)
        manifest.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        (run_dir / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        _write_json(manifest_path, manifest)
        raise


def reevaluate_hf_experiment(
    spec: SurgeryExperimentSpec,
    *,
    run_dir: str | Path,
    surgery_checkpoint: str | Path,
    evaluation_runner: Callable[..., dict[str, Any]] | None = None,
) -> Path:
    """Rerun HF safety A/B evaluation against a preserved surgery checkpoint."""
    if not spec.evaluation_enabled:
        raise BenchConfigError("the selected experiment does not enable HF evaluation")

    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != spec.name:
        raise BenchConfigError("run manifest experiment does not match the selected config")
    if manifest.get("spec_sha256") != _sha256(spec.source_path):
        raise BenchConfigError("run manifest config hash does not match the selected config")

    model = _require_mapping(manifest.get("model"), "run manifest model")
    stages = _require_mapping(manifest.get("stages"), "run manifest stages")
    evaluation_stage = _require_mapping(stages.get("evaluation"), "run manifest evaluation stage")
    if evaluation_stage.get("status") != "complete":
        raise BenchConfigError("the run does not have a complete HF evaluation stage")

    baseline_checkpoint = Path(
        _require_text(model.get("checkpoint"), "baseline checkpoint")
    ).resolve()
    surgery_checkpoint = Path(surgery_checkpoint).expanduser().resolve()
    if not baseline_checkpoint.is_dir() or not surgery_checkpoint.is_dir():
        raise FileNotFoundError("baseline or surgery checkpoint is no longer available")

    artifact_stage = stages.get("artifact")
    artifact_marker: dict[str, Any] | None = None
    if isinstance(artifact_stage, dict) and artifact_stage.get("mode") == "capsule":
        expected_surgery_id = _require_text(
            artifact_stage.get("surgery_id"), "run manifest artifact surgery_id"
        )
        marker_path = surgery_checkpoint / "surgery-artifact.json"
        if not marker_path.is_file():
            raise BenchConfigError(
                "capsule-backed HF reevaluation requires a rehydrated surgery checkpoint marker"
            )
        artifact_marker = _require_mapping(
            json.loads(marker_path.read_text(encoding="utf-8")),
            "rehydrated surgery checkpoint marker",
        )
        if artifact_marker.get("surgery_id") != expected_surgery_id:
            raise BenchConfigError(
                "rehydrated surgery checkpoint marker does not match the run artifact"
            )

    expected_hashes = spec.model.get("expected_sha256", {})
    verified_files = _verify_checkpoint_files(baseline_checkpoint, expected_hashes)
    report = preflight_experiment(spec, output_root=run_dir.parent, skip_gguf=True)
    if not report["ready"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["failures"]))

    evaluation_path = (run_dir / "evaluation" / "hf-results.json").resolve()
    recorded_path = Path(
        _require_text(evaluation_stage.get("results"), "HF evaluation results path")
    ).resolve()
    if recorded_path != evaluation_path or not evaluation_path.is_file():
        raise BenchConfigError("recorded HF evaluation result is missing or outside the run")

    started_at = _utc_now()
    try:
        evaluation_result = _evaluate_safety_checkpoints(
            baseline_checkpoint,
            surgery_checkpoint,
            spec=spec,
            evaluation_runner=evaluation_runner,
        )
        summaries = _safety_evaluation_summaries(evaluation_result)
    except BaseException as error:
        evaluation_stage.setdefault("reevaluation_failures", []).append(
            {
                "started_at": started_at,
                "ended_at": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        manifest["stages"]["evaluation"] = evaluation_stage
        _write_json(manifest_path, manifest)
        raise

    history_dir = evaluation_path.parent / "history"
    history_name = datetime.now(timezone.utc).strftime("hf-results-%Y%m%dt%H%M%S%fz.json")
    history_path = history_dir / history_name
    history_dir.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        raise FileExistsError(f"HF evaluation history entry already exists: {history_path}")
    previous_hash = _sha256(evaluation_path)
    shutil.copy2(evaluation_path, history_path)
    if _sha256(history_path) != previous_hash:
        raise RuntimeError("archived HF evaluation result failed hash verification")

    ended_at = _utc_now()
    evaluation_stage.setdefault("history", []).append(
        {
            "evaluated_at": evaluation_stage.get(
                "reevaluated_at", evaluation_stage.get("ended_at")
            ),
            "results": str(history_path),
            "results_sha256": previous_hash,
            "summaries": evaluation_stage.get("summaries"),
        }
    )
    _write_json(evaluation_path, evaluation_result)
    evaluation_stage.update(
        {
            "results": str(evaluation_path),
            "results_sha256": _sha256(evaluation_path),
            "summaries": summaries,
            "reevaluated_at": ended_at,
            "reevaluation_runtime": {
                "started_at": started_at,
                "ended_at": ended_at,
                "baseline_checkpoint": str(baseline_checkpoint),
                "baseline_verified_files": verified_files,
                "surgery_checkpoint": str(surgery_checkpoint),
                "surgery_artifact": artifact_marker,
                "git": {
                    "commit": _git_value(["rev-parse", "HEAD"]),
                    "status": _git_value(["status", "--short"]),
                },
                "python": sys.version,
                "platform": platform.platform(),
            },
        }
    )
    manifest["stages"]["evaluation"] = evaluation_stage
    _write_json(manifest_path, manifest)
    return run_dir


def reevaluate_gguf_experiment(
    spec: SurgeryExperimentSpec,
    *,
    run_dir: str | Path,
) -> Path:
    """Rerun llama.cpp smoke prompts without repeating surgery or conversion."""
    if not spec.gguf_enabled:
        raise BenchConfigError("the selected experiment does not enable GGUF evaluation")
    run_dir = Path(run_dir).expanduser().resolve()
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != spec.name:
        raise BenchConfigError("run manifest experiment does not match the selected config")
    if manifest.get("spec_sha256") != _sha256(spec.source_path):
        raise BenchConfigError("run manifest config hash does not match the selected config")

    stages = _require_mapping(manifest.get("stages"), "run manifest stages")
    gguf_stage = _require_mapping(stages.get("gguf"), "run manifest GGUF stage")
    if gguf_stage.get("status") != "complete":
        raise BenchConfigError("the run does not have a complete GGUF stage")
    model_records = _require_mapping(gguf_stage.get("models"), "run manifest GGUF models")
    expected_labels = {"surgery"}
    if spec.gguf["compare_baseline"]:
        expected_labels.add("baseline")
    if set(model_records) != expected_labels:
        raise BenchConfigError("run manifest GGUF model set does not match the selected config")

    directory = (run_dir / "gguf").resolve()
    models: dict[str, Path] = {}
    model_hashes: dict[str, str] = {}
    for label in sorted(expected_labels):
        record = _require_mapping(model_records[label], f"run manifest {label} GGUF")
        path = Path(_require_text(record.get("path"), f"{label} GGUF path")).resolve()
        if path.parent != directory or not path.is_file():
            raise BenchConfigError(f"{label} GGUF is missing or outside the run directory")
        recorded_hash = _require_text(record.get("sha256"), f"{label} GGUF hash")
        actual_hash = _sha256(path)
        if actual_hash != recorded_hash:
            raise BenchConfigError(f"{label} GGUF hash no longer matches the run manifest")
        models[label] = path
        model_hashes[label] = actual_hash

    report = preflight_experiment(spec, output_root=run_dir.parent, skip_gguf=False)
    if not report["ready"]:
        raise RuntimeError("preflight failed: " + "; ".join(report["failures"]))
    tools = discover_gguf_tools(spec)
    started_at = _utc_now()
    cases = _gguf_evaluation_cases(spec)
    evaluations = {
        label: [
            _llama_completion(
                model,
                str(case["prompt"]),
                spec=spec,
                tools=tools,
                case_name=case["name"],
                expected=case["expected"],
            )
            for case in cases
        ]
        for label, model in models.items()
    }
    summaries = {
        label: summary
        for label, rows in evaluations.items()
        if (summary := _gguf_evaluation_summary(rows)) is not None
    }
    result = {
        "models": {
            label: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": model_hashes[label],
            }
            for label, path in models.items()
        },
        "evaluations": evaluations,
        "evaluation_summaries": summaries,
    }
    _write_json(directory / "smoke-results.json", result)

    previous_evaluations = gguf_stage.get("evaluations")
    if previous_evaluations is not None:
        gguf_stage.setdefault("smoke_history", []).append(
            {
                "evaluated_at": gguf_stage.get("smoke_evaluated_at", gguf_stage.get("ended_at")),
                "evaluations": previous_evaluations,
            }
        )
    ended_at = _utc_now()
    gguf_stage.update(
        {
            **result,
            "smoke_evaluated_at": ended_at,
            "smoke_runtime": {
                "started_at": started_at,
                "ended_at": ended_at,
                "llama_cli": report["gguf"]["llama_cli"],
                "git": {
                    "commit": _git_value(["rev-parse", "HEAD"]),
                    "status": _git_value(["status", "--short"]),
                },
                "python": sys.version,
                "platform": platform.platform(),
            },
        }
    )
    manifest["stages"]["gguf"] = gguf_stage
    _write_json(manifest_path, manifest)
    return run_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abliteralus-surgery",
        description="Run a reproducible HF surgery and optional GGUF A/B experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "postprocess", "reevaluate-hf", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name in {"preflight", "run"}:
            command.add_argument("--output-root", type=Path)
            command.add_argument("--skip-gguf", action="store_true")
        if name == "run":
            command.add_argument("--run-id")
            command.add_argument("--offline", action="store_true")
        elif name in {"postprocess", "reevaluate-hf", "smoke"}:
            command.add_argument("--run-dir", required=True, type=Path)
            if name == "reevaluate-hf":
                command.add_argument("--surgery-checkpoint", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_experiment_spec(args.config)
    if args.command == "preflight":
        report = preflight_experiment(spec, output_root=args.output_root, skip_gguf=args.skip_gguf)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ready"] else 2
    if args.command == "postprocess":
        print(postprocess_experiment(spec, run_dir=args.run_dir))
        return 0
    if args.command == "reevaluate-hf":
        print(
            reevaluate_hf_experiment(
                spec,
                run_dir=args.run_dir,
                surgery_checkpoint=args.surgery_checkpoint,
            )
        )
        return 0
    if args.command == "smoke":
        print(reevaluate_gguf_experiment(spec, run_dir=args.run_dir))
        return 0
    run_dir = run_experiment(
        spec,
        output_root=args.output_root,
        run_id=args.run_id,
        skip_gguf=args.skip_gguf,
        offline=args.offline,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
