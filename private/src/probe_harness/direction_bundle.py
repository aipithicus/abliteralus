"""Strict consumer for complete guard-study direction run directories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from safetensors import safe_open
import torch

from .contracts import json_sha256
from .errors import ProbeHarnessError


BUNDLE_SCHEMA = "https://aipithicus.org/schemas/probe-harness/direction-bundle-v1"
SUPPORTED_METRIC = "unsafe_minus_safe_logit_margin"
_LAYER_KEY = re.compile(r"^layer\.(\d+)\.direction$")
_DATASET_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FILES = (
    "run-manifest.json",
    "dataset-contract.json",
    "causal-map.json",
    "directions.safetensors",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeHarnessError(f"could not read direction-bundle JSON {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ProbeHarnessError(f"direction-bundle JSON is not an object: {path.name}")
    return value


def _required_int(value: Mapping[str, Any], key: str, *, context: str) -> int:
    candidate = value.get(key)
    if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate <= 0:
        raise ProbeHarnessError(f"{context} {key} must be a positive integer")
    return candidate


def _required_finite(value: Mapping[str, Any], key: str, *, context: str) -> float:
    candidate = value.get(key)
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
        raise ProbeHarnessError(f"{context} {key} must be numeric")
    result = float(candidate)
    if not math.isfinite(result):
        raise ProbeHarnessError(f"{context} {key} must be finite")
    return result


def _normalized_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProbeHarnessError(f"{context} checkpoint path is missing")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ProbeHarnessError(f"{context} checkpoint identity is not an absolute path")
    return candidate.resolve()


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


@dataclass(frozen=True, slots=True)
class DirectionEntry:
    """One named direction and the layer/statistics needed to interpret it."""

    name: str
    tensor_key: str
    kind: str
    layer_index: int | None
    vector: torch.Tensor
    statistics: Mapping[str, float]

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tensor_key": self.tensor_key,
            "kind": self.kind,
            "layer_index": self.layer_index,
            "hidden_size": int(self.vector.numel()),
            "l2_norm": float(self.vector.norm().item()),
            "statistics": dict(self.statistics),
        }


@dataclass(slots=True)
class DirectionBundle:
    """A validated immutable view over one complete guard-study run."""

    run_directory: Path
    identity: dict[str, Any]
    directions: dict[str, DirectionEntry]

    @classmethod
    def load(
        cls,
        run_directory: str | Path,
        *,
        subject_identity: Mapping[str, Any],
    ) -> "DirectionBundle":
        run = Path(run_directory).resolve()
        if not run.is_dir():
            if run.is_file() and run.name == "directions.safetensors":
                raise ProbeHarnessError(
                    "load the complete guard-study run directory, not loose directions.safetensors"
                )
            raise ProbeHarnessError(f"guard-study run directory does not exist: {run}")
        files = {name: run / name for name in _REQUIRED_FILES}
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise ProbeHarnessError(
                "guard-study direction bundle is incomplete; missing: " + ", ".join(missing)
            )

        manifest = _read_json_object(files["run-manifest.json"])
        dataset_contract = _read_json_object(files["dataset-contract.json"])
        causal_map = _read_json_object(files["causal-map.json"])
        if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            raise ProbeHarnessError("guard-study run manifest is not a complete schema-v1 run")
        if manifest.get("weights_mutated") is not False:
            raise ProbeHarnessError("direction bundle does not declare an unmodified fit checkpoint")

        subject_checkpoint = _normalized_path(
            subject_identity.get("checkpoint_path"),
            context="subject",
        )
        bundle_checkpoint = _normalized_path(manifest.get("checkpoint"), context="bundle")
        if not _same_path(subject_checkpoint, bundle_checkpoint):
            raise ProbeHarnessError(
                "direction bundle checkpoint identity does not match the resident subject"
            )

        hidden_size = _required_int(subject_identity, "hidden_size", context="subject")
        layer_count = _required_int(subject_identity, "num_layers", context="subject")
        manifest_dataset = manifest.get("dataset")
        if not isinstance(manifest_dataset, Mapping):
            raise ProbeHarnessError("direction bundle manifest has no dataset identity")
        dataset_digest = manifest_dataset.get("content_sha256")
        if not isinstance(dataset_digest, str) or not _DATASET_DIGEST.fullmatch(dataset_digest):
            raise ProbeHarnessError("direction bundle dataset digest is invalid")
        if dataset_contract.get("content_sha256") != dataset_digest:
            raise ProbeHarnessError(
                "direction bundle dataset digest does not match its run manifest"
            )

        tensor_path = files["directions.safetensors"]
        try:
            with safe_open(str(tensor_path), framework="pt", device="cpu") as archive:
                metadata = dict(archive.metadata() or {})
                tensors = {
                    key: archive.get_tensor(key).detach().float().cpu().contiguous()
                    for key in sorted(archive.keys())
                }
        except Exception as error:
            raise ProbeHarnessError(f"could not read directions.safetensors: {error}") from error
        metric = metadata.get("metric")
        if metric != SUPPORTED_METRIC or causal_map.get("metric") != metric:
            raise ProbeHarnessError(
                f"direction bundle metric must be {SUPPORTED_METRIC!r}"
            )
        if metadata.get("dataset_sha256") != dataset_digest:
            raise ProbeHarnessError(
                "directions.safetensors dataset digest does not match the run manifest"
            )

        per_layer = causal_map.get("per_layer")
        if not isinstance(per_layer, Mapping):
            raise ProbeHarnessError("direction bundle causal map has no per-layer statistics")
        directions: dict[str, DirectionEntry] = {}
        learned_layers: set[int] = set()
        for tensor_key, tensor in tensors.items():
            if tensor.ndim != 1 or int(tensor.numel()) != hidden_size:
                raise ProbeHarnessError(
                    f"direction tensor {tensor_key!r} does not match hidden size {hidden_size}"
                )
            if not torch.isfinite(tensor).all() or float(tensor.norm().item()) == 0.0:
                raise ProbeHarnessError(f"direction tensor {tensor_key!r} is non-finite or zero")
            match = _LAYER_KEY.fullmatch(tensor_key)
            if match is not None:
                layer_index = int(match.group(1))
                if layer_index < 0 or layer_index >= layer_count:
                    raise ProbeHarnessError(
                        f"direction tensor {tensor_key!r} is outside the subject layer range"
                    )
                stats_value = per_layer.get(str(layer_index))
                if not isinstance(stats_value, Mapping):
                    raise ProbeHarnessError(
                        f"direction bundle has no statistics for layer {layer_index}"
                    )
                statistics = {
                    key: _required_finite(
                        stats_value,
                        key,
                        context=f"layer {layer_index}",
                    )
                    for key in (
                        "projection_std",
                        "safe_projection_mean",
                        "unsafe_projection_mean",
                        "projection_gap",
                    )
                }
                if statistics["projection_std"] <= 0.0 or statistics["projection_gap"] <= 0.0:
                    raise ProbeHarnessError(
                        f"layer {layer_index} direction scale or orientation is invalid"
                    )
                if not math.isclose(float(tensor.norm().item()), 1.0, abs_tol=1e-3):
                    raise ProbeHarnessError(f"learned layer {layer_index} direction is not unit norm")
                name = f"learned.layer_{layer_index}"
                directions[name] = DirectionEntry(
                    name=name,
                    tensor_key=tensor_key,
                    kind="learned",
                    layer_index=layer_index,
                    vector=tensor,
                    statistics=statistics,
                )
                learned_layers.add(layer_index)
            elif tensor_key == "control.label_axis.direction":
                name = "control.label_axis"
                directions[name] = DirectionEntry(
                    name=name,
                    tensor_key=tensor_key,
                    kind="label-axis-control",
                    layer_index=None,
                    vector=tensor,
                    statistics={},
                )
            else:
                raise ProbeHarnessError(f"unsupported direction tensor key: {tensor_key}")

        expected_layers = set(range(layer_count))
        if learned_layers != expected_layers:
            missing_layers = sorted(expected_layers - learned_layers)
            extra_layers = sorted(learned_layers - expected_layers)
            raise ProbeHarnessError(
                "direction bundle layer coverage does not match the subject; "
                f"missing={missing_layers}, extra={extra_layers}"
            )
        if "control.label_axis" not in directions:
            raise ProbeHarnessError("direction bundle has no label-axis control")

        file_identity = {
            name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        }
        identity_basis = {
            "schema": BUNDLE_SCHEMA,
            "schema_version": 1,
            "kind": "guard-study-run",
            "checkpoint_path": str(bundle_checkpoint),
            "dataset_sha256": dataset_digest,
            "metric": metric,
            "hidden_size": hidden_size,
            "num_layers": layer_count,
            "files": file_identity,
            "direction_names": sorted(directions),
        }
        identity = {
            **identity_basis,
            "run_directory": str(run),
            "digest": json_sha256(identity_basis),
        }
        return cls(run_directory=run, identity=identity, directions=directions)

    def resolve(self, name: str) -> DirectionEntry:
        if name in self.directions:
            return self.directions[name]
        matches = [entry for key, entry in self.directions.items() if key.startswith(name)]
        if not matches:
            raise ProbeHarnessError(f"unknown direction: {name}")
        if len(matches) > 1:
            raise ProbeHarnessError(f"ambiguous direction prefix: {name}")
        return matches[0]

    def project(self, name: str, activation: torch.Tensor) -> dict[str, Any]:
        """Project one decision vector or token sequence onto a named learned direction."""

        entry = self.resolve(name)
        if entry.layer_index is None:
            raise ProbeHarnessError(
                f"direction {entry.name} targets the post-final-norm hidden state, not a layer capture"
            )
        values = activation.detach().float().cpu()
        if values.ndim not in {1, 2} or int(values.shape[-1]) != int(entry.vector.numel()):
            raise ProbeHarnessError(
                f"captured activation does not match direction {entry.name}"
            )
        raw = values @ entry.vector
        activation_norm = values.norm(dim=-1)
        direction_norm = float(entry.vector.norm().item())
        denominator = activation_norm * direction_norm
        cosine = torch.where(denominator > 0, raw / denominator, torch.zeros_like(raw))
        scale = float(entry.statistics["projection_std"])
        midpoint = (
            float(entry.statistics["safe_projection_mean"])
            + float(entry.statistics["unsafe_projection_mean"])
        ) / 2.0
        standardized = raw / scale
        midpoint_standardized = (raw - midpoint) / scale
        raw_values = raw.reshape(-1)
        cosine_values = cosine.reshape(-1)
        standardized_values = standardized.reshape(-1)
        midpoint_values = midpoint_standardized.reshape(-1)
        profile = [
            {
                "position": index,
                "raw_projection": float(raw_value),
                "cosine_similarity": float(cosine_values[index].item()),
                "projection_over_fit_std": float(standardized_values[index].item()),
                "midpoint_standardized_projection": float(midpoint_values[index].item()),
            }
            for index, raw_value in enumerate(raw_values.tolist())
        ]
        return {
            "direction": entry.summary(),
            "position_count": len(profile),
            "raw_projection_min": float(raw_values.min().item()),
            "raw_projection_max": float(raw_values.max().item()),
            "raw_projection_mean": float(raw_values.mean().item()),
            "raw_projection_last": float(raw_values[-1].item()),
            "profile": profile,
        }

    def summary(self) -> dict[str, Any]:
        ordered = sorted(
            self.directions.values(),
            key=lambda entry: (
                entry.layer_index is None,
                entry.layer_index if entry.layer_index is not None else entry.name,
            ),
        )
        return {
            **self.identity,
            "directions": [entry.summary() for entry in ordered],
        }
