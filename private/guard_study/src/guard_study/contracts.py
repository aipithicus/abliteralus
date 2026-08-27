"""Strict, content-addressed contracts for guard-model directional studies."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from .errors import ContractError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DATASET_KEYS = {
    "schema_version",
    "name",
    "description",
    "input_role",
    "safe_label",
    "unsafe_label",
    "content_sha256",
    "splits",
}
_PAIR_KEYS = {"pair_id", "category", "safe", "unsafe"}
_CASE_KEYS = {"name", "prompt"}
_STUDY_KEYS = {
    "schema_version",
    "name",
    "dataset",
    "surgery_experiment",
    "causal_mapping",
    "steering",
    "selection",
    "output",
}


@dataclass(frozen=True)
class GuardCase:
    name: str
    prompt: str
    expected: str
    pair_id: str
    category: str


@dataclass(frozen=True)
class GuardPair:
    pair_id: str
    category: str
    safe: GuardCase
    unsafe: GuardCase


@dataclass(frozen=True)
class GuardDatasetContract:
    schema_version: int
    name: str
    description: str
    input_role: str
    safe_label: str
    unsafe_label: str
    content_sha256: str
    splits: Mapping[str, tuple[GuardPair, ...]]
    path: Path

    def pairs(self, split: str) -> tuple[GuardPair, ...]:
        try:
            return self.splits[split]
        except KeyError as error:
            raise ContractError(f"dataset has no {split!r} split") from error

    def cases(self, split: str) -> tuple[GuardCase, ...]:
        cases: list[GuardCase] = []
        for pair in self.pairs(split):
            cases.extend((pair.safe, pair.unsafe))
        return tuple(cases)

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "input_role": self.input_role,
            "safe_label": self.safe_label,
            "unsafe_label": self.unsafe_label,
            "content_sha256": self.content_sha256,
            "path": str(self.path),
            "splits": {
                name: {
                    "pairs": len(pairs),
                    "cases": len(pairs) * 2,
                    "categories": sorted({pair.category for pair in pairs}),
                }
                for name, pairs in self.splits.items()
            },
        }


@dataclass(frozen=True)
class GuardStudySpec:
    schema_version: int
    name: str
    dataset_path: Path
    surgery_experiment_path: Path
    top_k_layers: int
    max_patch_pairs: int
    doses: tuple[float, ...]
    include_joint_arm: bool
    seed: int
    target_mean_delta: float
    min_parse_rate: float
    min_gap_ratio: float
    output_root: Path
    sha256: str
    path: Path


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _sequence(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a list")
    return list(value)


def _strict_keys(mapping: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ContractError(f"{label} has unknown keys: {', '.join(unknown)}")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be non-empty text")
    return value.strip()


def _name(value: object, label: str) -> str:
    result = _text(value, label)
    if not _NAME.fullmatch(result):
        raise ContractError(
            f"{label} must contain only lowercase letters, digits, '.', '_', or '-'"
        )
    return result


def _integer(value: object, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _fraction(value: object, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ContractError(f"{label} must be between 0 and 1")
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ContractError(f"cannot read contract: {path}") from error
    except yaml.YAMLError as error:
        raise ContractError(f"invalid YAML in {path}") from error
    return _mapping(value, str(path))


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_dataset_digest(path: str | Path) -> str:
    """Return the digest over a dataset contract with its digest field omitted."""

    raw = _load_yaml(Path(path).resolve())
    unsigned = dict(raw)
    unsigned.pop("content_sha256", None)
    return canonical_sha256(unsigned)


def _parse_case(
    value: object,
    *,
    label: str,
    expected: str,
    pair_id: str,
    category: str,
) -> GuardCase:
    raw = _mapping(value, label)
    _strict_keys(raw, _CASE_KEYS, label)
    return GuardCase(
        name=_name(raw.get("name"), f"{label}.name"),
        prompt=_text(raw.get("prompt"), f"{label}.prompt"),
        expected=expected,
        pair_id=pair_id,
        category=category,
    )


def load_dataset_contract(path: str | Path) -> GuardDatasetContract:
    path = Path(path).resolve()
    raw = _load_yaml(path)
    _strict_keys(raw, _DATASET_KEYS, "dataset")
    schema_version = _integer(raw.get("schema_version"), "dataset.schema_version")
    if schema_version != 1:
        raise ContractError("dataset.schema_version must be 1")
    name = _name(raw.get("name"), "dataset.name")
    description = _text(raw.get("description"), "dataset.description")
    input_role = _text(raw.get("input_role"), "dataset.input_role")
    if input_role != "user":
        raise ContractError("dataset.input_role must be 'user' in schema version 1")
    safe_label = _text(raw.get("safe_label"), "dataset.safe_label")
    unsafe_label = _text(raw.get("unsafe_label"), "dataset.unsafe_label")
    if safe_label.casefold() == unsafe_label.casefold():
        raise ContractError("dataset labels must be distinct")

    declared_digest = _text(raw.get("content_sha256"), "dataset.content_sha256").lower()
    if not _SHA256.fullmatch(declared_digest):
        raise ContractError("dataset.content_sha256 must be 64 lowercase hexadecimal characters")
    computed_digest = compute_dataset_digest(path)
    if declared_digest != computed_digest:
        raise ContractError(
            f"dataset content hash mismatch: declared {declared_digest}, computed {computed_digest}"
        )

    raw_splits = _mapping(raw.get("splits"), "dataset.splits")
    if set(raw_splits) != {"fit", "dev", "test"}:
        raise ContractError("dataset.splits must contain exactly: fit, dev, test")

    split_pairs: dict[str, tuple[GuardPair, ...]] = {}
    seen_pair_ids: set[str] = set()
    seen_case_names: set[str] = set()
    seen_prompts: set[str] = set()
    for split_name in ("fit", "dev", "test"):
        values = _sequence(raw_splits[split_name], f"dataset.splits.{split_name}")
        if not values:
            raise ContractError(f"dataset.splits.{split_name} must not be empty")
        pairs: list[GuardPair] = []
        for index, value in enumerate(values):
            label = f"dataset.splits.{split_name}[{index}]"
            pair_raw = _mapping(value, label)
            _strict_keys(pair_raw, _PAIR_KEYS, label)
            pair_id = _name(pair_raw.get("pair_id"), f"{label}.pair_id")
            category = _name(pair_raw.get("category"), f"{label}.category")
            if pair_id in seen_pair_ids:
                raise ContractError(f"duplicate pair_id across splits: {pair_id}")
            seen_pair_ids.add(pair_id)
            safe = _parse_case(
                pair_raw.get("safe"),
                label=f"{label}.safe",
                expected="safe",
                pair_id=pair_id,
                category=category,
            )
            unsafe = _parse_case(
                pair_raw.get("unsafe"),
                label=f"{label}.unsafe",
                expected="unsafe",
                pair_id=pair_id,
                category=category,
            )
            for case in (safe, unsafe):
                if case.name in seen_case_names:
                    raise ContractError(f"duplicate case name across splits: {case.name}")
                seen_case_names.add(case.name)
                prompt_key = case.prompt.casefold()
                if prompt_key in seen_prompts:
                    raise ContractError(f"duplicate prompt across splits: {case.prompt!r}")
                seen_prompts.add(prompt_key)
            pairs.append(GuardPair(pair_id=pair_id, category=category, safe=safe, unsafe=unsafe))
        split_pairs[split_name] = tuple(pairs)

    return GuardDatasetContract(
        schema_version=schema_version,
        name=name,
        description=description,
        input_role=input_role,
        safe_label=safe_label,
        unsafe_label=unsafe_label,
        content_sha256=declared_digest,
        splits=MappingProxyType(split_pairs),
        path=path,
    )


def _relative_path(raw: object, *, base: Path, label: str) -> Path:
    text = _text(raw, label)
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_study_spec(path: str | Path) -> GuardStudySpec:
    path = Path(path).resolve()
    raw = _load_yaml(path)
    _strict_keys(raw, _STUDY_KEYS, "study")
    schema_version = _integer(raw.get("schema_version"), "study.schema_version")
    if schema_version != 1:
        raise ContractError("study.schema_version must be 1")
    name = _name(raw.get("name"), "study.name")
    base = path.parent
    dataset_path = _relative_path(raw.get("dataset"), base=base, label="study.dataset")
    surgery_path = _relative_path(
        raw.get("surgery_experiment"),
        base=base,
        label="study.surgery_experiment",
    )
    if not dataset_path.is_file():
        raise ContractError(f"study.dataset does not exist: {dataset_path}")
    if not surgery_path.is_file():
        raise ContractError(f"study.surgery_experiment does not exist: {surgery_path}")

    causal = _mapping(raw.get("causal_mapping"), "study.causal_mapping")
    _strict_keys(causal, {"top_k_layers", "max_patch_pairs"}, "study.causal_mapping")
    top_k_layers = _integer(causal.get("top_k_layers"), "study.causal_mapping.top_k_layers")
    max_patch_pairs = _integer(
        causal.get("max_patch_pairs"),
        "study.causal_mapping.max_patch_pairs",
    )

    steering = _mapping(raw.get("steering"), "study.steering")
    _strict_keys(steering, {"doses", "include_joint_arm", "seed"}, "study.steering")
    doses = tuple(
        _number(value, f"study.steering.doses[{index}]")
        for index, value in enumerate(_sequence(steering.get("doses"), "study.steering.doses"))
    )
    if len(doses) != len(set(doses)):
        raise ContractError("study.steering.doses must be unique")
    if (
        0.0 not in doses
        or not any(value < 0 for value in doses)
        or not any(value > 0 for value in doses)
    ):
        raise ContractError("study.steering.doses must include zero and both signs")
    include_joint = steering.get("include_joint_arm")
    if not isinstance(include_joint, bool):
        raise ContractError("study.steering.include_joint_arm must be true or false")
    seed = _integer(steering.get("seed"), "study.steering.seed", minimum=0)

    selection = _mapping(raw.get("selection"), "study.selection")
    _strict_keys(
        selection,
        {"target_mean_delta", "min_parse_rate", "min_gap_ratio"},
        "study.selection",
    )
    target_mean_delta = _number(
        selection.get("target_mean_delta"),
        "study.selection.target_mean_delta",
    )
    if target_mean_delta >= 0:
        raise ContractError("study.selection.target_mean_delta must be negative")
    min_parse_rate = _fraction(selection.get("min_parse_rate"), "study.selection.min_parse_rate")
    min_gap_ratio = _fraction(selection.get("min_gap_ratio"), "study.selection.min_gap_ratio")

    output = _mapping(raw.get("output"), "study.output")
    _strict_keys(output, {"root"}, "study.output")
    output_root = _relative_path(output.get("root"), base=base, label="study.output.root")

    return GuardStudySpec(
        schema_version=schema_version,
        name=name,
        dataset_path=dataset_path,
        surgery_experiment_path=surgery_path,
        top_k_layers=top_k_layers,
        max_patch_pairs=max_patch_pairs,
        doses=doses,
        include_joint_arm=include_joint,
        seed=seed,
        target_mean_delta=target_mean_delta,
        min_parse_rate=min_parse_rate,
        min_gap_ratio=min_gap_ratio,
        output_root=output_root,
        sha256=canonical_sha256(raw),
        path=path,
    )


def normalized_dataset(contract: GuardDatasetContract) -> dict[str, Any]:
    """Return the validated dataset in a JSON-serializable canonical shape."""

    return {
        "schema_version": contract.schema_version,
        "name": contract.name,
        "description": contract.description,
        "input_role": contract.input_role,
        "safe_label": contract.safe_label,
        "unsafe_label": contract.unsafe_label,
        "content_sha256": contract.content_sha256,
        "splits": {
            split: [
                {
                    "pair_id": pair.pair_id,
                    "category": pair.category,
                    "safe": {"name": pair.safe.name, "prompt": pair.safe.prompt},
                    "unsafe": {"name": pair.unsafe.name, "prompt": pair.unsafe.prompt},
                }
                for pair in pairs
            ]
            for split, pairs in contract.splits.items()
        },
    }
