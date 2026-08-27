from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from guard_study.contracts import (
    canonical_sha256,
    load_dataset_contract,
    load_study_spec,
)
from guard_study.errors import ContractError


def _dataset_mapping() -> dict:
    def pair(name: str, safe_prompt: str, unsafe_prompt: str) -> dict:
        return {
            "pair_id": name,
            "category": "test-category",
            "safe": {"name": f"{name}-safe", "prompt": safe_prompt},
            "unsafe": {"name": f"{name}-unsafe", "prompt": unsafe_prompt},
        }

    value = {
        "schema_version": 1,
        "name": "test-dataset",
        "description": "A deterministic unit-test contract.",
        "input_role": "user",
        "safe_label": "safe",
        "unsafe_label": "unsafe",
        "splits": {
            "fit": [pair("fit-pair", "fit safe", "fit unsafe")],
            "dev": [pair("dev-pair", "dev safe", "dev unsafe")],
            "test": [pair("test-pair", "test safe", "test unsafe")],
        },
    }
    value["content_sha256"] = canonical_sha256(value)
    return value


def _write_yaml(path: Path, value: dict) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_dataset_contract_is_content_addressed_and_non_overlapping(tmp_path):
    path = _write_yaml(tmp_path / "dataset.yaml", _dataset_mapping())

    contract = load_dataset_contract(path)

    assert contract.content_sha256 == _dataset_mapping()["content_sha256"]
    assert [case.expected for case in contract.cases("fit")] == ["safe", "unsafe"]
    assert contract.summary()["splits"]["test"]["pairs"] == 1


def test_dataset_contract_rejects_content_changed_without_new_digest(tmp_path):
    value = _dataset_mapping()
    value["splits"]["test"][0]["safe"]["prompt"] = "changed after hashing"
    path = _write_yaml(tmp_path / "dataset.yaml", value)

    with pytest.raises(ContractError, match="content hash mismatch"):
        load_dataset_contract(path)


def test_dataset_contract_rejects_prompt_leakage_between_splits(tmp_path):
    value = _dataset_mapping()
    value["splits"]["test"][0]["safe"]["prompt"] = "fit safe"
    unsigned = dict(value)
    unsigned.pop("content_sha256")
    value["content_sha256"] = canonical_sha256(unsigned)
    path = _write_yaml(tmp_path / "dataset.yaml", value)

    with pytest.raises(ContractError, match="duplicate prompt across splits"):
        load_dataset_contract(path)


def test_study_contract_requires_a_bidirectional_dose_grid(tmp_path):
    dataset = _write_yaml(tmp_path / "dataset.yaml", _dataset_mapping())
    surgery = tmp_path / "surgery.yaml"
    surgery.write_text("schema_version: 1\n", encoding="utf-8")
    study_value = {
        "schema_version": 1,
        "name": "test-study",
        "dataset": dataset.name,
        "surgery_experiment": surgery.name,
        "causal_mapping": {"top_k_layers": 2, "max_patch_pairs": 1},
        "steering": {
            "doses": [-1.0, 0.0, 1.0],
            "include_joint_arm": True,
            "seed": 7,
        },
        "selection": {
            "target_mean_delta": -0.25,
            "min_parse_rate": 1.0,
            "min_gap_ratio": 0.8,
        },
        "output": {"root": "outputs"},
    }
    study_path = _write_yaml(tmp_path / "study.yaml", study_value)

    study = load_study_spec(study_path)

    assert study.dataset_path == dataset.resolve()
    assert study.doses == (-1.0, 0.0, 1.0)

    study_value["steering"]["doses"] = [0.0, 1.0]
    _write_yaml(study_path, study_value)
    with pytest.raises(ContractError, match="both signs"):
        load_study_spec(study_path)
