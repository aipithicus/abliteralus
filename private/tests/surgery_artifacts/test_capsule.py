from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from abliteralus.analysis.numerical_contracts import project_weight_against_direction
from safetensors.torch import load_file, save_file
from surgery_artifacts.errors import UnsupportedExportError
from surgery_artifacts.exporters import export_peft

from surgery_artifacts import (
    ArtifactRegistry,
    CapsuleFormatError,
    create_capsule,
    rehydrate_capsule,
    validate_capsule,
)


def _checkpoint(path: Path, tensors: dict[str, torch.Tensor], *, marker: str) -> Path:
    path.mkdir()
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "marker": marker}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return path


def _sharded_checkpoint(
    path: Path,
    tensors: dict[str, torch.Tensor],
    *,
    marker: str,
) -> Path:
    path.mkdir()
    names = sorted(tensors)
    split = max(1, len(names) // 2)
    groups = (names[:split], names[split:])
    weight_map: dict[str, str] = {}
    for index, group in enumerate(groups, start=1):
        if not group:
            continue
        filename = f"model-{index:05d}-of-00002.safetensors"
        save_file({name: tensors[name] for name in group}, path / filename)
        weight_map.update({name: filename for name in group})
    total_size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    (path / "config.json").write_text(
        json.dumps({"architectures": ["TinyForCausalLM"], "marker": marker}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    return path


def _low_rank_pair(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, torch.Tensor], list[dict[str, object]]]:
    generator = torch.Generator().manual_seed(11)
    base_tensors = {
        "model.layers.0.q_proj.weight": torch.randn(
            (16, 12), generator=generator, dtype=torch.float32
        ),
        "model.embed_tokens.weight": torch.randn(
            (4, 12), generator=generator, dtype=torch.float32
        ),
    }
    direction = torch.randn((12,), generator=generator, dtype=torch.float32)
    target_tensors = {name: value.clone() for name, value in base_tensors.items()}
    target_tensors["model.layers.0.q_proj.weight"] = project_weight_against_direction(
        target_tensors["model.layers.0.q_proj.weight"],
        direction,
        norm_preserve=False,
        regularization=0.2,
    ).weight
    base = _checkpoint(tmp_path / "base", base_tensors, marker="base")
    target = _checkpoint(tmp_path / "target", target_tensors, marker="target")
    (target / "legacy-adapter.pt").write_bytes(b"not copied")
    hints = [
        {
            "parameter": "model.layers.0.q_proj.weight",
            "direction": direction,
            "norm_preserve": False,
            "regularization": 0.2,
            "projection_row_fraction": 1.0,
            "max_norm_ratio": 1.1,
        }
    ]
    return base, target, target_tensors, hints


def test_capsule_round_trip_registry_and_overlay(tmp_path: Path) -> None:
    base, target, target_tensors, hints = _low_rank_pair(tmp_path)
    capsule = tmp_path / "capsule"

    result = create_capsule(
        base,
        target,
        capsule,
        base_identity={
            "source": "aipithicus/tiny",
            "revision": "main",
            "resolved_revision": "a" * 40,
        },
        recipe={"method": "test"},
        mutation_hints=hints,
    )

    validation = validate_capsule(capsule)
    assert validation.surgery_id == result.surgery_id
    assert result.codecs == {"project_direction": 1}
    manifest = json.loads((capsule / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["omitted_unsafe_files"] == ["legacy-adapter.pt"]

    rehydrated = rehydrate_capsule(capsule, base, tmp_path / "rehydrated")
    actual = load_file(rehydrated / "model.safetensors")
    assert actual.keys() == target_tensors.keys()
    for name, expected in target_tensors.items():
        assert torch.equal(actual[name], expected)
    config = json.loads((rehydrated / "config.json").read_text(encoding="utf-8"))
    assert config["marker"] == "target"
    assert not (rehydrated / "legacy-adapter.pt").exists()

    registry = ArtifactRegistry(tmp_path / "registry")
    entry = registry.add(capsule, ref="tiny/first")
    assert entry.created is True
    assert registry.resolve("tiny/first") == entry.path
    assert registry.resolve(entry.surgery_id) == entry.path
    assert registry.add(capsule, ref="tiny/first").created is False


def test_capsule_tamper_is_detected(tmp_path: Path) -> None:
    base, target, _, hints = _low_rank_pair(tmp_path)
    capsule = tmp_path / "capsule"
    create_capsule(
        base,
        target,
        capsule,
        base_identity={"source": "aipithicus/tiny"},
        mutation_hints=hints,
    )
    with (capsule / "recipe.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")

    with pytest.raises(CapsuleFormatError, match="checksum mismatch"):
        validate_capsule(capsule)


def test_surgery_identity_is_independent_of_storage_codec(tmp_path: Path) -> None:
    base, target, _, hints = _low_rank_pair(tmp_path)

    hinted = create_capsule(
        base,
        target,
        tmp_path / "hinted",
        base_identity={"source": "aipithicus/tiny"},
        mutation_hints=hints,
    )
    unhinted = create_capsule(
        base,
        target,
        tmp_path / "unhinted",
        base_identity={"source": "aipithicus/tiny"},
    )

    assert hinted.codecs == {"project_direction": 1}
    assert unhinted.codecs != hinted.codecs
    assert unhinted.surgery_id == hinted.surgery_id


def test_surgery_identity_is_independent_of_safetensors_sharding(tmp_path: Path) -> None:
    base, target, target_tensors, hints = _low_rank_pair(tmp_path)
    base_tensors = load_file(base / "model.safetensors")
    sharded_base = _sharded_checkpoint(
        tmp_path / "base-sharded", base_tensors, marker="base"
    )
    sharded_target = _sharded_checkpoint(
        tmp_path / "target-sharded", target_tensors, marker="target"
    )
    identity = {"source": "aipithicus/tiny", "resolved_revision": "a" * 40}

    single = create_capsule(
        base,
        target,
        tmp_path / "single-capsule",
        base_identity=identity,
        mutation_hints=hints,
    )
    sharded = create_capsule(
        sharded_base,
        sharded_target,
        tmp_path / "sharded-capsule",
        base_identity=identity,
        mutation_hints=hints,
    )

    assert sharded.surgery_id == single.surgery_id


def test_sequential_norm_preserving_projections_rehydrate_exactly(tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(23)
    before = torch.randn((16, 12), generator=generator, dtype=torch.float16)
    directions = [
        torch.randn((12,), generator=generator, dtype=torch.float16),
        torch.randn((12,), generator=generator, dtype=torch.float16),
    ]
    after = before
    hints = []
    for direction in directions:
        after = project_weight_against_direction(
            after,
            direction,
            norm_preserve=True,
            regularization=0.05,
        ).weight
        hints.append(
            {
                "parameter": "model.layers.0.q_proj.weight",
                "direction": direction,
                "norm_preserve": True,
                "regularization": 0.05,
                "projection_row_fraction": 1.0,
                "max_norm_ratio": 1.1,
            }
        )
    base = _checkpoint(
        tmp_path / "base",
        {"model.layers.0.q_proj.weight": before},
        marker="base",
    )
    target = _checkpoint(
        tmp_path / "target",
        {"model.layers.0.q_proj.weight": after},
        marker="target",
    )
    capsule = tmp_path / "capsule"

    result = create_capsule(
        base,
        target,
        capsule,
        base_identity={"source": "aipithicus/tiny"},
        mutation_hints=hints,
    )

    assert result.codecs == {"project_direction": 2}
    rehydrated = rehydrate_capsule(capsule, base, tmp_path / "rehydrated")
    assert torch.equal(
        load_file(rehydrated / "model.safetensors")["model.layers.0.q_proj.weight"],
        after,
    )


def test_peft_export_preserves_native_low_rank_payload(tmp_path: Path) -> None:
    base, target, _, hints = _low_rank_pair(tmp_path)
    capsule = tmp_path / "capsule"
    create_capsule(
        base,
        target,
        capsule,
        base_identity={"source": "aipithicus/tiny"},
        mutation_hints=hints,
    )

    adapter = export_peft(capsule, tmp_path / "peft")

    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["base_model_name_or_path"] == "aipithicus/tiny"
    assert config["target_modules"] == ["model.layers.0.q_proj"]


def test_peft_export_refuses_non_low_rank_capsule(tmp_path: Path) -> None:
    before = {"model.embed_tokens.weight": torch.zeros((8, 8), dtype=torch.float16)}
    after = {name: value.clone() for name, value in before.items()}
    after["model.embed_tokens.weight"][2] = 1
    base = _checkpoint(tmp_path / "base", before, marker="base")
    target = _checkpoint(tmp_path / "target", after, marker="target")
    capsule = tmp_path / "capsule"
    create_capsule(base, target, capsule, base_identity={"source": "aipithicus/tiny"})

    with pytest.raises(UnsupportedExportError, match="verified additive low-rank"):
        export_peft(capsule, tmp_path / "peft")
