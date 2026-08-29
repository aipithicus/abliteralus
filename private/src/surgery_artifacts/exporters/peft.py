"""Export exactly additive low-rank capsule operations as a PEFT LoRA adapter."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .._staging import create_staging_directory
from ..canonical_json import pretty_text
from ..errors import CapsuleFormatError, UnsupportedExportError
from ..hashing import sha256_file, tensor_sha256
from ..validation import validate_capsule


def export_peft(capsule: str | Path, output: str | Path) -> Path:
    """Create a PEFT LoRA directory when every native operation is representable."""

    capsule_root = validate_capsule(capsule).path
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"PEFT export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(capsule_root / "manifest.json")
    operations = manifest["operations"]
    if any(not _peft_compatible(operation) for operation in operations):
        kinds = sorted({str(operation.get("kind")) for operation in operations})
        raise UnsupportedExportError(
            "PEFT export requires exclusively verified additive low-rank projection operations; "
            f"capsule contains {', '.join(kinds)}"
        )

    adapter_tensors: dict[str, torch.Tensor] = {}
    target_modules: list[str] = []
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    factors: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    payload_path = capsule_root / "delta.safetensors"
    with safe_open(payload_path, framework="pt", device="cpu") as payload:
        for operation in operations:
            parameter = operation.get("parameter")
            if not isinstance(parameter, str) or not parameter.endswith(".weight"):
                raise UnsupportedExportError(
                    f"PEFT cannot address capsule parameter {parameter!r} as a module weight"
                )
            module = parameter[: -len(".weight")]
            records = operation.get("payload")
            if not isinstance(records, dict) or not {"a", "b"}.issubset(records):
                raise CapsuleFormatError("low-rank operation does not contain A/B payloads")
            a = payload.get_tensor(records["a"]["key"]).contiguous()
            b = payload.get_tensor(records["b"]["key"]).contiguous()
            if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
                raise CapsuleFormatError(f"low-rank payload shapes are invalid for {parameter}")
            if int(a.shape[0]) <= 0:
                raise CapsuleFormatError(f"low-rank payload is empty for {parameter}")
            pair = factors.setdefault(module, ([], []))
            pair[0].append(a)
            pair[1].append(b)

        for module, (a_parts, b_parts) in sorted(factors.items()):
            a = torch.cat(a_parts, dim=0)
            b = torch.cat(b_parts, dim=1)
            rank = int(a.shape[0])
            prefix = f"base_model.model.{module}"
            adapter_tensors[f"{prefix}.lora_A.weight"] = a
            adapter_tensors[f"{prefix}.lora_B.weight"] = b
            target_modules.append(module)
            rank_pattern[module] = rank
            alpha_pattern[module] = rank

    base = manifest.get("base")
    source = base.get("source") if isinstance(base, dict) else None
    revision = base.get("resolved_revision") if isinstance(base, dict) else None
    if not isinstance(source, str) or not source:
        raise UnsupportedExportError("capsule base identity has no PEFT base model source")
    ranks = sorted(set(rank_pattern.values()))
    default_rank = ranks[-1]
    config: dict[str, Any] = {
        "alpha_pattern": alpha_pattern if len(ranks) > 1 else {},
        "auto_mapping": None,
        "base_model_name_or_path": source,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": default_rank,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": default_rank,
        "rank_pattern": rank_pattern if len(ranks) > 1 else {},
        "revision": revision,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }

    temporary = create_staging_directory(
        destination.parent,
        prefix=f".{destination.name}.tmp-",
    )
    try:
        save_file(
            adapter_tensors,
            temporary / "adapter_model.safetensors",
            metadata={"format": "pt", "surgery_id": manifest["surgery_id"]},
        )
        (temporary / "adapter_config.json").write_text(pretty_text(config), encoding="utf-8")
        export_manifest = {
            "schema_version": 1,
            "format": "peft-lora",
            "surgery_id": manifest["surgery_id"],
            "adapter_sha256": sha256_file(temporary / "adapter_model.safetensors"),
            "operation_count": len(operations),
            "verified": True,
        }
        (temporary / "surgery-export.json").write_text(
            pretty_text(export_manifest), encoding="utf-8"
        )
        _verify_export(temporary, operations, capsule_root)
        os.replace(temporary, destination)
        _verify_export(destination, operations, capsule_root)
        return destination
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _verify_export(
    directory: Path,
    operations: list[dict[str, Any]],
    capsule_root: Path,
) -> None:
    config = _read_json(directory / "adapter_config.json")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise UnsupportedExportError("PEFT adapter configuration failed verification")
    with safe_open(
        capsule_root / "delta.safetensors", framework="pt", device="cpu"
    ) as native, safe_open(
        directory / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as exported:
        expected_keys: set[str] = set()
        expected_factors: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
        for operation in operations:
            module = operation["parameter"][: -len(".weight")]
            pair = expected_factors.setdefault(module, ([], []))
            pair[0].append(native.get_tensor(operation["payload"]["a"]["key"]))
            pair[1].append(native.get_tensor(operation["payload"]["b"]["key"]))
        for module, (a_parts, b_parts) in expected_factors.items():
            for expected, peft_name in (
                (torch.cat(a_parts, dim=0), "lora_A.weight"),
                (torch.cat(b_parts, dim=1), "lora_B.weight"),
            ):
                key = f"base_model.model.{module}.{peft_name}"
                expected_keys.add(key)
                exported_tensor = exported.get_tensor(key)
                if tensor_sha256(expected) != tensor_sha256(exported_tensor):
                    raise UnsupportedExportError(f"PEFT payload changed during export: {key}")
        if set(exported.keys()) != expected_keys:
            raise UnsupportedExportError("PEFT export contains unexpected adapter tensors")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CapsuleFormatError(f"JSON file is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise CapsuleFormatError(f"JSON file must contain an object: {path}")
    return value


def _peft_compatible(operation: object) -> bool:
    if not isinstance(operation, dict):
        return False
    if operation.get("kind") == "add_low_rank":
        return True
    return operation.get("kind") == "project_direction" and operation.get("metadata", {}).get(
        "peft_compatible"
    ) is True
