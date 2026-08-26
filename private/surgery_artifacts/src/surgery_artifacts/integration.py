"""Private artifact-stage integration for the public ABLITERALUS surgery runner."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

from abliteralus.artifact_contracts import ArtifactStageContext
from abliteralus.surgery_bench import load_experiment_spec, run_experiment

from .capsule import create_capsule
from .errors import ArtifactError, UnsupportedExportError
from .exporters import export_llama_cpp, export_peft
from .hints import ProjectionHintRecorder
from .validation import validate_capsule


def run_artifact_stage(context: ArtifactStageContext) -> dict[str, Any]:
    """Build, verify, export, and only then retire a temporary HF checkpoint."""

    spec = context.spec
    artifact = spec.artifact
    capsule_path = context.run_dir / "artifact"
    try:
        run_manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
        model = run_manifest.get("model", {})
        base_identity = {
            "source": spec.model["source"],
            "revision": spec.model["revision"],
            "resolved_revision": model.get("resolved_revision"),
        }
        recipe = {
            "experiment": spec.name,
            "spec_path": str(spec.source_path),
            "spec_sha256": run_manifest.get("spec_sha256"),
            "pipeline": spec.pipeline,
            "prompts": spec.prompts,
            "model": spec.model,
            "git": run_manifest.get("git"),
            "runtime": run_manifest.get("runtime"),
        }
        result = create_capsule(
            context.base_checkpoint,
            context.target_checkpoint,
            capsule_path,
            base_identity=base_identity,
            recipe=recipe,
            require_exact=artifact["require_exact"],
            max_low_rank=artifact["max_low_rank"],
            mutation_hints=context.mutation_hints,
        )
        validation = validate_capsule(result.path)
    except Exception as error:
        if artifact["fallback"] == "full-checkpoint":
            return {
                "status": "fallback",
                "mode": "full-checkpoint",
                "reason": f"{type(error).__name__}: {error}",
                "checkpoint": str(context.target_checkpoint),
            }
        raise

    exports: dict[str, dict[str, Any]] = {}
    export_root = context.run_dir / "artifact-exports"
    for export_name in artifact["exports"]:
        try:
            if export_name == "peft":
                path = export_peft(result.path, export_root / "peft")
            elif export_name == "llama_cpp":
                converter = _discover_llama_lora_converter(spec)
                if converter is None:
                    raise UnsupportedExportError(
                        "convert_lora_to_gguf.py is unavailable; set "
                        "ABLITERALUS_LLAMA_CPP_LORA_CONVERTER"
                    )
                path = export_llama_cpp(
                    result.path,
                    export_root / "adapter.gguf",
                    converter=converter,
                    base_checkpoint=context.base_checkpoint,
                )
            else:  # Strict experiment parsing makes this unreachable.
                raise UnsupportedExportError(f"unknown artifact export: {export_name}")
            exports[export_name] = {"status": "complete", "path": str(path)}
        except (ArtifactError, OSError, subprocess.SubprocessError) as error:
            exports[export_name] = {
                "status": "unavailable",
                "reason": f"{type(error).__name__}: {error}",
            }

    cleanup_error: str | None = None
    try:
        removed = _remove_temporary_checkpoint(context)
    except OSError as error:
        removed = False
        cleanup_error = f"{type(error).__name__}: {error}"
    response = {
        "mode": "capsule",
        "path": str(result.path),
        "surgery_id": validation.surgery_id,
        "operation_count": result.operation_count,
        "changed_tensor_count": result.changed_tensor_count,
        "unchanged_tensor_count": result.unchanged_tensor_count,
        "payload_bytes": result.payload_bytes,
        "total_bytes": result.total_bytes,
        "codecs": result.codecs,
        "exports": exports,
        "full_checkpoint_removed": removed,
    }
    if cleanup_error is not None:
        response["checkpoint_cleanup_error"] = cleanup_error
    return response


def _remove_temporary_checkpoint(context: ArtifactStageContext) -> bool:
    target = context.target_checkpoint.resolve()
    run_dir = context.run_dir.resolve()
    if target.parent != run_dir or target.name != "hf":
        raise ArtifactError("refusing to remove a checkpoint outside the run's hf directory")
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _discover_llama_lora_converter(spec: Any) -> Path | None:
    candidates: list[Path] = []
    configured = os.environ.get("ABLITERALUS_LLAMA_CPP_LORA_CONVERTER")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = os.environ.get("LLAMA_CPP_ROOT") or os.environ.get("ABLITERALUS_LLAMA_CPP_ROOT")
    if root:
        candidates.append(Path(root).expanduser() / "convert_lora_to_gguf.py")
    gguf_converter = spec.gguf.get("converter")
    if gguf_converter:
        candidates.append(Path(gguf_converter).expanduser().parent / "convert_lora_to_gguf.py")
    candidates.extend(
        [
            Path.cwd() / "llama.cpp" / "convert_lora_to_gguf.py",
            Path.cwd().parent / "llama.cpp" / "convert_lora_to_gguf.py",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surgery-artifact experiment",
        description="Run ABLITERALUS with the private capsule artifact stage.",
    )
    parser.add_argument("operation", choices=("run",))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_experiment_spec(args.config)
    mutation_sink = ProjectionHintRecorder()
    run_dir = run_experiment(
        spec,
        output_root=args.output_root,
        run_id=args.run_id,
        skip_gguf=args.skip_gguf,
        offline=args.offline,
        artifact_stage=run_artifact_stage,
        mutation_sink=mutation_sink,
    )
    print(run_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
