"""Conditional llama.cpp GGUF LoRA export through a pinned local converter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .._staging import staging_directory
from ..canonical_json import pretty_text
from ..errors import UnsupportedExportError
from ..hashing import sha256_file
from ..validation import validate_capsule
from .peft import export_peft


def export_llama_cpp(
    capsule: str | Path,
    output: str | Path,
    *,
    converter: str | Path,
    base_checkpoint: str | Path | None = None,
    outtype: str = "f16",
    timeout_seconds: float = 1800.0,
) -> Path:
    """Convert a representable capsule through llama.cpp's official LoRA script."""

    validation = validate_capsule(capsule)
    converter_path = Path(converter).expanduser().resolve()
    if not converter_path.is_file():
        raise UnsupportedExportError(f"llama.cpp converter is unavailable: {converter_path}")
    if outtype not in {"f32", "f16", "bf16", "q8_0", "auto"}:
        raise ValueError("llama.cpp LoRA outtype must be f32, f16, bf16, q8_0, or auto")
    if timeout_seconds <= 0:
        raise ValueError("llama.cpp converter timeout must be positive")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"llama.cpp export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with staging_directory(
        destination.parent,
        prefix=f".{destination.name}.llama-",
    ) as temporary:
        peft = export_peft(validation.path, temporary / "peft")
        candidate = temporary / "adapter.gguf"
        command = [
            sys.executable,
            str(converter_path),
            str(peft),
            "--outfile",
            str(candidate),
            "--outtype",
            outtype,
        ]
        if base_checkpoint is not None:
            base = Path(base_checkpoint).expanduser().resolve()
            if not base.is_dir():
                raise UnsupportedExportError(f"llama.cpp base directory is unavailable: {base}")
            command.extend(["--base", str(base)])
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            close_fds=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise UnsupportedExportError(
                f"llama.cpp LoRA conversion failed with exit code {completed.returncode}: "
                f"{detail[-1000:]}"
            )
        if not candidate.is_file() or candidate.stat().st_size < 8:
            raise UnsupportedExportError("llama.cpp converter did not produce a GGUF adapter")
        if candidate.read_bytes()[:4] != b"GGUF":
            raise UnsupportedExportError("llama.cpp converter output has no GGUF magic")
        os.replace(candidate, destination)

    provenance = {
        "schema_version": 1,
        "format": "llama.cpp-gguf-lora",
        "surgery_id": validation.surgery_id,
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
        "outtype": outtype,
        "converter": {
            "path": str(converter_path),
            "sha256": sha256_file(converter_path),
            "git_commit": _git_commit(converter_path.parent),
        },
        "verified": {"gguf_magic": True, "conversion_exit_code": 0},
    }
    sidecar = destination.with_suffix(destination.suffix + ".surgery-export.json")
    sidecar.write_text(pretty_text(provenance), encoding="utf-8")
    return destination


def _git_commit(start: Path) -> str | None:
    for candidate in (start, *start.parents):
        if not (candidate / ".git").exists():
            continue
        try:
            completed = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
                close_fds=True,
            )
        except OSError:
            return None
        commit = completed.stdout.strip()
        return commit if completed.returncode == 0 and commit else None
    return None
