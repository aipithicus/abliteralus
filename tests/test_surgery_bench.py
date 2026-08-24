"""Contract tests for the reproducible surgery experiment bench."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import yaml

import abliteralus.surgery_bench as bench
from abliteralus.surgery_bench import (
    BenchConfigError,
    discover_gguf_tools,
    load_experiment_spec,
    preflight_experiment,
    run_experiment,
)


pytestmark = pytest.mark.cpu


def _write_config(
    tmp_path: Path,
    *,
    model_source: str | None = None,
    gguf: dict | None = None,
) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "unit-mini",
                "model": {
                    "source": model_source or str(checkpoint),
                    "device": "cpu",
                    "dtype": "float32",
                },
                "pipeline": {"method": "basic", "n_directions": 1},
                "prompts": {"harmful": 3, "harmless": 2, "jailbreak": 0},
                "output": {"root": str(tmp_path / "outputs")},
                "gguf": gguf or {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_checked_in_profiles_are_valid_and_pin_model_revisions():
    repo = Path(__file__).resolve().parents[1]
    local = load_experiment_spec(repo / "experiments/surgery/local-qwen25-0.5b.yaml")
    remote = load_experiment_spec(repo / "experiments/surgery/lightning-qwen25-7b.yaml")

    assert local.model["revision"] == "c89bee90d9f811437d9735454613c35b4a3c4dc8"
    assert local.gguf_enabled is True
    assert local.gguf["compare_baseline"] is True
    assert remote.model["revision"] == "a09a35458c702b33eeacc393d103063234e8bc28"
    assert remote.gguf_enabled is False


def test_direct_gguf_surgery_is_rejected(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    path = _write_config(tmp_path, model_source=str(gguf))

    with pytest.raises(BenchConfigError, match="postprocess"):
        load_experiment_spec(path)


def test_unknown_pipeline_key_is_rejected(tmp_path):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["pipeline"]["typoed_strength"] = 1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(BenchConfigError, match="unknown pipeline keys"):
        load_experiment_spec(path)


def test_preflight_without_gguf_is_cpu_executable(tmp_path):
    spec = load_experiment_spec(_write_config(tmp_path))

    report = preflight_experiment(spec)

    assert report["ready"] is True
    assert report["gguf"]["required"] is False
    assert report["failures"] == []


def test_tool_discovery_finds_sibling_llama_source_checkout(tmp_path):
    portable = tmp_path / "portable"
    binary_dir = portable / "llama.cpp-build" / "bin"
    source_dir = portable / "llama.cpp"
    binary_dir.mkdir(parents=True)
    source_dir.mkdir()
    quantizer = binary_dir / "llama-quantize"
    cli = binary_dir / "llama-cli"
    converter = source_dir / "convert_hf_to_gguf.py"
    for path in (quantizer, cli, converter):
        path.write_text("", encoding="utf-8")

    spec = load_experiment_spec(
        _write_config(
            tmp_path,
            gguf={
                "enabled": True,
                "quantizer": str(quantizer),
                "llama_cli": str(cli),
            },
        )
    )
    tools = discover_gguf_tools(spec)

    assert tools.converter == converter.resolve()
    assert tools.llama_cpp_root == source_dir.resolve()


def test_run_records_manifest_and_enforces_prompt_limits(tmp_path):
    spec = load_experiment_spec(_write_config(tmp_path))
    captured: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.output_dir = Path(kwargs["output_dir"])

        def run(self):
            self.output_dir.mkdir(parents=True)
            (self.output_dir / "config.json").write_text("{}", encoding="utf-8")
            return self.output_dir

    run_dir = run_experiment(
        spec,
        run_id="test-run",
        pipeline_factory=FakePipeline,
    )
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))

    assert manifest["status"] == "complete"
    assert manifest["stages"]["surgery"]["status"] == "complete"
    assert len(captured["harmful_prompts"]) == 3
    assert len(captured["harmless_prompts"]) == 2
    assert captured["model_name"] == str((tmp_path / "checkpoint").resolve())


def test_failed_pipeline_leaves_diagnostic_manifest(tmp_path):
    spec = load_experiment_spec(_write_config(tmp_path))

    class BrokenPipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            raise RuntimeError("synthetic surgery failure")

    with pytest.raises(RuntimeError, match="synthetic surgery failure"):
        run_experiment(spec, run_id="failed-run", pipeline_factory=BrokenPipeline)

    run_dir = tmp_path / "outputs" / "unit-mini" / "failed-run"
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"] == {
        "message": "synthetic surgery failure",
        "type": "RuntimeError",
    }
    assert manifest["stages"]["surgery"]["status"] == "failed"
    assert (run_dir / "failure-traceback.log").is_file()


def test_local_model_source_must_be_a_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"weights")

    with pytest.raises(BenchConfigError, match="checkpoint directory"):
        load_experiment_spec(_write_config(tmp_path, model_source=str(checkpoint)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update(schema_version=2), "schema_version"),
        (lambda raw: raw.update(name="Bad Name"), "name must start"),
        (lambda raw: raw["prompts"].update(harmful=0), "must both be positive"),
        (lambda raw: raw["pipeline"].update(quantization="3bit"), "4bit"),
        (lambda raw: raw["pipeline"].update(n_directions=0), "positive integer"),
        (lambda raw: raw["pipeline"].update(norm_preserve="yes"), "true or false"),
        (lambda raw: raw["model"].update(dtype="float64"), "model.dtype"),
        (lambda raw: raw["gguf"].update(outtype="int4"), "gguf.outtype"),
    ],
)
def test_spec_rejects_invalid_contract_values(tmp_path, mutate, message):
    path = _write_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(raw)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(BenchConfigError, match=message):
        load_experiment_spec(path)


def test_snapshot_resolution_passes_pin_and_offline_policy(tmp_path, monkeypatch):
    path = _write_config(tmp_path, model_source="owner/model")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["model"]["revision"] = "0123456789abcdef"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    spec = load_experiment_spec(path)
    snapshot = tmp_path / "hub/models--owner--model/snapshots/0123456789abcdef"
    snapshot.mkdir(parents=True)
    captured = {}

    def fake_download(**kwargs):
        captured.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)

    assert bench.resolve_model_checkpoint(spec, offline=True) == snapshot.resolve()
    assert captured["repo_id"] == "owner/model"
    assert captured["revision"] == "0123456789abcdef"
    assert captured["local_files_only"] is True
    assert "*.gguf" in captured["ignore_patterns"]


def test_usage_output_counts_as_successful_quantizer_probe(tmp_path, monkeypatch):
    executable = tmp_path / "llama-quantize"
    executable.write_text("", encoding="utf-8")
    tools = bench.GGUFTools(None, executable, None, None)

    monkeypatch.setattr(
        bench.subprocess,
        "run",
        lambda *_args, **_kwargs: bench.subprocess.CompletedProcess(
            [], 1, stdout="usage: llama-quantize model type\nmore", stderr=""
        ),
    )
    result = bench._probe_executable(executable, tools)

    assert result["usable"] is True
    assert result["version"] == "usage: llama-quantize model type"


def test_gguf_stage_records_same_prompt_ab_and_hashes(tmp_path, monkeypatch):
    spec = load_experiment_spec(
        _write_config(
            tmp_path,
            gguf={
                "enabled": True,
                "compare_baseline": True,
                "smoke_prompts": ["first", "second"],
            },
        )
    )
    monkeypatch.setattr(
        bench, "discover_gguf_tools", lambda _spec: bench.GGUFTools(None, None, None, None)
    )

    def fake_convert(_checkpoint, *, label, directory, **_kwargs):
        output = directory / f"{label}.gguf"
        output.write_bytes(label.encode("utf-8"))
        return output

    monkeypatch.setattr(bench, "_convert_checkpoint", fake_convert)
    monkeypatch.setattr(
        bench,
        "_llama_completion",
        lambda model, prompt, **_kwargs: {"model": model.stem, "prompt": prompt},
    )
    baseline = tmp_path / "baseline"
    surgery = tmp_path / "surgery"
    baseline.mkdir()
    surgery.mkdir()

    result = bench._run_gguf_stage(
        baseline,
        surgery,
        run_dir=tmp_path / "run",
        spec=spec,
    )

    assert [row["prompt"] for row in result["evaluations"]["baseline"]] == [
        "first",
        "second",
    ]
    assert [row["prompt"] for row in result["evaluations"]["surgery"]] == [
        "first",
        "second",
    ]
    assert len(result["models"]["baseline"]["sha256"]) == 64
    assert (tmp_path / "run/gguf/smoke-results.json").is_file()


def test_skip_gguf_is_recorded_without_requiring_tools(tmp_path):
    path = _write_config(tmp_path, gguf={"enabled": True})
    spec = load_experiment_spec(path)

    class FakePipeline:
        def __init__(self, **kwargs):
            self.output_dir = Path(kwargs["output_dir"])

        def run(self):
            self.output_dir.mkdir(parents=True)
            return self.output_dir

    run_dir = run_experiment(
        spec,
        run_id="skip-gguf",
        skip_gguf=True,
        pipeline_factory=FakePipeline,
    )
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))

    assert manifest["stages"]["gguf"] == {"reason": "--skip-gguf", "status": "skipped"}


def test_postprocess_recovers_failed_run_without_repeating_surgery(tmp_path, monkeypatch):
    spec = load_experiment_spec(_write_config(tmp_path, gguf={"enabled": True}))
    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            self.output_dir = Path(kwargs["output_dir"])

        def run(self):
            self.output_dir.mkdir(parents=True)
            return self.output_dir

    run_dir = run_experiment(
        spec,
        run_id="recover-gguf",
        skip_gguf=True,
        pipeline_factory=FakePipeline,
    )
    manifest_path = run_dir / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "failed",
            "error": {"type": "RuntimeError", "message": "converter missing"},
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        bench,
        "preflight_experiment",
        lambda *_args, **_kwargs: {"ready": True, "failures": []},
    )

    def fake_stage(baseline, surgery, **_kwargs):
        calls.append((baseline, surgery))
        return {"models": {}, "evaluations": {}}

    monkeypatch.setattr(bench, "_run_gguf_stage", fake_stage)

    assert bench.postprocess_experiment(spec, run_dir=run_dir) == run_dir
    recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recovered["status"] == "complete"
    assert recovered["stages"]["surgery"]["status"] == "complete"
    assert recovered["stages"]["gguf"]["resumed"] is True
    assert recovered["recoveries"][0]["error"]["message"] == "converter missing"
    assert len(calls) == 1


def test_logged_command_preserves_unicode_log_on_narrow_console(tmp_path, monkeypatch):
    console_bytes = io.BytesIO()
    console = io.TextIOWrapper(console_bytes, encoding="ascii")
    monkeypatch.setattr(bench.sys, "stdout", console)
    log_path = tmp_path / "unicode.log"

    bench._run_logged_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'\\xc4\\xa0\\n')",
        ],
        log_path=log_path,
    )
    console.flush()

    assert b"\\u0120" in console_bytes.getvalue()
    assert log_path.read_text(encoding="utf-8") == "Ġ\n"


def test_llama_completion_parser_removes_conversation_framing():
    raw = """Loading model...
build : fixture

> Explain a safe topic.

I am sorry, but I cannot assist with that.

[ Prompt: 10 t/s | Generation: 5 t/s ]

Exiting...
"""

    completion, mode = bench._extract_llama_completion(raw, "Explain a safe topic.")

    assert completion == "I am sorry, but I cannot assist with that."
    assert mode == "conversation-marker"


def test_smoke_reevaluation_reuses_verified_gguf_models(tmp_path, monkeypatch):
    spec = load_experiment_spec(
        _write_config(
            tmp_path,
            gguf={
                "enabled": True,
                "compare_baseline": True,
                "smoke_prompts": ["first", "second"],
            },
        )
    )
    run_dir = tmp_path / "outputs/unit-mini/reevaluate"
    gguf_dir = run_dir / "gguf"
    gguf_dir.mkdir(parents=True)
    model_records = {}
    for label in ("baseline", "surgery"):
        path = (gguf_dir / f"{label}.gguf").resolve()
        path.write_bytes(label.encode("utf-8"))
        model_records[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": bench._sha256(path),
        }
    manifest_path = run_dir / "run-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": spec.name,
                "spec_sha256": bench._sha256(spec.source_path),
                "stages": {
                    "gguf": {
                        "status": "complete",
                        "ended_at": "earlier",
                        "models": model_records,
                        "evaluations": {"baseline": [{"old": True}]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        bench,
        "preflight_experiment",
        lambda *_args, **_kwargs: {
            "ready": True,
            "failures": [],
            "gguf": {"llama_cli": {"version": "fixture"}},
        },
    )
    monkeypatch.setattr(
        bench, "discover_gguf_tools", lambda _spec: bench.GGUFTools(None, None, None, None)
    )
    monkeypatch.setattr(
        bench,
        "_llama_completion",
        lambda model, prompt, **_kwargs: {"model": model.stem, "prompt": prompt},
    )
    monkeypatch.setattr(
        bench,
        "_git_value",
        lambda arguments: "fixture-commit" if arguments == ["rev-parse", "HEAD"] else "",
    )

    assert bench.reevaluate_gguf_experiment(spec, run_dir=run_dir) == run_dir.resolve()
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = updated["stages"]["gguf"]
    assert [row["prompt"] for row in stage["evaluations"]["baseline"]] == [
        "first",
        "second",
    ]
    assert stage["smoke_history"][0]["evaluations"] == {"baseline": [{"old": True}]}
    assert stage["smoke_runtime"]["git"] == {
        "commit": "fixture-commit",
        "status": "",
    }
    assert (gguf_dir / "smoke-results.json").is_file()
