from __future__ import annotations

import json
from pathlib import Path

from lab_bench.cli import main


class FakeBench:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def journal_add(self, **kwargs):
        self.calls.append(("add", kwargs))
        return {"entry_id": "entry-001", **kwargs}

    def journal_verify(self):
        self.calls.append(("verify", {}))
        return {"valid": False, "error": "broken record hash chain"}


def _install_bench(monkeypatch, bench: FakeBench) -> None:
    monkeypatch.setattr("lab_bench.cli.load_config", lambda _path: object())
    monkeypatch.setattr("lab_bench.cli.LabBench", lambda _config: bench)


def test_journal_add_dispatches_structured_relations_and_files(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    body_file = tmp_path / "observation.md"
    body_file.write_text("Guard margin crossed baseline.", encoding="utf-8")
    data_file = tmp_path / "metrics.json"
    data_file.write_text('{"margin_delta": 0.18}', encoding="utf-8")
    bench = FakeBench()
    _install_bench(monkeypatch, bench)

    result = main(
        [
            "journal",
            "add",
            "--kind",
            "experiment.observation",
            "--title",
            "Positive steering sweep",
            "--body-file",
            str(body_file),
            "--data-file",
            str(data_file),
            "--tag",
            "llama-guard",
            "--run-id",
            "guard-baseline-001",
            "--model",
            "meta-llama/Llama-Guard-3-1B",
            "--artifact",
            "guard/sweep-positive-002",
            "--relation",
            "study=guard-directionality-v1",
        ]
    )

    assert result == 0
    operation, payload = bench.calls[0]
    assert operation == "add"
    assert payload["body"] == "Guard margin crossed baseline."
    assert payload["data"] == {"margin_delta": 0.18}
    assert payload["sharing"] == "private"
    assert payload["relations"] == [
        {"type": "study", "target": "guard-directionality-v1"},
        {"type": "run", "target": "guard-baseline-001"},
        {"type": "model", "target": "meta-llama/Llama-Guard-3-1B"},
        {"type": "artifact", "target": "guard/sweep-positive-002"},
    ]
    assert json.loads(capsys.readouterr().out)["entry_id"] == "entry-001"


def test_journal_verify_uses_distinct_invalid_exit_code(monkeypatch, capsys) -> None:
    bench = FakeBench()
    _install_bench(monkeypatch, bench)

    assert main(["journal", "verify"]) == 2
    assert bench.calls == [("verify", {})]
    assert json.loads(capsys.readouterr().out)["valid"] is False
