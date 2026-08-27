from __future__ import annotations

import json
from pathlib import Path

from research_journal.cli import main


def test_cli_add_list_verify_and_show(tmp_path: Path, capsys) -> None:
    path = tmp_path / "journal.jsonl"
    assert (
        main(
            [
                "--journal",
                str(path),
                "add",
                "--kind",
                "observation",
                "--title",
                "Guard margin rose",
                "--relation",
                "run=guard-001",
            ]
        )
        == 0
    )
    entry = json.loads(capsys.readouterr().out)

    assert main(["--journal", str(path), "verify"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True

    assert main(["--journal", str(path), "list", "--relation", "run=guard-001"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["entry_id"] == entry["entry_id"]

    assert main(["--journal", str(path), "show", entry["entry_id"]]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "Guard margin rose"


def test_cli_repair_is_preview_first(tmp_path: Path, capsys) -> None:
    path = tmp_path / "journal.jsonl"
    assert main(["--journal", str(path), "add", "--kind", "note", "--title", "One"]) == 0
    capsys.readouterr()
    with path.open("ab") as handle:
        handle.write(b"broken")

    assert main(["--journal", str(path), "verify"]) == 2
    capsys.readouterr()
    assert main(["--journal", str(path), "repair"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["needed"] is True
    assert "dry run" in captured.err
    assert main(["--journal", str(path), "repair", "--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is True


def test_cli_rejects_two_standard_input_consumers(tmp_path: Path, capsys) -> None:
    path = tmp_path / "journal.jsonl"

    assert (
        main(
            [
                "--journal",
                str(path),
                "add",
                "--kind",
                "note",
                "--title",
                "One",
                "--body-file",
                "-",
                "--data-file",
                "-",
            ]
        )
        == 1
    )
    assert "cannot both read standard input" in capsys.readouterr().err
    assert not path.exists()
