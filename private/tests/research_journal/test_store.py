from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from research_journal import JournalCorruptionError, JournalFormatError, ResearchJournal


def _append_from_process(payload: tuple[str, int]) -> str:
    path, index = payload
    return ResearchJournal(path).append(
        kind="observation",
        title=f"Process observation {index}",
    )["entry_id"]


def test_first_append_initializes_a_canonical_hash_chained_journal(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = ResearchJournal(path)

    entry = journal.append(
        kind="hypothesis",
        title="Steering should raise the unsafe margin",
        tags=["guard", "sweep", "guard"],
        relations=[{"type": "run", "target": "guard-sweep-001"}],
        data={"strengths": [-1.0, 0.0, 1.0]},
    )

    inspection = journal.inspect()
    assert inspection.valid
    assert inspection.initialized
    assert inspection.record_count == 2
    assert inspection.entry_count == 1
    assert inspection.journal_id == entry["journal_id"]
    assert inspection.head_hash == entry["record_hash"]
    assert entry["sharing"] == "private"
    assert entry["tags"] == ["guard", "sweep"]
    lines = path.read_bytes().splitlines(keepends=True)
    assert len(lines) == 2
    assert all(line.endswith(b"\n") and b"\r" not in line for line in lines)
    assert all(json.loads(line)["schema_version"] == 1 for line in lines)


def test_list_filters_and_get_preserve_entry_relations(tmp_path: Path) -> None:
    journal = ResearchJournal(tmp_path / "journal.jsonl")
    first = journal.append(kind="hypothesis", title="First", tags=["guard"])
    second = journal.append(
        kind="result",
        title="Second",
        tags=["guard", "sweep"],
        relations=[
            {"type": "run", "target": "run-002"},
            {"type": "artifact", "target": "sha256:" + "a" * 64},
        ],
        sharing="candidate",
    )

    assert journal.get_entry(first["entry_id"])["title"] == "First"
    assert journal.list_entries(limit=10, kind="result") == [second]
    assert journal.list_entries(
        limit=10,
        tags=["sweep"],
        relations=[{"type": "run", "target": "run-002"}],
    ) == [second]


def test_committed_journal_cursor_avoids_a_second_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.jsonl"
    ResearchJournal(path).append(kind="note", title="First")
    journal = ResearchJournal(path)

    def reject_scan(*, collect_records: bool):
        pytest.fail(f"unexpected full journal scan (collect_records={collect_records})")

    monkeypatch.setattr(journal, "_scan", reject_scan)
    journal.append(kind="note", title="Second")

    assert len(path.read_bytes().splitlines()) == 3
    assert path.with_name(f"{path.name}.transactions.jsonl").is_file()
    assert path.with_suffix(".jidx").is_file()


def test_concurrent_writers_serialize_without_lost_entries(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"

    def append(index: int) -> str:
        return ResearchJournal(path).append(
            kind="observation",
            title=f"Concurrent observation {index}",
        )["entry_id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        entry_ids = list(executor.map(append, range(24)))

    inspection = ResearchJournal(path).inspect()
    assert inspection.valid
    assert inspection.entry_count == 24
    assert len(set(entry_ids)) == 24


def test_concurrent_processes_share_one_append_lease(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    work = [(str(path), index) for index in range(12)]

    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        entry_ids = list(executor.map(_append_from_process, work))

    inspection = ResearchJournal(path).inspect()
    assert inspection.valid
    assert inspection.entry_count == 12
    assert len(set(entry_ids)) == 12


@pytest.mark.parametrize(
    "kwargs",
    [
        {"data": {"HF_" + "TOKEN": "not-a-real-token"}},
        {"data": {"api key": "not-a-real-key"}},
        {"body": "Authorization: " + "Bearer " + "a" * 26},
        {"body": "token=" + "a" * 26},
        {"body": "https://" + "researcher:" + "password123" + "@example.invalid/model"},
        {"body": "-" * 5 + "BEGIN PRIVATE KEY" + "-" * 5},
    ],
)
def test_sensitive_material_is_rejected_before_initialization(tmp_path: Path, kwargs: dict) -> None:
    path = tmp_path / "journal.jsonl"

    with pytest.raises(JournalFormatError, match="sensitive|credential|private key"):
        ResearchJournal(path).append(kind="note", title="Unsafe input", **kwargs)

    assert not path.exists()


def test_token_count_is_not_mistaken_for_a_secret(tmp_path: Path) -> None:
    journal = ResearchJournal(tmp_path / "journal.jsonl")

    entry = journal.append(kind="result", title="Counts", data={"token_count": 128})

    assert entry["data"] == {"token_count": 128}


def test_invalid_tail_blocks_append_and_repair_preserves_a_backup(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = ResearchJournal(path)
    journal.append(kind="note", title="Before crash")
    original = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    inspection = journal.inspect()
    assert not inspection.valid
    assert inspection.valid_prefix_bytes == len(original)
    with pytest.raises(JournalCorruptionError, match="preview repair"):
        journal.append(kind="note", title="Blocked")

    preview = journal.repair()
    assert preview.needed and not preview.applied
    assert preview.removed_bytes == len(b'{"partial":')
    assert path.read_bytes() != original

    repaired = journal.repair(apply=True)
    assert repaired.applied
    assert repaired.backup is not None
    assert repaired.backup.read_bytes() == original + b'{"partial":'
    assert path.read_bytes() == original
    assert path.with_suffix(".jidx").is_file()
    assert journal.inspect().valid
    journal.append(kind="note", title="After repair")
    assert path.with_suffix(".jidx").is_file()
    assert journal.inspect().entry_count == 2


def test_content_tampering_breaks_the_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    journal = ResearchJournal(path)
    journal.append(kind="result", title="Original")
    payload = path.read_bytes().replace(b'"Original"', b'"Tampered"')
    path.write_bytes(payload)

    inspection = journal.inspect()
    assert not inspection.valid
    assert inspection.error_line == 2
    assert "record_hash" in inspection.error


def test_non_finite_and_non_object_data_fail_closed(tmp_path: Path) -> None:
    journal = ResearchJournal(tmp_path / "journal.jsonl")

    with pytest.raises(JournalFormatError, match="non-finite"):
        journal.append(kind="result", title="NaN", data={"score": float("nan")})
    with pytest.raises(JournalFormatError, match="JSON object"):
        journal.append(kind="result", title="List", data=[1, 2])  # type: ignore[arg-type]


def test_invalid_unicode_fails_as_a_format_error(tmp_path: Path) -> None:
    journal = ResearchJournal(tmp_path / "journal.jsonl")

    with pytest.raises(JournalFormatError, match="invalid Unicode"):
        journal.append(kind="note", title="surrogate \ud800")
    with pytest.raises(JournalFormatError, match="invalid Unicode"):
        journal.append(kind="result", title="Data", data={"value": "\ud800"})


def test_symbolic_link_journal_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_bytes(b"")
    link = tmp_path / "journal.jsonl"
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")

    journal = ResearchJournal(link)

    inspection = journal.inspect()
    assert not inspection.valid
    assert "symbolic link" in inspection.error
    with pytest.raises(JournalCorruptionError, match="symbolic link"):
        journal.append(kind="note", title="Refused")
    assert target.read_bytes() == b""


def test_empty_and_missing_journals_are_valid_uninitialized_stores(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    missing = ResearchJournal(path).inspect()
    assert missing.valid and not missing.exists and not missing.initialized

    path.write_bytes(b"")
    empty = ResearchJournal(path).inspect()
    assert empty.valid and empty.exists and not empty.initialized
