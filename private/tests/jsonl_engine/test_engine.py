from __future__ import annotations

import os
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jsonl_engine.engine as engine_module
import pytest

from jsonl_engine import (
    JsonlCorruptionError,
    JsonlFormatError,
    JsonlStore,
    UncommittedTailError,
    encode_record,
    read_index,
)


def test_batched_appends_preserve_the_data_file_generation(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)

    first = engine.append([{"case": "a"}, {"case": "b"}], metadata={"tier": "screen"})
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        opened_jidx = engine.jidx_path.stat()
        second = engine.append([{"case": "c"}, {"case": "d"}])
        named = path.stat()
        named_jidx = engine.jidx_path.stat()

    assert (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
    assert (opened_jidx.st_dev, opened_jidx.st_ino) == (
        named_jidx.st_dev,
        named_jidx.st_ino,
    )
    assert engine.jidx_path == tmp_path / "observations.jidx"
    assert first.appended_records == 2
    assert second.generation == first.generation + 1
    assert second.record_count == 4
    assert second.metadata == {"tier": "screen"}
    assert engine.read_records() == [
        {"case": "a"},
        {"case": "b"},
        {"case": "c"},
        {"case": "d"},
    ]
    assert read_index(engine.jidx_path, path).offsets == (
        0,
        len(encode_record({"case": "a"})),
        len(encode_record({"case": "a"})) + len(encode_record({"case": "b"})),
        len(encode_record({"case": "a"}))
        + len(encode_record({"case": "b"}))
        + len(encode_record({"case": "c"})),
    )
    assert engine.read_record(0) == {"case": "a"}
    assert engine.read_record(-1) == {"case": "d"}
    assert engine.read_range(1, 3) == [{"case": "b"}, {"case": "c"}]
    assert len(engine.transaction_path.read_bytes().splitlines()) == 3
    assert engine.verify().record_count == 4


def test_concurrent_writers_serialize_without_lost_batches(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"

    def append(index: int) -> int:
        receipt = JsonlStore(path).append([{"assignment": index}])
        return receipt.generation

    with ThreadPoolExecutor(max_workers=6) as executor:
        generations = list(executor.map(append, range(12)))

    records = JsonlStore(path).read_records()
    assert sorted(record["assignment"] for record in records) == list(range(12))
    assert sorted(generations) == list(range(1, 13))
    assert JsonlStore(path).verify().record_count == 12


def test_failed_commit_rolls_back_data_and_transaction_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"case": "committed"}])
    original_data = path.read_bytes()
    original_transactions = engine.transaction_path.read_bytes()
    real_append = engine_module._append_durable

    def fail_transaction(target: Path, payload: bytes) -> None:
        if target == engine.transaction_path:
            real_append(target, payload[: max(1, len(payload) // 3)])
            raise OSError("simulated transaction-table write failure")
        real_append(target, payload)

    monkeypatch.setattr(engine_module, "_append_durable", fail_transaction)
    with pytest.raises(OSError, match="simulated"):
        engine.append([{"case": "rolled-back"}])

    assert path.read_bytes() == original_data
    assert engine.transaction_path.read_bytes() == original_transactions
    assert JsonlStore(path).read_records() == [{"case": "committed"}]
    assert read_index(engine.jidx_path, path).is_current()


def test_default_recovery_discards_bytes_beyond_the_commit_boundary(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"case": "committed"}])
    with path.open("ab") as handle:
        handle.write(encode_record({"case": "orphan"}))
        handle.flush()
        os.fsync(handle.fileno())

    engine.append([{"case": "after-recovery"}])

    assert engine.read_records() == [
        {"case": "committed"},
        {"case": "after-recovery"},
    ]
    assert engine.verify().record_count == 2


def test_fail_closed_recovery_preserves_an_uncommitted_tail(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    engine = JsonlStore(path, recover_uncommitted=False)
    engine.append([{"case": "committed"}])
    committed_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    with pytest.raises(UncommittedTailError) as raised:
        engine.append([{"case": "blocked"}])

    assert raised.value.committed_bytes == committed_size
    assert path.stat().st_size > committed_size


def test_prefix_repair_preserves_backup_and_rebuilds_derived_state(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    store = JsonlStore(path, recover_uncommitted=False)
    store.append([{"case": "a"}, {"case": "b"}])
    committed = path.read_bytes()
    invalid_tail = b'{"partial":'
    with path.open("ab") as handle:
        handle.write(invalid_tail)

    scan = store.inspect_prefix(collect_records=True)
    assert not scan.valid
    assert scan.valid_prefix_bytes == len(committed)
    assert scan.records == ({"case": "a"}, {"case": "b"})

    receipt = store.repair_prefix(scan.valid_prefix_bytes)

    assert receipt.backup_path.read_bytes() == committed + invalid_tail
    assert receipt.removed_bytes == len(invalid_tail)
    assert path.read_bytes() == committed
    assert store.jidx_path.is_file()
    assert store.transaction_path.is_file()
    assert store.verify().record_count == 2
    assert store.read_record(1) == {"case": "b"}


def test_same_size_external_tampering_is_detected_before_append(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"value": "alpha"}])
    path.write_bytes(path.read_bytes().replace(b"alpha", b"omega"))

    with pytest.raises(JsonlCorruptionError, match="data hash"):
        engine.append([{"value": "later"}])
    with pytest.raises(JsonlCorruptionError, match="data hash"):
        engine.verify()


def test_full_reads_validate_the_committed_data_hash(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"value": "alpha"}])
    original_stat = path.stat()
    path.write_bytes(path.read_bytes().replace(b"alpha", b"omega"))
    os.utime(
        path,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    with pytest.raises(JsonlCorruptionError, match="data hash"):
        engine.read_records()


def test_missing_jidx_is_rebuilt_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"case": "a"}, {"case": "b"}, {"case": "c"}])
    original = engine.jidx_path.read_bytes()
    engine.jidx_path.unlink()

    assert engine.read_record(1) == {"case": "b"}
    assert engine.jidx_path.read_bytes() == original
    assert engine.rebuild_index().offsets == read_index(engine.jidx_path, path).offsets
    assert engine.jidx_path.read_bytes() == original


def test_parsed_jidx_is_reused_within_one_committed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"case": "a"}, {"case": "b"}, {"case": "c"}])
    assert engine.read_record(0) == {"case": "a"}

    def reject_index_reload(*args, **kwargs):
        pytest.fail(f"unexpected JIDX reload: args={args}, kwargs={kwargs}")

    monkeypatch.setattr(engine_module, "read_index", reject_index_reload)
    assert engine.read_record(2) == {"case": "c"}


def test_tampered_jidx_fails_verification_and_indexed_read_repairs_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)
    engine.append([{"case": "a"}, {"case": "b"}, {"case": "c"}])
    original = read_index(engine.jidx_path, path)
    payload = bytearray(engine.jidx_path.read_bytes())
    struct.pack_into("<q", payload, 28 + 8, original.offsets[1] + 1)
    engine.jidx_path.write_bytes(payload)

    with pytest.raises(JsonlCorruptionError, match="offset hash"):
        engine.verify()

    assert engine.read_record(1) == {"case": "b"}
    assert read_index(engine.jidx_path, path).offsets == original.offsets
    assert engine.verify().record_count == 3


def test_partition_ranges_drive_indexed_factorial_reads(tmp_path: Path) -> None:
    path = tmp_path / "battery.jsonl"
    engine = JsonlStore(path)
    screen = {"tier": "screen", "factors": {"temperature": 0.0, "persona": "plain"}}
    deep = {"tier": "deep", "factors": {"temperature": 0.7, "persona": "plain"}}
    engine.append([{"case": "a"}, {"case": "b"}], partition=screen)
    engine.append([{"case": "c"}], partition=deep)
    engine.append([{"case": "d"}], partition=screen)

    ranges = engine.partition_ranges({"tier": "screen"})
    assert [(item.start_record, item.stop_record) for item in ranges] == [(0, 2), (3, 4)]
    assert ranges[0].partition == screen
    assert engine.read_partition({"factors": {"temperature": 0.0}}) == [
        {"case": "a"},
        {"case": "b"},
        {"case": "d"},
    ]
    assert engine.record_count() == 4


def test_nonportable_records_fail_before_the_data_file_is_mutated(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    engine = JsonlStore(path)

    with pytest.raises(JsonlFormatError, match="non-portable"):
        engine.append([{"score": float("nan")}])

    assert not path.exists()
    assert engine.transaction_path.exists()
    assert engine.jidx_path.exists()
    assert engine.verify().record_count == 0
