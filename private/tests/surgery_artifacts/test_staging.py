"""Tests for access-policy-preserving atomic staging paths."""

from __future__ import annotations

from surgery_artifacts._staging import (
    create_staging_directory,
    staging_directory,
    write_staging_text,
)


def test_staging_directory_is_unique_and_context_cleanup_is_scoped(tmp_path):
    first = create_staging_directory(tmp_path, prefix=".stage-")
    second = create_staging_directory(tmp_path, prefix=".stage-")

    assert first.parent == tmp_path
    assert second.parent == tmp_path
    assert first != second
    assert first.is_dir()
    assert second.is_dir()

    with staging_directory(tmp_path, prefix=".managed-") as managed:
        (managed / "payload").write_text("value", encoding="utf-8")
        assert managed.is_dir()
    assert not managed.exists()
    assert first.is_dir()
    assert second.is_dir()


def test_staging_text_is_exclusive_and_uses_requested_shape(tmp_path):
    staged = write_staging_text(
        tmp_path,
        prefix=".ref.",
        suffix=".tmp",
        text="payload\n",
    )

    assert staged.parent == tmp_path
    assert staged.name.startswith(".ref.")
    assert staged.name.endswith(".tmp")
    assert staged.read_text(encoding="utf-8") == "payload\n"
