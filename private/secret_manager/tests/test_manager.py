from __future__ import annotations

import json
import subprocess

import pytest
from secret_manager.errors import PrivilegedSecretError, ProviderError, SecretUnavailable
from secret_manager.manager import SecretManager


def test_run_injects_references_only_and_leaves_parent_unchanged(
    manager: SecretManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = {
        "PATH": "safe-path",
        "HF_TOKEN": "stale-parent-value",
        "LIGHTNING_API_KEY": "cross-profile-value",
    }
    before = dict(parent)
    observed: dict = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(arguments, 17, stdout=b"", stderr=b"")

    monkeypatch.setattr("secret_manager.proton_pass.subprocess.run", fake_run)
    completed = manager.run("local", ["worker", "--dry-run"], environ=parent)

    assert completed.returncode == 17
    assert parent == before
    assert observed["arguments"][1:3] == ["run", "--"]
    assert observed["arguments"][-2:] == ["worker", "--dry-run"]
    assert observed["environment"]["HF_TOKEN"] == ("pass://vault-share-id==/read-item-id==/secret")
    assert "LIGHTNING_API_KEY" not in observed["environment"]
    assert "stale-parent-value" not in repr(observed)
    assert "cross-profile-value" not in repr(observed)


def test_unmanaged_pass_reference_fails_before_provider_scan(manager: SecretManager) -> None:
    with pytest.raises(ProviderError, match="UNRELATED_REFERENCE"):
        manager.run(
            "local",
            ["worker"],
            environ={"UNRELATED_REFERENCE": "pass://another/vault/value"},
        )


def test_privileged_profile_requires_explicit_gate(manager: SecretManager) -> None:
    with pytest.raises(PrivilegedSecretError):
        manager.references_for_profile("admin")

    _, references = manager.references_for_profile("admin", allow_privileged=True)
    assert references["HF_TOKEN"].endswith("/admin-item-id==/secret")


def test_check_returns_boolean_metadata_only(
    manager: SecretManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "DO-NOT-LEAK-RESOLVED-VALUE"

    def fake_run(arguments, **kwargs):
        assert sentinel not in repr(arguments)
        assert sentinel not in repr(kwargs)
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=json.dumps({"HF_TOKEN": True}).encode(),
            stderr=b"",
        )

    monkeypatch.setattr("secret_manager.proton_pass.subprocess.run", fake_run)

    assert manager.check("local", environ={"PATH": "safe"}) == {"HF_TOKEN": True}


def test_broker_resolution_uses_direct_field_reference(
    manager: SecretManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = b"broker-secret-sentinel\n"
    observed: dict = {}

    def fake_run(arguments, **kwargs):
        observed["arguments"] = arguments
        assert kwargs["stderr"] is subprocess.DEVNULL
        return subprocess.CompletedProcess(arguments, 0, stdout=sentinel, stderr=b"")

    monkeypatch.setattr("secret_manager.proton_pass.subprocess.run", fake_run)
    resolved = manager.resolve_for_broker("HF_TOKEN")

    assert resolved == sentinel.rstrip()
    assert observed["arguments"][1:3] == ["item", "view"]
    assert observed["arguments"][3] == "pass://vault-share-id==/read-item-id==/secret"


def test_broker_never_exposes_privileged_or_unmapped_names(manager: SecretManager) -> None:
    with pytest.raises(SecretUnavailable):
        manager.resolve_for_broker("LIGHTNING_API_KEY")
    with pytest.raises(SecretUnavailable):
        manager.resolve_for_broker("HF_TOKEN", profile_name="admin")


def test_provider_error_does_not_include_provider_output(
    manager: SecretManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = "PROVIDER-STDERR-SECRET"

    def fake_run(arguments, **kwargs):
        return subprocess.CompletedProcess(arguments, 9, stdout=b"", stderr=sentinel.encode())

    monkeypatch.setattr("secret_manager.proton_pass.subprocess.run", fake_run)
    with pytest.raises(ProviderError) as captured:
        manager.resolve_for_broker("HF_TOKEN")
    assert sentinel not in str(captured.value)
