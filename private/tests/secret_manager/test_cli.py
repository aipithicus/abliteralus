from __future__ import annotations

from pathlib import Path

from secret_manager.errors import SecretUnavailable

from secret_manager import broker, cli

CONFIG_TEXT = """\
schema_version = 1

[providers.proton]
type = "proton-pass"
executable = "pass-cli"

[credentials.hf-read]
provider = "proton"
vault = "LOCATOR-VAULT"
item = "LOCATOR-ITEM"
field = "secret"

[profiles.local]
environment = { HF_TOKEN = "hf-read" }

[broker]
default_profile = "local"
allowed_profiles = ["local"]
"""


def test_inventory_hides_references_by_default(tmp_path: Path, capsys) -> None:
    config = tmp_path / "mapping.toml"
    config.write_text(CONFIG_TEXT, encoding="utf-8")

    assert cli.main(["--config", str(config), "inventory"]) == 0
    output = capsys.readouterr().out
    assert "hf-read" in output
    assert "LOCATOR-VAULT" not in output
    assert "LOCATOR-ITEM" not in output


def test_inventory_can_explicitly_show_references(tmp_path: Path, capsys) -> None:
    config = tmp_path / "mapping.toml"
    config.write_text(CONFIG_TEXT, encoding="utf-8")

    assert cli.main(["--config", str(config), "inventory", "--show-references"]) == 0
    assert "pass://LOCATOR-VAULT/LOCATOR-ITEM/secret" in capsys.readouterr().out


def test_broker_success_writes_only_value(monkeypatch, capsysbinary) -> None:
    class FakeManager:
        def resolve_for_broker(self, name, *, profile_name=None):
            assert name == "HF_TOKEN"
            assert profile_name == "local"
            return b"secret-value"

    class Factory:
        @staticmethod
        def from_config():
            return FakeManager()

    monkeypatch.setattr(broker, "SecretManager", Factory)
    monkeypatch.setenv("SECRET_MANAGER_PROFILE", "local")

    assert broker.main(["HF_TOKEN"]) == 0
    captured = capsysbinary.readouterr()
    assert captured.out == b"secret-value"
    assert captured.err == b""


def test_broker_wrong_argument_count_is_a_hard_failure() -> None:
    assert broker.main([]) == 1
    assert broker.main(["HF_TOKEN", "EXTRA"]) == 1


def test_broker_unavailable_credential_uses_contract_exit_code(monkeypatch) -> None:
    class FakeManager:
        def resolve_for_broker(self, name, *, profile_name=None):
            raise SecretUnavailable(name)

    class Factory:
        @staticmethod
        def from_config():
            return FakeManager()

    monkeypatch.setattr(broker, "SecretManager", Factory)

    assert broker.main(["HF_TOKEN"]) == 2


def test_privileged_flags_parse_after_profile_before_child_boundary() -> None:
    args = cli._parser().parse_args(
        [
            "run",
            "hf-admin",
            "--allow-privileged",
            "--yes",
            "--",
            "python",
            "worker.py",
            "--child-flag",
        ]
    )

    assert args.allow_privileged is True
    assert args.yes is True
    assert args.child_command == ["python", "worker.py", "--child-flag"]
