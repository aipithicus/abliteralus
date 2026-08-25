from __future__ import annotations

import sys
from pathlib import Path

import pytest
from lab_bench.config import parse_config
from lab_bench.errors import LabBenchError


def sample_mapping(repository: Path) -> dict:
    return {
        "schema_version": 1,
        "repository": str(repository),
        "python": sys.executable,
        "secret_config": str(repository / "private/secret_manager/secrets.local.toml"),
        "profiles": {
            "local_surgery": "local-surgery",
            "local_inference": "local-inference",
            "lightning_control": "lightning-control",
            "lightning_surgery": "lightning-surgery",
            "lightning_inference": "lightning-inference",
        },
        "lightning": {
            "teamspace": "owner/teamspace",
            "studio": "abliteralus-lab",
            "control_machine": "CPU-4",
            "surgery_machine": "L40S",
            "inference_machine": "L40S",
            "forward_environment": ["HF_TOKEN"],
        },
        "ssh": {
            "executable": sys.executable,
            "destination": "researcher@example.invalid",
            "port": 22,
            "identity_file": "",
        },
    }


def test_config_resolves_paths_and_machine_purposes() -> None:
    repository = Path.cwd()
    config = parse_config(sample_mapping(repository), path=repository / "lab.local.toml")

    assert config.repository == repository.resolve()
    assert config.python == Path(sys.executable).resolve()
    assert config.lightning.control_machine == "CPU-4"
    assert config.lightning.forward_environment == ("HF_TOKEN",)


def test_teamspace_requires_owner_and_name() -> None:
    raw = sample_mapping(Path.cwd())
    raw["lightning"]["teamspace"] = "just-one-name"

    with pytest.raises(LabBenchError, match="OWNER/TEAMSPACE"):
        parse_config(raw)


def test_unknown_configuration_keys_fail_closed() -> None:
    raw = sample_mapping(Path.cwd())
    raw["ssh"]["strict_host_key_checking"] = False

    with pytest.raises(LabBenchError, match="unknown keys"):
        parse_config(raw)


def test_ssh_destination_cannot_be_an_option() -> None:
    raw = sample_mapping(Path.cwd())
    raw["ssh"]["destination"] = "-oProxyCommand=untrusted"

    with pytest.raises(LabBenchError, match="option prefix"):
        parse_config(raw)
