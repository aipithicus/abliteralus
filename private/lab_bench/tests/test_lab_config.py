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
    assert config.lightning.remote_root == ".abliteralus"
    assert config.lightning.forward_environment == ("HF_TOKEN",)
    assert config.lightning.allocation_timeout_seconds == 900.0
    assert config.lightning.allocation_retry_seconds == 30.0
    assert config.lightning.pending_policy == "fail"
    assert config.lightning.surgery_fallback_machines == ()
    assert config.lightning.surgery_max_runtime_seconds is None
    assert config.artifacts.registry == repository.resolve() / "outputs/artifact-registry"
    assert config.journal.path == repository.resolve() / "outputs/research-journal/journal.jsonl"


def test_explicit_journal_path_is_config_relative(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    raw = sample_mapping(repository)
    raw["journal"] = {"path": "state/experiments.jsonl"}
    config_path = tmp_path / "config" / "lab.local.toml"

    config = parse_config(raw, path=config_path)

    assert config.journal.path == config_path.parent / "state/experiments.jsonl"


def test_journal_path_requires_jsonl_suffix() -> None:
    raw = sample_mapping(Path.cwd())
    raw["journal"] = {"path": "outputs/research-journal/journal.json"}

    with pytest.raises(LabBenchError, match="must name a .jsonl file"):
        parse_config(raw)


def test_config_parses_unattended_allocation_policy() -> None:
    raw = sample_mapping(Path.cwd())
    raw["lightning"].update(
        {
            "allocation_timeout_seconds": 1200,
            "allocation_retry_seconds": 15.5,
            "pending_policy": "adopt",
            "surgery_fallback_machines": ["H100", "L40S"],
            "inference_fallback_machines": ["A100-80GB"],
            "control_max_runtime_seconds": 3600,
            "surgery_max_runtime_seconds": 14400,
            "inference_max_runtime_seconds": 1800,
        }
    )

    config = parse_config(raw)

    assert config.lightning.allocation_timeout_seconds == 1200.0
    assert config.lightning.allocation_retry_seconds == 15.5
    assert config.lightning.pending_policy == "adopt"
    assert config.lightning.surgery_fallback_machines == ("H100", "L40S")
    assert config.lightning.inference_fallback_machines == ("A100-80GB",)
    assert config.lightning.control_max_runtime_seconds == 3600
    assert config.lightning.surgery_max_runtime_seconds == 14400
    assert config.lightning.inference_max_runtime_seconds == 1800


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


def test_remote_root_rejects_parent_traversal() -> None:
    raw = sample_mapping(Path.cwd())
    raw["lightning"]["remote_root"] = "../shared"

    with pytest.raises(LabBenchError, match="relative POSIX path"):
        parse_config(raw)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("allocation_timeout_seconds", 0),
        ("allocation_retry_seconds", float("inf")),
        ("pending_policy", "guess"),
        ("surgery_fallback_machines", ["H100", "h100"]),
        ("surgery_max_runtime_seconds", 0),
    ],
)
def test_invalid_allocation_policy_fails_closed(key, value) -> None:
    raw = sample_mapping(Path.cwd())
    raw["lightning"][key] = value

    with pytest.raises(LabBenchError):
        parse_config(raw)
