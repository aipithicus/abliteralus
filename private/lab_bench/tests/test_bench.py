from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from lab_bench.bench import LabBench
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
            "allocation_timeout_seconds": 1200,
            "allocation_retry_seconds": 20,
            "pending_policy": "adopt",
            "surgery_fallback_machines": ["H100"],
            "inference_fallback_machines": ["A100-80GB"],
            "control_max_runtime_seconds": 3600,
            "surgery_max_runtime_seconds": 14400,
            "inference_max_runtime_seconds": 3600,
        },
        "ssh": {
            "executable": sys.executable,
            "destination": "researcher@example.invalid",
            "port": 22,
            "identity_file": "",
        },
    }


class FakeManager:
    def __init__(self, return_code: int = 0) -> None:
        self.return_code = return_code
        self.calls: list[dict] = []

    def run(self, profile, command, **kwargs):
        self.calls.append({"profile": profile, "command": list(command), **kwargs})
        return subprocess.CompletedProcess(command, self.return_code)


@pytest.fixture
def configured_bench() -> tuple[LabBench, FakeManager]:
    repository = Path.cwd()
    config = parse_config(sample_mapping(repository), path=repository / "lab.local.toml")
    manager = FakeManager(return_code=7)
    return LabBench(config, manager=manager), manager


def test_online_local_surgery_uses_local_profile(configured_bench) -> None:
    bench, manager = configured_bench

    assert bench.local_surgery(["run", "--config", "experiment.yaml"]) == 7
    call = manager.calls[0]
    assert call["profile"] == "local-surgery"
    assert call["command"][1:3] == ["-m", "surgery_artifacts.integration"]
    assert str(Path("private") / "surgery_artifacts" / "src") in call["environ"]["PYTHONPATH"]
    assert call["command"][-3:] == ["run", "--config", "experiment.yaml"]


def test_offline_local_surgery_does_not_load_secret_manager(
    configured_bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench, manager = configured_bench
    observed: dict = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("lab_bench.bench.subprocess.run", fake_run)

    assert bench.local_surgery(["run", "--config", "experiment.yaml", "--offline"]) == 0
    assert not manager.calls
    assert observed["command"][-1] == "--offline"


def test_anonymous_local_surgery_strips_hub_credentials(
    configured_bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench, manager = configured_bench
    observed: dict = {}
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-child")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "must-not-reach-child")

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        observed["environ"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("lab_bench.bench.subprocess.run", fake_run)

    assert bench.local_surgery(["run", "--config", "experiment.yaml"], anonymous_hub=True) == 0
    assert not manager.calls
    assert "HF_TOKEN" not in observed["environ"]
    assert "HUGGING_FACE_HUB_TOKEN" not in observed["environ"]
    assert observed["environ"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert observed["command"][-3:] == ["run", "--config", "experiment.yaml"]


def test_lightning_plan_adds_defaults_without_credentials(
    configured_bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench, manager = configured_bench
    observed: dict = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("lab_bench.bench.subprocess.run", fake_run)

    assert bench.lightning_surgery(["plan", "--config", "experiment.yaml"]) == 0
    command = observed["command"]
    assert not manager.calls
    assert ["--teamspace", "owner/teamspace"] == command[
        command.index("--teamspace") : command.index("--teamspace") + 2
    ]
    assert ["--studio", "abliteralus-lab"] == command[
        command.index("--studio") : command.index("--studio") + 2
    ]
    assert command.count("--forward-env") == 1
    assert command[command.index("--forward-env") + 1] == "HF_TOKEN"
    assert command[command.index("--remote-root") + 1] == ".abliteralus"
    assert command[command.index("--allocation-timeout") + 1] == "1200.0"
    assert command[command.index("--allocation-retry") + 1] == "20.0"
    assert command[command.index("--pending-policy") + 1] == "adopt"
    assert command[command.index("--fallback-machine") + 1] == "H100"
    assert command[command.index("--max-runtime") + 1] == "14400"


def test_lightning_run_uses_surgery_profile_and_preserves_overrides(configured_bench) -> None:
    bench, manager = configured_bench

    assert (
        bench.lightning_surgery(
            [
                "run",
                "--config",
                "experiment.yaml",
                "--machine",
                "A10G",
                "--forward-env",
                "HF_TOKEN",
            ]
        )
        == 7
    )
    call = manager.calls[0]
    assert call["profile"] == "lightning-surgery"
    assert call["command"].count("--forward-env") == 1
    assert call["command"][call["command"].index("--machine") + 1] == "A10G"


def test_studio_control_and_inference_use_distinct_profiles(configured_bench) -> None:
    bench, manager = configured_bench

    assert bench.studio_status() == 7
    assert bench.studio_exec(["python", "server.py"], detached=True, wait_seconds=2.0) == 7

    assert manager.calls[0]["profile"] == "lightning-control"
    assert manager.calls[1]["profile"] == "lightning-inference"
    assert "--remote-command" in manager.calls[1]["command"]
    remote = manager.calls[1]["command"][manager.calls[1]["command"].index("--remote-command") + 1]
    assert remote == "python server.py"
    assert manager.calls[1]["command"].count("--forward-env") == 1


def test_studio_start_uses_bounded_allocator_and_purpose_fallback(configured_bench) -> None:
    bench, manager = configured_bench

    assert bench.studio_start(purpose="inference", max_runtime=1800) == 7
    call = manager.calls[0]
    command = call["command"]

    assert call["profile"] == "lightning-control"
    assert command[1:4] == ["-m", "abliteralus.lightning_surgery", "start"]
    assert command[command.index("--machine") + 1] == "L40S"
    assert command[command.index("--fallback-machine") + 1] == "A100-80GB"
    assert command[command.index("--allocation-timeout") + 1] == "1200.0"
    assert command[command.index("--pending-policy") + 1] == "adopt"
    assert command[command.index("--max-runtime") + 1] == "1800"


def test_studio_provision_uses_control_profile_without_forwarding_hub_token(
    configured_bench,
) -> None:
    bench, manager = configured_bench

    assert (
        bench.studio_provision(
            experiment_config="experiments/surgery/lightning-qwen25-7b.yaml",
            max_runtime=3600,
        )
        == 7
    )
    call = manager.calls[0]
    command = call["command"]

    assert call["profile"] == "lightning-control"
    assert command[2:4] == ["abliteralus.lightning_surgery", "provision"]
    assert command[command.index("--machine") + 1] == "CPU-4"
    assert command[command.index("--max-runtime") + 1] == "3600"
    assert command[command.index("--remote-root") + 1] == ".abliteralus"
    assert "--forward-env" not in command
    assert "HF_TOKEN" not in command


def test_studio_doctor_uses_control_profile_without_forwarding_secrets(
    configured_bench,
) -> None:
    bench, manager = configured_bench

    assert (
        bench.studio_doctor(experiment_config="experiments/surgery/lightning-qwen25-7b.yaml") == 7
    )
    call = manager.calls[0]

    assert call["profile"] == "lightning-control"
    assert call["command"][2:4] == ["abliteralus.lightning_surgery", "doctor"]
    assert "--forward-env" not in call["command"]


def test_studio_provision_dry_run_does_not_load_secret_manager(
    configured_bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench, manager = configured_bench
    observed: dict = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("lab_bench.bench.subprocess.run", fake_run)

    assert (
        bench.studio_provision(
            experiment_config="experiments/surgery/lightning-qwen25-7b.yaml",
            dry_run=True,
        )
        == 0
    )
    assert not manager.calls
    assert "--dry-run" in observed["command"]


def test_placeholder_teamspace_fails_before_secret_or_api_access() -> None:
    repository = Path.cwd()
    raw = sample_mapping(repository)
    raw["lightning"]["teamspace"] = "SET_ME/SET_ME"
    config = parse_config(raw, path=repository / "lab.local.toml")
    manager = FakeManager()
    bench = LabBench(config, manager=manager)

    with pytest.raises(LabBenchError, match="teamspace is not configured"):
        bench.studio_status()
    assert not manager.calls


def test_ssh_keeps_host_verification_defaults_and_quotes_remote_command(
    configured_bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench, _manager = configured_bench
    observed: dict = {}

    def fake_run(command, **kwargs):
        observed["command"] = list(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("lab_bench.bench.subprocess.run", fake_run)

    assert bench.ssh(["python", "chat client.py"]) == 0
    command = observed["command"]
    assert "StrictHostKeyChecking=no" not in command
    assert command[-2] == "researcher@example.invalid"
    assert command[-1] == "python 'chat client.py'"
    assert command[0] == str(Path(sys.executable).resolve())
