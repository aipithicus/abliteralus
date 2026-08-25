"""Contract tests for privacy-scoped Lightning Studio orchestration."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest
import yaml

import abliteralus.lightning_surgery as lightning
from abliteralus.lightning_surgery import (
    LightningConfigError,
    _runtime_paths,
    build_lightning_plan,
    build_provision_plan,
    build_runtime_layout,
    build_runtime_bundle,
    execute_lightning_plan,
    execute_provision_plan,
    execute_runtime_doctor,
)
from abliteralus.surgery_bench import load_experiment_spec


pytestmark = pytest.mark.cpu


def _spec(tmp_path: Path, *, local_source: bool = False):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "name": "remote-unit",
                "model": {
                    "source": str(checkpoint) if local_source else "owner/model",
                    "device": "cpu",
                    "dtype": "float32",
                },
                "pipeline": {"method": "basic"},
                "prompts": {"harmful": 2, "harmless": 2},
                "gguf": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    return load_experiment_spec(config)


def _git(repo: Path, *arguments: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "abliteralus").mkdir(parents=True)
    (repo / "private").mkdir()
    (repo / "abliteralus/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "private/secret.txt").write_text("not for upload\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("runtime\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return repo


def test_runtime_selection_excludes_private_and_test_material():
    selected = _runtime_paths(
        [
            "abliteralus/core.py",
            "private/sync.py",
            "tests/test_core.py",
            "pyproject.toml",
            "uv.lock",
            "README.md",
        ]
    )

    assert selected == [
        "README.md",
        "abliteralus/core.py",
        "pyproject.toml",
        "uv.lock",
    ]


def test_plan_is_shell_safe_and_contains_no_forwarded_secret_value(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")

    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="abliteralus-surgery",
        machine="L40S",
        run_id="run-1",
        forwarded_environment=["HF_TOKEN"],
    )

    encoded = json.dumps(plan.to_dict())
    assert "super-secret-token" not in encoded
    assert plan.forwarded_environment == ("HF_TOKEN",)
    assert "uv sync" not in plan.command
    assert "pip install" not in plan.command
    assert plan.runtime.key in plan.command
    assert plan.runtime.uv_version == "0.12.4"
    assert "private" not in plan.command


def test_runtime_identity_is_lock_addressed_and_uses_persistent_caches(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)

    first = build_runtime_layout(repo, spec)
    (repo / "uv.lock").write_text("version = 2\n", encoding="utf-8")
    second = build_runtime_layout(repo, spec)

    assert first.key != second.key
    assert first.environment == f".abliteralus/runtimes/{first.key}"
    assert first.uv_cache == ".abliteralus/cache/uv"
    assert first.hf_home == ".abliteralus/cache/huggingface"


def test_provision_plan_owns_install_and_sync_work(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)

    plan = build_provision_plan(
        spec,
        repo_root=repo,
        teamspace="owner/research",
        studio="studio",
        machine="CPU-4",
        max_runtime=3600,
    )

    assert "uv==0.12.4" in plan.command
    assert "uv sync --frozen --no-dev --no-install-project" in plan.command
    assert "UV_PROJECT_ENVIRONMENT" in plan.command
    assert plan.runtime.environment in plan.command
    assert "HF_TOKEN" not in json.dumps(plan.to_dict())
    assert plan.max_runtime == 3600

    compile(lightning._runtime_probe_script(plan.runtime, write_marker=True), "probe", "exec")
    compile(lightning._runtime_probe_script(plan.runtime, write_marker=False), "probe", "exec")


@pytest.mark.parametrize("teamspace", ["", "owner", "owner/team/extra", "-bad/team"])
def test_plan_rejects_ambiguous_teamspace(tmp_path, teamspace):
    with pytest.raises(LightningConfigError):
        build_lightning_plan(
            _spec(tmp_path),
            teamspace=teamspace,
            studio="studio",
            machine="L40S",
        )


def test_bundle_is_deterministic_and_excludes_private_tree(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_manifest = build_runtime_bundle(repo, spec, first)
    second_manifest = build_runtime_bundle(repo, spec, second)

    assert first_manifest["sha256"] == second_manifest["sha256"]
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert "abliteralus/module.py" in names
    assert "experiment.yaml" in names
    assert "bundle-manifest.json" in names
    assert "private/secret.txt" not in names


def test_bundle_rejects_uncommitted_runtime_change(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)
    (repo / "abliteralus/module.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        build_runtime_bundle(repo, spec, tmp_path / "dirty.zip")


def test_bundle_rejects_untracked_runtime_file(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)
    (repo / "abliteralus/new_runtime.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="uncommitted changes"):
        build_runtime_bundle(repo, spec, tmp_path / "untracked.zip")


def test_plan_rejects_local_checkpoint_source(tmp_path):
    with pytest.raises(LightningConfigError, match="OWNER/MODEL"):
        build_lightning_plan(
            _spec(tmp_path, local_source=True),
            teamspace="owner/research",
            studio="studio",
            machine="L40S",
        )


class _FakeMachine:
    @classmethod
    def from_str(cls, value):
        return f"machine:{value}"


class _FakeStudio:
    instances = []
    initial_status = "Stopped"
    remote_exit_code = 0
    doctor_exit_code = 0
    provision_exit_code = 0
    initial_environment = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.status = self.initial_status
        self.started = []
        self.stopped = False
        self.uploaded = []
        self.commands = []
        self.environment = dict(self.initial_environment)
        self.__class__.instances.append(self)

    def start(self, **kwargs):
        self.started.append(kwargs)
        self.status = "Running"

    def stop(self):
        self.stopped = True

    def set_env(self, values, partial):
        assert partial is True
        self.environment.update(values)

    @property
    def env(self):
        return dict(self.environment)

    def delete_env(self, name):
        del self.environment[name]

    def upload_file(self, path, remote_path):
        self.uploaded.append((path, remote_path))

    def run_with_exit_code(self, command):
        self.commands.append(command)
        if "abliteralus.surgery_bench" in command:
            return "remote output\n", self.remote_exit_code
        if "pip install" in command:
            return "provisioned\n", self.provision_exit_code
        if "runtime-manifests" in command:
            return "doctor output\n", self.doctor_exit_code
        return "prepared\n", 0

    def download_file(self, _remote, local):
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_text("downloaded\n", encoding="utf-8")

    def download_folder(self, _remote, local):
        Path(local).mkdir(parents=True, exist_ok=True)


def _stub_bundle(_repo, _specification, destination, **_kwargs):
    destination.write_bytes(b"bundle")
    return {"path": str(destination), "sha256": "fixture", "bytes": 6}


def test_provision_materializes_runtime_and_stops_owned_compute(tmp_path, monkeypatch):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)
    plan = build_provision_plan(
        spec,
        repo_root=repo,
        teamspace="owner/research",
        studio="studio",
        machine="CPU-4",
        max_runtime=3600,
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.provision_exit_code = 0

    result = execute_provision_plan(
        plan,
        spec=spec,
        repo_root=repo,
        local_output=tmp_path / "provision-result",
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )
    studio = _FakeStudio.instances[-1]

    assert result["runtime_key"] == plan.runtime.key
    assert studio.started == [
        {"machine": "machine:CPU-4", "interruptible": False, "max_runtime": 3600}
    ]
    assert studio.stopped is True
    assert len(studio.commands) == 2
    assert (tmp_path / "provision-result/provision-result.json").is_file()


def test_runtime_doctor_is_read_only_and_requires_running_studio(tmp_path):
    repo = _minimal_repo(tmp_path)
    runtime = build_runtime_layout(repo, _spec(tmp_path))
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Running"
    _FakeStudio.doctor_exit_code = 0

    result = execute_runtime_doctor(
        runtime,
        teamspace="owner/research",
        studio="studio",
        studio_class=_FakeStudio,
    )
    remote = _FakeStudio.instances[-1]

    assert result["ready"] is True
    assert remote.started == []
    assert remote.stopped is False
    assert remote.uploaded == []


def test_execute_stops_only_compute_it_started(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-2",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 0

    result = execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=tmp_path,
        local_output=tmp_path / "result",
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )
    studio = _FakeStudio.instances[-1]

    assert result["remote_exit_code"] == 0
    assert studio.started == [{"machine": "machine:L40S", "interruptible": False}]
    assert studio.stopped is True
    assert (tmp_path / "result/lightning-result.json").is_file()


def test_execute_does_not_take_ownership_of_running_studio(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-3",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Running"
    _FakeStudio.remote_exit_code = 0

    execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=tmp_path,
        local_output=tmp_path / "result-running",
        reuse_running=True,
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )
    studio = _FakeStudio.instances[-1]

    assert studio.started == []
    assert studio.stopped is False


def test_remote_failure_still_stops_owned_compute(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-4",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 7

    with pytest.raises(RuntimeError, match="exit code 7"):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-failed",
            studio_class=_FakeStudio,
            machine_class=_FakeMachine,
        )

    assert _FakeStudio.instances[-1].stopped is True
    result = json.loads(
        (tmp_path / "result-failed/lightning-result.json").read_text(encoding="utf-8")
    )
    assert result["remote_exit_code"] == 7


def test_unprovisioned_runtime_fails_before_upload_and_stops_owned_compute(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-unprovisioned",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    monkeypatch.setattr(_FakeStudio, "doctor_exit_code", 5)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"

    with pytest.raises(RuntimeError, match="run provision"):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-unprovisioned",
            studio_class=_FakeStudio,
            machine_class=_FakeMachine,
        )

    studio = _FakeStudio.instances[-1]
    assert studio.uploaded == []
    assert studio.stopped is True


def test_plan_can_explicitly_skip_unprovisioned_remote_gguf(tmp_path):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-skip",
        skip_gguf=True,
    )

    assert plan.skip_gguf is True
    assert "--skip-gguf" in plan.command


def test_forwarded_environment_must_exist(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-env",
        forwarded_environment=["MISSING_FIXTURE_TOKEN"],
    )
    monkeypatch.delenv("MISSING_FIXTURE_TOKEN", raising=False)
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"

    with pytest.raises(LightningConfigError, match="unset"):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-env",
            studio_class=_FakeStudio,
            machine_class=_FakeMachine,
        )

    assert _FakeStudio.instances[-1].stopped is True


def test_forwarded_environment_is_restored_after_run(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-env-restore",
        forwarded_environment=["HF_TOKEN", "NEW_TOKEN"],
    )
    monkeypatch.setenv("HF_TOKEN", "local-token")
    monkeypatch.setenv("NEW_TOKEN", "temporary-token")
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    monkeypatch.setattr(_FakeStudio, "initial_environment", {"HF_TOKEN": "studio-token"})
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 0

    execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=tmp_path,
        local_output=tmp_path / "result-env-restore",
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )

    assert _FakeStudio.instances[-1].environment == {"HF_TOKEN": "studio-token"}


def test_running_studio_requires_explicit_reuse(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-owned",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Running"

    with pytest.raises(RuntimeError, match="already running"):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-owned",
            studio_class=_FakeStudio,
            machine_class=_FakeMachine,
        )

    assert _FakeStudio.instances[-1].stopped is False


def test_collect_all_uses_folder_transfer(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="L40S",
        run_id="run-all",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 0

    result = execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=tmp_path,
        local_output=tmp_path / "result-all",
        collect="all",
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )

    assert result["downloaded"] == ["artifacts/"]
    assert (tmp_path / "result-all/artifacts").is_dir()


def test_cli_normalizes_unexpected_provider_errors(monkeypatch, capsys):
    def fail(_argv=None):
        raise RuntimeError("provider response headers must not reach the console")

    monkeypatch.setattr(lightning, "_main", fail)

    assert lightning.main([]) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err
    assert "provider response headers" not in captured.err
