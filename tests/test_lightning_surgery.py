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
    LightningAllocationError,
    LightningConfigError,
    _allocation_policy,
    _ensure_studio_running,
    _runtime_paths,
    build_lightning_plan,
    build_provision_plan,
    build_runtime_layout,
    build_runtime_bundle,
    execute_lightning_plan,
    execute_provision_plan,
    execute_runtime_doctor,
    execute_studio_start,
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
        ["git", "-c", "commit.gpgsign=false", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "abliteralus").mkdir(parents=True)
    (repo / "deps/uv").mkdir(parents=True)
    (repo / "private/surgery_artifacts/src/surgery_artifacts").mkdir(parents=True)
    (repo / "abliteralus/module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "private/secret.txt").write_text("not for upload\n", encoding="utf-8")
    (repo / "private/surgery_artifacts/src/surgery_artifacts/__init__.py").write_text(
        "FORMAT = 1\n", encoding="utf-8"
    )
    (repo / "pyproject.toml").write_text("[project]\nname='x'\nversion='0'\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (repo / "README.md").write_text("runtime\n", encoding="utf-8")
    repository_pin = Path(__file__).resolve().parents[1] / "deps/uv/pin.json"
    (repo / "deps/uv/pin.json").write_bytes(repository_pin.read_bytes())
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
            "deps/uv/pin.json",
        ]
    )

    assert selected == [
        "README.md",
        "abliteralus/core.py",
        "deps/uv/pin.json",
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
    assert "private/surgery_artifacts/src" in plan.command
    assert "private/secret_manager" not in plan.command
    assert "${PYTHONPATH:+:$PYTHONPATH}" not in plan.command
    assert "XDG_CONFIG_HOME" in plan.command
    assert "PYTHONNOUSERSITE=1" in plan.command
    assert "compgen -A variable" in plan.command


def test_plan_records_explicit_ordered_allocation_policy(tmp_path):
    plan = build_lightning_plan(
        _spec(tmp_path),
        teamspace="owner/research",
        studio="abliteralus-surgery",
        machine="h200",
        run_id="run-h200",
        allocation_timeout_seconds=1200,
        allocation_retry_seconds=20,
        fallback_machines=["H100", "h100", "L40S"],
        pending_policy="adopt",
    )

    assert plan.machine == "H200"
    assert plan.allocation.fallback_machines == ("H100", "L40S")
    assert plan.to_dict()["allocation"] == {
        "timeout_seconds": 1200.0,
        "retry_seconds": 20.0,
        "fallback_machines": ["H100", "L40S"],
        "pending_policy": "adopt",
    }


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
    assert first.uv_platform == "linux-x86_64-gnu"
    assert first.uv_archive_url.endswith("uv-x86_64-unknown-linux-gnu.tar.gz")


def test_runtime_identity_includes_exact_uv_pin_bytes(tmp_path):
    repo = _minimal_repo(tmp_path)
    spec = _spec(tmp_path)

    first = build_runtime_layout(repo, spec)
    pin_path = repo / "deps/uv/pin.json"
    pin_path.write_bytes(pin_path.read_bytes() + b"\n")
    second = build_runtime_layout(repo, spec)

    assert first.uv_pin_sha256 != second.uv_pin_sha256
    assert first.key != second.key


def test_runtime_layout_requires_repository_uv_contract(tmp_path):
    repo = _minimal_repo(tmp_path)
    (repo / "deps/uv/pin.json").unlink()

    with pytest.raises(LightningConfigError, match="repository uv toolchain is invalid"):
        build_runtime_layout(repo, _spec(tmp_path))


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

    assert "pip install" not in plan.command
    assert "python -m venv" not in plan.command
    assert plan.runtime.uv_archive_url in plan.command
    assert plan.runtime.uv_archive_sha256 in plan.command
    assert "urllib.request.urlopen" in plan.command
    assert "uv sync --frozen --no-dev --no-install-project" in plan.command
    assert "UV_PROJECT_ENVIRONMENT" in plan.command
    assert 'UV_PYTHON="$ABLITERALUS_BOOTSTRAP_PYTHON"' in plan.command
    assert "ABLITERALUS_BOOTSTRAP_PYTHON" in plan.command
    assert "XDG_CONFIG_HOME" in plan.command
    assert "PYTHONNOUSERSITE=1" in plan.command
    assert "compgen -A variable" in plan.command
    assert plan.runtime.environment in plan.command
    assert "HF_TOKEN" not in json.dumps(plan.to_dict())
    assert plan.max_runtime == 3600

    compile(lightning._runtime_probe_script(plan.runtime, write_marker=True), "probe", "exec")
    compile(lightning._runtime_probe_script(plan.runtime, write_marker=False), "probe", "exec")
    compile(lightning._remote_uv_install_script(plan.runtime), "uv-installer", "exec")


@pytest.mark.parametrize("teamspace", ["", "owner", "owner/team/extra", "-bad/team"])
def test_plan_rejects_ambiguous_teamspace(tmp_path, teamspace):
    with pytest.raises(LightningConfigError):
        build_lightning_plan(
            _spec(tmp_path),
            teamspace=teamspace,
            studio="studio",
            machine="L40S",
        )


def test_bundle_is_deterministic_and_allowlists_only_artifact_private_tree(tmp_path):
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
    assert "deps/uv/pin.json" in names
    assert "experiment.yaml" in names
    assert "bundle-manifest.json" in names
    assert "private/surgery_artifacts/src/surgery_artifacts/__init__.py" in names
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
        if "surgery_artifacts.integration" in command:
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


class OutOfCapacityError(RuntimeError):
    pass


def test_allocator_uses_only_explicit_fallbacks(tmp_path):
    class Studio:
        status = "Stopped"
        machine = None

    attempted: list[str] = []

    def start(machine: str, _remaining: float, status) -> str:
        attempted.append(machine)
        status({"status": "pending", "observed_at": "fixture", "elapsed_seconds": 1.0})
        return "out_of_capacity" if machine == "H200" else "running"

    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=60,
        allocation_retry_seconds=5,
        fallback_machines=["H100"],
        pending_policy="fail",
    )
    result = _ensure_studio_running(
        Studio(),
        requested_machine="H200",
        policy=policy,
        reuse_running=False,
        start_attempt=start,
        journal_path=tmp_path / "allocation.json",
    )

    assert attempted == ["H200", "H100"]
    assert result.selected_machine == "H100"
    assert [attempt["outcome"] for attempt in result.attempts] == [
        "out_of_capacity",
        "running",
    ]
    journal = json.loads((tmp_path / "allocation.json").read_text(encoding="utf-8"))
    assert journal["attempts"][1]["status_polls"][0]["status"] == "pending"


def test_allocator_strict_mode_retries_only_requested_machine():
    class Studio:
        status = "Stopped"

    now = [0.0]
    attempted: list[str] = []

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    def start(machine: str, _remaining: float, _status) -> str:
        attempted.append(machine)
        return "out_of_capacity"

    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=5,
        allocation_retry_seconds=2,
        fallback_machines=[],
        pending_policy="fail",
    )
    with pytest.raises(LightningAllocationError, match="allocation timeout"):
        _ensure_studio_running(
            Studio(),
            requested_machine="H200",
            policy=policy,
            reuse_running=False,
            start_attempt=start,
            clock=clock,
            sleeper=sleep,
        )

    assert attempted
    assert set(attempted) == {"H200"}


def test_allocator_does_not_stop_compute_when_studio_state_changed_externally():
    class Studio:
        status = "Stopped"

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    remote = Studio()

    def start(_machine: str, _remaining: float, _status) -> str:
        remote.status = "Running"
        return "state_changed"

    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=5,
        allocation_retry_seconds=1,
        fallback_machines=[],
        pending_policy="fail",
    )
    with pytest.raises(LightningAllocationError, match="refusing to assume ownership"):
        _ensure_studio_running(
            remote,
            requested_machine="H200",
            policy=policy,
            reuse_running=False,
            start_attempt=start,
        )

    assert remote.stopped is False


def test_allocator_stops_pending_compute_after_start_timeout():
    class Studio:
        status = "Stopped"

        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True
            self.status = "Stopped"

    remote = Studio()

    def start(_machine: str, _remaining: float, _status) -> str:
        remote.status = "Pending"
        return "timeout"

    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=5,
        allocation_retry_seconds=1,
        fallback_machines=[],
        pending_policy="fail",
    )
    with pytest.raises(LightningAllocationError, match="did not complete"):
        _ensure_studio_running(
            remote,
            requested_machine="H200",
            policy=policy,
            reuse_running=False,
            start_attempt=start,
        )

    assert remote.stopped is True


def test_headless_start_supervisor_polls_pending_until_running(monkeypatch):
    class Remote:
        def __init__(self):
            self.statuses = iter(["Pending", "Running"])

        @property
        def status(self):
            return next(self.statuses)

    class Process:
        def __init__(self):
            self.terminated = False

        def poll(self):
            return 0 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            if not self.terminated:
                raise subprocess.TimeoutExpired("fixture", timeout)
            assert timeout == 5
            return 0

        def kill(self):
            pytest.fail("graceful worker termination should succeed")

    process = Process()
    monkeypatch.setattr(lightning.subprocess, "Popen", lambda *_args, **_kwargs: process)
    observations: list[dict] = []

    outcome = lightning._subprocess_start_attempt(
        Remote(),
        "owner/research",
        "studio",
        "H200",
        interruptible=False,
        max_runtime=3600,
        timeout_seconds=30,
        poll_seconds=5,
        status_callback=observations.append,
    )

    assert outcome == "running"
    assert [observation["status"] for observation in observations] == ["pending", "running"]
    assert process.terminated is True


def test_allocator_requires_explicit_pending_reconciliation():
    class Studio:
        status = "Pending"

    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=5,
        allocation_retry_seconds=1,
        fallback_machines=[],
        pending_policy="fail",
    )
    with pytest.raises(LightningAllocationError, match="Pending"):
        _ensure_studio_running(
            Studio(),
            requested_machine="H200",
            policy=policy,
            reuse_running=False,
            start_attempt=lambda _machine, _remaining, _status: pytest.fail("must not start"),
        )


def test_allocator_can_adopt_pending_machine_without_new_start():
    class Studio:
        machine = "H200"

        def __init__(self):
            self.reads = 0

        @property
        def status(self):
            self.reads += 1
            return "Pending" if self.reads < 3 else "Running"

    now = [0.0]
    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=5,
        allocation_retry_seconds=1,
        fallback_machines=[],
        pending_policy="adopt",
    )
    result = _ensure_studio_running(
        Studio(),
        requested_machine="H200",
        policy=policy,
        reuse_running=False,
        start_attempt=lambda _machine, _remaining, _status: pytest.fail("must not start"),
        clock=lambda: now[0],
        sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert result.mode == "adopted_pending"
    assert result.attempts[0]["outcome"] == "adopted_pending"


def test_studio_start_retries_capacity_with_explicit_fallback(tmp_path):
    class CapacityStudio(_FakeStudio):
        instances = []

        def start(self, **kwargs):
            self.started.append(kwargs)
            if kwargs["machine"] == "machine:H200":
                raise OutOfCapacityError("fixture capacity miss")
            self.status = "Running"

    CapacityStudio.initial_status = "Stopped"
    policy = _allocation_policy(
        primary_machine="H200",
        allocation_timeout_seconds=60,
        allocation_retry_seconds=5,
        fallback_machines=["H100"],
        pending_policy="fail",
    )
    result = execute_studio_start(
        teamspace="owner/research",
        studio="studio",
        machine="H200",
        allocation=policy,
        local_output=tmp_path / "start-result",
        studio_class=CapacityStudio,
        machine_class=_FakeMachine,
    )
    remote = CapacityStudio.instances[-1]

    assert [call["machine"] for call in remote.started] == ["machine:H200", "machine:H100"]
    assert result["allocation"]["selected_machine"] == "H100"
    assert remote.stopped is False
    assert (tmp_path / "start-result/studio-start-result.json").is_file()


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
    assert result["allocation"]["selected_machine"] == "L40S"
    assert result["timing"]["total_seconds"] >= 0
    assert result["timing"]["remote_seconds"] >= 0
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


def test_interrupted_start_stops_pending_compute_and_records_attempt(tmp_path, monkeypatch):
    class InterruptedStudio(_FakeStudio):
        instances = []

        def start(self, **kwargs):
            self.started.append(kwargs)
            self.status = "Pending"
            raise KeyboardInterrupt

    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="H200",
        run_id="run-interrupted-start",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    InterruptedStudio.initial_status = "Stopped"

    with pytest.raises(KeyboardInterrupt):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-interrupted-start",
            studio_class=InterruptedStudio,
            machine_class=_FakeMachine,
        )

    remote = InterruptedStudio.instances[-1]
    assert remote.stopped is True
    allocation = json.loads(
        (tmp_path / "result-interrupted-start/allocation.json").read_text(encoding="utf-8")
    )
    assert allocation["state"] == "interrupted"
    assert allocation["attempts"][-1]["outcome"] == "interrupted"


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


def test_collect_capsule_records_local_verification_before_release(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="H200",
        run_id="run-capsule",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    monkeypatch.setattr(
        lightning,
        "_download_capsule",
        lambda _studio, _plan, _destination: (
            ["run-manifest.json", "artifact/"],
            {
                "surgery_id": "a" * 64,
                "operation_count": 4,
                "file_count": 5,
                "total_bytes": 1024,
                "verified": True,
            },
        ),
    )
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 0

    result = execute_lightning_plan(
        plan,
        spec=spec,
        repo_root=tmp_path,
        local_output=tmp_path / "result-capsule",
        collect="capsule",
        studio_class=_FakeStudio,
        machine_class=_FakeMachine,
    )

    assert result["capsule_validation"]["verified"] is True
    assert result["capsule_validation"]["surgery_id"] == "a" * 64
    assert _FakeStudio.instances[-1].stopped is True


def test_failed_capsule_run_collects_diagnostics_without_requesting_capsule(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    plan = build_lightning_plan(
        spec,
        teamspace="owner/research",
        studio="studio",
        machine="H200",
        run_id="run-capsule-failed",
    )
    monkeypatch.setattr(lightning, "build_runtime_bundle", _stub_bundle)
    monkeypatch.setattr(
        lightning,
        "_download_capsule",
        lambda *_args, **_kwargs: pytest.fail("failed runs must not request a capsule"),
    )
    _FakeStudio.instances.clear()
    _FakeStudio.initial_status = "Stopped"
    _FakeStudio.remote_exit_code = 7

    with pytest.raises(RuntimeError, match="exit code 7"):
        execute_lightning_plan(
            plan,
            spec=spec,
            repo_root=tmp_path,
            local_output=tmp_path / "result-capsule-failed",
            collect="capsule",
            studio_class=_FakeStudio,
            machine_class=_FakeMachine,
        )

    result = json.loads(
        (tmp_path / "result-capsule-failed/lightning-result.json").read_text(encoding="utf-8")
    )
    assert "run-manifest.json" in result["downloaded"]
    assert result["capsule_validation"] is None
    assert _FakeStudio.instances[-1].stopped is True


def test_cli_normalizes_unexpected_provider_errors(monkeypatch, capsys):
    def fail(_argv=None):
        raise RuntimeError("provider response headers must not reach the console")

    monkeypatch.setattr(lightning, "_main", fail)

    assert lightning.main([]) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err
    assert "provider response headers" not in captured.err
