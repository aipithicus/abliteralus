from __future__ import annotations

from argparse import Namespace

import pytest
from lab_bench._studio_worker import StudioWorkerError, execute


class FakeMachine:
    @staticmethod
    def from_str(name):
        return f"machine:{name}"


class FakeStudio:
    last = None
    initial_environment = {}
    initial_status = "running"

    def __init__(self, *, name, teamspace, create_ok):
        type(self).last = self
        self.name = name
        self.teamspace = teamspace
        self.create_ok = create_ok
        self.status = type(self).initial_status
        self.machine = "L40S"
        self.public_ip = "203.0.113.10"
        self.env = dict(type(self).initial_environment)
        self.deleted = []
        self.started = None
        self.stopped = False

    def set_env(self, values, *, partial):
        assert partial is True
        self.env.update(values)

    def delete_env(self, name):
        self.deleted.append(name)
        self.env.pop(name, None)

    def run_with_exit_code(self, command):
        assert self.env["HF_TOKEN"] == "ephemeral-token"
        return f"ran {command}", 0

    def run_and_detach(self, command, *, timeout):
        assert self.env["HF_TOKEN"] == "ephemeral-token"
        return f"started {command}", None

    def start(self, **kwargs):
        self.started = kwargs
        self.status = "running"

    def stop(self):
        self.stopped = True
        self.status = "stopped"


def test_remote_execution_restores_existing_studio_environment(monkeypatch) -> None:
    FakeStudio.initial_environment = {"HF_TOKEN": "previous-token"}
    FakeStudio.initial_status = "running"
    monkeypatch.setenv("HF_TOKEN", "ephemeral-token")
    args = Namespace(
        operation="exec",
        studio="abliteralus-lab",
        teamspace="owner/teamspace",
        remote_command="python server.py",
        forward_env=["HF_TOKEN"],
    )

    payload, return_code = execute(
        args,
        studio_class=FakeStudio,
        machine_class=FakeMachine,
    )

    assert return_code == 0
    assert payload["remote_output"] == "ran python server.py"
    assert FakeStudio.last.env["HF_TOKEN"] == "previous-token"
    assert FakeStudio.last.deleted == []


def test_detached_execution_deletes_new_studio_environment(monkeypatch) -> None:
    FakeStudio.initial_environment = {}
    FakeStudio.initial_status = "running"
    monkeypatch.setenv("HF_TOKEN", "ephemeral-token")
    args = Namespace(
        operation="detach",
        studio="abliteralus-lab",
        teamspace="owner/teamspace",
        remote_command="python server.py",
        forward_env=["HF_TOKEN"],
        wait_seconds=1.0,
    )

    payload, return_code = execute(
        args,
        studio_class=FakeStudio,
        machine_class=FakeMachine,
    )

    assert return_code == 0
    assert payload["remote_exit_code"] is None
    assert "HF_TOKEN" not in FakeStudio.last.env
    assert FakeStudio.last.deleted == ["HF_TOKEN"]


def test_start_uses_requested_machine_without_forwarding_secrets() -> None:
    FakeStudio.initial_environment = {}
    FakeStudio.initial_status = "stopped"
    args = Namespace(
        operation="start",
        studio="abliteralus-lab",
        teamspace="owner/teamspace",
        machine="CPU-4",
        interruptible=False,
        max_runtime=None,
    )

    payload, return_code = execute(
        args,
        studio_class=FakeStudio,
        machine_class=FakeMachine,
    )

    assert return_code == 0
    assert payload["changed"] is True
    assert FakeStudio.last.started["machine"] == "machine:CPU-4"


def test_start_refuses_silent_machine_mismatch() -> None:
    FakeStudio.initial_environment = {}
    FakeStudio.initial_status = "running"
    args = Namespace(
        operation="start",
        studio="abliteralus-lab",
        teamspace="owner/teamspace",
        machine="CPU-4",
        interruptible=False,
        max_runtime=None,
    )

    with pytest.raises(StudioWorkerError, match="different machine"):
        execute(
            args,
            studio_class=FakeStudio,
            machine_class=FakeMachine,
        )
