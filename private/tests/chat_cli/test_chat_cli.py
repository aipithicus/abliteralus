from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "chat_cli" / "chat.py"
SPEC = importlib.util.spec_from_file_location("abliteralus_private_chat_cli", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
chat_cli = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = chat_cli
SPEC.loader.exec_module(chat_cli)


class _Response:
    def __init__(self, payload: dict, *, status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_llama_guard_profile_uses_typed_content_and_template() -> None:
    profile = chat_cli.get_profile("llama-guard-3")

    assert profile.messages("hello") == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
    assert profile.chat_template is not None
    template = profile.chat_template.read_text(encoding="utf-8")
    assert "<BEGIN UNSAFE CONTENT CATEGORIES>" in template
    assert "raise_exception" not in template


def test_embedded_profile_uses_plain_string_content() -> None:
    profile = chat_cli.get_profile("embedded")

    assert profile.messages("hello") == [{"role": "user", "content": "hello"}]
    assert profile.chat_template is None


def test_server_command_contains_explicit_profile_template(tmp_path: Path) -> None:
    executable = tmp_path / "llama-server.exe"
    model = tmp_path / "subject.gguf"
    template = tmp_path / "subject.jinja"
    executable.write_bytes(b"")
    model.write_bytes(b"")
    template.write_text("{{ messages }}", encoding="utf-8")
    config = chat_cli.ServerConfig(
        executable=str(executable),
        model=model,
        host="127.0.0.1",
        port=18080,
        context_size=4096,
        gpu_layers="all",
        threads=4,
        parallel=1,
        alias="subject",
        chat_template=template,
        startup_timeout=30.0,
        extra_args=("--no-warmup",),
    )

    command = config.command()

    assert command[0] == str(executable)
    assert command[command.index("--model") + 1] == str(model)
    assert command[command.index("--chat-template-file") + 1] == str(template.resolve())
    assert command[-1] == "--no-warmup"


def test_client_posts_openai_chat_request() -> None:
    captured: dict = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response({"choices": [{"message": {"role": "assistant", "content": "unsafe\nS1"}}]})

    client = chat_cli.LlamaServerClient("http://127.0.0.1:8080", opener=opener)
    result = client.chat(
        model="subject",
        messages=[{"role": "user", "content": "prompt"}],
        max_tokens=16,
        temperature=0.0,
        timeout=20.0,
    )

    assert result.content == "unsafe\nS1"
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["timeout"] == 20.0
    assert captured["payload"] == {
        "model": "subject",
        "messages": [{"role": "user", "content": "prompt"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": False,
    }


def test_resolve_model_path_accepts_absolute_gguf(tmp_path: Path) -> None:
    model = tmp_path / "subject.gguf"
    model.write_bytes(b"GGUF")

    assert chat_cli.resolve_model_path(str(model)) == model.resolve()


def test_resolve_model_path_rejects_non_gguf(tmp_path: Path) -> None:
    model = tmp_path / "subject.bin"
    model.write_bytes(b"not gguf")

    with pytest.raises(chat_cli.ChatCliError, match="must be a GGUF"):
        chat_cli.resolve_model_path(str(model))


def test_hf_subcommand_dispatches_to_probe_harness(monkeypatch) -> None:
    captured = {}

    def fake_main(arguments):
        captured["arguments"] = arguments
        return 17

    monkeypatch.setitem(sys.modules, "probe_harness.cli", SimpleNamespace(main=fake_main))

    assert chat_cli.main(["hf", "--prompt", "hello"]) == 17
    assert captured["arguments"] == ["--prompt", "hello"]
