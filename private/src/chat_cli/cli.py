#!/usr/bin/env python3
"""Run a small interactive client against a managed local llama.cpp server."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .errors import ChatCliError

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_ALIAS = "local-subject"
WINDOWS_MISSING_DLL_CODES = {-1073741515, 3221225781}


@dataclass(frozen=True, slots=True)
class Profile:
    """Model-specific request and server-template behavior."""

    name: str
    content_mode: str
    chat_template: Path | None
    default_max_tokens: int
    default_temperature: float

    def messages(self, prompt: str) -> list[dict[str, Any]]:
        if self.content_mode == "text-blocks":
            content: str | list[dict[str, str]] = [{"type": "text", "text": prompt}]
        elif self.content_mode == "string":
            content = prompt
        else:  # pragma: no cover - profiles are declared below, not user-authored yet.
            raise ChatCliError(f"unsupported profile content mode: {self.content_mode}")
        return [{"role": "user", "content": content}]


PROFILES = {
    "embedded": Profile(
        name="embedded",
        content_mode="string",
        chat_template=None,
        default_max_tokens=256,
        default_temperature=0.8,
    ),
    "llama-guard-3": Profile(
        name="llama-guard-3",
        content_mode="text-blocks",
        chat_template=SCRIPT_DIRECTORY / "templates" / "llama-guard-3.jinja",
        default_max_tokens=16,
        default_temperature=0.0,
    ),
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _port(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return parsed


def _temperature(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("temperature must be a finite non-negative number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-cli",
        description=(
            "Keep a llama.cpp model resident in a local server and submit independent "
            "console prompts to its OpenAI-compatible chat endpoint."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="repository-relative or absolute GGUF path (omit when attaching to a server)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="llama-guard-3",
        help="model-specific template and request representation (default: llama-guard-3)",
    )
    parser.add_argument(
        "--server-url",
        help="attach to an existing llama-server instead of starting a managed process",
    )
    parser.add_argument(
        "--server-command",
        default="llama-server",
        help="llama-server command name or explicit executable path (default: llama-server)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=_port,
        default=0,
        help="managed-server port; 0 selects an available local port (default: 0)",
    )
    parser.add_argument("--ctx-size", type=_positive_int, default=4096)
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--parallel", type=_positive_int, default=1)
    parser.add_argument("--alias", default=DEFAULT_ALIAS, help="server/API model alias")
    parser.add_argument("--max-tokens", type=_positive_int)
    parser.add_argument("--temperature", type=_temperature)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra llama-server argument; repeat and use --server-arg=--flag for flags",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="submit a prompt and exit instead of opening the console; repeatable",
    )
    parser.add_argument("--show-json", action="store_true", help="show the full API response")
    parser.add_argument(
        "--show-server-log",
        action="store_true",
        help="mirror managed llama-server output to stderr",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved configuration without starting or contacting a server",
    )
    return parser


def get_profile(name: str) -> Profile:
    profile = PROFILES[name]
    if profile.chat_template is not None and not profile.chat_template.is_file():
        raise ChatCliError(f"profile template does not exist: {profile.chat_template}")
    return profile


def resolve_model_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ChatCliError(f"model does not exist: {candidate}") from error
    if not resolved.is_file():
        raise ChatCliError(f"model is not a file: {resolved}")
    if resolved.suffix.casefold() != ".gguf":
        raise ChatCliError(f"model must be a GGUF file: {resolved}")
    return resolved


def resolve_server_command(value: str) -> str:
    has_path_component = Path(value).is_absolute() or "/" in value or "\\" in value
    if has_path_component:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = REPOSITORY / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ChatCliError(f"llama-server executable does not exist: {candidate}") from error
        if not resolved.is_file():
            raise ChatCliError(f"llama-server executable is not a file: {resolved}")
        return str(resolved)

    resolved_command = shutil.which(value)
    if resolved_command is None:
        raise ChatCliError(
            f"could not find {value!r} on PATH; pass --server-command with an explicit path"
        )
    return resolved_command


def select_port(host: str, requested: int) -> int:
    if requested:
        return requested
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((host, 0))
            return int(listener.getsockname()[1])
    except OSError as error:
        raise ChatCliError(f"could not select a local port for host {host!r}: {error}") from error


def normalize_server_url(value: str) -> str:
    normalized = value.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ChatCliError(f"server URL must be an HTTP(S) URL: {value!r}")
    if parsed.query or parsed.fragment:
        raise ChatCliError("server URL must not contain a query or fragment")
    return normalized


@dataclass(frozen=True, slots=True)
class ServerConfig:
    executable: str
    model: Path
    host: str
    port: int
    context_size: int
    gpu_layers: str
    threads: int
    parallel: int
    alias: str
    chat_template: Path | None
    startup_timeout: float
    extra_args: tuple[str, ...] = ()
    show_log: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def command(self) -> list[str]:
        command = [
            self.executable,
            "--model",
            str(self.model),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--gpu-layers",
            self.gpu_layers,
            "--threads",
            str(self.threads),
            "--parallel",
            str(self.parallel),
            "--alias",
            self.alias,
            "--no-webui",
            "--log-colors",
            "off",
            "--log-verbosity",
            "2",
        ]
        if self.chat_template is not None:
            command.extend(["--jinja", "--chat-template-file", str(self.chat_template.resolve())])
        command.extend(self.extra_args)
        return command


@dataclass(frozen=True, slots=True)
class ChatResult:
    content: str
    response: Mapping[str, Any]


class LlamaServerClient:
    """Small stdlib client for the llama.cpp OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        *,
        opener: Callable[..., Any] = urlopen,
    ) -> None:
        self.base_url = normalize_server_url(base_url)
        self._opener = opener

    def health(self, *, timeout: float = 1.0) -> bool:
        request = Request(f"{self.base_url}/health", method="GET")
        try:
            with self._opener(request, timeout=timeout) as response:
                return int(getattr(response, "status", 200)) == 200
        except HTTPError as error:
            if error.code == 503:
                return False
            return False
        except (OSError, URLError):
            return False

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> ChatResult:
        payload = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response_payload = self._request_json(request, timeout=timeout)
        try:
            message = response_payload["choices"][0]["message"]
            content = message["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ChatCliError(
                "server response did not contain choices[0].message.content"
            ) from error
        if not isinstance(content, str):
            raise ChatCliError("server response content was not text")
        return ChatResult(content=content, response=response_payload)

    def _request_json(self, request: Request, *, timeout: float) -> dict[str, Any]:
        try:
            with self._opener(request, timeout=timeout) as response:
                raw = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            detail = body.strip() or str(error.reason)
            raise ChatCliError(f"server returned HTTP {error.code}: {detail}") from error
        except (OSError, URLError) as error:
            raise ChatCliError(f"could not contact llama-server: {error}") from error
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ChatCliError("server returned a non-JSON response") from error
        if not isinstance(payload, dict):
            raise ChatCliError("server returned a JSON value that was not an object")
        return payload


class ManagedServer:
    """Own one llama-server process and wait until its health endpoint is ready."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lines: deque[str] = deque(maxlen=200)

    def __enter__(self) -> "ManagedServer":
        self.start()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.stop()

    def start(self) -> None:
        if self.process is not None:
            raise ChatCliError("managed llama-server is already started")
        try:
            self.process = subprocess.Popen(
                self.config.command(),
                cwd=REPOSITORY,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
        except OSError as error:
            raise ChatCliError(f"could not start llama-server: {error}") from error

        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        client = LlamaServerClient(self.config.base_url)
        deadline = time.monotonic() + self.config.startup_timeout
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                message = self._early_exit_message(return_code)
                self.stop()
                raise ChatCliError(message)
            if client.health(timeout=1.0):
                return
            time.sleep(0.2)
        message = (
            f"llama-server did not become ready within {self.config.startup_timeout:.1f}s"
            + self._log_suffix()
        )
        self.stop()
        raise ChatCliError(message)

    def stop(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self.process = None

    def _read_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            value = line.rstrip("\r\n")
            self._lines.append(value)
            if self.config.show_log:
                print(f"[llama-server] {value}", file=sys.stderr)

    def _early_exit_message(self, return_code: int) -> str:
        message = f"llama-server exited before becoming ready (exit code {return_code})"
        if return_code in WINDOWS_MISSING_DLL_CODES:
            message += "; a required runtime DLL is not available on the child process PATH"
        return message + self._log_suffix()

    def _log_suffix(self) -> str:
        lines = [line for line in self._lines if line]
        if not lines:
            return ""
        tail = "\n".join(lines[-30:])
        return f"\n--- llama-server output ---\n{tail}"


def emit_result(result: ChatResult, *, show_json: bool) -> None:
    print(result.content.strip())
    if show_json:
        print(json.dumps(result.response, indent=2, ensure_ascii=False, sort_keys=True))


def submit_prompt(
    client: LlamaServerClient,
    *,
    profile: Profile,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
    show_json: bool,
) -> None:
    result = client.chat(
        model=model,
        messages=profile.messages(prompt),
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=request_timeout,
    )
    emit_result(result, show_json=show_json)


def run_console(
    client: LlamaServerClient,
    *,
    profile: Profile,
    model: str,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
    show_json: bool,
) -> int:
    print("Ready. Each prompt is an independent request. Use :help or :quit.")
    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt.strip():
            continue
        command = prompt.strip().casefold()
        if command in {":quit", ":exit"}:
            return 0
        if command == ":help":
            print(":help  show this help")
            print(":quit  stop the client and its managed server")
            print("Prefix a literal leading colon with another colon, for example ::topic.")
            continue
        if prompt.startswith("::"):
            prompt = prompt[1:]
        try:
            submit_prompt(
                client,
                profile=profile,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=request_timeout,
                show_json=show_json,
            )
        except ChatCliError as error:
            print(f"chat-cli: {error}", file=sys.stderr)


def _dry_run_payload(
    *,
    profile: Profile,
    server_url: str,
    model_alias: str,
    max_tokens: int,
    temperature: float,
    command: Sequence[str] | None,
) -> dict[str, Any]:
    return {
        "profile": profile.name,
        "server_url": server_url,
        "api_model": model_alias,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "managed_server_command": list(command) if command is not None else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0].casefold() == "hf":
        try:
            from probe_harness.cli import main as hf_main
        except ImportError:
            print(
                "chat-cli: the HF harness is unavailable; run through the private "
                "repository environment",
                file=sys.stderr,
            )
            return 2
        return hf_main(arguments[1:])

    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        profile = get_profile(args.profile)
        max_tokens = args.max_tokens or profile.default_max_tokens
        temperature = profile.default_temperature if args.temperature is None else args.temperature
        if args.startup_timeout <= 0 or args.request_timeout <= 0:
            raise ChatCliError("timeouts must be greater than zero")
        if not args.alias.strip():
            raise ChatCliError("alias must not be empty")

        if args.server_url:
            if args.model is not None:
                raise ChatCliError("do not pass a model path when attaching with --server-url")
            base_url = normalize_server_url(args.server_url)
            if args.dry_run:
                print(
                    json.dumps(
                        _dry_run_payload(
                            profile=profile,
                            server_url=base_url,
                            model_alias=args.alias,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            command=None,
                        ),
                        indent=2,
                    )
                )
                return 0
            client = LlamaServerClient(base_url)
            if not client.health(timeout=2.0):
                raise ChatCliError(f"llama-server is not ready at {base_url}")
            return _run_requests(args, client, profile, max_tokens, temperature)

        if args.model is None:
            raise ChatCliError("a GGUF model path is required unless --server-url is used")
        model = resolve_model_path(args.model)
        executable = resolve_server_command(args.server_command)
        port = select_port(args.host, args.port)
        config = ServerConfig(
            executable=executable,
            model=model,
            host=args.host,
            port=port,
            context_size=args.ctx_size,
            gpu_layers=args.gpu_layers,
            threads=args.threads,
            parallel=args.parallel,
            alias=args.alias,
            chat_template=profile.chat_template,
            startup_timeout=args.startup_timeout,
            extra_args=tuple(args.server_arg),
            show_log=args.show_server_log,
        )
        if args.dry_run:
            print(
                json.dumps(
                    _dry_run_payload(
                        profile=profile,
                        server_url=config.base_url,
                        model_alias=args.alias,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        command=config.command(),
                    ),
                    indent=2,
                )
            )
            return 0

        print(f"Starting llama-server for {model.name} ...", file=sys.stderr)
        with ManagedServer(config):
            print(f"llama-server ready at {config.base_url}", file=sys.stderr)
            client = LlamaServerClient(config.base_url)
            return _run_requests(args, client, profile, max_tokens, temperature)
    except ChatCliError as error:
        print(f"chat-cli: {error}", file=sys.stderr)
        return 2


def _run_requests(
    args: argparse.Namespace,
    client: LlamaServerClient,
    profile: Profile,
    max_tokens: int,
    temperature: float,
) -> int:
    if args.prompt:
        for index, prompt in enumerate(args.prompt):
            if index:
                print()
            submit_prompt(
                client,
                profile=profile,
                prompt=prompt,
                model=args.alias,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=args.request_timeout,
                show_json=args.show_json,
            )
        return 0
    return run_console(
        client,
        profile=profile,
        model=args.alias,
        max_tokens=max_tokens,
        temperature=temperature,
        request_timeout=args.request_timeout,
        show_json=args.show_json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
