"""Interactive chat and probe CLI for local models and managed servers."""

from .cli import (
    PROFILES,
    ChatResult,
    LlamaServerClient,
    ManagedServer,
    Profile,
    ServerConfig,
    emit_result,
    get_profile,
    main,
    normalize_server_url,
    resolve_model_path,
    resolve_server_command,
    run_console,
    select_port,
    submit_prompt,
)
from .errors import ChatCliError

__all__ = [
    "ChatCliError",
    "ChatResult",
    "LlamaServerClient",
    "ManagedServer",
    "Profile",
    "PROFILES",
    "ServerConfig",
    "emit_result",
    "get_profile",
    "main",
    "normalize_server_url",
    "resolve_model_path",
    "resolve_server_command",
    "run_console",
    "select_port",
    "submit_prompt",
]
