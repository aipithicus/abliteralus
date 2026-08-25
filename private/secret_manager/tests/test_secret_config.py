from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
from secret_manager.config import discover_config_path, parse_config
from secret_manager.errors import ConfigError


def sample_mapping() -> dict:
    return {
        "schema_version": 1,
        "providers": {
            "proton": {
                "type": "proton-pass",
                "executable": sys.executable,
                "timeout_seconds": 5.0,
            }
        },
        "credentials": {
            "hf-read": {
                "provider": "proton",
                "vault": "vault-share-id==",
                "item": "read-item-id==",
                "field": "secret",
            },
            "hf-admin": {
                "provider": "proton",
                "vault": "vault-share-id==",
                "item": "admin-item-id==",
                "field": "secret",
                "privileged": True,
            },
            "lightning-user": {
                "provider": "proton",
                "vault": "vault-share-id==",
                "item": "user-item-id==",
                "field": "secret",
            },
            "lightning-key": {
                "provider": "proton",
                "vault": "vault-share-id==",
                "item": "key-item-id==",
                "field": "secret",
            },
        },
        "profiles": {
            "local": {"environment": {"HF_TOKEN": "hf-read"}},
            "control": {
                "environment": {
                    "LIGHTNING_USER_ID": "lightning-user",
                    "LIGHTNING_API_KEY": "lightning-key",
                }
            },
            "admin": {"environment": {"HF_TOKEN": "hf-admin"}},
        },
        "broker": {
            "default_profile": "local",
            "allowed_profiles": ["local"],
        },
    }


def test_config_parses_profiles_and_privilege() -> None:
    config = parse_config(sample_mapping(), path=Path("mapping.toml"))

    assert config.profiles["local"].environment == {"HF_TOKEN": "hf-read"}
    assert config.credentials["hf-admin"].privileged is True
    assert config.broker.default_profile == "local"


def test_unknown_keys_fail_closed() -> None:
    raw = sample_mapping()
    raw["credentials"]["hf-read"]["value"] = "must-never-be-supported"

    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config(raw)


@pytest.mark.parametrize("bad_segment", ["vault/item", "vault?query", "vault#fragment"])
def test_reference_segments_reject_uri_structure(bad_segment: str) -> None:
    raw = deepcopy(sample_mapping())
    raw["credentials"]["hf-read"]["vault"] = bad_segment

    with pytest.raises(ConfigError, match="reserved reference character"):
        parse_config(raw)


def test_broker_default_must_be_allowlisted() -> None:
    raw = sample_mapping()
    raw["broker"]["allowed_profiles"] = []

    with pytest.raises(ConfigError, match="default_profile must be listed"):
        parse_config(raw)


def test_config_discovery_prefers_explicit_then_environment(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.toml"
    environment = tmp_path / "environment.toml"

    assert (
        discover_config_path(explicit, environ={"SECRET_MANAGER_CONFIG": str(environment)})
        == explicit
    )
    assert (
        discover_config_path(None, environ={"SECRET_MANAGER_CONFIG": str(environment)})
        == environment
    )
