from __future__ import annotations

import sys
from pathlib import Path

import pytest
from secret_manager.config import parse_config
from secret_manager.manager import SecretManager


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
            "local": {
                "environment": {"HF_TOKEN": "hf-read"},
            },
            "control": {
                "environment": {
                    "LIGHTNING_USER_ID": "lightning-user",
                    "LIGHTNING_API_KEY": "lightning-key",
                }
            },
            "admin": {
                "environment": {"HF_TOKEN": "hf-admin"},
            },
        },
        "broker": {
            "default_profile": "local",
            "allowed_profiles": ["local"],
        },
    }


@pytest.fixture
def manager(tmp_path: Path) -> SecretManager:
    config = parse_config(sample_mapping(), path=tmp_path / "secrets.local.toml")
    return SecretManager(config)
