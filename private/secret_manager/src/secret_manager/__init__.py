"""Ephemeral, profile-based secret delivery."""

from .config import SecretManagerConfig, load_config
from .manager import SecretManager

__all__ = ["SecretManager", "SecretManagerConfig", "load_config"]
__version__ = "0.1.0"
