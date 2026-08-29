"""Failure types whose messages are safe to show without provider output."""


class SecretManagerError(RuntimeError):
    """Base class for normalized secret-manager failures."""


class ConfigError(SecretManagerError):
    """The non-secret mapping configuration is invalid."""


class ProviderError(SecretManagerError):
    """A secret provider could not safely perform the requested operation."""


class PrivilegedSecretError(SecretManagerError):
    """A privileged credential was requested without an explicit gate."""


class SecretUnavailable(SecretManagerError):
    """A broker name is not mapped to an available non-privileged credential."""
