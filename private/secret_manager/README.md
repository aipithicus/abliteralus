# Aipithicus secret manager

This package turns non-secret Proton Pass locators into short-lived process
environments. Its normal path is `pass-cli run`, so the Python parent retains only
`pass://` references and Proton Pass masks matching values in child output.

It deliberately does not manage SSH keys, write decrypted dotenv files, persist
credential values in user or machine environment variables, or cache vault exports.
The authenticated Proton Pass CLI session is the local root of trust.

## Configuration

Copy `secrets.example.toml` to the ignored `secrets.local.toml` and replace only the
vault Share ID and item IDs. The `field = "secret"` setting addresses the Secret
field in Proton's API Credential items. Item titles and the separate API Key label
remain organizational metadata; the configured item ID makes lookup unambiguous.

Config discovery order is:

1. `--config PATH`;
2. the non-secret `SECRET_MANAGER_CONFIG` environment variable;
3. `%APPDATA%/Aipithicus/secret-manager.toml` on Windows;
4. `$XDG_CONFIG_HOME/aipithicus/secret-manager.toml` or
   `~/.config/aipithicus/secret-manager.toml` elsewhere.

## Commands

```text
secret-manager --config secrets.local.toml inventory
secret-manager --config secrets.local.toml check local-surgery
secret-manager --config secrets.local.toml run local-surgery -- python worker.py
```

`inventory` shows aliases and environment-variable names, not values. References are
also hidden unless `--show-references` is explicitly requested. `check` runs an
ephemeral child that emits booleans only. The check path allows at least 15 seconds
for cold provider-wrapper and Python startup; a larger configured provider timeout
is preserved.

Privileged profiles require `--allow-privileged` and a typed confirmation. Automation
must additionally pass `--yes`. Broker-enabled profiles can never address privileged
credentials.

```text
secret-manager --config secrets.local.toml run hf-admin \
  --allow-privileged --yes -- python explicit_admin_task.py
```

## OBLITERATUS broker

The `secret-broker` entry point implements the executable contract expected by
`OBLITERATUS_SECRET_COMMAND`: exactly one normalized environment-variable argument,
only the value on stdout, exit 2 when unavailable, and a generic nonzero failure for
malformed invocation or provider/configuration errors. Provider stderr is discarded.

Set these non-secret deployment values before starting OBLITERATUS:

```text
SECRET_MANAGER_CONFIG=<absolute path to secrets.local.toml>
SECRET_MANAGER_PROFILE=local-inference
OBLITERATUS_SECRET_COMMAND=<absolute path to secret-broker.exe>
```

The broker necessarily materializes one value in its short-lived process memory.
Everyday batch commands should prefer `secret-manager run`, where Python never sees
the resolved value.

## Local development install

From the ABLITERALUS repository root:

```text
uv tool install --editable ./private/secret_manager
```

Re-run that command after changing the editable tool configuration. The generated
absolute `secret-broker.exe` path can then be used by OBLITERATUS on Windows.
