# Repository uv toolchain

`pin.json` is the single authority for the uv executable used by ABLITERALUS.
It records the release, official platform archives, archive digests, and the
digests of every deployed executable.

The executable payload is deliberately ignored by Git and restored beneath this
directory. Mutable caches and download staging belong under `.scratch`, never
under `deps` or a user-global cache.

On Windows, restore or verify the payload with the repository's portable Python:

```powershell
pwsh -NoProfile -File deps/uv/restore-uv.ps1
```

The restore operation is explicit and infrequent. Normal launchers do not install,
self-update, search `PATH`, or silently substitute another uv version. They fail
with an actionable error when this payload is absent or does not match the pin.

For normal Windows commands, use the non-installing runner:

```powershell
pwsh -NoProfile -File deps/uv/run-uv.ps1 --version
```

It verifies the installed payload before every invocation, fixes uv's Python to the
portable runtime under `deps/python`, and scopes temporary files and caches to
`.scratch`. It also isolates configuration discovery from machine-level uv files by
redirecting uv's user/system roots into `.scratch`, while retaining project configuration
and removing ambient uv/Python configuration variables.
Machine-local settings therefore cannot silently change repository operations.
`run_uv.py` provides the same contract when invoked by an already selected Python on
another platform. Lightning provisioning consumes the Linux asset from the same manifest.
