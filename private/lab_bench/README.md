# ABLITERALUS lab bench

`lab-bench` is a thin local control plane. It composes the public experiment code;
it does not create a second surgery implementation.

- Local surgery runs through `surgery_artifacts.integration`, which composes the
  public runner with the private artifact stage.
- Remote surgery invokes `python -m abliteralus.lightning_surgery`.
- Studio lifecycle, status, ports, and generic remote commands use Lightning's SDK.
- Interactive shells use the system OpenSSH client and the existing OS SSH
  key/agent. SSH private keys are not moved into Proton Pass.
- Every credential-bearing local subprocess is launched through `secret_manager`.

The tracked example contains no account IDs or secrets. Copy `lab.example.toml` to
the ignored `lab.local.toml`, set `OWNER/TEAMSPACE`, and optionally paste the SSH
destination shown by Lightning for the Studio. Keep normal OpenSSH host-key checking
enabled; this tool does not add permissive SSH options.

## Install

From the ABLITERALUS repository root:

```text
deps/uv/uv.exe pip install --python .venv/Scripts/python.exe --editable private/secret_manager --editable private/surgery_artifacts --editable private/lab_bench
```

This installs both console commands into the existing project venv; it does not
register Python with Windows or create a global tool environment. Activate the venv
or invoke `.venv/Scripts/lab-bench.exe` directly.

Use `--config private/lab_bench/lab.local.toml` explicitly at first. Later, the
non-secret `LAB_BENCH_CONFIG` variable can point to that file if desired.

## Local surgery and inference

```text
lab-bench --config private/lab_bench/lab.local.toml local-surgery run --config experiments/example.yaml

lab-bench --config private/lab_bench/lab.local.toml local-surgery --anonymous-hub run --config experiments/public-mirror.yaml

lab-bench --config private/lab_bench/lab.local.toml local-inference -- python path/to/chat_client.py
```

Only an online local `run` receives the `local-surgery` Hub profile. Offline runs,
preflight, postprocessing, and smoke tests run without a credential environment.
Use `--anonymous-hub` only for public sources: it bypasses Proton Pass, removes
inherited Hugging Face token variables, and disables implicit cached credentials
for the child process.

## Lightning surgery

Planning is local and credential-free:

```text
lab-bench --config private/lab_bench/lab.local.toml lightning-surgery plan --config experiments/example.yaml
```

Provision the Studio runtime explicitly before the first run, and again only when
`uv.lock`, the pinned uv version, or the requested `base`/`gguf` dependency variant
changes. `--dry-run` prints the credential-free plan without contacting Lightning.

```text
lab-bench --config private/lab_bench/lab.local.toml studio provision --experiment-config experiments/example.yaml --dry-run

lab-bench --config private/lab_bench/lab.local.toml studio provision --experiment-config experiments/example.yaml
```

Provisioning uses the control profile, normally starts the inexpensive control
machine, installs the repository-pinned uv into an isolated tool environment, and
materializes a dependency-only environment keyed by the exact lockfile hash. It
stops compute that it started unless `--keep-running` is explicit. Package and Hub
caches live beneath the configured persistent `lightning.remote_root` (default
`.abliteralus` in the Studio home). `--max-runtime SECONDS` sets Lightning's remote
compute lease; it is independent of the local allocation-wait deadline.

To inspect that environment without changing it, start the Studio and run doctor:

```text
lab-bench --config private/lab_bench/lab.local.toml studio start --purpose control
lab-bench --config private/lab_bench/lab.local.toml studio doctor --experiment-config experiments/example.yaml
lab-bench --config private/lab_bench/lab.local.toml studio stop
```

A run receives the Lightning surgery profile. `HF_TOKEN` is passed to the existing
launcher, forwarded to the Studio only around the remote command, and restored or
deleted afterward. The launcher stops compute that it started unless its explicit
`--keep-running` option is used. Before uploading or exposing the Hub token, it
checks the exact provisioned runtime. A normal run never installs uv and never runs
`uv sync`; a missing or stale runtime fails with an instruction to provision it.

```text
lab-bench --config private/lab_bench/lab.local.toml lightning-surgery run --config experiments/example.yaml --collect capsule
```

For a dedicated single-controller H200 Studio, the non-secret policy can be kept in
`lab.local.toml`:

```toml
[lightning]
surgery_machine = "H200"
allocation_timeout_seconds = 1200
allocation_retry_seconds = 30
pending_policy = "adopt"
surgery_fallback_machines = []
surgery_max_runtime_seconds = 14400
```

An empty fallback list is strict: the controller retries H200 until the 20-minute
deadline and then fails. Add, for example, `surgery_fallback_machines = ["H100"]`
only when that downgrade is acceptable for the experiment. `pending_policy =
"adopt"` resumes a request already queued by this dedicated Studio and takes
responsibility for stopping it after the run. The safer shared-Studio default is
`"fail"`; `"stop"` cancels a stale Pending request before retrying.

Use both bounds for unattended work:

```text
lab-bench --config private/lab_bench/lab.local.toml lightning-surgery run --config experiments/example.yaml --max-runtime 14400 --collect capsule
```

The command-line value overrides the configured four-hour surgery lease when a
particular experiment needs a different ceiling.

The headless supervisor polls accepted `Pending` starts on the configured retry
cadence and writes each observation into `allocation.json`. It also records
allocation and remote-phase timing in `lightning-result.json`, and stops Pending or
Running compute it may have requested when startup times out or is interrupted.
The remote lease remains the final bound if the local controller itself is killed
too abruptly to perform API cleanup.

## Durable surgery artifacts

The `[artifacts]` config points at the local content-addressed registry. Lightning
is compute and staging; it is not the durable store.

```text
lab-bench --config private/lab_bench/lab.local.toml artifact verify outputs/lightning/RUN/artifact
lab-bench --config private/lab_bench/lab.local.toml artifact register outputs/lightning/RUN/artifact --ref qwen25-7b/experiment-001
lab-bench --config private/lab_bench/lab.local.toml artifact resolve qwen25-7b/experiment-001
lab-bench --config private/lab_bench/lab.local.toml artifact rehydrate qwen25-7b/experiment-001 --base D:/models/Qwen2.5-7B-Instruct --output outputs/rehydrated/experiment-001
```

Registration validates the capsule before an immutable object is committed and
updates refs atomically. Rehydration refuses a base whose tensor hashes differ.

## Studio lifecycle and brief inference

```text
lab-bench --config private/lab_bench/lab.local.toml studio status
lab-bench --config private/lab_bench/lab.local.toml studio start --purpose inference
lab-bench --config private/lab_bench/lab.local.toml studio detach -- python server.py
lab-bench --config private/lab_bench/lab.local.toml studio ports --add 8000
lab-bench --config private/lab_bench/lab.local.toml studio stop
```

`studio exec` quotes each supplied argument before sending one remote shell command.
`studio detach` is intended for an inference server or another bounded background
session. The configured forwarded variables are restored immediately after launch;
the spawned remote process retains its inherited copy.

Proton's `run` command forwards basic input/output but is not a full pseudo-terminal.
For a genuinely interactive terminal, use direct OpenSSH:

```text
lab-bench --config private/lab_bench/lab.local.toml ssh
lab-bench --config private/lab_bench/lab.local.toml ssh -- python remote_chat.py
```

Start or stop the Studio through the API commands separately. This keeps the API
credential plane and the SSH key plane independent and auditable.
