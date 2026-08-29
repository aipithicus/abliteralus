# ABLITERALUS lab bench

`lab-bench` is a thin local control plane. It composes the public experiment code;
it does not create a second surgery implementation.

- Local surgery runs through `surgery_artifacts.integration`, which composes the
  public runner with the private artifact stage.
- Remote surgery invokes `python -m lab_bench.lightning`.
- Studio lifecycle, status, ports, and generic remote commands use Lightning's SDK.
- Interactive shells use the system OpenSSH client and the existing OS SSH
  key/agent. SSH private keys are not moved into Proton Pass.
- Every credential-bearing local subprocess is launched through `secret_manager`.
- Research notes and result metadata are written through the integrity-checked
  `research_journal` module.

The tracked example contains no account IDs or secrets. Copy `lab.example.toml` to
the ignored `lab.local.toml`, set `OWNER/TEAMSPACE`, and optionally paste the SSH
destination shown by Lightning for the Studio. Keep normal OpenSSH host-key checking
enabled; this tool does not add permissive SSH options.

## Install

From the ABLITERALUS repository root, synchronize the one private project with
the repository-owned uv executable:

```powershell
pwsh -NoProfile -File deps/uv/run-uv.ps1 sync --project private --group dev
```

This installs every private console command into `private/.venv`; it does not
modify the public-shadow environment, register Python with Windows, or create a
global tool environment. Activate that environment or invoke
`private/.venv/Scripts/lab-bench.exe` directly.

Use `--config private/config/lab.local.toml` explicitly at first. Later, the
non-secret `LAB_BENCH_CONFIG` variable can point to that file if desired.

## Run workspaces and tests

Every local surgery, Lightning controller, inference child, and managed pytest
session receives a unique workspace beneath:

```text
.scratch/runs/<kind>/<UTC-run-id>/
```

The controller creates the leaf atomically with the repository's normal inherited
permissions, then sets `TEMP`, `TMP`, and `TMPDIR` before the child interpreter
starts. It also exposes `ABLITERALUS_RUN_ID` and `ABLITERALUS_WORK_DIR` to the child.
These directories are disposable and are removed after either success or failure;
durable checkpoints, capsules, and study results continue to live under `outputs/`.
The small `run-context.json` records only paths, timestamps, status, and controller
PID; it never records the child command, environment, or resolved secret values.
Use `--keep-workdir` before a command's forwarded arguments when debugging.

The managed test launcher replaces the former shared `.scratch/pytest` base and
keeps coverage state inside the same run workspace:

```text
lab-bench --config private/config/lab.local.toml test -- tests/test_run_paths.py
lab-bench --config private/config/lab.local.toml test --cwd private -- -q
lab-bench --config private/config/lab.local.toml test --keep-workdir -- private/tests/lab_bench
```

A trusted repository-local Codex `PreToolUse` hook in
[`../../.codex/hooks.json`](../../.codex/hooks.json) blocks shell commands that
invoke pytest outside this controller. Review new or changed definitions with
Codex `/hooks`; Codex skips an untrusted project hook. The hook is an agent
guardrail, while this launcher remains the owner of workspace allocation,
environment isolation, coverage state, and cleanup.

A controller killed before its `finally` block may leave a workspace behind.
Inspection is read-only, and cleanup is preview-first. A workspace whose active
controller process still exists is never selected:

```text
lab-bench --config private/config/lab.local.toml runs list
lab-bench --config private/config/lab.local.toml runs clean --older-than-hours 24
lab-bench --config private/config/lab.local.toml runs clean --older-than-hours 24 --apply
```

Hugging Face and uv caches are intentionally shared at `.scratch/cache/huggingface`
and `.scratch/cache/uv`; they are caches rather than run evidence and are not part
of stale-run cleanup.

## Storage creep and reclamation

The storage census partitions the repository so growth is visible without treating
all large data as garbage. Sizes are logical file bytes; links and Windows reparse
points are reported but never followed. An ordinary report is read-only:

```text
lab-bench --config private/config/lab.local.toml storage report
lab-bench --config private/config/lab.local.toml storage report --json
```

Use `--record` at the cadence you want to measure. Each aggregate-only snapshot is
an immutable JSON file under `.scratch/storage/snapshots`; it contains category
sizes, counts, timestamps, and bounded scan diagnostics, but no commands,
environments, prompts, or secrets.

```text
lab-bench --config private/config/lab.local.toml storage report --record
lab-bench --config private/config/lab.local.toml storage history --limit 12
```

Reclamation requires one or more explicit categories and is always preview-first.
The age test uses the newest observed modification anywhere in a selected path.
Hugging Face and uv caches are reclaimed only as whole cache roots, which avoids
leaving an internally inconsistent partial cache. Cache application is refused
while a managed run is active.

```text
# Routine lightweight cleanup preview, then application.
lab-bench --config private/config/lab.local.toml storage clean --category runs --category tool-state --older-than-days 7
lab-bench --config private/config/lab.local.toml storage clean --category runs --category tool-state --older-than-days 7 --apply

# More conservative whole-cache reclamation.
lab-bench --config private/config/lab.local.toml storage clean --category huggingface-cache --category uv-cache --older-than-days 30
lab-bench --config private/config/lab.local.toml storage clean --category huggingface-cache --category uv-cache --older-than-days 30 --apply
```

`outputs/`, the artifact registry, `.venv`, `deps`, Git data, unclassified scratch
state, and repository source are report-only and cannot be passed as reclaim
categories. A path with scan errors is shown as `INCOMPLETE` and is omitted from
reclamation rather than guessed at. Caches selected through externally overridden
`HF_HOME` or `UV_CACHE_DIR` paths outside this repository are intentionally outside
the census and cleanup authority. The `uv-cache` category also accounts for the
older workspace path `.scratch/uv-cache`, allowing it to age out safely after the
canonical `.scratch/cache/uv` path takes over.

## Private research journal

The journal is an ABLITERALUS-owned implementation under `private/src/research_journal`;
it has no runtime import or path dependency on Codex Scientiae. By default,
`lab-bench` stores it at `outputs/research-journal/journal.jsonl`. Each line is
canonical UTF-8 JSON, append operations are serialized across processes, and every
record links to the previous record by SHA-256. The chain detects truncation,
reordering, and edits; it is an integrity check, not a cryptographic signature.
The shared JSONL engine appends batches in place and records each committed batch in
one adjacent `.transactions.jsonl` table. A binary `{stem}.jidx` supplies indexed
record offsets and is extended in place during ordinary append; neither mechanism
creates per-entry shard files.

Entries are explicit rather than inferred from subprocesses. Relate a note to its
durable evidence using the shorthand relation flags, and put machine-readable
measurements in a JSON object:

```text
lab-bench --config private/config/lab.local.toml journal add --kind experiment.plan --title "Llama Guard positive steering sweep" --run-id guard-sweep-001 --model meta-llama/Llama-Guard-3-1B --tag llama-guard

lab-bench --config private/config/lab.local.toml journal add --kind experiment.observation --title "Layer 12 crossed baseline" --body-file outputs/guard-sweep-001/observation.md --data-file outputs/guard-sweep-001/metrics.json --run-id guard-sweep-001 --artifact guard/sweep-positive-002

lab-bench --config private/config/lab.local.toml journal list --model meta-llama/Llama-Guard-3-1B --limit 10
lab-bench --config private/config/lab.local.toml journal show ENTRY_UUID
lab-bench --config private/config/lab.local.toml journal verify
```

`--relation TYPE=TARGET` adds domain-specific links beyond the built-in `--run-id`,
`--model`, and `--artifact` shorthands. Entry sharing state defaults to `private`;
`candidate` and `approved` record curation intent only. Nothing uploads, syncs, or
submits to OBLITERATUS automatically. The writer rejects known credential field
names and common token/private-key patterns, but the journal should still never be
used for secrets.

Recovery is preview-first. If verification finds an incomplete or corrupt suffix,
the apply step preserves the entire original in a timestamped `.bak` file before
replacing the journal with its last valid prefix:

```text
lab-bench --config private/config/lab.local.toml journal repair
lab-bench --config private/config/lab.local.toml journal repair --apply
```

The lower-level `research-journal --journal PATH ...` command is available for
other local projects that want the same portable store without depending on the
ABLITERALUS control plane.

## Local surgery and inference

```text
lab-bench --config private/config/lab.local.toml local-surgery run --config private/experiments/example.yaml

lab-bench --config private/config/lab.local.toml local-surgery --anonymous-hub run --config private/experiments/public-mirror.yaml

lab-bench --config private/config/lab.local.toml local-inference -- python path/to/chat_client.py
```

For surgery `run` operations, the controller's generated ID is also supplied as
`--run-id`, so the disposable controller workspace, durable local output, and
Lightning remote plan share one correlation identifier. An explicit forwarded
`--run-id` remains authoritative and is never silently renamed.

Only an online local `run` receives the `local-surgery` Hub profile. Offline runs,
preflight, postprocessing, and smoke tests run without a credential environment.
Use `--anonymous-hub` only for public sources: it bypasses Proton Pass, removes
inherited Hugging Face token variables, and disables implicit cached credentials
for the child process. Unless `HF_HOME` is explicitly set, local surgery caches
Hub files beneath the ignored workspace-local `.scratch/cache/huggingface` tree.

## Lightning surgery

Planning is local and credential-free:

```text
lab-bench --config private/config/lab.local.toml lightning-surgery plan --config private/experiments/example.yaml
```

Provision the Studio runtime explicitly before the first run, and again only when
`uv.lock`, the pinned uv version, or the requested `base`/`gguf` dependency variant
changes. `--dry-run` prints the credential-free plan without contacting Lightning.

```text
lab-bench --config private/config/lab.local.toml studio provision --experiment-config private/experiments/example.yaml --dry-run

lab-bench --config private/config/lab.local.toml studio provision --experiment-config private/experiments/example.yaml
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
lab-bench --config private/config/lab.local.toml studio start --purpose control
lab-bench --config private/config/lab.local.toml studio doctor --experiment-config private/experiments/example.yaml
lab-bench --config private/config/lab.local.toml studio stop
```

A run receives the Lightning surgery profile. `HF_TOKEN` is passed to the existing
launcher, forwarded to the Studio only around the remote command, and restored or
deleted afterward. The launcher stops compute that it started unless its explicit
`--keep-running` option is used. Before uploading or exposing the Hub token, it
checks the exact provisioned runtime. A normal run never installs uv and never runs
`uv sync`; a missing or stale runtime fails with an instruction to provision it.

```text
lab-bench --config private/config/lab.local.toml lightning-surgery run --config private/experiments/example.yaml --collect capsule
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
lab-bench --config private/config/lab.local.toml lightning-surgery run --config private/experiments/example.yaml --max-runtime 14400 --collect capsule
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
lab-bench --config private/config/lab.local.toml artifact verify outputs/lightning/RUN/artifact
lab-bench --config private/config/lab.local.toml artifact register outputs/lightning/RUN/artifact --ref qwen25-7b/experiment-001
lab-bench --config private/config/lab.local.toml artifact resolve qwen25-7b/experiment-001
lab-bench --config private/config/lab.local.toml artifact rehydrate qwen25-7b/experiment-001 --base D:/models/Qwen2.5-7B-Instruct --output outputs/rehydrated/experiment-001
```

Registration validates the capsule before an immutable object is committed and
updates refs atomically. Rehydration refuses a base whose tensor hashes differ.

## Studio lifecycle and brief inference

```text
lab-bench --config private/config/lab.local.toml studio status
lab-bench --config private/config/lab.local.toml studio start --purpose inference
lab-bench --config private/config/lab.local.toml studio detach -- python server.py
lab-bench --config private/config/lab.local.toml studio ports --add 8000
lab-bench --config private/config/lab.local.toml studio stop
```

`studio exec` quotes each supplied argument before sending one remote shell command.
`studio detach` is intended for an inference server or another bounded background
session. The configured forwarded variables are restored immediately after launch;
the spawned remote process retains its inherited copy.

Proton's `run` command forwards basic input/output but is not a full pseudo-terminal.
For a genuinely interactive terminal, use direct OpenSSH:

```text
lab-bench --config private/config/lab.local.toml ssh
lab-bench --config private/config/lab.local.toml ssh -- python remote_chat.py
```

Start or stop the Studio through the API commands separately. This keeps the API
credential plane and the SSH key plane independent and auditable.
