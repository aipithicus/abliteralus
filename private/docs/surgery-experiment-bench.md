# Surgery experiment bench

This bench gives local and Lightning runs the same validated YAML contract,
source revision, ABLITERALUS entry point, artifact layout, and run manifest. It
keeps two distinct claims separate:

1. ABLITERALUS surgery operates on a Transformers/Safetensors checkpoint.
2. GGUF is produced afterward for deployment and same-runtime A/B inference.

Loading a GGUF through Transformers dequantizes it, so a small GGUF file is not
evidence that surgery itself fits in the same amount of memory.

## Local miniature lane

The checked-in profile pins `Qwen/Qwen2.5-0.5B-Instruct`, uses eight harmful and
eight harmless probes, performs one-direction basic surgery in float16, converts
both the untouched and operated checkpoints with the same llama.cpp converter,
quantizes both to `Q4_K_M`, and runs the same deterministic llama.cpp prompts.
The baseline conversion is important: comparing an official GGUF against a
locally converted operated model would confound surgery with converter and
quantizer differences.

From Nushell:

```nu
(
  pwsh -NoProfile -File deps/uv/run-uv.ps1 run --frozen --extra gguf abliteralus-surgery preflight
    --config private/experiments/surgery/local-qwen25-0.5b.yaml
)

(
  lab-bench --config private/config/lab.local.toml local-surgery run
    --config private/experiments/surgery/local-qwen25-0.5b.yaml
)
```

`lab-bench` defaults `HF_HOME` to the ignored workspace-local
`.scratch/cache/huggingface` tree when the caller has not selected another cache.

The workspace-local `.scratch` tree is ignored by Git. Allow roughly 8 GB of free
disk for the source snapshot, operated checkpoint, temporary float GGUFs, and
the two final quantized GGUFs. The runner removes float GGUF intermediates only
after quantization succeeds.

### Llama Guard safety-label lane

`private/experiments/surgery/local-llama-guard-3-1b.yaml` pins the full-precision
`meta-llama/Llama-Guard-3-1B` Safetensors checkpoint. The Hugging Face account
behind `HF_TOKEN` must first be granted access to the gated repository. Surgery
uses eight built-in harmful/harmless pairs; evaluation uses a separate set of
four unsafe and four safe prompts.

Llama Guard's chat template requires typed text-content blocks. The shared chat
renderer validates that the rendered conversation still contains the source
prompt and falls back between plain and typed content representations. The
profile skips the ordinary assistant-coherence/refusal verifier and instead
loads the untouched and operated checkpoints sequentially, recording generated
verdicts and the emitted-label-step `unsafe - safe` logit margin in
`evaluation/hf-results.json`. This keeps peak VRAM bounded to one evaluated model.
Both profiles preserve the source checkpoint's BF16 tensor contract so an exact
capsule does not degenerate into a full-checkpoint FP16 conversion. CUDA preflight
fails closed when a BF16 profile is selected on a GPU without BF16 support.

While official access is pending,
`private/experiments/surgery/local-llama-guard-3-1b-mirror.yaml` pins the public
`project-free-llama/Llama-Guard-3-1B` mirror. Its 13 repository objects match
the pinned Meta repository object-for-object; the profile records both commits
and the official `model.safetensors` SHA-256. The runner hashes the downloaded
weights before surgery and carries the verified identity into the run manifest
and artifact capsule. The mirror also retains Meta's license and use policy;
using it does not replace accepting or complying with those terms.

Public mirrors do not need a Hub credential. `--anonymous-hub` bypasses Proton
Pass, removes inherited Hub token variables, and disables implicit cached-token
delivery for that child process:

```nu
(
  lab-bench --config private/config/lab.local.toml
    local-surgery --anonymous-hub run
    --config private/experiments/surgery/local-llama-guard-3-1b-mirror.yaml
)
```

If evaluator code changes after a capsule-backed run, rehydrate the exact surgery
checkpoint and rerun only the Transformers A/B evaluation:

```nu
(
  lab-bench --config private/config/lab.local.toml artifact rehydrate
    outputs/surgery/<experiment>/<run-id>/artifact
    --base <resolved-baseline-checkpoint>
    --output .scratch/verification/<surgery-id>
)

(
  lab-bench --config private/config/lab.local.toml
    local-surgery --anonymous-hub reevaluate-hf
    --config private/experiments/surgery/local-llama-guard-3-1b-mirror.yaml
    --run-dir outputs/surgery/<experiment>/<run-id>
    --surgery-checkpoint .scratch/verification/<surgery-id>
)
```

`reevaluate-hf` verifies the pinned baseline hashes and the rehydrated capsule's
surgery marker before loading either model. It preserves the prior result under
`evaluation/history/`, writes the replacement atomically, and records the new
runtime, Git state, hashes, and summaries in `run-manifest.json`.

Run the HF surgery/evaluation/capsule lane without waiting for llama.cpp tooling:

```nu
(
  lab-bench --config private/config/lab.local.toml local-surgery run
    --config private/experiments/surgery/local-llama-guard-3-1b.yaml
    --skip-gguf
)
```

The full command omits `--skip-gguf`; its GGUF smoke stage reuses the labelled
cases and records per-class summaries. A shift in unsafe verdicts is exploratory
evidence only. A valid result must also preserve safe controls and avoid a global
or unparsable-label collapse.

Tool discovery checks, in order, explicit YAML paths, the
`ABLITERALUS_LLAMA_*` environment variables, `LLAMA_CPP_ROOT`, executable paths,
and nearby `llama.cpp` source directories. Useful overrides are:

```nu
$env.ABLITERALUS_LLAMA_CPP_ROOT = 'D:/path/to/llama.cpp'
$env.ABLITERALUS_LLAMA_QUANTIZE = 'D:/path/to/llama-quantize.exe'
$env.ABLITERALUS_LLAMA_CLI = 'D:/path/to/llama-cli.exe'
```

On Windows the runner adds the active PyTorch `lib` directory to child-process
`PATH`. This lets CUDA llama.cpp builds find the same CUDA runtime DLLs already
used by PyTorch without modifying the global environment.

Each run writes:

```text
outputs/surgery/<experiment>/<run-id>/
  run-manifest.json
  surgery.log
  evaluation/
    hf-results.json
    history/
      hf-results-<timestamp>.json
  artifact/
    manifest.json
    tensor-manifest.json
    delta.safetensors
    recipe.json
    SHA256SUMS
  gguf/
    baseline.Q4_K_M.gguf
    surgery.Q4_K_M.gguf
    smoke-results.json
    *.convert.log
    *.quantize.log
```

With `artifact.mode: capsule`, `hf/` is a temporary staging checkpoint. It is
removed only after optional GGUF work and exact capsule validation complete. If
capsule construction fails and `artifact.fallback` is `full-checkpoint`, `hf/`
is retained and the run manifest records the fallback. Pickle, `.pt`, and other
executable weight containers are never copied into a capsule.

The manifest records the requested and resolved model revisions, config hash,
repository state, preflight evidence, stage results, GGUF hashes, and raw A/B
completions. These smoke results are plumbing and non-regression evidence; a
larger prompt set and controlled statistics are still required for a behavioral
claim.

Use `--skip-gguf` to exercise the HF surgery lane without converter tooling, and
`--offline` after the pinned snapshot is present in `HF_HOME`.

If conversion or quantization fails after the HF checkpoint was saved, fix the
preflight issue and resume only that stage:

```nu
(
  pwsh -NoProfile -File deps/uv/run-uv.ps1 run --frozen --extra gguf abliteralus-surgery postprocess
    --config private/experiments/surgery/local-qwen25-0.5b.yaml
    --run-dir outputs/surgery/qwen25-0.5b-local-mini/RUN_ID
)
```

The retry replaces only partial generated GGUF files, records the previous
failure under `recoveries`, and never repeats surgery.

To rerun only the deterministic llama.cpp prompts after changing or upgrading
the local inference executable, keep the verified GGUF files in place and use:

```nu
(
  pwsh -NoProfile -File deps/uv/run-uv.ps1 run --frozen --extra gguf abliteralus-surgery smoke
    --config private/experiments/surgery/local-qwen25-0.5b.yaml
    --run-dir outputs/surgery/qwen25-0.5b-local-mini/RUN_ID
)
```

This command verifies both GGUF hashes against the run manifest before loading
them, preserves the prior evaluations under `smoke_history`, and records the
exact Git checkout and llama.cpp probe under `smoke_runtime`.

## Lightning AI scale lane

The Lightning control plane separates infrequent runtime provisioning from each
experiment launch. Provisioning:

1. Verify that all tracked and untracked runtime inputs are committed.
2. Build a deterministic bundle containing only `abliteralus/`, the exact
   `private/src/surgery_artifacts/` implementation subtree, `pyproject.toml`,
   `uv.lock`, `README.md`, and the selected experiment YAML.
3. Restore the digest-bound Linux `uv` declared by `deps/uv/pin.json` into an
   isolated, versioned tool directory without invoking `pip` or searching `PATH` for uv.
4. Materialize a dependency-only environment keyed by the exact lockfile hash
   and the requested `base` or `gguf` dependency variant.
5. Clear ambient uv/Python configuration, retain project configuration, and keep uv,
   Hugging Face, XDG, bytecode, configuration, and temporary state beneath the
   persistent Studio home.
6. Stop compute if and only if this invocation started it.

The AI Development Studio image's `python` command is the one explicit bootstrap
prerequisite. Provisioning invokes it in isolated mode once, records its exact
`sys.executable`, and fixes uv to that interpreter with Python downloads disabled.
The uv executable itself, its archive, and all application dependencies remain bound
to `deps/uv/pin.json` and `uv.lock`.

A normal surgery launch then:

1. Starts the requested Studio machine if the Studio is stopped, retrying capacity
   misses within a bounded local allocation window.
2. Runs a read-only doctor check for the exact lock-addressed runtime before
   uploading code or forwarding a Hub token.
3. Uploads the deterministic experiment bundle and runs it directly with the
   provisioned environment's Python. It does not install uv or run `uv sync`.
4. Collects summary metadata by default and releases compute it started.

The bundle explicitly excludes `.git/`, every other `private/` subtree, `ci/`,
tests, local caches, and unrelated worktree content. A dirty allowlisted package
or lockfile is rejected so a remote result cannot claim a commit that does not
contain the executed code.
Lightning profiles must use a remote-addressable `OWNER/MODEL` source; local
checkpoint directories are never added to the upload bundle.

Materialize the current pinned Lightning SDK through the repository uv executable,
then authenticate:

```nu
pwsh -NoProfile -File deps/uv/run-uv.ps1 sync --project private --group dev
private/.venv/Scripts/lightning.exe login
```

The current pinned Lightning SDK requires Python 3.11 or newer; the core
ABLITERALUS package continues to support Python 3.10.

Planning is local, does not import the SDK, and does not start paid compute:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe plan
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
    --machine L40S
    --interruptible
)
```

Provisioning can also be reviewed locally before it contacts Lightning:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe provision
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
    --machine CPU-4
    --dry-run
)
```

Run the real provision operation before the first experiment and again only
when `uv.lock`, the pinned uv version, or the selected dependency variant changes:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe provision
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
    --machine CPU-4
)
```

`doctor` is deliberately non-provisioning and requires the Studio to already be
running. It validates the uv version, runtime marker, lock hash, Python, and required
imports without forwarding `HF_TOKEN`:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe doctor
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
)
```

Run only after reviewing that plan:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe run
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
    --machine L40S
    --interruptible
)
```

For a strict unattended H200 request, leave the fallback list empty and set both a
local allocation deadline and a remote compute lease:

```nu
(
  private/.venv/Scripts/abliteralus-lightning.exe run
    --config private/experiments/surgery/lightning-qwen25-7b.yaml
    --teamspace OWNER/TEAMSPACE
    --studio abliteralus-surgery
    --machine H200
    --allocation-timeout 1200
    --allocation-retry 30
    --pending-policy adopt
    --max-runtime 14400
    --collect capsule
)
```

`--fallback-machine` is repeatable and ordered, but never implicit. For example,
`--fallback-machine H100 --fallback-machine L40S` permits those machines only after
an H200 capacity miss. Omit the flags when the experiment requires H200.

Starting GPU compute incurs Lightning charges. The default run remains attached
until surgery finishes and then releases compute it started, including after a
remote failure. If the Studio was already running, the launcher requires
`--reuse-running` and will not stop or switch that existing machine. Use
`--keep-running` only when intentionally retaining compute after the run. When the
launcher owns the start, `--max-runtime SECONDS` forwards Lightning's requested
remote run duration. It does not bound the local wait for capacity; that is
`--allocation-timeout` (900 seconds by default).

A Studio already `Pending` is ambiguous after a prior controller exit. The default
`--pending-policy fail` refuses to take ownership. `adopt` waits for that request,
then owns and releases it; use this only for a dedicated single-controller Studio.
`stop` cancels the Pending request and begins a fresh bounded allocation cycle.
The headless supervisor polls an accepted `Pending` start on the configured retry
cadence. Allocation attempts and timestamped status observations are written
incrementally to `allocation.json`. Successful
result manifests also separate allocation, remote-command, and total elapsed time.
If startup times out or is interrupted, the controller reconciles the Studio and
stops any Pending or Running compute it may have requested. The remote
`--max-runtime` lease is still necessary for a hard local process or machine loss
that prevents cleanup code from running.

No credential is forwarded implicitly. For a gated model, opt in by naming the
already-set local variable:

```nu
... --forward-env HF_TOKEN
```

Forwarded values are restored to their prior Studio state after the command,
including when a run fails or reuses an already-running Studio.

The default `--collect summary` downloads manifests and logs. `--collect capsule`
downloads the native capsule, validates every local checksum and semantic id,
and records that verification in `lightning-result.json` before owned compute is
released. `--collect all` downloads the whole output directory; `--collect none`
leaves every artifact remote. Collection does not delete remote data and does not
claim that a capsule has entered the durable local registry.

## Capsule registry and rehydration

The exact native capsule is the durable source of truth. Register it locally only
after verification:

```nu
(
  lab-bench --config private/config/lab.local.toml artifact register
    outputs/lightning/RUN/artifact
    --ref qwen25-7b/experiment-001
)
```

Rehydration requires the exact base tensor identity recorded by the capsule and
performs a second full tensor-hash pass after writing the checkpoint:

```nu
(
  lab-bench --config private/config/lab.local.toml artifact rehydrate
    qwen25-7b/experiment-001
    --base D:/models/Qwen2.5-7B-Instruct
    --output outputs/rehydrated/qwen25-7b-experiment-001
)
```

Capsule v1 can encode replayable ABLITERALUS projections, sparse row/element
updates, dense additions, and exact tensor replacement fallback. PEFT LoRA is a
derived export only when every operation is additive and tolerance-verified;
norm-preserving scale operations correctly remain native-only. A llama.cpp GGUF
LoRA is derived through the official `convert_lora_to_gguf.py`, and its converter
file hash and Git commit are recorded. Neither adapter format replaces the native
exact capsule.

The default persistent layout is:

```text
$HOME/.abliteralus/
  tools/uv/<version>/
  runtimes/<lock-hash>-<variant>/
  runtime-manifests/<lock-hash>-<variant>.json
  cache/uv/
  cache/huggingface/
  cache/xdg/
  cache/python-bytecode/
  configuration/user/
  configuration/system/uv/uv.toml
  data/xdg/
  temp/
  bundles/
  runs/<run-id>/
```

The initial 7B profile deliberately leaves GGUF disabled. It proves the larger
Safetensors surgery and capsule lane without also provisioning a compiler and a
pinned llama.cpp build. A remote GGUF lane should pin and preflight that
toolchain before being used for cross-run comparisons.

Lightning's current SDK supports programmatic Studio start/stop, command
execution, uploads, and downloads; see the official
[Studio SDK documentation](https://lightning.ai/docs/overview/sdk/studio) and
[Lightning SDK repository](https://github.com/Lightning-AI/sdk).

## Quantized destructive surgery caveat

The checked-in profiles use floating surgery. The existing packed 4/8-bit
destructive path is still experimental: multi-direction operation may perform
repeated lossy dequantize/project/requantize cycles. Do not use this bench to
claim that packed-surgery fidelity or speed has improved until the dedicated
one-window oracle, conversion-count tests, and isolated CUDA benchmark are in
place.
