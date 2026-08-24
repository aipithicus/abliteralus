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
$env.HF_HOME = ((pwd) | path join .codex hf)

uv run --frozen --extra gguf abliteralus-surgery preflight \
  --config experiments/surgery/local-qwen25-0.5b.yaml

uv run --frozen --extra gguf abliteralus-surgery run \
  --config experiments/surgery/local-qwen25-0.5b.yaml
```

The workspace-local `.codex` cache is ignored by Git. Allow roughly 8 GB of free
disk for the source snapshot, operated checkpoint, temporary float GGUFs, and
the two final quantized GGUFs. The runner removes float GGUF intermediates only
after quantization succeeds.

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
  hf/
  gguf/
    baseline.Q4_K_M.gguf
    surgery.Q4_K_M.gguf
    smoke-results.json
    *.convert.log
    *.quantize.log
```

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
uv run --frozen --extra gguf abliteralus-surgery postprocess \
  --config experiments/surgery/local-qwen25-0.5b.yaml \
  --run-dir outputs/surgery/qwen25-0.5b-local-mini/<run-id>
```

The retry replaces only partial generated GGUF files, records the previous
failure under `recoveries`, and never repeats surgery.

To rerun only the deterministic llama.cpp prompts after changing or upgrading
the local inference executable, keep the verified GGUF files in place and use:

```nu
uv run --frozen --extra gguf abliteralus-surgery smoke \
  --config experiments/surgery/local-qwen25-0.5b.yaml \
  --run-dir outputs/surgery/qwen25-0.5b-local-mini/<run-id>
```

This command verifies both GGUF hashes against the run manifest before loading
them and preserves the prior evaluations under `smoke_history`.

## Lightning AI scale lane

The Lightning launcher is a blocking orchestrator around a persistent Studio:

1. Verify that all tracked and untracked runtime inputs are committed.
2. Build a deterministic bundle containing only `abliteralus/`, `pyproject.toml`,
   `uv.lock`, `README.md`, and the selected experiment YAML.
3. Start the requested Studio machine if the Studio is stopped.
4. Upload the bundle, install the pinned `uv`, sync the frozen environment, and
   run the same `abliteralus-surgery` module.
5. Collect summary metadata by default.
6. Stop compute if and only if this invocation started it.

The bundle explicitly excludes `.git/`, `private/`, `ci/`, tests, local caches,
and unrelated worktree content. A dirty package or lockfile is rejected so a
remote result cannot claim a commit that does not contain the executed code.
Lightning profiles must use a remote-addressable `OWNER/MODEL` source; local
checkpoint directories are never added to the upload bundle.

Install the current pinned Lightning SDK and authenticate:

```nu
uv sync --extra lightning
lightning login
```

The current pinned Lightning SDK requires Python 3.11 or newer; the core
ABLITERALUS package continues to support Python 3.10.

Planning is local, does not import the SDK, and does not start paid compute:

```nu
uv run --frozen --extra lightning abliteralus-lightning plan \
  --config experiments/surgery/lightning-qwen25-7b.yaml \
  --teamspace OWNER/TEAMSPACE \
  --studio abliteralus-surgery \
  --machine L40S \
  --interruptible
```

Run only after reviewing that plan:

```nu
uv run --frozen --extra lightning abliteralus-lightning run \
  --config experiments/surgery/lightning-qwen25-7b.yaml \
  --teamspace OWNER/TEAMSPACE \
  --studio abliteralus-surgery \
  --machine L40S \
  --interruptible
```

Starting GPU compute incurs Lightning charges. The default run remains attached
until surgery finishes and then releases compute it started, including after a
remote failure. If the Studio was already running, the launcher requires
`--reuse-running` and will not stop or switch that existing machine. Use
`--keep-running` only when intentionally retaining compute after the run.

No credential is forwarded implicitly. For a gated model, opt in by naming the
already-set local variable:

```nu
... --forward-env HF_TOKEN
```

Forwarded values are restored to their prior Studio state after the command,
including when a run fails or reuses an already-running Studio.

The default `--collect summary` downloads manifests and logs while leaving the
large checkpoint on persistent Studio storage. `--collect all` downloads the
whole output directory; `--collect none` leaves every artifact remote.

The initial 7B profile deliberately leaves GGUF disabled. It proves the larger
HF surgery lane without also provisioning a compiler and a pinned llama.cpp
build. A remote GGUF lane should pin and preflight that toolchain before being
used for cross-run comparisons.

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
