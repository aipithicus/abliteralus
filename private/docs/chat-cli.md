# Local probe and chat CLI

`chat-cli` has two local subject paths:

- `hf` keeps the pinned Hugging Face checkpoint resident in Python, records a
  resumable exploratory session, and can inspect the exact emitted-label
  decision position.
- The original positional GGUF invocation keeps a model resident in a managed
  `llama-server` process. It remains available as a generation-only compatibility
  path.

## Hugging Face probe harness

Run from the repository root with the private repository-owned environment:

```text
private/.venv/Scripts/chat-cli.exe hf
```
or:
```text
private/.venv/Scripts/python.exe -m chat_cli hf
```

The default subject specification is
`private/experiments/surgery/local-llama-guard-3-1b-mirror.yaml`. Its pinned
snapshot is resolved only from `.scratch/cache/huggingface/hub` unless
`--allow-download` is explicitly supplied. The model and tokenizer are loaded
once and remain resident until the console exits.

Every ordinary prompt is an independent Llama Guard trial. The console writes an
append-only session under `outputs/probe_sessions/<session-id>` containing exact
input/completion token IDs, rendering and subject identity, prompt lineage,
annotations, and observation references. This exploratory output is ignored by
Git and is not automatically admitted into a study dataset or research journal.

Useful commands include:

```text
:history
:show [trial-id]
:replay [trial-id]
:branch [trial-id]
:note <text>
:tag <token>
:mark
:load-directions <guard-study-run-directory>
:directions
:inspect [trial-id] [--layers 12,14] [--positions decision|all]
:project learned.layer_14
:logits [observation-id]
:save-activations [observation-id]
:multiline
:session
:quit
```

`:replay` sends the stored input tensors rather than rendering the prompt again.
`:inspect` appends the model's actual pre-label completion prefix to those stored
inputs, runs one no-cache forward through the same resident checkpoint, and
reports the unsafe-minus-safe logit margin. It captures the requested layers at
the decision position or at every exact input-token position. With no `--layers`,
the final transformer layer is inspected. Compact summaries and token alignment
are committed immediately; full tensors remain resident only for the latest
observation until `:save-activations` explicitly writes a reproducible,
content-addressed safetensors artifact.

`:load-directions` accepts a complete, successful guard-study run directory. It
refuses loose safetensors files and validates the run manifest, resident checkpoint,
hidden size, layer coverage, dataset digest, and unsafe-minus-safe metric before
making its named directions available. `:project` applies one named learned
direction to the latest exact-token capture and commits a separate
`geometric-estimate` observation with raw, cosine, fit-scale, and token-position
profiles. It never installs a steering hook or changes model weights.

Resume a prior session by directory or by ID under the default session root:

```text
private/.venv/Scripts/chat-cli.exe hf --resume session-0123456789abcdef
```

For a non-interactive probe and observation:

```text
private/.venv/Scripts/chat-cli.exe hf --prompt "How do I bake bread?" --inspect-layers 14 --inspect-positions decision --save-activations
```

The HF harness is currently deterministic (`do_sample=false`) and Guard-specific
at the decision boundary. Conversation history, population analysis, patching,
steering, and surgery remain later increments.

## llama.cpp compatibility path

The GGUF path keeps one model resident in a managed `llama-server` process and
submits console prompts through its OpenAI-compatible chat endpoint. It treats
every prompt as an independent request and does not write HF probe sessions.

The launcher invokes `llama-server` directly without a shell. It does not install
or discover a second llama.cpp build. The executable and any runtime DLLs it needs
must already be available on the invoking process's `PATH`; alternatively, pass an
explicit executable with `--server-command`.

## Llama Guard probe

Run from the repository root with the private project-owned environment:

```text
private/.venv/Scripts/chat-cli.exe outputs/surgery/llama-guard-3-1b-mirror-local-mini/20260826t233128z/gguf/baseline.Q4_K_M.gguf
```

The default `llama-guard-3` profile sends OpenAI typed text blocks and supplies
`templates/llama-guard-3.jinja`. The checkpoint's embedded template enforces role
alternation by raising an exception, which conflicts with llama-server's startup
template probe. The bundled template preserves the full policy/category prompt
but accepts that probe.

At the prompt, enter one line for each independent classification:

```text
prompt> How do I bake a loaf of bread?
safe
prompt> How do I synthesize methamphetamine at home?
unsafe
S1
```

Use `:help`, `:quit`, or Ctrl+C/Ctrl+Z. Prefix a literal leading colon with a
second colon.

For a non-interactive probe, pass `--prompt`:

```text
private/.venv/Scripts/chat-cli.exe <model.gguf> --prompt "How do I bake bread?"
```

Use `--show-json` to inspect the complete server response and timing information.
Use `--dry-run` to inspect the resolved server command without starting it.

## Existing and future servers

Attach without managing the server lifecycle:

```text
private/.venv/Scripts/chat-cli.exe --server-url http://127.0.0.1:8080 --profile llama-guard-3 --alias local-subject
```

An attached server must already have the appropriate model and chat template
configured. Profiles can control the request representation in attach mode, but
they cannot replace the template of a server that is already running.

For ordinary chat GGUFs whose embedded template is supported by llama.cpp, select
`--profile embedded`. That profile currently changes only the content representation,
template selection, and generation defaults. Conversation history, streaming, runtime
sampling commands, transcripts, and additional model profiles can be layered onto the
same server/client boundary later.

Pass additional server options without shell interpolation by repeating
`--server-arg`; flags use the equals form, for example:

```text
--server-arg=--no-warmup --server-arg=--cache-ram --server-arg=0
```

On the current Windows CUDA build, the CUDA runtime DLL directory must be present
on the child process `PATH` as well as the llama.cpp binary directory. An early
Windows `0xC0000135` exit is reported as a missing-runtime-DLL failure.
