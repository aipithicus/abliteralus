This is a scratch project for R&D for making contributions to the Obliteratus open source project. This is a shadow of a fork of the source for the purpose of user privacy and security. Automation and tracking are in place under `aipithicus-issues` for development work, issue tracking, and pull request management.

## Surgery experiment lanes

The repository includes one experiment contract that runs both on a small local
GPU and in a Lightning AI Studio. Surgery is performed on pinned Hugging Face
Safetensors checkpoints; optional GGUF conversion and llama.cpp A/B inference
happen afterward.

- Local miniature profile: `experiments/surgery/local-qwen25-0.5b.yaml`
- Lightning scale profile: `experiments/surgery/lightning-qwen25-7b.yaml`
- Operator guide: [`docs/surgery-experiment-bench.md`](docs/surgery-experiment-bench.md)

## Repository toolchain

The repository declares uv once in [`deps/uv/pin.json`](deps/uv/pin.json), including
platform archive and installed-file SHA-256 digests. Runtime code resolves only the
verified executable under `deps/uv`; it never searches `PATH` or installs a tool as a
side effect of launching an experiment. Normal shell commands go through
`deps/uv/run-uv.ps1`, which also keeps uv's Python, cache, and temporary state local,
redirects user/system uv configuration roots into `.scratch`, and removes ambient
uv/Python environment overrides before launching the tool. Repository `pyproject.toml`
configuration remains authoritative.

On Windows, restore or verify the declared payload with:

```powershell
pwsh -NoProfile -File deps/uv/restore-uv.ps1
```

Mutable caches and temporary state belong under the ignored `.scratch/` tree, not
under `deps/` or the operating system's global temporary directory.

