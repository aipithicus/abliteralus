# ABLITERALUS guard study

`guard-study` runs a non-destructive, signed dose-response experiment against a
generative guard model. It uses ABLITERALUS contrastive directions, last-token
activation patching, and temporary forward hooks. It never writes model weights.

The tracked pilot keeps three boundaries separate:

- `fit` extracts one unsafe-minus-safe direction per layer and maps causal effects;
- `dev` selects the smallest signed dose satisfying the declared margin, parse-rate,
  and class-gap constraints;
- `test` is not tokenized or evaluated unless a dev candidate satisfies those
  constraints.

Llama Guard emits a formatting token before its `safe` or `unsafe` token. The
runner therefore uses deterministic baseline generation to locate that actual label
position, freezes the emitted prefix, and performs causal mapping and the full dose
sweep at that comparable decision position. Activation-patch effects are reported
directly in unsafe-minus-safe logit units. After selection, the winner and matched
orthogonal sham are also greedily generated from the original dev and test prompts;
this verifies that fixed-prefix effects survive the model's real generation path.

The dataset carries a SHA-256 digest over its canonical content with the digest
field omitted. Any prompt or split change therefore requires an explicit new
digest. This prevents accidental fit/dev/test drift while preserving readable YAML.

## Install

From the repository root, synchronize the private project with the repository's
portable uv:

```powershell
pwsh -NoProfile -File deps/uv/run-uv.ps1 sync --project private --group dev
```

The private lock owns ABLITERALUS, Torch, Transformers, PyYAML, and Safetensors
without installing private modules into the public-shadow environment. No
launcher installs dependencies; synchronization is an explicit provisioning
action after project metadata changes.

## Validate and run

```text
private/.venv/Scripts/guard-study.exe validate \
  private/experiments/studies/datasets/llama-guard-3-1b-directional-pilot-v1.yaml \
  --study private/experiments/studies/local-llama-guard-3-1b-directional-pilot.yaml

private/.venv/Scripts/guard-study.exe run \
  private/experiments/studies/local-llama-guard-3-1b-directional-pilot.yaml \
  --offline
```

Omit `--offline` only when the pinned checkpoint must be resolved from the Hub, and
launch that command through `lab-bench local-inference` so `HF_TOKEN` exists only
inside the credential-bearing subprocess.

Each run writes a manifest, normalized dataset snapshot, `causal-map.json`, compact
direction tensors, and steering results beneath ignored `outputs/studies/`. The
manifest and result both state `weights_mutated: false`. Directions are experimental
measurements, not deployable safety artifacts.

The metric is directional rather than normative: a negative change makes the guard
more likely to emit `safe`, while a positive change makes it more likely to emit
`unsafe`. Moving toward `safe` can suppress false positives, but it can also suppress
true unsafe detections. Treat accuracy, class-conditional behavior, sham controls,
and full-generation verification as separate evidence; a signed margin shift alone
is not a safety improvement.

The checked-in dataset is deliberately a pilot for verifying mechanics. Expand and
rehash the matched pairs before treating confidence intervals or category effects as
population-level evidence.
