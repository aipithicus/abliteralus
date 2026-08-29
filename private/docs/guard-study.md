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

## Layer selection

`causal_mapping.strategy` chooses how the causal map ranks layers, and
`causal_mapping.strategy_params` carries whatever that strategy declares. Unknown
strategy names and undeclared parameters are rejected at load time, so a typo
cannot silently become a no-op inside a run.

- `mean_absolute_effect` (the default when `strategy` is absent) ranks by raw patch
  magnitude. Where the effect saturates with depth, this returns the plateau: on
  Llama-Guard-3-1B it selects layers 15, 13, 14.
- `differential_effect` ranks by the n-th finite difference, with `order` defaulting
  to 1 and layers before the first treated as zero. First order asks where causal
  power *appears* rather than where it has accumulated; on the same map it selects
  layers 9, 8, 10 — the seam rather than the plateau after it.

Orders above 1 difference the measurement noise as many times as the signal, so
they need a logit resolution well below the step being resolved. Under bfloat16 the
pre-seam layers of the tracked pilot sit within a few units in the last place of
zero, which order 2 amplifies.

The `low_causal_layer` control stays pinned to `mean_absolute_effect` regardless of
the configured strategy. It exists to supply an inert layer, and a differential
score near zero means a flat region of the plateau, not an inactive layer.

## Precision

A guard margin is a difference of two logits of similar magnitude, so its error
floor tracks the size of the *operands*, not of the effect being measured. At a
margin near 13 logits the representable spacing is 0.0625 in bfloat16, 0.0078 in
float16, and about 1e-6 in float32. A causal effect of 0.05 is therefore not small
in bfloat16, it is unrepresentable.

The optional `precision` block makes this a declared parameter rather than an
inherited accident:

```yaml
precision:
  compute: inherit     # inherit | float32 | bfloat16 | float16
  readout: float32     # float32 | float64
  allow_tf32: false
```

- `compute` defaults to `inherit`, taking the dtype from the surgery experiment so
  that the two specs cannot silently disagree. Declaring it here overrides that
  dtype for the study alone and leaves surgery runs untouched.
- `readout` governs how the safe/unsafe margin is measured. Rather than subtracting
  two already-rounded head logits, the runner projects the final hidden state onto
  the raw `W_U[unsafe] - W_U[safe]` axis at this precision. That is one dot product
  instead of a vocabulary-wide head, and it stays valid when the head itself is
  quantized. The projection is checked against the model's own logits on one probe
  case before the run trusts it, so a checkpoint whose reported final hidden state
  is not the post-norm residual fails at start rather than reporting wrong margins.
- `allow_tf32` is pinned for the duration of the run and recorded in the manifest.
  This matters on Tensor Core hardware: a float32 matmul may be computed in TF32,
  whose significand is 11 bits — the same width as float16. A run that declares
  float32 and silently receives TF32 would otherwise report a resolution it never had.

Rows now carry both `unsafe_minus_safe_logit_margin` (the projected measurement)
and `head_logit_margin` (what the model's own head emitted), so the two can be
compared directly.

`causal-map.json` carries a `resolution` block naming the readout dtype, the
compute dtype, whichever of the two binds, the peak margin observed, and the
resulting `effect_floor`. A float32 readout removes the cancellation term but not
the rounding already carried by the residual, so the floor follows the coarser
dtype. Layers whose mean effect falls below it are listed in `unresolved_layers`
and marked `resolved: false`: they are not measured as inert, they are unmeasured.

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
