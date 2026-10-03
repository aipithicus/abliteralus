# ABLITERALUS

Research harness for studying how behaviors are represented and causally implemented inside open-weight language models. Built on a selectively synced copy of [OBLITERATUS](upstream-url) (AGPL-3.0).

## Research components

- **Guard study:** non-destructive, signed dose-response experiments on Llama Guard. Per-layer contrastive directions, last-token activation patching via temporary forward hooks, causal layer mapping, fit/dev/test separation, and matched orthogonal sham controls. [Details](private/docs/guard-study.md)
- **Probe harness:** live, interactive probing of model internals with recorded observations.
- **Surgery bench:** reproducible weight-surgery runs with validated configs, run manifests, and confound-controlled A/B comparison. [Details](private/docs/surgery-experiment-bench.md)
- **Subspace geometry:** Grassmann- and Stiefel-manifold analysis of directions across layers.

## Layout

- `abliteralus/`: OBLITERATUS-compatible source with bug fixes and added analysis modules.
- `private/`: the independent research lab, with experiments, studies, and tooling.
