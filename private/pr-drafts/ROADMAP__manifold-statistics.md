# Roadmap — robust manifold statistics

Source: `sol-kisungyou-thermomapper-review.md` (Kisung You, MoMPCA / product-manifold
median) plus the ThermoMapper estimator tier at
`D:/aghado01/ThermoMapper/src/maths/geometry/`.

Bookkeeping follows the model in [feat__grassmann-metrics.md](feat__grassmann-metrics.md):
each entry becomes a pristine branch cut from its stated base, merged into main but
never rebased. Nothing here is built yet.

## What the review actually establishes

Two objects that must not be conflated, and the review is right to insist on it:

| Object | Radial weight | Answers |
|---|---|---|
| Weiszfeld scatter `C_W` | `r · uuᵀ` | Shape of the observed tangent cloud |
| Sandwich covariance `V_α = A⁻¹SA⁻ᵀ` | `A: (I−uuᵀ)/r`, `S: uuᵀ` | Sampling uncertainty of the *estimator* |

`C_W⁻¹` whitens individual residuals; `V_α⁻¹` standardizes error in the final
estimate. They have different radial moments and answer different questions. Keep
the names distinct (`tangent_scatter` vs `median_asymptotic_covariance`).

The metric hierarchy — You's scalar `H_α` → block-anisotropic → fully coupled — is
the axis along which contributions get progressively more ambitious.

## Where ABLITERALUS stands against the ThermoMapper reference

ThermoMapper already implements this whole stack in C#. That is the single biggest
de-risking factor: these are ports of working code, not new derivations.

| ThermoMapper | ABLITERALUS | Status |
|---|---|---|
| `manifolds/GrassmannManifold.cs` | `analysis/grassmann.py` | shipped (metrics) |
| `estimators/intrinsic/ManifoldMean.cs` | `karcher_mean` | built, deferred (`feat/grassmann-statistics`) |
| `estimators/intrinsic/ManifoldMedian.cs` | — | missing |
| `estimators/intrinsic/ScatterAccumulator.cs` | — | missing |
| `estimators/intrinsic/KarcherScatter.cs` | — | missing |
| `estimators/intrinsic/WeiszfeldScatter.cs` | — | missing |
| `manifolds/ScaledManifold.cs` (`H_α`) | — | missing |
| `manifolds/RiemannianProductManifold.cs` | — | missing |

## The reframing that matters

The review's "practical division of labor" points at the consensus refusal subspace
across layers. That is the *wrong first target*, for two reasons found in the code:

1. **The real outlier problem is upstream of everything.**
   [abliterate.py:1387](../../abliteralus/abliterate.py:1387) reduces per-prompt
   activations to a layer mean with `torch.stack(...).mean(dim=0)`. `_harmful_acts`
   is a `dict[int, list[Tensor]]` — one tensor per prompt. Every refusal direction
   in the project is built on an arithmetic mean, which has a breakdown point of
   zero. A handful of anomalous prompts move every direction downstream.
   This site has large `n` (prompt count), lives in plain Euclidean `R^d`, and needs
   **none** of the Grassmann machinery.

2. **Cross-layer clusters are too small for a scatter matrix.**
   Cluster representatives are chosen at
   [informed_pipeline.py:490](../../abliteralus/informed_pipeline.py:490) by
   `max(cluster, key=norm)`. Clusters typically hold 2–5 layers. A `d × d` scatter
   from 3 points is rank ≤ 2 in `d ≈ 4096`; a Mahalanobis metric there is
   regularization, not estimation. Consensus (median/mean) is fine at that scale.
   Scatter is not.

So: robust **location** first, at the prompt site where `n` is large; robust
**shape** only where a real cloud exists; consensus geometry on the layer axis;
inference last.

## PR queue

| # | Branch | Base | Depends on | Scope |
|---|---|---|---|---|
| R1 | `feat/robust-activation-location` | upstream main | none | Weiszfeld/IRLS geometric median in `R^d`, opt-in, replacing the mean at the prompt-reduction site |
| R2 | `feat/prompt-anomaly-scatter` | R1 | R1 | Weiszfeld tangent scatter + regularized Mahalanobis distance per prompt; flags anomalous prompts as a diagnostic |
| R3 | `feat/cross-layer-subspace-pipeline` | `feat/grassmann-cross-layer` | cross-layer PR + whitened-SVD fix | Feed `refusal_subspaces` (rank-k) into cross-layer analysis instead of `quick_directions` (rank-1); contract test against silent rank-1 fallback |
| R4 | `feat/grassmann-consensus` | `feat/grassmann-statistics` | R3 + statistics PR | Replace `argmax norm` cluster representative with a Karcher-mean consensus subspace |
| R5 | `feat/grassmann-median` | R4 | R4 | Geometric median on `Gr(k,d)`; robust consensus where a cluster holds an anomalous layer |
| R6 | `feat/manifold-protocol` | R5 | R5 | Extract a manifold protocol (`log`/`exp`/`dist`) so estimators stop being Grassmann-specific |
| R7 | `feat/product-manifold` | R6 | R6 | `R^p × Gr(k,d)` with scaled metric `H_α`; the MoMPCA structure |
| R8 | `feat/subspace-uncertainty` | R7 | R4 (bootstrap) / R7 (sandwich) | Confidence region for the consensus subspace |

**R1 and R2 are independent of the entire Grassmann stack.** They can be submitted
alongside the metrics PR rather than queued behind it, which is what makes this
roadmap safe to start: value lands before the geometry tower is finished.

R4 is also what finally gives `feat/grassmann-statistics` a production consumer —
until it exists, that PR stays deferred as its draft says.

## Caveats that decide whether the later PRs are real

- **Layers are not i.i.d.** You's asymptotics assume independent nodes. Transformer
  layers are a dependent sequence, so the sandwich covariance has no clean
  interpretation on the layer axis. Robust consensus across layers is descriptive
  and fine; *inference* across layers is not. R8 should therefore bootstrap over
  prompts or data splits — genuine replicates — and treat the sandwich form as a
  product-manifold refinement, not the headline.

- **The scatter is not high-breakdown, and ThermoMapper's own comment says otherwise.**
  `WeiszfeldScatter.cs:5` states the weighting is "consistent with the median's
  breakdown-point guarantees." The review contradicts this correctly: a point at
  radius `R` contributes `R·uuᵀ`, so influence grows linearly rather than
  quadratically, but one arbitrarily remote point still makes the scatter unbounded.
  It is a Weiszfeld-*weighted* robust scatter, not a high-breakdown estimator. The
  Python port must not inherit that claim, and the C# comment is worth correcting.

- **Start block-diagonal.** Full cross-factor precision raises gauge,
  intrinsic-coordinate, singularity, and `D²` scaling problems all at once. R7
  should ship `H_α` and the block-anisotropic form; the coupled metric is a separate
  decision with its own evidence bar.

- **Every PR past R3 needs an empirical claim.** Upstream will reasonably ask
  whether robustness changes surgery outcomes, not just whether the math is right.
  R1 is easy to demonstrate (perturb a few prompts, show the direction moves under
  mean and not under median). R5–R8 need a measured effect on excision quality or
  they are unshippable regardless of correctness.
