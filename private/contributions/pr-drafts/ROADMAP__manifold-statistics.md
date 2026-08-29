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
| R1 | `feat/robust-activation-location` | upstream main | none | Scale-calibrated **median-of-means** over prompt blocks replacing the arithmetic mean at the prompt-reduction site; the Weiszfeld/IRLS geometric median is the `K = n` limit. Opt-in |
| R2 | `feat/prompt-anomaly-scatter` | R1 | R1 | Leave-one-prompt-out jackknife influence plus Weiszfeld tangent scatter and regularized Mahalanobis distance; flags anomalous prompts. Jackknife influence alone already answers "which prompts drive this direction" without needing a `d x d` scatter |
| R3 | `feat/cross-layer-subspace-pipeline` | `feat/grassmann-cross-layer` | cross-layer PR + whitened-SVD fix | Feed `refusal_subspaces` (rank-k) into cross-layer analysis instead of `quick_directions` (rank-1); contract test against silent rank-1 fallback |
| R4 | `feat/grassmann-consensus` | `feat/grassmann-statistics` | R3 + statistics PR | Replace `argmax norm` cluster representative with a Karcher-mean consensus subspace |
| R5 | `feat/grassmann-median` | R4 | R4 | Geometric median on `Gr(k,d)`; robust consensus where a cluster holds an anomalous layer |
| R6 | `feat/manifold-protocol` | R5 | R5 | Extract a manifold protocol (`log`/`exp`/`dist`) so estimators stop being Grassmann-specific |
| R7 | `feat/product-manifold` | R6 | R6 | `R^p × Gr(k,d)` with scaled metric `H_α`; the MoMPCA structure. **`α` must not be fit by naive joint minimization** — that is provably degenerate (see caveats) |
| R8 | `feat/subspace-uncertainty` | R4 | R4 | **Node bootstrap as the primary route** — its validity is proven in the MoMPCA paper, so this needs citing rather than justifying. Intrinsic effective-sample-size diagnostic alongside. Sandwich covariance is a secondary refinement needing R7 |

**R1 and R2 are independent of the entire Grassmann stack.** They can be submitted
alongside the metrics PR rather than queued behind it, which is what makes this
roadmap safe to start: value lands before the geometry tower is finished.

R4 is also what finally gives `feat/grassmann-statistics` a production consumer —
until it exists, that PR stays deferred as its draft says.

## Caveats that decide whether the later PRs are real

- **The binding constraint is effective sample size, not the i.i.d. label.**
  Dependence alone is not disqualifying, and the sandwich is a poor target for that
  objection: `A⁻¹SA⁻ᵀ` exists precisely *because* models are misspecified. The
  standard repair is mechanical — replace the meat matrix `S = Var(U_i)` with a HAC
  (Newey–West) or cluster-robust form summing score autocovariances across layer
  lags. The layer sequence is smooth and locally correlated, which is the friendliest
  case for that repair, and the clusters this codebase already computes are natural
  dependence blocks.
  Small `n` is the sharper worry — HAC autocovariance estimates are themselves noisy —
  but the literature already answers it, and more cleanly than HAC does. You's own
  work supplies both halves: the MoMPCA paper proves **fixed-node** (small-`n`)
  non-Gaussian limits and **finite-sample high-probability** median-of-means bounds, so
  validity does not hinge on a large-`n` asymptotic at all; and the intrinsic-ESS paper
  gives a coordinate-free effective-sample-size diagnostic whose lag-window estimator is
  consistent under **absolute regularity** — a mixing condition, not independence.
  So the assumption actually being stretched is "Markov chain" to "dependent sequence,"
  which is a far milder stretch than i.i.d., and it is one the estimator family is
  already built to absorb. Compute the ESS, report it next to any interval, and prefer
  resampling over HAC wherever both are available.

- **Separate relative use from absolute use.** Ranking layers or clusters by stability
  survives a miscalibrated variance; quoting a 95% region does not. Under positively
  correlated scores a naive `S` underestimates variance, so intervals come out too
  narrow — anti-conservative, not meaningless. That failure mode is checkable:
  bootstrapping over prompts gives a large-`n` reference interval to calibrate the
  layer-axis one against. R8 should ship the relative ordering first and quote
  coverage only once such a check exists.

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

## Sampling-based routes

Resampling is not an add-on here; it is what the source papers actually do.

- **Median-of-means is itself a partitioning scheme.** Split the sample into `K`
  blocks, take a mean per block, take the geometric median of the block means. It
  buys sub-Gaussian concentration under finite variance and tolerates a fraction of
  corrupted blocks, and it is what the MoMPCA paper builds on. This is why R1 is
  specified as MoM rather than a plain geometric median: same code path, strictly
  better guarantees, and blocks are where dependence gets absorbed.

- **Node bootstrap has proven validity** in the MoMPCA paper. R8 therefore cites
  rather than argues, which is the difference between a shippable PR and a research
  claim an upstream reviewer has to referee.

- **Jackknife is the cheapest useful thing on the list.** Leave-one-prompt-out at the
  reduction site directly answers "which prompts move this direction," at `n` refits
  of an estimator that is already cheap. It needs no scatter matrix, no manifold, and
  no asymptotics — which is why R2 leads with it. A block (delete-`d`) jackknife over
  contiguous layer ranges is the analogous move on the layer axis and respects the
  sequence structure that makes plain resampling awkward there.

- **Permutation tests answer a different question** — whether the cluster structure is
  real at all — and are worth keeping distinct from uncertainty about a given subspace.

## References

Corpus: `D:/aghado01/graveyard/codex-scientiae/bibliotecha/corpora/KisungYou`

| Paper | File | Bears on |
|---|---|---|
| Scale-Calibrated Median-of-Means for Robust Distributed PCA | `2605.20681v1.md` | R1, R7, R8 — MoM estimator, node-bootstrap validity, fixed-node limits, bad-node influence |
| Geometric medians on product manifolds | `2505.18844v3.md` | R5, R7 — the base product-median construction |
| Scale selection for geometric medians on product manifolds | `2605.08001v1.md` | R7 — `α` identifiability; naive joint minimization is degenerate |
| Intrinsic effective sample size for manifold-valued MCMC via kernel discrepancy | `2605.03266v1.md` | R8 — coordinate-free ESS; lag-window estimator under absolute regularity; geodesic Gaussian kernels are not generally PD on curved spaces |
| PCA, SVD, and Centering of Data | `2307.15213v2.md` | R1 — centering choice interacts with the location estimator |
| Data transforming augmentation for heteroscedastic models | `1911.02748v2.md` | Background — the early sampling work (MCMC/DA, Gibbs and EM acceleration); relevant to estimator convergence, not to inference here |

### Background briefs

Deeper discussion lives in the sibling issues vault at
`D:/aipithicus/aipithicus-issues/thermomapper/briefs/`:

| Brief | Bears on |
|---|---|
| `weiszfeld-scatter-vs-sandwich-covariance.md` | R1, R2, R8 — what the scatter estimates vs what the sandwich estimates; the breakdown-point correction; the metric hierarchy |
| `scale-selection-degeneracy.md` | R7 — why `α` cannot be fit by joint minimization, and the three sanctioned alternatives |
| `intrinsic-ess-for-spc.md` | R8 — the ESS construction, and the positive-definiteness trap for geodesic kernels |
| `dta-adaptive-homogenization-spc.md` | ThermoMapper-side; background on data-side vs metric-side homogenization |
