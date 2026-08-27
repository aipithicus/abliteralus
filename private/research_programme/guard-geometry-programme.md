# The Geometry of a Verdict: a research programme on directional structure in a generative safety classifier

Status: draft v0.1 — 2026-08-26. Scope: near-term (weeks, not quarters). Prep:
Llama Guard 3 1B via `private/guard_study`. This document is the working
specification for the methods-testbed phase; the study YAMLs under
`experiments/studies/` and the guard_study contracts remain the operational
source of truth for any individual run.

---

## 1. Motivation and framing

Directional-intervention results on instruction-tuned chat models rest on a
specific anatomy: a general-purpose capability core with refusal attached late
in post-training as a behavioral gate. There, harm *perception* and refusal
*behavior* are separable, a single residual-stream direction largely mediates
the gate, and chain-of-thought gives the network room to repair a lesion
downstream. Llama Guard 3 1B has none of these properties. The safety verdict
**is** the trained task; it is emitted as (essentially) the first generated
token with no downstream computation to compensate; and the model is small
enough that resampling-based statistics are cheap.

Pilot interventions on the guard do not reproduce the chat-model picture. This
programme treats that divergence as the object of study rather than a failure
to replicate. The prep has the character of a psychophysics preparation:

- a **continuous, calibrated measurand** — the logit margin between the safe
  and unsafe label tokens — instead of fuzzy behavioral judging;
- a **fixed stimulus template**, so nuisance variation is shared across cases;
- **no downstream repair**, so what is cut is what changes;
- **cheap forward passes**, so bootstrap confidence intervals and permutation
  tests are affordable as a matter of routine.

The scientific question, stated once: **how is the safety verdict organized
geometrically in the residual stream, and which estimator mathematics recovers
that organization as causally potent, statistically stable structure?** Every
hypothesis below is an aspect of this question with a scalar or figure as its
answer.

**Upstream continuity.** This is the guard-side instantiation of the
OBLITERATUS concept-cone agenda ([docs/theory_journal.md](../../docs/theory_journal.md)
§2.3): chat-side refusal is a *polyhedral concept cone* — per-category
directions sharing a common half-space (Wollschläger et al., ICML 2025) — but
chat preps suffer fuzzy category attribution and conflate content *detection*
with refusal *enforcement* (§2.4 there). The guard has machine-verified
S-code labels and no enforcement pathway: its category cones are pure
detection geometry. Cone-structure results established here (half-space
tests, generator estimation, DSI calibration) flow back upstream.

## 2. Conceptual vocabulary

Terms used throughout, fixed here so results are communicable without the
codebase in hand.

- **Verdict axis / verdict subspace** — direction(s) in a layer's residual
  stream whose manipulation changes the emitted safety label.
- **Readout vs computation** — a direction may carry the verdict because the
  answer is being *written* toward the unembedding (readout) or because the
  judgment is being *formed* (computation). The unembedding difference vector
  `W_U[unsafe] − W_U[safe]` defines the readout axis; alignment with it,
  profiled by depth, separates the two.
- **Potency** — dose required to move the verdict; summarized as ED50, the
  dose at which the sign-appropriate flip rate reaches 50% (equivalently the
  margin zero-crossing), read off the dose–response curve.
- **Sham-normalization** — every potency claim is stated relative to the
  matched orthogonal-control arm at the same dose; effects a random orthogonal
  vector also produces are concussion, not mechanism.
- **Stability** — dispersion of an estimator's output across bootstrap
  resamples of the fitting data, measured intrinsically: directions and
  subspaces are points on the Grassmannian, dispersion is geodesic distance to
  a robust center. An estimator can be potent and unstable; that combination
  means it is fitting noise that happens to point somewhere effective.
- **Verdict rank** — the effective dimensionality of the verdict structure:
  one shared axis, or a union of per-category (S-code) detectors plus a shared
  component.
- **Template covariance** — activation variance shared across all cases due to
  the fixed guard prompt scaffolding; the dominant nuisance in raw-metric
  estimation.

## 3. Retrospective: primitives already in main, and the concepts they implement

The mathematical contributions merged over the past weeks were developed
against chat-refusal geometry. In this programme they change role: from
exploratory tools to **measurement instruments**. Inventory, by concept:

**Subspaces as data points** — [abliteralus/analysis/grassmann.py](../../abliteralus/analysis/grassmann.py)
implements the Grassmannian as a metric space: orthonormalization with
rank-revealing tolerance, principal angles, geodesic and projection distances,
log/exp maps, Karcher mean, pairwise distance matrices, and the diameter bound
`max_geodesic_distance(rank, dim)` for normalization. This is the substrate of
every stability metric in §5–§6. Lineage: `feat/grassmann` /
`feat/grassmann-*` branches; P2/P3 review corrections are landed in main.

**Depth as a profile, not a choice** —
[abliteralus/analysis/cross_layer.py](../../abliteralus/analysis/cross_layer.py)
(`CrossLayerAlignmentAnalyzer`) compares rank-k subspaces across layers on the
Grassmannian, giving the verdict subspace a depth profile rather than a single
"best layer".

**Metric correction** — [abliteralus/analysis/whitened_svd.py](../../abliteralus/analysis/whitened_svd.py)
(`WhitenedSVDExtractor`) extracts directions in the covariance-whitened
(Mahalanobis) metric with Tikhonov regularization and eigenvalue truncation,
reporting condition number and effective rank. The fix chain
`bb61464 → cf21a97 → 1570f76 → 2465e55` made its principal-angle diagnostics
numerically trustworthy (orthonormalized rank-revealing bases, CPU-side angle
computation); those diagnostics are load-bearing for H3/H5.

**The null family** — [abliteralus/analysis/steering_vectors.py](../../abliteralus/analysis/steering_vectors.py)
(`SteeringVectorFactory.from_contrastive_pairs`) is estimator #0: the raw
difference-in-means. Every method arm is judged against it.

**Category structure** — [abliteralus/analysis/concept_geometry.py](../../abliteralus/analysis/concept_geometry.py)
(`ConceptConeAnalyzer`) already implements the polyhedral-cone machinery from
the upstream agenda: per-category directions, the Direction Specificity Index
(`DSI = 1 − mean|cos(d_cat, d_other)|`), a linear-vs-polyhedral classifier
(currently the heuristic `mean_cos < 0.8 or cone_dim > 2`), and
most-polyhedral-layer selection. H1/P5 reuses this and upgrades its two soft
spots: the classifier threshold gets calibrated against ground-truth S-code
labels, and the theory journal's flagged gap — *"SVD components are not
aligned with cone generators"* — is answered by the SN-mixture components
(§4.5), which estimate the cone's extreme-ray clusters directly. The theory
journal's bootstrap-cosine stability recommendation (§2.5 there) is likewise
the primitive ancestor of H5's Grassmann stability scorecard.

**Comparators from the erasure family** —
[abliteralus/analysis/leace.py](../../abliteralus/analysis/leace.py) (closed-form
linear concept erasure) and the Wasserstein modules
([wasserstein_optimal.py](../../abliteralus/analysis/wasserstein_optimal.py),
[wasserstein_transfer.py](../../abliteralus/analysis/wasserstein_transfer.py))
are held in reserve as later arms; they are not in the near-term critical path.

**Claims discipline** — [abliteralus/analysis/spectral_certification.py](../../abliteralus/analysis/spectral_certification.py)
grades eigen-structure claims by certification level; spectrum-based statements
in H1 should pass through it rather than being read off a scree plot.

**Control architecture** — `private/guard_study` already implements what a
methods comparison needs: dataset contracts with content digests
(`contracts.py`), dose-grid steering with per-layer natural scales
(`interventions.py`), matched orthogonal sham arms, a low-causal-layer negative
control, a label-axis (readout) comparator arm, causal mapping via last-token
patching, and selection gates on parse rate and projection-gap ratio
(`runner.py`). The single hardwired choice is the estimator itself
(`runner._contrastive_directions`); opening that seam is Phase 1.

**Run hygiene** — lab_bench run workspaces (`abliteralus/run_paths.py`) give
every run an isolated, atomically allocated, self-describing workdir; the
surgery-bench run-directory allocation is race-free. Baselines are reproducible
from a commit plus a dataset digest.

## 4. Method imports from the You corpus

Corpus: `D:\aghado01\graveyard\codex-scientiae\bibliotecha\corpora\KisungYou`
(mdnav-indexed; see run `20260827_060753`). Implementation references: the
author's R package clones under
`D:\aghado01\codex-scientiae\ingestion\gauntlet\kisungyou` and prior art in
`D:\aghado01\ThermoMapper`. Six imports, in dependency order:

### 4.1 mxPBF two-sample gates (arXiv:2112.02580)

Per coordinate `j`, with pooled/per-class variance estimates and
`γ = (n ∨ p)^(−α)`:

```
log B₁₀(j) = ½·log(γ/(1+γ)) + (n/2)·log( n·σ̂²_pooled,j / (n₁σ̂²_X,j + n₂σ̂²_Y,j) )
```

The test statistic is `max_j B₁₀(j)`; consistency holds for
`α > 2(1+ε₀)/(1−3√(C₁ε₀))` and the detectable region is rate-optimal in
max-norm (signal ≥ C·log(n∨p)/n). Closed form, no MCMC, trivially vectorized.

**Role**: evidence gate. Run per layer (is there any class-separable signal
here?) and per S-code (which categories can support their own direction?)
*before* fitting estimators. **Deliberate twist**: mxPBF is powerful when the
difference is sparse in the analysis basis. Running it in both the neuron
basis and the whitened-PCA basis makes basis-dependence itself an instrument —
a coarse probe of how basis-aligned the verdict signal is.

### 4.2 Centering discipline (arXiv:2307.15213)

Propositions 1–5: when the mean is large relative to within-class spread, the
top singular vector of an uncentered matrix collapses onto the mean direction,
the remaining SVD is nearly unchanged, and eigenvalues interlace. For guard
activations the (class or pooled) mean is template-dominated and large, so
this regime is the *expected* one, not an edge case.

**Role**: taxonomy discipline for the estimator family. Every estimator must
declare which moment it removes: **uncentered** (top direction absorbs the
pooled mean — nuisance), **pooled-centered** (between-class difference remains
in the data; top PCs drift toward it), **class-centered** (PCs capture
within-class covariance — the correct whitening basis). Raw mean-diff is the
between-class axis in the raw metric; whitened mean-diff is the same axis in
the within-class Mahalanobis metric — i.e., the family interpolates toward the
Fisher discriminant, which is the honest way to describe it to an audience.

### 4.3 Scale-calibrated median-of-means on ℝᵖ × Gr(r,p) (arXiv:2605.20681)

Blocks `B₁..B_K` (for us: bootstrap resamples or disjoint prompt blocks) each
yield a local estimate `θ_k = (μ̂_k, Û_k)`. Aggregate as the geometric median
under the scaled product metric

```
d_α(θ, θ′)² = α·‖μ − μ′‖² + (2−α)·d_Gr(𝒰, 𝒱)²
```

with the robust radial calibration
`α̂ = 2·τ̂_U / (τ̂_μ + τ̂_U)`, where `τ̂_μ, τ̂_U` are per-tangent-dimension
squared block medians of deviation from a preliminary center. Key theoretical
import: the subspace block of the node-error covariance carries **eigengap
denominators** — subspace uncertainty explodes as the spectrum flattens. That
is the quantitative form of the fat-spectrum instability prediction (H5) and
the reason stability must be measured, not assumed.

**Role**: robust aggregation for the stability scorecard (§P4) and the
MoM-robust estimator arm (§P3). Implementable entirely with `grassmann.py`
primitives plus a Weiszfeld loop (§4.4).

### 4.4 Product-manifold geometric median (arXiv:2505.18844)

Damped Riemannian Weiszfeld: iterate weighted Fréchet-mean steps with weights
`w̃_i ∝ w_i / d_ℳ(current, x_i)`, the coupling entering only through the joint
distance in the denominator; fall back to subgradient steps (step size
`η₀/√k`) near singularities; regularize small distances by ε. Uniqueness holds
in a geodesic ball whose radius depends on curvature; Grassmannians are
compact with known curvature bounds, and our per-layer estimates cluster
tightly, so the local regime applies.

**Role**: (a) the solver for §4.3; (b) a **robust cross-layer joint object** —
the median across `∏_ℓ Gr(k, d)` of per-layer verdict subspaces, replacing the
current joint arm's independent stapling, with breakdown-point robustness to a
bad layer.

### 4.5 Spherical-normal mixtures on S^(d−1) (arXiv:2106.06375)

EM for `Σ_k π_k f_SN(x | μ_k, λ_k)`: E-step is standard soft assignment;
M-step locations are **weighted Fréchet means on the sphere** (great-circle
metric) and concentrations solve a 1-D convex problem; k-means initialization;
hard-assignment heuristic available; homogeneous-λ variant recovers geodesic
k-means as λ→∞.

**Role**: the rank instrument (H1). Per-S-code and per-resample unit
directions are points on the hypersphere; fitting SN mixtures for K = 1, 2, …
and selecting by BIC answers "how many verdict directions?" with a model
comparison rather than a scree-plot squint, and per-component `λ_k` doubles as
a likelihood-based stability number. **Implementation note**: directions are
axes (±v indistinguishable); canonicalize signs before fitting (align to
positive margin effect, or cluster with absolute-cosine affinity first).

### 4.6 Persistence landscapes + energy statistics (arXiv:2208.12435)

Pipeline: point cloud → Vietoris–Rips complex → persistence diagram →
persistence landscape Λ. Landscapes live in `L²(ℕ×ℝ)`, a separable Hilbert
space, which has strong negative type — therefore the **energy distance**
two-sample and k-sample statistics validly characterize equality of
distributions there, with permutation p-values (and the DISCO within/between
decomposition as an ANOVA analogue).

**Role**: collateral-damage instrument (H6). Per arm and dose, subsample
last-position activations, compute landscapes, and run the k-sample energy
test across doses against sham. The question it answers: does an effective
dose *translate* the activation cloud or *tear* it?

### 4.7 Metric hygiene (arXiv:2504.16318)

Residual-stream norms carry information; raw cosine can mislead when norms are
semantic. All alignment diagnostics report whitened-metric cosine alongside
raw cosine, and margin-weighted variants where a scalar summary is needed.

### 4.8 Riemannian Gaussian mixtures on hyperbolic space (arXiv:2604.24895)

All computation in the **hyperboloid (Lorentz) model** `H^d` — closed-form
distance `arcosh(−⟨x,y⟩_L)`, exp/log maps, tangent projection — with the
Poincaré ball reserved for visualization (isometric; projection formulas
provided). Isotropic Riemannian Gaussians `∝ exp(−d(x,μ)²/2σ²)` with an exact
erfc finite-sum normalizing constant; weighted MLE splits into a weighted
Fréchet mean (unique — `H^d` is Hadamard, so the location problem is *strongly
geodesically convex globally*, unlike on the sphere or Grassmannian) plus a
one-dimensional strictly convex scale profile; exact and generalized
(majorization–minimization) EM for mixtures with monotonicity guarantees.

**Role**: the clustering/model-selection instrument for the hierarchy arm
(H7), applied to hyperbolically embedded direction sets. Note the negative
curvature *simplifies* the statistics relative to our spherical and Grassmann
instruments — global uniqueness where the compact manifolds give only local.

### 4.9 Anisotropic hyperbolic wrapped normal inference (arXiv:2605.00363)

`HWN(μ, Σ)`: push a Euclidean `N(0, Σ)` through parallel transport and the
exponential map; the density is the tangent-space Gaussian times the
volume-growth correction `(r/sinh r)^(d−1)`. Shell-constrained profile MLE
(eigenvalues of Σ clipped into `[λ₋, λ₊]` — the same spectral-regularization
discipline as our whitening extractor), with asymptotic normality and
efficient profile (Wald) inference for the location through the
Schur-complement information.

**Role**: proper confidence statements for hyperbolic locations in the
hierarchy arm, and full-covariance (anisotropic) fits where 4.8's isotropic
model is too rigid.

### 4.10 HOPE: neurons as Hilbert–Schmidt operators (arXiv:2607.21366)

Source: `bibliotecha_archive/md-shelf/2607.21366v1-latex.md`. Each neuron is
lifted to a rank-1 Hilbert–Schmidt operator `f_i = g_i ⊗ w_out,i` in
`L²(𝒳, P_𝒳; ℝ^c)`, where `P_𝒳` is a Maximum-Entropy Gaussian surrogate built
from stored normalization statistics (or a one-time calibration pass). The
neuron Gram matrix `⟨f_i, f_j⟩ = K(i,j)·⟨w_out,i, w_out,j⟩` has closed-form
ReLU kernels; capacity `‖f_i‖` is scale- and shape-invariant by construction
(kills the reparametrization symmetry that breaks magnitude pruning). Pruning
and merging are unified as constrained rank-1 projection with an axiomatically
derived distortion cost `𝒥` (`𝒥_prune = N‖f_i‖/(E_a − ‖f_i‖)`; merge
analogous), driven by a greedy receding-horizon rate–distortion loop selecting
`argmin 𝒥_k/ΔP_k`.

**Roles** (pivot before direct):
(a) *Core-vs-slack as the compression dual of verdict-as-task* — the H8
instrument: track the verdict margin along a progressive-compression
trajectory; structure that **is** the task should resist compression longest.
(b) *Merging-as-consolidation* — a confound control for H1: if measured
verdict rank is inflated by feature fragmentation across correlated neurons,
consolidating merges shrink it; genuine category detectors survive.
(c) *Methodological through-line* — the same "replace the convenient inner
product with an honest one" move as whitening (activation space) and the
landscape Hilbert space (topology): here applied to *neuron* space, giving the
programme's geometric instruments a dual metric space (neurons-as-functions)
to act on.
**Direct application** (deferred): a HOPE-style strategy for the surgery
bench requires re-deriving the cross-kernels for RMSNorm + SwiGLU (the gated
form `SiLU(w_g·x)·(w_u·x)` makes the kernel a bivariate Gaussian integral —
tractable numerically, not erfc-closed-form) and the calibration-pass
surrogate. Real lift; scoped out of the near term.

### 4.11 Embedding condensation as the mechanistic backdrop (arXiv:2602.00217)

Source: `codex-scientiae/ingestion/inventory/2602.00217v3/` (PDF; Liu et al.,
ICML 2026). **Embedding condensation**: with depth, pairwise cosine
similarities of a transformer's token embeddings drift toward 1 — the
representation collapses into a narrow cone. Severity is strongly
size-dependent (severe at ≤1B, resisted at 32B; confirmed in a
confounder-controlled family varying only MLP width), present at
*initialization* (matching the Geshkovski et al. infinite-depth clustering
theory for random QKV), counteracted but not eliminated by pre-training, and
**not transferred by knowledge distillation** — behavioral matching does not
transfer geometry. Their training-side fix (a log-sum-exp dispersion loss
over pairwise angles) is out of scope here (we do not train the guard), but
their *diagnostic* imports directly: layer-wise pairwise-cosine distributions
stacked into a depth heatmap, summarized by the Spearman/Kendall rank
correlation between mean cosine and layer index.

**Roles**: (a) mechanism for H3 — Llama Guard 3 1B sits squarely in the
severe-condensation size class, so the raw metric fights *two* stacked
nuisances: the task-independent condensation cone and the template
covariance; whitening is the measurement-time counterpart of their
training-time dispersion loss. The cone axis is also essentially the mean
direction that uncentered SVD's top singular vector absorbs (§4.2) — the
estimator taxonomy, the condensation phenomenon, and the centering theory are
one story. (b) A cheap P2 companion diagnostic: the guard's condensation
profile from the existing observation hooks, cross-referenced against the
evidence heatmap and the `WhitenedSVDExtractor` condition-number/effective-
rank reports. (c) Communicable framing support: distillation transferring
behavior but not geometry is independent evidence that geometry must be
interrogated directly — the programme's premise.

## 5. Hypotheses

Each hypothesis states its measurand, prediction, and falsifier. Both outcomes
are interpretable; none is decorative.

**H1 — Verdict rank and cone structure.** The verdict is not rank-1: per-S-code
whitened difference directions span a subspace with effective rank > 1,
organized as a **polyhedral cone** — category detectors sharing a common
half-space around a shared axis — mirroring the upstream chat-refusal finding
in a prep with ground-truth labels. (The structure ladder: subspace rank
here → half-space cone here → hierarchy depth in H7.)
*Measurand*: singular spectrum of the stacked per-category directions
(certified via `spectral_certification`); SN-mixture BIC curve over K;
per-S-code DSI; the half-space test (existence of an axis with strictly
positive cosine to every category direction — candidate axes: shared top
singular vector, label axis).
*Prediction*: BIC prefers K ≥ 2; leading singular value < ~75% of energy;
half-space test passes (cone, not a mere subspace); mixture components
recover generator clusters that SVD components miss.
*Falsifier*: K = 1 preferred and a single dominant singular value — the guard
verdict is chat-refusal-like after all, and rank-k arms (P6) should then show
no potency advantage; the programme pivots to asking why the *pilot*
interventions diverged. A separate partial outcome: rank > 1 but the
half-space test fails — category detectors are genuinely antagonistic rather
than cone-structured, which would itself contradict the upstream polyhedral
picture and be worth reporting loudly.

**H2 — Dose asymmetry.** Inducing "unsafe" is easier than suppressing it:
ED50(+) < ED50(−) on matched cases.
*Measurand*: signed ED50s from the existing dose grid, bootstrap CIs.
*Prediction*: strict inequality, outside overlapping CIs.
*Falsifier*: symmetry (verdict behaves like a balanced two-sided readout) or
reversal (the "default-on safe prior" picture is wrong).
*Note*: partially answerable from already-collected pilot tables before any
new code.

**H3 — Whitening potency.** Two stacked nuisances dominate the raw metric at
this scale — the size-dependent condensation cone (§4.11) and the template
covariance — so the whitened estimator finds a more causally aligned axis:
ED50(whitened) < ED50(raw mean-diff), sham-normalized, both signs.
*Measurand*: ED50 ratio with bootstrap CI; condensation profile (§4.11)
reported alongside as the mechanistic covariate.
*Falsifier*: ratio ≈ 1 — within-class covariance is effectively isotropic
inside the verdict-relevant subspace, itself a publishable observation about
classifier fine-tuning (and, given the measured condensation profile, a
surprising one).
*Cross-scale corollary (not near-term)*: since condensation attenuates with
model size, whitening's advantage should shrink on larger guard variants — a
directional prediction recorded now for any future multi-scale run.

**H4 — Computation upstream of transcription.** There exist layers where
causal patching moves the verdict but readout-axis alignment is low.
*Measurand*: depth profiles of (a) patched-margin causal effect, (b)
whitened cosine between layer direction and the unembedding difference axis.
*Prediction*: the causal-effect peak precedes the alignment peak by ≥ 2 layers.
*Falsifier*: profiles co-located — the verdict is computed at the readout, and
"steering" is just writing the answer; interventional claims must then be
reframed as readout manipulation.

**H5 — Potency–stability Pareto.** Whitened and MoM-robust estimators
dominate raw mean-diff on the joint (potency, stability) front; stability
degrades with spectrum flatness as the eigengap theory predicts.
*Measurand*: per estimator, ED50 vs Grassmann dispersion (median geodesic
distance of resample fits to their MoM center, normalized by
`max_geodesic_distance`); overlay predicted eigengap-driven uncertainty.
*Falsifier*: raw mean-diff on the front — the sophistication does not pay for
itself at this scale, an important negative result for the methods paper.

**H6 — Translation, not tearing.** At doses up to ED50, steering translates
the activation cloud without changing its topology relative to sham.
*Measurand*: k-sample energy statistic on persistence landscapes across doses
vs sham, permutation p-values.
*Prediction*: non-significant topology change at |dose| ≤ ED50 while margins
flip; significance only at the grid extremes (±2σ).
*Falsifier*: topology breaks at or below ED50 — measured "flips" are cloud
destruction, and every potency number in H2–H5 must be reinterpreted with a
collateral-damage covariate.

**H7 — Verdict hierarchy (conditional on H1: K > 1).** The category detectors
are not flat siblings: the direction set is tree-like — a shared "unsafe"
parent with S-code children — and is therefore better represented in
hyperbolic than in spherical or Euclidean geometry, with the radial coordinate
carrying generality/confidence (the information cosine discards, per §4.7).
*Measurand*: (a) scaled Gromov δ-hyperbolicity of the whitened geodesic
distance matrix over per-category and per-resample directions; (b) embedding
distortion into `H^r` vs `S^r` vs `ℝ^r` at matched dimension r; (c) mixture
fits in `H^r` via §4.8 with §4.9 confidence statements; (d) radial ordering of
embedded case activations by margin magnitude.
*Prediction*: δ small relative to diameter; hyperbolic distortion strictly
lowest; shared-axis direction embeds radially inward of the category
directions.
*Falsifier*: no hyperbolic distortion advantage — the taxonomy is flat, the
spherical mixture description (H1) is complete, and the hierarchy framing is
dropped from the writeup.
*Discipline*: geometries are compared by δ and distortion, never by
likelihood across manifolds (different sample spaces); the norm→radius map is
fixed before fitting (see §8).

**H8 — The verdict is core, not slack.** Because the verdict is the guard's
trained task, its machinery should be among the *last* structure destroyed by
capacity-ordered progressive compression — in sharp contrast to chat-model
safety behavior, which is empirically shallow and dies early under
compression. This is the compression dual of the programme's central framing.
*Measurand*: verdict margin and parse rate tracked along a progressive
ablation trajectory ordered by a capacity-style dispensability score
(approximated near-term with existing surgery-bench strategies; HOPE-proper
per §4.10 later); the margin-vs-parameters-removed curve and its knee,
sham-compared against random-order ablation at matched parameter counts.
*Prediction*: a long flat plateau — margin degradation lags the random-order
control substantially, with a late cliff; neurons whose output weights align
with the verdict subspace (from H1/P5) are over-represented in the surviving
core.
*Falsifier*: margin degrades at or faster than the random-order control — the
verdict machinery is not privileged capacity, and the verdict-as-task framing
in §1 loses its strongest structural support. This is the programme's riskiest
prediction and is deliberately kept falsifiable.
*Extension (optional)*: the same trajectory run on a chat model's refusal
behavior yields the two-panel "task vs veneer" comparison — the most
communicable figure the lab's surgery machinery can produce.

## 6. Experimental trajectory

Phases are ordered so that every method arm consumes the same frozen inputs
and every claim has its control before its effect. Compute is not the
bottleneck at 1B; implementation discipline is.

**P0 — Freeze the baseline.** Commit the working tree (sliced); run the pilot
study as-is on main; archive the complete dose–response tables, causal map,
and manifests (not just the selected winner) keyed by dataset digest + commit.
*Exit gate*: a second run from the same commit reproduces the archived tables.

**P1 — Open the estimator seam.** `DirectionEstimator` protocol in
guard_study (`fit(safe_acts, unsafe_acts, layer) → directions, natural_scale,
diagnostics`), registry keyed from the study YAML
(`steering.estimator: mean-diff | ...`); refactor the current mean-diff path
into estimator #0.
*Exit gate*: regression identity — estimator #0 reproduces P0 outputs exactly.

**P2 — Evidence gates.** Implement mxPBF (α set per the consistency bound;
report sensitivity in {α, 2α}); run per layer and per S-code, in neuron and
whitened bases. Alongside, the condensation profile (§4.11): layer-wise
pairwise-cosine distributions with the Spearman-ρ depth summary — near-free
from the same activation captures.
*Exit gate*: layer × category evidence heatmap + condensation profile; layer
set for P3 chosen from them (with the causal map), not by convention.

**P3 — Estimator arms.** Centering variants (4.2 taxonomy), whitened-SVD
(existing extractor behind the protocol), MoM-robust (block medians over
prompt blocks). Full dose grids with matched shams. Reserve arm: RDO-style
gradient search (the theory journal's §2.3 recommendation) — directly
optimize the direction against the differentiable logit margin; the guard
makes this cleaner than any chat prep (no behavioral judge in the loop).
Expected role: the potency ceiling that the H5 scorecard tests for stability.
*Exit gate*: ED50 table (arm × sign) with bootstrap CIs → **H2, H3 resolved.**

**P4 — Stability scorecard.** B ≈ 200 bootstrap resamples per estimator;
Grassmann dispersion about the α̂-calibrated MoM center (4.3–4.4).
*Exit gate*: potency × stability Pareto figure → **H5 resolved.**

**P5 — Rank and cone instrument.** Per-S-code directions (categories passing
the P2 gate); sign canonicalization; singular spectrum + SN-mixture BIC scan;
DSI per category and the half-space test via `ConceptConeAnalyzer`, with its
polyhedral threshold calibrated against the S-code ground truth.
*Exit gate*: verdict-rank + cone-structure estimate with certification level
→ **H1 resolved**; decision point: K > 1 authorizes P6's rank-k arms.

**P6 — Subspace arms and depth profiles.** Rank-k projection and steering
arms; cross-layer joint arm as product-manifold Weiszfeld median; depth
profiles of causal effect vs readout alignment.
*Exit gate*: **H4 resolved**; rank-k vs rank-1 potency comparison recorded.

**P7 — Topology collateral.** Landscape pipeline on subsampled clouds per
arm × dose; k-sample energy tests vs sham.
*Exit gate*: **H6 resolved**; collateral covariate joined to the ED50 table.

**P8 — Hierarchy arm (conditional: opens only if P5 finds K > 1).** Gromov δ
and matched-dimension distortion comparison on the P5 direction set;
hyperboloid-model embedding with a pre-registered norm→radius map; Riemannian
Gaussian mixture fits (§4.8) with profile-likelihood inference (§4.9);
Poincaré-disk projection for the figure only.
*Exit gate*: **H7 resolved**; δ/distortion table + verdict-map figure.

**P9 — Compression trajectory (independent of P8; needs P5 outputs for the
core-alignment analysis only).** Capacity-ordered progressive ablation via
existing surgery-bench strategies with a random-order control at matched
parameter counts; margin/parse tracking per step; overlap analysis between
the surviving core and the verdict subspace. HOPE-proper (SwiGLU kernels,
merging, consolidation control for H1) is a stretch goal behind the same
interface, not a gate.
*Exit gate*: **H8 resolved**; margin-vs-capacity trajectory figure.

Retrospective analyses of P0 pilot data (notably H2) may be reported as soon
as available; they do not wait for the phase ladder.

*Beyond the near term*: with guard category cones established as pure
detection geometry, comparing them against a chat model's refusal cones (via
`analysis/cross_model_transfer.py`) would decompose chat refusal into a
detection component (shared with the guard) and an enforcement residual —
the geometric form of the theory journal's §2.4 decomposition. Recorded as an
arc, not a phase.

## 7. Deliverable figures

One figure per hypothesis, plus two framing figures; this is the skeleton of
the eventual writeup.

1. Dose–response margin curves, all arms, sham band shaded (framing).
2. mxPBF evidence heatmap, layer × S-code, both bases (P2; framing + basis
   probe).
3. Signed ED50 table/forest plot with CIs (H2, H3).
4. Potency × stability Pareto scatter with eigengap overlay (H5).
5. Spectrum + mixture-BIC panel (H1).
6. Depth profiles: causal effect vs readout alignment (H4).
7. Landscape energy distance vs dose, with permutation bands (H6).
8. Verdict map on the Poincaré disk: category directions and case activations,
   radius = generality/confidence, with δ/distortion table inset (H7;
   conditional).
9. Margin vs parameters-removed trajectory, capacity-ordered vs random-order,
   knee marked (H8); optional second panel: chat-refusal comparator
   ("task vs veneer").

Narrative arc for human consumption: *a safety classifier's verdict is not a
gate bolted onto a capability core but a task-defining structure; here is its
rank, its depth, its metric, and the estimator mathematics that measures it
without fooling itself.*

## 8. Threats to validity

- **Gaussianity**: mxPBF and SN models assume it; activations only
  approximate it. Gates are used as evidence rankings, not calibrated tests;
  permutation and bootstrap procedures back every headline claim.
- **Single model, single template**: conclusions are about this prep.
  Replication path: the mirror surgery config
  (`experiments/surgery/local-llama-guard-3-1b-mirror.yaml` lineage) and,
  later, Guard variants at other scales. The scale caveat now has a mechanism
  and a direction: condensation severity falls with model size (§4.11), so
  estimator rankings measured at 1B — especially whitening's margin over raw
  mean-diff — may legitimately reorder at larger scale, and H3's cross-scale
  corollary predicts *which way*.
- **Axis ambiguity**: antipodal direction pairs must be canonicalized before
  any spherical statistics; the policy (sign of margin effect) is part of the
  spec, not a per-analysis choice.
- **Steering surface**: the hook steers the final position during prefill and
  every position during generation (verdict and category tokens alike); this
  is the intended surface and must be stated in writeups.
- **Readout confound**: late-layer potency may be transcription, not
  computation — H4 exists to bound this, and no potency claim should be
  interpreted without its depth profile.
- **Selection gates**: `min_parse_rate: 1.0` is strict; arms that degrade
  formatting die silently. Report gate attrition alongside winners.
- **Geometry shopping**: manifolds are analysis representations, not causal
  surfaces — the model's readout is linear, so interventions stay in the
  residual stream's own coordinates regardless of which geometry describes
  the direction set best. Cross-geometry claims use δ-hyperbolicity and
  distortion at matched dimension, never likelihoods across different sample
  spaces; and the norm→radius embedding map for H7 is a researcher degree of
  freedom that must be fixed (and recorded) before any hyperbolic fit runs.
- **HOPE provenance and transfer gap**: 2607.21366 is proof-of-concept scale,
  derived for BatchNorm + positively homogeneous activations on FC/conv
  architectures; its closed-form kernels do not transfer to RMSNorm + SwiGLU
  without re-derivation. H8's near-term form therefore uses the framework's
  *concept* (capacity-ordered progressive reduction with a matched random
  control), not its specific estimator; any "HOPE score" claim waits for the
  SwiGLU kernels.

## 9. Assets

| Asset | Location |
| --- | --- |
| Guard study package (contracts, runner, interventions) | `private/guard_study/` |
| Pilot study + dataset specs | `experiments/studies/` |
| Geometry primitives | `abliteralus/analysis/` (grassmann, whitened_svd, cross_layer, concept_geometry, steering_vectors, leace, wasserstein_*, spectral_certification) |
| You corpus (markdown, mdnav-indexed) | `D:\aghado01\graveyard\codex-scientiae\bibliotecha\corpora\KisungYou` |
| Author implementation clones (R) | `D:\aghado01\codex-scientiae\ingestion\gauntlet\kisungyou` |
| Prior art (own) | `D:\aghado01\ThermoMapper` |
| Upstream cone theory + open gaps | [docs/theory_journal.md](../../docs/theory_journal.md) (§2.3–2.5, Part VI) |

Corpus papers on the critical path: 2112.02580 (mxPBF), 2307.15213
(centering), 2605.20681 (scale-calibrated MoM), 2505.18844 (product medians),
2106.06375 (SN mixtures), 2208.12435 (landscapes + energy tests), 2504.16318
(cosine hygiene). Conditional path (H7/P8): 2604.24895 (hyperbolic Riemannian
Gaussian mixtures), 2605.00363 (anisotropic HWN inference). Conditional path
(H8/P9): 2607.21366 (HOPE; `bibliotecha_archive/md-shelf/`). Mechanistic
backdrop (H3/P2): 2602.00217 (embedding condensation;
`codex-scientiae/ingestion/inventory/`, PDF). Reserve:
2209.03318 / 2603.14815 / 2509.11435 (Wasserstein family), 2601.10992 (metric
scaling), 2605.08001 (median scale selection).
