# feat(analysis): add Grassmann log/exp maps and the Karcher mean

<!-- DEFERRED — do not submit until a production consumer exists. See "Submission
posture" below. Paste into the upstream PR template when it is time; scrub before posting. -->

## PR package

| Field | Value |
|---|---|
| Local branch | `feat/grassmann-statistics` @ `6cb5be1` |
| Branch base | `feat/grassmann-metrics` @ `069bf13` |
| Upstream target | `feat/grassmann-metrics` initially; retarget to `upstream/main` once that PR merges |
| Prerequisite PRs | `feat/grassmann-metrics` (hard dependency — extends the same module) |
| Dependents | none; a prospective `feat/cross-layer-distance-clustering` would consume Grassmann centroids |
| Already in ABLITERALUS main | yes, via merge `43c52fd` |
| Fork materialization | not yet materialized — deferred |
| Source | authored fresh off the metrics branch; content quarried from archived `feat/grassmann` @ `a0877ae` (the manifold-statistics half of commit `605843b`) |

Exclusions — must not enter the fork branch: `uv.lock`, `.python-version`, and
everything under `private/`.

### Submission posture

Deferred by design. Nothing in the codebase consumes these functions, and an
upstream reviewer is entitled to ask what a new public API is for. The branch is
kept pristine and merged locally so the code is available the moment a consumer
lands — most plausibly Grassmann centroids replacing mean-principal-cosine
clustering in representative-layer selection. Submit it *with* that consumer, or
immediately before it, rather than on its own.

Note that this branch is deliberately cut from `feat/grassmann-metrics` and not
from `feat/grassmann-cross-layer`: it has no dependency on the analyzer work and
should not inherit that PR's review fate.

## Summary

- Problem: averaging refusal subspaces cannot be done by averaging basis matrices, because the result depends on an arbitrary SVD gauge. There is no well-defined "mean subspace" without a tangent space to average in.
- Change: the Riemannian logarithm and exponential for Gr(k, d), and the Fréchet (Karcher) mean obtained by iterating them.
- User-visible effect: none. New functions with no callers.
- Linked issue: Closes #TBD

### Design decisions worth a reviewer's attention

1. **The maps are the point, the mean is the payoff.** `log_map` sends a subspace to a horizontal tangent at another, where ordinary averaging is meaningful; `exp_map` sends the average back to the manifold. `karcher_mean` iterates `Y ← exp_Y(mean_i log_Y(Z_i))` to a fixed point.
2. **Gauge and permutation invariance are tested**, since those are exactly the properties a naive basis average fails: the mean of `[Q @ y, z]` matches the mean of `[z, y]`.
3. **Uniqueness holds below the cut locus.** The log map is unique for principal angles below π/2; at the cut locus one valid minimizing geodesic is returned. The docstring and module notes say so, and the mean is documented as intended for clustered inputs. This is a real limitation, not an oversight.
4. **Verified against closed forms**: `‖log_Y(Z)‖_F` equals the geodesic distance, the tangent is horizontal (`Y Δᵀ = 0`), `exp_Y(log_Y(Z))` returns to `Z`, the mean of two points sits at the midpoint `exp_Y(½ log_Y(Z))`, and a mean of bounded perturbations stays within the perturbation radius of its centre.
5. **Device and dtype contract matches the metrics module**: computation on CPU float64 copies, but subspaces produced by `exp_map` / `karcher_mean` return on the device and dtype of the input they were derived from, since these are values callers keep rather than diagnostics they read.

### Files

- `obliteratus/analysis/grassmann.py`
- `tests/test_grassmann.py`

## Risk and trust assessment

- Risk surfaces touched: TBD
- Untrusted inputs or external dependencies: none
- Remote code, deserialization, credentials, subprocess, network, or filesystem impact: none
- Compatibility or migration impact: none — additive, no callers

## Test evidence

Exact head SHA: `6cb5be1` (local); fork head TBD

| Check | Result | Evidence or notes |
|---|---|---|
| Focused regression/contract tests | pass (local) | `pytest tests/test_grassmann.py` — 37 passed (24 metrics + 13 statistics) |
| Negative and boundary tests | pass (local) | shape mismatch, empty input list, single-element mean, float16/bfloat16 round trips |
| `python -m ruff check --select F app.py obliteratus tests scripts` | TBD | |
| `uv lock --check` | TBD | |
| Selected PR core/risk tests | TBD | |
| Package build, when package inputs changed | not applicable | |
| Import and CLI smoke checks | TBD | |
| Applicable risk-surface checks | not applicable | |
| Conditional hardware/service gates | not applicable | |

Coverage or mutation impact:

- Changed-line: TBD

## Research or performance evidence

Not applicable.
