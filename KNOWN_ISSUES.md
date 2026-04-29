# Known issues

Tracks substantive issues discovered post-merge that are not parity gaps
but real model/numerical concerns. File a corresponding GitHub issue
when starting work.

## Posterior of `Rt` and `infections` diverges from R reference

**Discovered:** 2026-04-29 via `test/validate_against_r.jl`.

**Symptoms (validate_against_r.jl, 1 chain × 1000 samples × 500 warmup):**

| Test                         | Rt corr | Rt med err | Inf corr | Inf med err |
|------------------------------|--------:|-----------:|---------:|------------:|
| test1 (Dirac GT, no delay)   |  -0.81  |     0.12   |   0.90   |    1.28     |
| test2 (LogNormal GT)         |  -0.87  |     0.41   |   0.93   |    0.96     |
| test3 (+ reporting delay)    |  +0.96  |     0.45   |   0.93   |    1.11     |
| test4 (full + week effect)   |  +0.95  |     0.44   |   0.93   |    1.10     |

Negative Rt correlation in tests 1 & 2 indicates the posterior median
Rt is essentially **flat** in Julia (~1.33 across all 30 days) while
R's median Rt declines from 1.34 to 1.10 — i.e. Julia's GP isn't
pulling Rt downwards as the data imply it should.

**Diagnostic findings (test1, 4 chains × 1000 warmup × 1000 samples,
seed=42):**

- `gp_alpha`: median **0.0100** (≈ prior mean 0.008), but **p99 = 0.86**
  and `mean = 0.066` — heavy right tail.
- `R0`: median 1.34, mean 1.41, sd 0.46, p99 3.85 — also heavy-tailed.
- Per-sample Rt trajectories are extremely scattered: at t=30 the
  `max` across 4000 samples is ~99, and the posterior `mean` of
  infections at t=30 is `~10^57` (driven by the heavy right tail).
- The `median` posterior trajectory remains stable, masking the
  pathology in summary tables but not in the full posterior.

**Investigation (2026-04-29):**

Three hypotheses were tested.

| Hypothesis                                    | Verdict |
|-----------------------------------------------|---------|
| (1) Likelihood scaling bug vs Stan            | NOT confirmed |
| (2) Sampler hitting divergences               | CONFIRMED |
| (3) `alpha = HalfNormal(0.01)` prior too tight | FALSE |

Details:

1. **Likelihood scaling.** Per-sample manual reproduction of
   `sum(_negbin2_logpmf(cases[t], expected_reports[t], 1/rod^2))`
   matches `chain[:loglikelihood]` within ~10 logL units for
   typical samples. Larger gaps appear only at extreme tail
   samples (e.g. R0=1.68, alpha=0.91) and are consistent with
   floating-point precision in regimes where the negbin pmf is
   essentially zero. So the likelihood is being computed correctly.

2. **Sampler divergences.** `chain[:numerical_error]` is `1.0` for
   ~4% of post-warmup samples at the default prior, rising to ~25%
   when the alpha prior is widened. `step_size` adapts down to
   ~0.03 (small), and `tree_depth` hits the cap of 12 in ~2% of
   samples. NUTS is genuinely struggling with the posterior
   geometry — the heavy right tail in `gp_alpha` is partly
   artefactual from divergent transitions wandering into stiff
   regions.

3. **Wider alpha prior doesn't recover R's behaviour.** Tested
   `Normal(0, σ)` with σ ∈ {0.01, 0.05, 0.10, 0.30}. In all four
   cases the posterior median Rt remains essentially flat
   (Rt[1] ≈ Rt[30]), and `gp_alpha` p99 remains very large
   (1.5–5.0 across all priors). Widening just adds more divergences
   without improving the trajectory.

**Verdict:** the bug is in the **posterior geometry / sampler**
interaction, not in the likelihood, the GP construction, or the
prior width. The most likely remaining culprits are:

- AdvancedHMC's mass-matrix adaptation. Stan's adaptor uses a
  windowed dual-averaging scheme; Turing's default
  `StanHMCAdaptor` should be similar, but in practice the chains
  are getting stuck. Worth trying:
  - longer warmup (R uses 1000; we tested 500–1000)
  - dense mass matrix instead of diagonal
  - JITTER initialisation across chains
- Subtle parameterisation difference in `log_R = log(R0) + cumsum(GP)`.
  R's Stan stores `eta` as the standard-normal noise and reconstructs
  GP every iteration; .jl does the same via `gp_z ~ filldist(Normal,…)`.
  Verified structurally identical, but a unit test feeding known
  `(R0, alpha, rho, eta)` to both implementations and comparing
  trajectories point-by-point would close this off definitively.

**Reproducer:** `test/validate_against_r.jl`, with reference CSVs in
`test/reference/`.
