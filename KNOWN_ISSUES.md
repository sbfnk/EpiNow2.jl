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

**Hypotheses (ranked by plausibility):**

1. **GP prior under-identification in this regime**. Default
   `alpha = HalfNormal(0.01)` is very tight and the data don't
   pull alpha far from prior mean for the median sample, but
   the posterior still has a heavy right tail where alpha can
   reach ~1.0 and produce divergent Rt trajectories. Sampler
   visits these regions, contributing to the mean explosion.
2. **Likelihood scaling differs from Stan**. Julia uses
   `Turing.@addlogprob! _negbin2_logpmf(...)` per observation;
   Stan uses `target += neg_binomial_2_lpmf`. Worth checking if
   either side double-applies an `obs_weight` factor or has a
   sign issue.
3. **`update_Rt` indexing bug**. The .jl `gp_cumsum` construction
   matches Stan's `update_Rt` structurally (verified 2026-04-29),
   so this is unlikely but not ruled out. Worth a unit test that
   feeds known `(R0, alpha, rho, eta)` into both implementations
   and compares Rt point-by-point.

**Reproducer:** `test/validate_against_r.jl` — already wired to use
the existing CSV reference data in `test/reference/`.

**Suggested next steps:**

1. Stan-vs-Julia direct logp check at fixed parameters: build a Turing
   `LogDensityProblem` from the model and call `logdensity(model, θ)`,
   then compare to Stan's logp at the same θ via `expose_stan_fns()` or
   manual reproduction of the Stan model in cmdstan.
2. Inspect divergences / treedepth in the .jl chain. If many divergences
   are firing, the heavy alpha tail is a sampler problem; if few, it's
   a posterior shape problem.
3. Try a stronger alpha prior (e.g. `HalfNormal(0.05)`) and re-validate;
   if R's behaviour reproduces, the default prior is too tight in the
   short-time-series regime.
