# Feature parity with EpiNow2 (R)

This document tracks which features of the R package
[EpiNow2](https://github.com/epiforecasts/EpiNow2) have been ported to
EpiNow2.jl, and the EpiNow2 R commit that defines the parity reference
at the time each row was last assessed. Bump the per-row reference when
you re-check a feature; bump the global reference at the top when you
do a sweep.

**Global parity reference:** EpiNow2 R `main` @ `8454b076` (post-1.8.0
dev), assessed 2026-04-28.

**Status legend:**

- ✅ **matched** — present and behaviourally equivalent to the R
  implementation at the referenced commit
- ⚠️ **partial** — present but with a subset of features, simplified
  semantics, or known gaps (see notes)
- ❌ **missing** — not implemented in EpiNow2.jl
- N/A — Stan-specific, R-platform-specific, or otherwise not applicable
  to a Julia/Turing port

When the R repo moves ahead, add a new row in **Recent upstream changes
to triage** at the bottom; close the loop by either implementing the
feature (and bumping the appropriate row) or marking it N/A with a
reason.

---

## Top-level estimation

| R function                | .jl status     | Ref commit | Notes |
|---------------------------|----------------|------------|-------|
| `epinow()`                | ✅ matched     | 8454b076   | |
| `estimate_infections()`   | ⚠️ partial     | 8454b076   | Default RW + GP + week effect + NegBin/Poisson works. Backcalc, breakpoints, population depletion, accumulation are wired but minimally tested vs R outputs. |
| `estimate_secondary()`    | ⚠️ partial     | 8454b076   | Core path works; the v1.8 S3 refactor (`epinowfit` class with `$fit`, `$args`, `$observations`) hasn't been mirrored. |
| `estimate_truncation()`   | ⚠️ partial     | 8454b076   | Implemented in `truncation.jl`; v1.8 S3 refactor (`obs` → `observations`, `data` → `args`) not yet mirrored. |
| `estimate_dist()`         | ⚠️ partial     | 8454b076   | `.jl` has `estimate_dist()` (utilities.jl) using maximum-likelihood fits. R's new `estimate_dist()` (post-1.8) uses Stan/MCMC with proper double censoring via `primarycensored`. Not yet mirrored. |
| `regional_epinow()`       | ✅ matched     | 8454b076   | |
| `forecast_infections()`   | ✅ matched     | 8454b076   | |
| `forecast_secondary()`    | ✅ matched     | 8454b076   | |
| `simulate_infections()`   | ✅ matched     | 8454b076   | |
| `simulate_secondary()`    | ✅ matched     | 8454b076   | |
| `estimate_delay()`        | N/A            | 8454b076   | Deprecated upstream in favour of `estimate_dist()`; track the latter instead. |

## Configuration options

| R function                    | .jl status     | Ref commit | Notes |
|-------------------------------|----------------|------------|-------|
| `generation_time_opts()` / `gt_opts()` | ✅ matched | 8454b076 | |
| `delay_opts()`                | ✅ matched     | 8454b076   | |
| `trunc_opts()`                | ✅ matched     | 8454b076   | |
| `rt_opts()`                   | ⚠️ partial     | 8454b076   | v1.8 changed `pop` to require `Fixed(pop)` — `.jl` accepts plain numeric. Adjusted vs unadjusted Rt outputs from v1.8 not mirrored. |
| `gp_opts()`                   | ⚠️ partial     | 8454b076   | Pre-v1.7 parameter names (`ls_mean`/`alpha_mean`) deprecation matched (`.jl` uses single-prior interface). v1.7 internal lengthscale rescaling matched. |
| `obs_opts()`                  | ✅ matched     | 8454b076   | |
| `backcalc_opts()`             | ✅ matched     | 8454b076   | |
| `forecast_opts()`             | ✅ matched     | 8454b076   | |
| `secondary_opts()`            | ✅ matched     | 8454b076   | |
| `inference_opts()`            | N/A (Julia)    | —          | Julia analogue of `stan_*_opts`; controls Turing/AdvancedHMC. |
| `stan_opts()` / `stan_*_opts()` | N/A          | —          | Stan-specific; `.jl` uses `inference_opts()`. |

## Model components

| Feature                                    | .jl status | Ref commit | Notes |
|--------------------------------------------|------------|------------|-------|
| Renewal-equation infections                | ✅ matched | 8454b076   | |
| Random-walk Rt                             | ✅ matched | 8454b076   | |
| Gaussian process Rt (HSGP, Matern/SE/periodic) | ✅ matched | 8454b076 | Vendored Matern spectral fix from R 1.6.1. |
| Breakpoints on Rt                          | ✅ matched | 8454b076   | |
| Population adjustment / depletion          | ⚠️ partial | 8454b076   | Implemented; v1.8 split into adjusted vs unadjusted Rt outputs not yet mirrored. |
| Day-of-week effect                         | ✅ matched | 8454b076   | |
| Truncation adjustment                      | ✅ matched | 8454b076   | |
| Uncertain delay/gt distributions           | ✅ matched | 8454b076   | Sampling-based, with optional prior weighting. |
| Back-calculation (no Rt)                   | ✅ matched | 8454b076   | |
| Accumulation of irregularly-reported data  | ⚠️ partial | 8454b076   | `accumulate::Vector{Bool}` column accepted in `EpiData`. R's `fill_missing()` with `initial_accumulate` not yet mirrored as a user-facing helper. |
| NegBin / Poisson likelihood                | ✅ matched | 8454b076   | |
| Power-likelihood weighting                 | ✅ matched | 8454b076   | |
| Initial growth from R0 (v1.7 change)       | ✅ matched | 8454b076   | |

## Distribution interface

| R feature                              | .jl status     | Ref commit | Notes |
|----------------------------------------|----------------|------------|-------|
| `Fixed()`, `Gamma()`, `LogNormal()`, `Normal()` constructors | N/A (Julia distributions) | — | `.jl` uses `Distributions.jl` types directly. |
| `NonParametric()`                      | ✅ matched (`NonParametricDist`) | 8454b076 | |
| `new_dist_spec()` / `dist_spec` algebra | ⚠️ partial    | 8454b076   | `UncertainDistribution` and `CompositeDelay` cover sampling + convolution. R's `+`, `==`, `c()`, `collapse()`, `fix_parameters()`, `bound_dist()`, `is_constrained()`, `get_distribution()`, `get_pmf()` algebra not exposed. |
| `discretise()`                         | ✅ matched     | 8454b076   | Vendored `primarycensored` — same numerical behaviour as R 1.8.0. |
| `convolve_and_scale()`                 | ✅ matched (`convolve_pmfs`) | 8454b076 | |

## Outputs / accessors

| R function                | .jl status      | Ref commit | Notes |
|---------------------------|-----------------|------------|-------|
| `get_samples()`           | ✅ matched      | 8454b076   | |
| `get_predictions()`       | ⚠️ partial      | 8454b076   | v1.8 added `format` arg for scoringutils (`"summary"`/`"sample"`/`"quantile"`). `.jl` returns DataFrame; multiple formats not exposed. |
| `get_parameters()`        | ✅ matched      | 8454b076   | |
| `get_imputed_reports()`   | ✅ matched      | 8454b076   | |
| `get_regional_results()`  | ✅ matched      | 8454b076   | |
| `get_distribution()`, `get_pmf()` | ❌ missing | 8454b076   | Tied to `dist_spec` algebra (see above). |
| `extract_samples()`, `extract_CrIs()`, `extract_inits()`, `extract_stan_param()` | N/A | — | Stan-output-shape utilities; `.jl` returns chains/DataFrames directly. |
| `summary()` S3 methods    | ⚠️ partial      | 8454b076   | v1.8 `epinowfit` S3 class with `$fit`/`$args`/`$observations` not mirrored. |
| `calc_CrI()`, `calc_CrIs()`, `calc_summary_measures()`, `calc_summary_stats()` | ❌ missing | 8454b076 | Summary helpers — `.jl` defers to `MCMCChains` / DataFrames. |
| `make_conf()`             | N/A             | —          | R-output-formatting helper. |

## Plotting

| R function           | .jl status                     | Ref commit | Notes |
|----------------------|--------------------------------|------------|-------|
| `plot_estimates()`   | ⚠️ partial (via `plot_summary`) | 8454b076   | CairoMakie extension covers headline plots; full `plot_estimates()` with `CrIs` arg etc. not at parity. |
| `plot_summary()`     | ✅ matched                     | 8454b076   | |
| `report_plots()`     | ✅ matched                     | 8454b076   | |

## Utilities

| R function                | .jl status     | Ref commit | Notes |
|---------------------------|----------------|------------|-------|
| `R_to_growth()`           | ✅ matched     | 8454b076   | |
| `growth_to_R()`           | ✅ matched     | 8454b076   | |
| `map_prob_change()`       | ✅ matched     | 8454b076   | |
| `prob_decrease()`         | ✅ matched     | 8454b076   | |
| `bootstrapped_dist_fit()` | ✅ matched     | 8454b076   | |
| `dist_fit()`              | ⚠️ partial     | 8454b076   | Internal helper; subset surfaced via `estimate_dist()`. |
| `add_breakpoints()`       | ❌ missing     | 8454b076   | R utility for adding a breakpoints column to a data frame. |
| `filter_leading_zeros()`  | ❌ missing     | 8454b076   | R-side data-cleaning helper. Argument was deprecated and removed from estimation functions in v1.9; standalone utility still exists. |
| `fill_missing()`          | ⚠️ partial     | 8454b076   | `.jl` accepts an `accumulate` column on `EpiData`; no user-facing `fill_missing` helper that derives it from intervals. |
| `convert_to_logmean()`, `convert_to_logsd()` | ❌ missing | 8454b076 | Trivial — easy to add when needed. |
| `opts_list()`             | ✅ matched     | 8454b076   | |

## Deprecated / removed upstream

Track these to keep the .jl interface aligned with R's deprecation timeline.

| Item | Removed in R | .jl status | Notes |
|------|--------------|------------|-------|
| `dist_skel()`, `apply_tolerance()`, `fix_dist()` | v1.8.0 | N/A | Never had Julia equivalents. |
| `gp_opts(matern_type)` | v1.8.0 | N/A | `.jl` uses single-prior interface. |
| `estimate_truncation(obs, model, weigh_delay_priors)` | v1.8.0 | N/A | `.jl` interface differs. |
| `gp_opts(ls_mean, ls_sd, ls_min, ls_max)` | v1.9.0 (errored since v1.8) | N/A | Already using single-prior interface. |
| `gp_opts(alpha_mean, alpha_sd)` | v1.9.0 | N/A | |
| `obs_opts(phi)` | v1.9.0 | N/A | `.jl` uses `dispersion` already. |
| `obs_opts(na)` | v1.9.0 | N/A | Never exposed in `.jl`; equivalent functionality via the `accumulate` column on `EpiData`. |
| `estimate_infections(filter_leading_zeros, zero_threshold, horizon)` | v1.9.0 | N/A | These args were never present in `.jl`; horizon lives on `forecast_opts()`. |
| `regional_epinow(horizon)` | v1.9.0 | N/A | `.jl` uses `forecast_opts()`. |
| `format_fit(burn_in, start_date)` | v1.9.0 | N/A | |

## Recent upstream changes to triage

When EpiNow2 R `main` advances past `8454b076`, add new entries here.
Convert each into a row in the relevant table above (or this section's
deprecation table) once assessed.

| R commit / PR | Summary | Triage status |
|---------------|---------|---------------|
| _none yet_    |         |               |

---

## How to update this document

When you assess a feature or close a gap:

1. Find the row in the relevant table.
2. Update the **`.jl` status** column (✅ / ⚠️ / ❌ / N/A).
3. Update the **Ref commit** column to the EpiNow2 R commit you
   diff'd against.
4. Add a one-line note if behaviour differs in any user-visible way.
5. If the assessment was a sweep across many features, also bump the
   **Global parity reference** at the top.

When EpiNow2 R lands a new feature you haven't yet looked at:

1. Add it to **Recent upstream changes to triage** with the commit/PR
   reference.
2. Triage at your next sweep.
