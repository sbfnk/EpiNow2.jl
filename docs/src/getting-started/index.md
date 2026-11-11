# [Getting started](@id getting-started)

EpiNow2.jl estimates the time-varying reproduction number, infections and short-term forecasts from a time series of reported cases. It ports the [EpiNow2](https://epiforecasts.io/EpiNow2/) R package to Julia and keeps its function names, options and defaults.

## A first example

Case data go in as a `DataFrame` with a `date` column and a `confirm` column of reported counts. The generation time and reporting delay are given as discretised distributions.

```julia
using EpiNow2
using Distributions

cases = first(example_confirmed(), 60)

result = epinow(
    cases;
    generation_time = gt_opts(discretise(LogNormal(1.6, 0.5); max = 14)),
    delays = delay_opts(discretise(LogNormal(0.5, 0.5); max = 14)),
)

summary(result)
```

By default Rt follows a Gaussian process (Matérn 3/2 kernel, Hilbert space approximation), observations are negative binomial with a day-of-week effect, and the last seven days are forecast. Each of these is set through its own option constructor: `rt_opts()`, `gp_opts()`, `obs_opts()` and `forecast_opts()`, while `inference_opts()` takes the place of the R package's `stan_opts()`.

## What the model does

- **Renewal equation**: infections follow from past infections, Rt and the generation time.
- **Rt prior**: a Gaussian process, a random walk, breakpoints, or none (back-calculation).
- **Delays**: one or more delay distributions, fixed or with uncertain parameters, map infections to reports.
- **Observation model**: negative binomial or Poisson, with day-of-week effects, scaling and right-truncation adjustment.
- **Population depletion**: optional susceptible depletion.
- **Further models**: `estimate_secondary()` for outcomes that follow a primary series (deaths after cases), `estimate_truncation()` for reporting delays from data snapshots, and `estimate_dist()` for censored delay data.

## Learning more

- [Estimating Rt and forecasting](../tutorials/estimate-infections-workflow.md) walks through a full analysis.
- [Multiple data streams](../tutorials/secondary.md) covers `estimate_secondary()`.
- The [Public API](@ref public-api) documents every function.
- Report a problem or ask a question on the [GitHub repository](https://github.com/sbfnk/EpiNow2.jl).
