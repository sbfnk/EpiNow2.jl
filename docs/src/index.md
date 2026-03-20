# EpiNow2.jl

*Estimate real-time case counts and time-varying epidemiological parameters.*

EpiNow2.jl is a Julia implementation of the
[EpiNow2](https://epiforecasts.io/EpiNow2/) methodology, providing an
opinionated, batteries-included interface for Rt estimation using
[Turing.jl](https://turinglang.org/) for Bayesian inference.

## Quick start

```julia
using EpiNow2, Distributions

result = epinow(
    data;
    generation_time = gt_opts(LogNormal(1.6, 0.5)),
    delays = delay_opts(LogNormal(0.5, 0.5))
)
plot(result)
```

## Features

- **Renewal equation model** with Hilbert space Gaussian process smoothing
  of Rt (Matérn and squared exponential kernels)
- **Uncertain delay distributions** — priors on generation time, incubation
  period, and reporting delay parameters, estimated jointly with Rt
- **Observation model** — negative binomial or Poisson likelihood with
  day-of-week effects and observation scaling
- **Truncation adjustment** — correct for right-truncated recent data
- **Forecasting** — project Rt and infections forward
- **Multi-region estimation** — `regional_epinow()` with Julia thread
  parallelism
- **Validated** — numerical results match R's EpiNow2 within 2%

## Getting started

See the [Workflow](estimate_infections_workflow.md) tutorial for a complete
walkthrough, or jump to the [API Reference](api.md) for function
documentation.
