# Workflow for Rt estimation and forecasting

This tutorial demonstrates the typical workflow for estimating reproduction
numbers and performing short-term forecasts using EpiNow2.jl.

## Setup

```@example workflow
using EpiNow2
using CSV, DataFrames, Dates, Distributions, Statistics
nothing # hide
```

## Data

EpiNow2 expects a `DataFrame` with columns `:date` and `:confirm`
(representing reported counts).

```@example workflow
pkgdir = dirname(dirname(pathof(EpiNow2)))
reported_cases = CSV.read(
    joinpath(pkgdir, "test", "reference", "example_confirmed_full.csv"),
    DataFrame
)
reported_cases.date = Date.(reported_cases.date)
reported_cases = first(reported_cases, 30)
first(reported_cases, 6)
```

## Parameters

### Delay distributions

Distributions can be **fixed** (known parameters) or **uncertain** (priors
on parameters, estimated jointly with Rt).

A fixed Gamma distribution:

```@example workflow
plot(Gamma(9.0, 1/3); max=10)
```

An uncertain Gamma (priors on shape and rate):

```@example workflow
uncertain_gamma = UncertainDistribution(
    (shape, rate) -> Gamma(shape, 1 / rate),
    [Normal(3.0, 2.0), truncated(Normal(1.0, 0.1); lower=0.01)],
    10.0
)
plot(uncertain_gamma)
```

### Generation time

```@example workflow
generation_time = UncertainDistribution(
    (shape, rate) -> Gamma(shape, 1 / rate),
    [truncated(Normal(1.4, 0.48); lower=0.01),
     truncated(Normal(0.38, 0.25); lower=0.01)],
    14.0
)
generation_time
```

```@example workflow
plot(generation_time)
```

### Reporting delays

Multiple delays are composed with `+`.

```@example workflow
incubation_period = UncertainDistribution(
    (μ, σ) -> LogNormal(μ, σ),
    [Normal(1.62, 0.064), truncated(Normal(0.418, 0.069); lower=0.01)],
    14.0
)
reporting_delay = LogNormal(0.5, 0.5)

delay = incubation_period + reporting_delay
delay
```

```@example workflow
plot(delay)
```

### Truncation

```@example workflow
trunc_opts(LogNormal(0.5, 0.5))
```

### Observation scaling

```@example workflow
obs_opts(scale=truncated(Normal(0.4, 0.01); lower=0.0, upper=1.0))
```

### Rt prior

```@example workflow
rt_opts()
```

## Estimation

```@example workflow
result = estimate_infections(
    reported_cases;
    generation_time = gt_opts(generation_time),
    delays = delay_opts(discretise(delay)),
    forecast = forecast_opts(horizon=7),
    inference = inference_opts(samples=1000, warmup=250, chains=2),
    verbose=false
)
result
```

## Interpretation

```@example workflow
plot(result)
```

### Reproduction number (last 7 days)

```@example workflow
cols = [:date, :median, :lower_50, :upper_50, :lower_90, :upper_90]
last(result.rt[!, cols], 7)
```

### Estimated infections (last 7 days)

```@example workflow
last(result.infections[!, cols], 7)
```

### Forecasts

```@example workflow
forecast_start = result.observations.date[end]
filter(r -> r.date > forecast_start, result.reports)[!, cols]
```

### Fitted parameters

```@example workflow
params = get_parameters(result)
param_summary = DataFrame(
    parameter = Symbol[], median = Float64[],
    lower_90 = Float64[], upper_90 = Float64[]
)
for (k, v) in sort(collect(params), by=first)
    any(ismissing, v) && continue
    push!(param_summary, (
        parameter=k, median=round(median(v), digits=4),
        lower_90=round(quantile(v, 0.05), digits=4),
        upper_90=round(quantile(v, 0.95), digits=4)
    ))
end
param_summary
```
