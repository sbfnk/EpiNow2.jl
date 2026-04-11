# Forecasting multiple data streams

This tutorial demonstrates how to use EpiNow2.jl to forecast multiple
epidemiological outcomes — for example, cases and hospitalisations — by
combining `estimate_infections()` with `estimate_secondary()`.

## Setup

```@example secondary
using EpiNow2
using CSV, DataFrames, Dates, Distributions, Statistics, Random
using CairoMakie
Random.seed!(6789)
nothing # hide
```

## Data

We use the example case data and simulate a hospitalisation time series,
assuming 10% of cases are hospitalised with a log-normal delay (mean ~7
days).

```@example secondary
pkgdir = dirname(dirname(pathof(EpiNow2)))
cases_df = CSV.read(
    joinpath(pkgdir, "test", "reference", "example_confirmed_full.csv"),
    DataFrame
)
cases_df.date = Date.(cases_df.date)
cases_df = first(cases_df, 60)

# Simulate hospitalisations: 10% of cases with a lognormal delay
primary = DataFrame(date=cases_df.date, primary=cases_df.confirm)
hosp = simulate_secondary(
    primary;
    delays = delay_opts(LogNormal(1.6, 0.5)),
    frac = 0.1
)

df = DataFrame(
    date = cases_df.date,
    cases = cases_df.confirm,
    hospitalisations = hosp.secondary
)
first(df, 6)
```

```@example secondary
fig = Figure(size=(600, 300))
ax = Axis(fig[1, 1]; xlabel="Date", ylabel="Count")
date_nums = Dates.value.(df.date .- df.date[1])
lines!(ax, date_nums, Float64.(df.cases); label="Cases")
lines!(ax, date_nums, Float64.(df.hospitalisations); label="Hospitalisations")
tick_step = max(1, nrow(df) ÷ 6)
ax.xticks = (date_nums[1:tick_step:end], string.(df.date[1:tick_step:end]))
ax.xticklabelrotation = π/6
axislegend(ax; position=:lt)
fig
```

## Estimating infections

We first estimate infections and forecast cases using
`estimate_infections()`. We use the confirmed cases and specify the
generation time, incubation period, and reporting delay.

```@example secondary
generation_time = UncertainDistribution(
    (shape, rate) -> Gamma(max(1e-6, shape), max(1e-6, 1 / rate)),
    [truncated(Normal(1.4, 0.48); lower=0.01),
     truncated(Normal(0.38, 0.25); lower=0.01)],
    14.0
)

incubation_period = UncertainDistribution(
    (μ, σ) -> LogNormal(μ, σ),
    [Normal(1.62, 0.064), truncated(Normal(0.418, 0.069); lower=0.01)],
    14.0
)

reporting_delay = LogNormal(0.5, 0.5)
delay = incubation_period + reporting_delay

est = estimate_infections(
    rename(df[!, [:date, :cases]], :cases => :confirm);
    generation_time = gt_opts(generation_time),
    delays = delay_opts(discretise(delay)),
    forecast = forecast_opts(horizon=14),
    inference = inference_opts(samples=500, warmup=250, chains=1),
    verbose=false
)
est
```

## Estimating the secondary relationship

We use `estimate_secondary()` to learn the delay and scaling between cases
and hospitalisations.

```@example secondary
sec_data = DataFrame(
    date = df.date,
    primary = df.cases,
    secondary = df.hospitalisations
)

sec = estimate_secondary(
    sec_data;
    delays = delay_opts(LogNormal(1.6, 0.5)),
    obs = obs_opts(week_effect=false),
    inference = inference_opts(samples=500, warmup=250, chains=1),
    burn_in=14,
    verbose=false
)
sec
```

## Visualising results

We combine the case forecast from `estimate_infections()` with the
fitted secondary relationship to show both data streams.

```@example secondary
fig = Figure(size=(600, 600))

# Cases panel
ax1 = Axis(fig[1, 1]; title="Confirmed cases", ylabel="Count",
           xticklabelrotation=π/6)
date_nums_r = Dates.value.(est.reports.date .- est.reports.date[1])
band!(ax1, date_nums_r, est.reports.lower_90, est.reports.upper_90;
      color=(:steelblue, 0.15))
band!(ax1, date_nums_r, est.reports.lower_50, est.reports.upper_50;
      color=(:steelblue, 0.3))
lines!(ax1, date_nums_r, est.reports.median; color=:steelblue, linewidth=2)
obs_nums = Dates.value.(df.date .- est.reports.date[1])
scatter!(ax1, obs_nums, Float64.(df.cases); color=:black, markersize=4)
fd = Dates.value(df.date[end] - est.reports.date[1])
vlines!(ax1, [fd]; color=:grey40, linestyle=:dot)
tick_idx = 1:max(1, length(date_nums_r) ÷ 6):length(date_nums_r)
ax1.xticks = (date_nums_r[tick_idx], string.(est.reports.date[tick_idx]))

# Hospitalisations panel
ax2 = Axis(fig[2, 1]; title="Hospitalisations", xlabel="Date", ylabel="Count",
           xticklabelrotation=π/6)
date_nums_s = Dates.value.(sec.predictions.date .- sec.predictions.date[1])
band!(ax2, date_nums_s, sec.predictions.lower_90, sec.predictions.upper_90;
      color=(:orange, 0.15))
band!(ax2, date_nums_s, sec.predictions.lower_50, sec.predictions.upper_50;
      color=(:orange, 0.3))
lines!(ax2, date_nums_s, sec.predictions.median; color=:orange, linewidth=2)
obs_nums_s = Dates.value.(df.date .- sec.predictions.date[1])
scatter!(ax2, obs_nums_s, Float64.(df.hospitalisations); color=:black, markersize=4)
tick_idx2 = 1:max(1, length(date_nums_s) ÷ 6):length(date_nums_s)
ax2.xticks = (date_nums_s[tick_idx2], string.(sec.predictions.date[tick_idx2]))

fig
```

### Fitted parameters

The fraction of cases that become hospitalisations and the delay
parameters:

```@example secondary
params = get_parameters(sec)
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
