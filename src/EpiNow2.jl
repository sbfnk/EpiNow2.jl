"""
    EpiNow2

Estimate real-time case counts and time-varying epidemiological parameters.

A Julia implementation of the EpiNow2 methodology, providing an opinionated,
batteries-included interface for Rt estimation using Turing.jl for inference.

# Quick start
```@example EpiNow2
using EpiNow2
result = epinow(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(
        discretise(example_incubation_period() + example_reporting_delay())
    ),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
summary(result)
```
"""
module EpiNow2

using ADTypes: ADTypes, AutoForwardDiff, AutoMooncake, AutoReverseDiff
using CSV: CSV
using CensoredDistributions: CensoredDistributions, double_interval_censored,
    ExponentiallyTilted
using DataFrames: DataFrames, DataFrame, Not, combine, groupby, innerjoin,
    insertcols!, nrow, select
using Dates: Dates, Date, Day
using Distributions: Distributions, Beta, Dirac, Dirichlet, Distribution,
    Exponential, Gamma, LogNormal, NegativeBinomial, Normal, Poisson,
    Truncated, Uniform, Weibull, cdf, fit, logpdf, params, pdf, scale, shape,
    truncated
using LinearAlgebra: LinearAlgebra
using MCMCChains: MCMCChains
using Mooncake: Mooncake
using PrecompileTools: @compile_workload, @setup_workload
using Random: Random
using ReverseDiff: ReverseDiff
using SpecialFunctions: loggamma, besseli
using Statistics: Statistics, mean, median, quantile, std
using Turing: Turing, @model, InitFromParams, InitFromUniform, MCMCThreads, NUTS,
    arraydist, filldist

# ── AD backends (re-exported from ADTypes for convenience in inference_opts) ─
export AutoForwardDiff, AutoReverseDiff, AutoMooncake

# ── Distribution system ────────────────────────────────────────────────
export NonParametricDist, UncertainDistribution, DelayDistribution,
    CompositeDelay
export discretise, convolve_pmfs

# ── Enums ──────────────────────────────────────────────────────────────
export GPKernel, matern, se, periodic
export ObsFamily, negbin, poisson
export ForecastMode, latest, project, estimate
export GPTarget, gp_Rt, gp_R0
export PopPeriod, pop_forecast, pop_all
export BackcalcPrior, bc_infections, bc_none, bc_growth_rate
export InferenceSampler, nuts
export SecondaryType, incidence, prevalence

# ── Options ─────────────────────────────────────────────────────────────
export gt_opts, delay_opts, trunc_opts, rt_opts, gp_opts, obs_opts,
    backcalc_opts, forecast_opts, inference_opts, secondary_opts

# ── Main inference functions ────────────────────────────────────────────
export epinow, estimate_infections, estimate_secondary,
    estimate_truncation, regional_epinow, example_confirmed,
    simulate_secondary, forecast_secondary,
    example_generation_time, example_incubation_period,
    example_reporting_delay

# ── Accessors ───────────────────────────────────────────────────────────
export get_samples, get_predictions, get_parameters, get_imputed_reports

# ── Utilities ──────────────────────────────────────────────────────────
export R_to_growth, growth_to_R, map_prob_change, prob_decrease,
    simulate_infections, forecast_infections, opts_list,
    estimate_dist, bootstrapped_dist_fit,
    get_regional_results,
    convert_to_logmean, convert_to_logsd,
    add_breakpoints, filter_leading_zeros, fill_missing,
    calc_CrI, calc_CrIs, calc_summary_stats, calc_summary_measures

include("distributions.jl")
include("options.jl")
include("data.jl")
include("model.jl")
include("inference.jl")
include("extract.jl")
include("epinow.jl")
include("secondary.jl")
include("truncation.jl")
include("regional.jl")
include("utilities.jl")
include("estimate_dist.jl")
include("r_bridge.jl")

"""
    report_plots(result; CrI = 0.9, forecast_date = nothing)

Plot the estimates of a fitted model, one figure per quantity: infections,
reported cases (with the observed data), Rt and growth rate. Requires
CairoMakie to be loaded.

# Arguments
- `result`: a fit from [`estimate_infections`](@ref) or [`epinow`](@ref).

# Keyword Arguments
- `CrI`: the credible interval to shade (default 0.9).
- `forecast_date`: date separating estimates from forecasts; `nothing`
  (default) uses the last date of the data.

# Returns
A `Dict` from `:infections`, `:reports`, `:rt` and `:growth_rate` to a
`Figure`.

```@example report_plots
using EpiNow2, CairoMakie
fit = estimate_infections(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
report_plots(fit)[:rt]
```
"""
function report_plots end

"""
    plot_summary(result; CrI = 0.9, forecast_date = nothing)

Plot Rt and reported cases (with the observed data) of a fitted model side
by side in one figure. Requires CairoMakie to be loaded.

# Arguments
- `result`: a fit from [`estimate_infections`](@ref) or [`epinow`](@ref).

# Keyword Arguments
- `CrI`: the credible interval to shade (default 0.9).
- `forecast_date`: date separating estimates from forecasts; `nothing`
  (default) uses the last date of the data.

# Returns
A `Figure`.

```@example plot_summary
using EpiNow2, CairoMakie
fit = estimate_infections(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
plot_summary(fit)
```
"""
function plot_summary end

export report_plots, plot_summary

# ── Precompile workload ───────────────────────────────────────────────────
# Run a tiny end-to-end fit at package precompile time so the model-specific
# JIT (Turing @model expansion, ReverseDiff tape, NUTS adaptation kernels)
# is baked into the precompile cache. Without this, the first user-facing
# `epinow()` / `estimate_infections()` call in a fresh session pays ~30-60 s
# of JIT before any sampling starts.
@setup_workload begin
    workload_data = example_confirmed()[1:30, :]
    workload_gt = gt_opts(example_generation_time())
    workload_delays = delay_opts(example_reporting_delay())
    @compile_workload begin
        estimate_infections(
            workload_data;
            generation_time = workload_gt,
            delays = workload_delays,
            rt = rt_opts(rw = 7),
            gp = nothing,
            inference = inference_opts(
                samples = 5, warmup = 5, chains = 1, progress = false
            ),
            verbose = false,
        )
    end
end

end # module
