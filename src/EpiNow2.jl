"""
    EpiNow2

Estimate real-time case counts and time-varying epidemiological parameters.

A Julia implementation of the EpiNow2 methodology, providing an opinionated,
batteries-included interface for Rt estimation using Turing.jl for inference.

# Quick start
```julia
using EpiNow2

result = epinow(
    data,
    generation_time = gt_opts(LogNormal(1.6, 0.5)),
    delays = delay_opts(LogNormal(0.5, 0.5))
)
summary(result)
plot(result)
```
"""
module EpiNow2

using ADTypes
using CSV
using CensoredDistributions
using Dates
using DataFrames
using Distributions
using LinearAlgebra
using MCMCChains
using Mooncake
using PrecompileTools
using Random
using ReverseDiff
using SpecialFunctions: loggamma, besseli
using Statistics
using Turing

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
       add_breakpoints, filter_leading_zeros

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
include("r_bridge.jl")

# Stubs for plotting functions (implemented by CairoMakie extension)
function report_plots end
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
            inference = inference_opts(
                samples = 5, warmup = 5, chains = 1, progress = false
            ),
            verbose = false,
        )
    end
end

end # module
