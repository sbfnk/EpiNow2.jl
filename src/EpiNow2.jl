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

using CairoMakie
using CensoredDistributions
using Dates
using DataFrames
using Distributions
using LinearAlgebra
using MCMCChains
using SpecialFunctions: loggamma
using Statistics
using Turing

# ── Distribution system ────────────────────────────────────────────────
export NonParametricDist, UncertainDistribution, DelayDistribution,
       CompositeDelay
export discretise, convolve_pmfs

# ── Options ─────────────────────────────────────────────────────────────
export gt_opts, delay_opts, trunc_opts, rt_opts, gp_opts, obs_opts,
       backcalc_opts, forecast_opts, inference_opts, secondary_opts

# ── Main inference functions ────────────────────────────────────────────
export epinow, estimate_infections, estimate_secondary,
       estimate_truncation, regional_epinow, example_confirmed

# ── Accessors ───────────────────────────────────────────────────────────
export get_samples, get_predictions, get_parameters

# ── Re-export Makie functions so plotting works without extra imports ───
using CairoMakie: save as save_figure
export plot, save_figure

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
include("plotting.jl")

end # module
