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
    generation_time = gt_opts(LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14)),
    delays = delay_opts(LogNormalSpec(meanlog=0.5, sdlog=0.5, max=14))
)
summary(result)
plot(result)
```
"""
module EpiNow2

using Dates
using DataFrames
using Distributions
using LinearAlgebra
using MCMCChains
using SpecialFunctions: gamma
using Turing

# ── Distribution specification ──────────────────────────────────────────
export DistSpec, LogNormalSpec, GammaSpec, NormalSpec, FixedSpec,
       NonParametricSpec
export discretise

# ── Options ─────────────────────────────────────────────────────────────
export gt_opts, delay_opts, trunc_opts, rt_opts, gp_opts, obs_opts,
       backcalc_opts, forecast_opts, inference_opts, secondary_opts

# ── Main inference functions ────────────────────────────────────────────
export epinow, estimate_infections, estimate_secondary,
       estimate_truncation, regional_epinow

# ── Accessors ───────────────────────────────────────────────────────────
export get_samples, get_predictions, get_parameters

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
