# ── Option types ─────────────────────────────────────────────────────────
#
# Each mirrors an EpiNow2 *_opts() function. These are plain structs with
# keyword constructors and sensible defaults. At model-assembly time they
# get translated into EpiAware model components.

# ── Generation time ──────────────────────────────────────────────────────

"""
    GTOpts(; dist, cdf_cutoff, weight_prior)

Generation time distribution options.

# Examples
```julia
gt_opts(LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14))
gt_opts()  # default: Fixed(1)
```
"""
Base.@kwdef struct GTOpts
    dist::DistSpec = FixedSpec(1.0)
    cdf_cutoff::Float64 = 0.001
    weight_prior::Bool = true
end

gt_opts(dist::DistSpec=FixedSpec(1.0); kwargs...) =
    GTOpts(; dist, kwargs...)

# ── Reporting delays ─────────────────────────────────────────────────────

Base.@kwdef struct DelayOpts
    dist::DistSpec = FixedSpec(0.0)
    cdf_cutoff::Float64 = 0.001
    weight_prior::Bool = true
end

delay_opts(dist::DistSpec=FixedSpec(0.0); kwargs...) =
    DelayOpts(; dist, kwargs...)

# ── Truncation ───────────────────────────────────────────────────────────

Base.@kwdef struct TruncOpts
    dist::DistSpec = FixedSpec(0.0)
    cdf_cutoff::Float64 = 0.001
    weight_prior::Bool = false
end

trunc_opts(dist::DistSpec=FixedSpec(0.0); kwargs...) =
    TruncOpts(; dist, kwargs...)

# ── Reproduction number ─────────────────────────────────────────────────

"""
    RtOpts(; prior, use_rt, rw, use_breakpoints, future, gp_on, pop, ...)

Options for time-varying reproduction number estimation.

# Key options
- `prior`: Prior on initial Rt (default: LogNormal(mean=1, sd=1))
- `use_rt`: Whether to use Rt model vs back-calculation (default: true)
- `rw`: Random walk period in days (0=none, 7=weekly) (default: 0)
- `gp_on`: Apply GP to `:Rt_minus_1` or `:R0` (default: `:Rt_minus_1`)
- `pop`: Susceptible population for depletion adjustment (default: 0=none)
"""
Base.@kwdef struct RtOpts
    prior::DistSpec = LogNormalSpec(mean=1.0, sd=1.0)
    use_rt::Bool = true
    rw::Int = 0
    use_breakpoints::Bool = true
    future::Symbol = :latest  # :latest, :project
    gp_on::Symbol = :Rt_minus_1  # :Rt_minus_1, :R0
    pop::Float64 = 0.0
    pop_period::Symbol = :forecast  # :forecast, :all
    pop_floor::Float64 = 1.0
    growth_method::Symbol = :infections  # :infections, :infectiousness
end

rt_opts(; kwargs...) = RtOpts(; kwargs...)

# ── Gaussian process ─────────────────────────────────────────────────────

"""
    GPOpts(; basis_prop, boundary_scale, ls, alpha, kernel, matern_order)

Options for the Gaussian process used to smooth Rt.

Setting `basis_prop=0` disables the GP entirely.
"""
Base.@kwdef struct GPOpts
    basis_prop::Float64 = 0.2
    boundary_scale::Float64 = 1.5
    ls::DistSpec = LogNormalSpec(mean=21.0, sd=7.0, max=60.0)
    alpha::DistSpec = NormalSpec(mean=0.0, sd=0.01)
    kernel::Symbol = :matern  # :matern, :se, :ou, :periodic
    matern_order::Float64 = 1.5
    w0::Float64 = 1.0
end

gp_opts(; kwargs...) = GPOpts(; kwargs...)

# ── Observation model ────────────────────────────────────────────────────

Base.@kwdef struct ObsOpts
    family::Symbol = :negbin  # :negbin, :poisson
    dispersion::DistSpec = NormalSpec(mean=0.0, sd=0.25)
    weight::Float64 = 1.0
    week_effect::Bool = true
    week_length::Int = 7
    scale::DistSpec = FixedSpec(1.0)
    likelihood::Bool = true
    return_likelihood::Bool = false
end

obs_opts(; kwargs...) = ObsOpts(; kwargs...)

# ── Back-calculation ─────────────────────────────────────────────────────

Base.@kwdef struct BackcalcOpts
    prior::Symbol = :reports  # :reports, :none, :infections
    prior_window::Int = 14
    rt_window::Int = 1
end

backcalc_opts(; kwargs...) = BackcalcOpts(; kwargs...)

# ── Forecasting ──────────────────────────────────────────────────────────

Base.@kwdef struct ForecastOpts
    horizon::Int = 7
    accumulate::Union{Int, Nothing} = nothing
end

forecast_opts(; kwargs...) = ForecastOpts(; kwargs...)

# ── Inference (replaces stan_opts) ───────────────────────────────────────

"""
    InferenceOpts(; sampler, samples, warmup, chains, seed, ...)

Options controlling the Turing.jl inference backend.
Replaces EpiNow2's `stan_opts()`.

# Samplers
- `:nuts` — No-U-Turn Sampler (default, exact)
- `:advi` — Automatic Differentiation Variational Inference
- `:pathfinder` — Pathfinder algorithm (requires Pathfinder.jl)

# Examples
```julia
# Default NUTS
inference_opts()

# Fast approximate inference
inference_opts(sampler=:advi, samples=2000)

# Tuned NUTS
inference_opts(
    chains=4, warmup=500, samples=1000,
    target_acceptance=0.95
)
```
"""
Base.@kwdef struct InferenceOpts
    sampler::Symbol = :nuts  # :nuts, :advi, :pathfinder
    samples::Int = 2000
    warmup::Int = 250
    chains::Int = 4
    seed::Union{Int, Nothing} = nothing
    target_acceptance::Float64 = 0.8
    max_treedepth::Int = 10
    progress::Bool = true
end

inference_opts(; kwargs...) = InferenceOpts(; kwargs...)

# ── Secondary model ──────────────────────────────────────────────────────

Base.@kwdef struct SecondaryOpts
    type::Symbol = :incidence  # :incidence, :prevalence
end

secondary_opts(; kwargs...) = SecondaryOpts(; kwargs...)
