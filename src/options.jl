# ── Option types ─────────────────────────────────────────────────────────
#
# Each mirrors an EpiNow2 *_opts() function. These are plain structs with
# keyword constructors and sensible defaults.

# ── Generation time ──────────────────────────────────────────────────────

"""
    GTOpts(; dist, weight_prior)

Generation time distribution options.

# Examples
```julia
gt_opts(LogNormal(1.6, 0.5))
gt_opts()  # default: Dirac(1)
```
"""
Base.@kwdef struct GTOpts
    dist::DelayDistribution = Dirac(1.0)
    weight_prior::Bool = true
end

gt_opts(dist::DelayDistribution=Dirac(1.0); kwargs...) =
    GTOpts(; dist, kwargs...)

# ── Reporting delays ─────────────────────────────────────────────────────

"""
    DelayOpts(; dist, weight_prior)

Reporting delay distribution options. Construct via `delay_opts(dist)`.
"""
Base.@kwdef struct DelayOpts
    dist::DelayDistribution = Dirac(0.0)
    weight_prior::Bool = true
end

delay_opts(dist::DelayDistribution=Dirac(0.0); kwargs...) =
    DelayOpts(; dist, kwargs...)

# ── Truncation ───────────────────────────────────────────────────────────

"""
    TruncOpts(; dist, weight_prior)

Right-truncation distribution options. Construct via `trunc_opts(dist)`.
"""
Base.@kwdef struct TruncOpts
    dist::DelayDistribution = Dirac(0.0)
    weight_prior::Bool = false
end

trunc_opts(dist::DelayDistribution=Dirac(0.0); kwargs...) =
    TruncOpts(; dist, kwargs...)

# ── Reproduction number ─────────────────────────────────────────────────

"""
    RtOpts(; prior, use_rt, rw, use_breakpoints, future, gp_on, pop, ...)

Options for time-varying reproduction number estimation.

- `prior`: Prior on initial Rt (default: LogNormal(mean=1, sd=1))
- `use_rt`: Whether to use Rt model vs back-calculation (default: true)
- `rw`: Random walk period in days (0=none, 7=weekly) (default: 0)
- `gp_on`: Apply GP to `:Rt_minus_1` or `:R0` (default: `:Rt_minus_1`)
- `pop`: Susceptible population for depletion adjustment (default: 0=none)
"""
Base.@kwdef struct RtOpts
    prior::Distribution = LogNormal(_moments_to_lognormal(1.0, 1.0)...)
    use_rt::Bool = true
    rw::Int = 0
    use_breakpoints::Bool = true
    future::Symbol = :latest
    gp_on::Symbol = :Rt_minus_1
    pop::Float64 = 0.0
    pop_period::Symbol = :forecast
    pop_floor::Float64 = 1.0
    growth_method::Symbol = :infections
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
    ls::Distribution = LogNormal(_moments_to_lognormal(21.0, 7.0)...)
    alpha::Distribution = Normal(0.0, 0.01)
    kernel::Symbol = :matern
    matern_order::Float64 = 1.5
    w0::Float64 = 1.0
end

gp_opts(; kwargs...) = GPOpts(; kwargs...)

# ── Observation model ────────────────────────────────────────────────────

"""
    ObsOpts(; family, dispersion, week_effect, scale, ...)

Observation model options. Construct via `obs_opts()`.

- `family`: `:negbin` or `:poisson`
- `week_effect`: day-of-week reporting effects (default: `true`)
- `scale`: fraction observed, as a `Distribution` for a prior or `Float64` for fixed
"""
Base.@kwdef struct ObsOpts
    family::Symbol = :negbin
    dispersion::Distribution = Normal(0.0, 0.25)
    weight::Float64 = 1.0
    week_effect::Bool = true
    week_length::Int = 7
    scale::Union{Float64, Distribution} = 1.0
    likelihood::Bool = true
    return_likelihood::Bool = false
end

obs_opts(; kwargs...) = ObsOpts(; kwargs...)

# ── Back-calculation ─────────────────────────────────────────────────────

Base.@kwdef struct BackcalcOpts
    prior::Symbol = :reports
    prior_window::Int = 14
    rt_window::Int = 1
end

backcalc_opts(; kwargs...) = BackcalcOpts(; kwargs...)

# ── Forecasting ──────────────────────────────────────────────────────────

"""
    ForecastOpts(; horizon, accumulate)

Forecasting options. Construct via `forecast_opts()`.

- `horizon`: number of days to forecast (default: `7`)
"""
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
"""
Base.@kwdef struct InferenceOpts
    sampler::Symbol = :nuts
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
    type::Symbol = :incidence
end

secondary_opts(; kwargs...) = SecondaryOpts(; kwargs...)

# ── Show methods ─────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", o::GTOpts)
    println(io, "Generation time options:")
    _show_tree(io, o.dist; indent=1)
end

function Base.show(io::IO, ::MIME"text/plain", o::DelayOpts)
    println(io, "Delay options:")
    _show_tree(io, o.dist; indent=1)
end

function Base.show(io::IO, ::MIME"text/plain", o::TruncOpts)
    println(io, "Truncation options:")
    _show_tree(io, o.dist; indent=1)
end

function Base.show(io::IO, ::MIME"text/plain", o::RtOpts)
    m = round(mean(o.prior), digits=2)
    s = round(std(o.prior), digits=2)
    println(io, "Rt options:")
    println(io, "  prior: $(_dist_family(o.prior))(mean=$m, sd=$s)")
    println(io, "  use_rt: $(o.use_rt)")
    o.rw > 0 && println(io, "  random walk period: $(o.rw)")
    println(io, "  gp_on: $(o.gp_on)")
    println(io, "  future: $(o.future)")
    o.pop > 0 && println(io, "  population: $(o.pop)")
end

function Base.show(io::IO, ::MIME"text/plain", o::ObsOpts)
    println(io, "Observation model options:")
    println(io, "  family: $(o.family)")
    println(io, "  week_effect: $(o.week_effect)")
    o.scale isa Distribution && println(io, "  scale: $(o.scale)")
end

function Base.show(io::IO, ::MIME"text/plain", o::InferenceOpts)
    println(io, "Inference options:")
    println(io, "  sampler: $(o.sampler)")
    println(io, "  samples: $(o.samples), warmup: $(o.warmup), chains: $(o.chains)")
end
