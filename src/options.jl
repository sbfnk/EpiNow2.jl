# ── Option types ─────────────────────────────────────────────────────────
#
# Each mirrors an EpiNow2 *_opts() function. These are plain structs with
# keyword constructors and sensible defaults.

# ── Enums for type-safe option values ────────────────────────────────────

@enum GPKernel matern se periodic
@enum ObsFamily negbin poisson
@enum ForecastMode latest project estimate
@enum GPTarget gp_Rt gp_R0
@enum PopPeriod pop_forecast pop_all
@enum BackcalcPrior bc_infections bc_none bc_growth_rate
@enum InferenceSampler nuts

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
    meanlog_prior::Distribution = Normal(0.0, 1.0)
    sdlog_prior::Distribution = truncated(Normal(0.5, 0.5); lower=0.01)
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
- `future`: How to handle Rt in forecast: `latest` (hold constant), `project` (extend GP), `estimate` (fix from `fixed_from` days before end) (default: `latest`)
- `fixed_from`: Days before end to fix Rt when `future=estimate` (default: 0)
- `gp_on`: Apply GP to `gp_Rt` (non-stationary) or `gp_R0` (stationary) (default: `gp_Rt`)
- `pop`: Susceptible population for depletion adjustment (default: 0=none)
"""
Base.@kwdef struct RtOpts{P<:Union{Float64, Distribution}}
    prior::Distribution = LogNormal(_moments_to_lognormal(1.0, 1.0)...)
    use_rt::Bool = true
    rw::Int = 0
    future::ForecastMode = latest
    fixed_from::Int = 0
    gp_on::GPTarget = gp_Rt
    pop::P = 0.0
    pop_period::PopPeriod = pop_forecast
    pop_floor::Float64 = 1.0
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
    ls::Distribution = truncated(LogNormal(_moments_to_lognormal(21.0, 7.0)...); upper=60.0)
    alpha::Distribution = Normal(0.0, 0.01)
    kernel::GPKernel = matern
    matern_order::Float64 = 1.5
    w0::Float64 = 1.0
end

gp_opts(; kwargs...) = GPOpts(; kwargs...)

# ── Observation model ────────────────────────────────────────────────────

"""
    ObsOpts(; family, dispersion, week_effect, scale, ...)

Observation model options. Construct via `obs_opts()`.

- `family`: `negbin` or `poisson` (from `ObsFamily` enum)
- `week_effect`: day-of-week reporting effects (default: `true`)
- `scale`: fraction observed, as a `Distribution` for a prior or `Float64` for fixed
"""
Base.@kwdef struct ObsOpts{S<:Union{Float64, Distribution}}
    family::ObsFamily = negbin
    dispersion::Distribution = Normal(0.0, 0.25)
    weight::Float64 = 1.0
    week_effect::Bool = true
    week_length::Int = 7
    scale::S = 1.0
    likelihood::Bool = true
end

obs_opts(; kwargs...) = ObsOpts(; kwargs...)

# ── Back-calculation ─────────────────────────────────────────────────────

"""
    BackcalcOpts(; prior, prior_window, rt_window)

Back-calculation (deconvolution) options. Used when `rt_opts(use_rt=false)`.

- `prior`: Prior mode — `:infections` (default, multiplicative correction),
  `:none` (pure GP), or `:growth_rate` (random walk)
- `prior_window`: Smoothing window for prior (default: 14)
- `rt_window`: Smoothing window for post-hoc Rt (default: 1)
"""
Base.@kwdef struct BackcalcOpts
    prior::BackcalcPrior = bc_infections
    prior_window::Int = 14
    rt_window::Int = 1
end

backcalc_opts(; kwargs...) = BackcalcOpts(; kwargs...)

# ── Forecasting ──────────────────────────────────────────────────────────

"""
    ForecastOpts(; horizon)

Forecasting options. Construct via `forecast_opts()`.

- `horizon`: number of days to forecast (default: `7`)
"""
Base.@kwdef struct ForecastOpts
    horizon::Int = 7
end

forecast_opts(; kwargs...) = ForecastOpts(; kwargs...)

# ── Inference (replaces stan_opts) ───────────────────────────────────────

"""
    InferenceOpts(; sampler, samples, warmup, chains, seed, adtype, ...)

Options controlling the Turing.jl inference backend.
Replaces EpiNow2's `stan_opts()`.

`adtype` selects the automatic-differentiation backend used by NUTS.
The default is `AutoForwardDiff()`. On the bench scenario (60 days,
1 chain, 250+250 iters, RW Rt + week effect + NegBin) ForwardDiff is
roughly 20× faster than `AutoReverseDiff(compile=false)`. The crossover
where reverse-mode would win lies far above the parameter dimensions
typical EpiNow2 models reach, so ForwardDiff is the right default.

`AutoReverseDiff(compile=true)` adds tape compilation for additional
speed but its frozen tape can hit `DomainError` on `log`/`sqrt`/etc.
when sampling visits parameter values whose intermediates drift just
below the recorded domain.

Other supported AD backends: `AutoMooncake()` or any
`ADTypes.AbstractADType`.
"""
Base.@kwdef struct InferenceOpts
    sampler::InferenceSampler = nuts
    samples::Int = 2000
    warmup::Int = 250
    chains::Int = 4
    seed::Union{Int, Nothing} = nothing
    target_acceptance::Float64 = 0.9
    max_treedepth::Int = 12
    progress::Bool = true
    adtype::ADTypes.AbstractADType = ADTypes.AutoForwardDiff()
end

inference_opts(; kwargs...) = InferenceOpts(; kwargs...)

# ── Secondary model ──────────────────────────────────────────────────────

@enum SecondaryType incidence prevalence

"""
    SecondaryOpts(; type)

Secondary observation model options. Use `secondary_opts(incidence)` or
`secondary_opts(prevalence)` for presets, or configure flags directly.

- `cumulative`: carry forward secondary observations
- `historic`: include convolved history
- `primary_hist_additive`: add (true) or subtract (false) history
- `current`: include current primary
- `primary_current_additive`: add (true) or subtract (false) current
"""
Base.@kwdef struct SecondaryOpts
    type::SecondaryType = incidence
    cumulative::Bool = false
    historic::Bool = true
    primary_hist_additive::Bool = true
    current::Bool = false
    primary_current_additive::Bool = false
end

function secondary_opts(type::SecondaryType=incidence; kwargs...)
    if type == incidence
        SecondaryOpts(;
            type=incidence, cumulative=false, historic=true,
            primary_hist_additive=true, current=false,
            primary_current_additive=false, kwargs...
        )
    elseif type == prevalence
        SecondaryOpts(;
            type=prevalence, cumulative=true, historic=true,
            primary_hist_additive=false, current=true,
            primary_current_additive=true, kwargs...
        )
    else
        SecondaryOpts(; type, kwargs...)
    end
end

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
    _has_pop(o) && println(io, "  population: $(o.pop)")
end

_has_pop(o::RtOpts) = o.pop isa Distribution || (o.pop isa Number && o.pop > 0)

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
    println(io, "  adtype: $(o.adtype)")
end
