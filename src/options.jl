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

"""
    GPKernel

Kernel of the approximate Gaussian process, passed as `kernel` to
[`gp_opts`](@ref). Values: [`matern`](@ref) (Matérn kernel, the default),
[`se`](@ref) (squared exponential) and [`periodic`](@ref).
"""
GPKernel

"""
    matern

Matérn kernel for the Gaussian process (a [`GPKernel`](@ref) value), the
default. Its order is set by `matern_order` in [`gp_opts`](@ref).

```@example matern
using EpiNow2
gp_opts(kernel = matern, matern_order = 2.5)
```
"""
matern

"""
    se

Squared exponential kernel for the Gaussian process (a [`GPKernel`](@ref)
value).

```@example se
using EpiNow2
gp_opts(kernel = se)
```
"""
se

"""
    periodic

Periodic kernel for the Gaussian process (a [`GPKernel`](@ref) value), with
fundamental frequency `w0` set in [`gp_opts`](@ref).

```@example periodic
using EpiNow2
gp_opts(kernel = periodic, w0 = 1.0)
```
"""
periodic

"""
    ObsFamily

Distribution of the observation model, passed as `family` to
[`obs_opts`](@ref). Values: [`negbin`](@ref) (negative binomial, the
default) and [`poisson`](@ref).
"""
ObsFamily

"""
    negbin

Negative binomial observation model (an [`ObsFamily`](@ref) value), the
default. Its overdispersion has the prior set by `dispersion` in
[`obs_opts`](@ref).

```@example negbin
using EpiNow2
obs_opts(family = negbin)
```
"""
negbin

"""
    poisson

Poisson observation model (an [`ObsFamily`](@ref) value), with no
overdispersion.

```@example poisson
using EpiNow2
obs_opts(family = poisson)
```
"""
poisson

"""
    ForecastMode

How Rt is treated beyond the data, passed as `future` to [`rt_opts`](@ref).
Values: [`latest`](@ref) (hold the last estimate, the default),
[`project`](@ref) (continue the Gaussian process) and [`estimate`](@ref)
(hold the estimate from a number of days before the end).
"""
ForecastMode

"""
    latest

Hold Rt at its last estimated value over the forecast horizon (a
[`ForecastMode`](@ref) value), the default.

```@example latest
using EpiNow2
rt_opts(future = latest)
```
"""
latest

"""
    project

Project Rt over the forecast horizon by continuing the Gaussian process (a
[`ForecastMode`](@ref) value).

```@example project
using EpiNow2
rt_opts(future = project)
```
"""
project

"""
    estimate

Hold Rt fixed from `fixed_from` days before the end of the data onwards (a
[`ForecastMode`](@ref) value), by default the seeding time. Useful when the
most recent estimates are unreliable because of right truncation.

```@example estimate
using EpiNow2
rt_opts(future = estimate, fixed_from = 4)
```
"""
estimate

"""
    GPTarget

What the Gaussian process is applied to, passed as `gp_on` to
[`rt_opts`](@ref). Values: [`gp_Rt`](@ref) (the previous Rt, the default)
and [`gp_R0`](@ref) (a global mean).
"""
GPTarget

"""
    gp_Rt

Apply the Gaussian process to the previous Rt, i.e. `Rt = Rt-1 * GP` (a
[`GPTarget`](@ref) value), the default.

```@example gp_Rt
using EpiNow2
rt_opts(gp_on = gp_Rt)
```
"""
gp_Rt

"""
    gp_R0

Apply the Gaussian process to a global mean, i.e. `Rt = R0 * GP` (a
[`GPTarget`](@ref) value). Real-time estimates revert towards the global
mean, which may not be desirable.

```@example gp_R0
using EpiNow2
rt_opts(gp_on = gp_R0)
```
"""
gp_R0

"""
    PopPeriod

When the susceptible population adjustment applies, passed as `pop_period`
to [`rt_opts`](@ref). Values: [`pop_forecast`](@ref) (forecasts only, the
default) and [`pop_all`](@ref) (data and forecasts).
"""
PopPeriod

"""
    pop_forecast

Apply the susceptible population adjustment to forecasts only (a
[`PopPeriod`](@ref) value), the default.

```@example pop_forecast
using EpiNow2
rt_opts(pop = 1.0e6, pop_period = pop_forecast)
```
"""
pop_forecast

"""
    pop_all

Apply the susceptible population adjustment to both the data period and
forecasts (a [`PopPeriod`](@ref) value).

```@example pop_all
using EpiNow2
rt_opts(pop = 1.0e6, pop_period = pop_all)
```
"""
pop_all

"""
    BackcalcPrior

Prior used when deconvolving without an Rt model, passed as `prior` to
[`backcalc_opts`](@ref). Values: [`bc_infections`](@ref) (the default),
[`bc_none`](@ref) and [`bc_growth_rate`](@ref).
"""
BackcalcPrior

"""
    bc_infections

Back-calculation prior that multiplies reported cases, shifted back by the
mean delay, by the Gaussian process (a [`BackcalcPrior`](@ref) value), the default.

```@example bc_infections
using EpiNow2
backcalc_opts(prior = bc_infections)
```
"""
bc_infections

"""
    bc_none

Back-calculation with no prior from reported cases: infections follow the
Gaussian process alone (a [`BackcalcPrior`](@ref) value). Real-time
performance is poor.

```@example bc_none
using EpiNow2
backcalc_opts(prior = bc_none)
```
"""
bc_none

"""
    bc_growth_rate

Back-calculation in which the Gaussian process acts on the growth rate, so
infections follow a random walk on the log scale seeded from delay-shifted
reported cases (a [`BackcalcPrior`](@ref) value).

```@example bc_growth_rate
using EpiNow2
backcalc_opts(prior = bc_growth_rate)
```
"""
bc_growth_rate

"""
    InferenceSampler

Sampler used for inference, passed as `sampler` to
[`inference_opts`](@ref). The only value is [`nuts`](@ref).
"""
InferenceSampler

"""
    nuts

The No-U-Turn sampler (an [`InferenceSampler`](@ref) value), the default and
only sampler.

```@example nuts
using EpiNow2
inference_opts(sampler = nuts)
```
"""
nuts

# ── Generation time ──────────────────────────────────────────────────────

"""
    GTOpts(; dist, weight_prior)

Generation time options. Construct with [`gt_opts`](@ref).
"""
Base.@kwdef struct GTOpts
    dist::Union{DelayDistribution, CompositeDelay} = Dirac(1.0)
    weight_prior::Bool = true
end

"""
    gt_opts(dist = Dirac(1.0); weight_prior = true)

Generation time distribution options.

# Arguments
- `dist`: the generation time distribution, any
  [`DelayDistribution`](@ref). If none is given a fixed generation time of 1
  day is assumed. Zero generation times are not
  supported, so the probability mass at zero is dropped and the rest
  renormalised.

# Keyword Arguments
- `weight_prior`: if `true` (default), priors on the parameters of `dist` are
  weighted by the number of observations, approximately placing an
  independent prior at each time step. If `false`, they are treated as a
  single parameter.

# Returns
A `GTOpts` object.

```@example gt_opts
using EpiNow2
gt_opts(example_generation_time())
```
"""
gt_opts(dist::Union{DelayDistribution, CompositeDelay} = Dirac(1.0); kwargs...) =
    GTOpts(; dist, kwargs...)

# ── Reporting delays ─────────────────────────────────────────────────────

"""
    DelayOpts(; dist, weight_prior)

Reporting delay options. Construct with [`delay_opts`](@ref).
"""
Base.@kwdef struct DelayOpts
    dist::Union{DelayDistribution, CompositeDelay} = Dirac(0.0)
    weight_prior::Bool = true
end

"""
    delay_opts(dist = Dirac(0.0); weight_prior = true)

Delay distribution options, for the delays between infection and report.

# Arguments
- `dist`: the delay distribution, any [`DelayDistribution`](@ref). To use
  several delays in sequence, e.g. incubation period plus reporting delay,
  add them with `+`; the parts can be fixed or have uncertain parameters.
  The default is a fixed distribution with all mass at 0, i.e. no delay.

# Keyword Arguments
- `weight_prior`: if `true` (default), priors on the parameters of `dist` are
  weighted by the number of observations; if `false` they are treated as a
  single parameter.

# Returns
A `DelayOpts` object.

```@example delay_opts
using EpiNow2
delay_opts(example_incubation_period() + example_reporting_delay())
```
"""
delay_opts(dist::Union{DelayDistribution, CompositeDelay} = Dirac(0.0); kwargs...) =
    DelayOpts(; dist, kwargs...)

# ── Truncation ───────────────────────────────────────────────────────────

"""
    TruncOpts(; dist, weight_prior)

Right-truncation options. Construct with [`trunc_opts`](@ref).
"""
Base.@kwdef struct TruncOpts
    dist::Union{DelayDistribution, CompositeDelay} = Dirac(0.0)
    weight_prior::Bool = false
end

"""
    trunc_opts(dist = Dirac(0.0); weight_prior = false)

Right-truncation distribution options.

# Arguments
- `dist`: a delay distribution reflecting the truncation, for example the
  `dist` returned by [`estimate_truncation`](@ref). The default is a fixed
  distribution at 0, i.e. no truncation.

# Keyword Arguments
- `weight_prior`: if `true`, priors on the truncation parameters are weighted
  by the number of observations; if `false` (default) they are treated as a
  single parameter.

# Returns
A `TruncOpts` object.

```@example trunc_opts
using EpiNow2
trunc_opts(discretise(example_reporting_delay(); max = 10))
```
"""
trunc_opts(dist::Union{DelayDistribution, CompositeDelay} = Dirac(0.0); kwargs...) =
    TruncOpts(; dist, kwargs...)

# ── Reproduction number ─────────────────────────────────────────────────

"""
    RtOpts(; prior, use_rt, rw, future, fixed_from, gp_on, pop, pop_period,
           pop_floor)

Options for the time-varying reproduction number. Construct with
[`rt_opts`](@ref).
"""
Base.@kwdef struct RtOpts{P <: Union{Float64, Distribution}}
    prior::Distribution = LogNormal(_moments_to_lognormal(1.0, 1.0)...)
    use_rt::Bool = true
    rw::Int = 0
    future::ForecastMode = latest
    fixed_from::Union{Int, Nothing} = nothing
    gp_on::GPTarget = gp_Rt
    pop::P = 0.0
    pop_period::PopPeriod = pop_forecast
    pop_floor::Float64 = 1.0
end

"""
    rt_opts(; prior, use_rt = true, rw = 0, future = latest, fixed_from = nothing,
            gp_on = gp_Rt, pop = 0.0, pop_period = pop_forecast,
            pop_floor = 1.0)

Time-varying reproduction number options.

# Keyword Arguments
- `prior`: prior on the initial reproduction number. Defaults to a lognormal
  distribution with mean 1 and standard deviation 1.
- `use_rt`: should Rt be used to generate infections and hence reported
  cases (default `true`)? If `false`, infections are back-calculated (see
  [`backcalc_opts`](@ref)).
- `rw`: step size of a random walk on Rt in days; 0 (default) means no random
  walk, 7 a weekly random walk. The random walk is combined with the Gaussian
  process unless `gp = nothing` is passed to the estimation function, as in
  R.
- `future`: how Rt is treated beyond the data, a [`ForecastMode`](@ref):
  `latest` (default), `project` or `estimate`.
- `fixed_from`: when `future = estimate`, the number of days before the end
  of the data from which Rt is held fixed. `nothing` (default) uses the
  seeding time, as R does: the rounded sum of the delay means, but at least
  the maximum generation time. Ignored for `latest` and `project`.
- `gp_on`: what the Gaussian process is applied to, a [`GPTarget`](@ref):
  `gp_Rt` (default) or `gp_R0`.
- `pop`: initial susceptible population size, as a number or a prior
  distribution. Used to adjust Rt for susceptible depletion; 0 (default)
  means no adjustment.
- `pop_period`: when the population adjustment applies, a
  [`PopPeriod`](@ref): `pop_forecast` (default) or `pop_all`.
- `pop_floor`: minimum susceptible population used when adjusting for
  depletion, to keep it numerically stable (default 1.0).

# Returns
An `RtOpts` object.

```@example rt_opts
using EpiNow2
rt_opts(rw = 7)
```
"""
rt_opts(; kwargs...) = RtOpts(; kwargs...)

# ── Gaussian process ─────────────────────────────────────────────────────

"""
    GPOpts(; basis_prop, boundary_scale, ls, alpha, kernel, matern_order, w0)

Options for the approximate Gaussian process. Construct with
[`gp_opts`](@ref).
"""
Base.@kwdef struct GPOpts
    basis_prop::Union{Nothing, Float64} = nothing
    boundary_scale::Union{Nothing, Float64} = nothing
    ls::Distribution = truncated(LogNormal(_moments_to_lognormal(21.0, 7.0)...); upper = 60.0)
    alpha::Distribution = Normal(0.0, 0.01)
    kernel::GPKernel = matern
    matern_order::Float64 = 1.5
    w0::Float64 = 1.0
end

"""
    gp_opts(; basis_prop = nothing, boundary_scale = nothing, ls, alpha,
            kernel = matern, matern_order = 1.5, w0 = 1.0)

Approximate Gaussian process options, for the Hilbert space approximation
used to smooth Rt (or infections when back-calculating). Pass `gp = nothing`
to the estimation functions to switch the Gaussian process off.

# Keyword Arguments
- `basis_prop`: proportion of time points to use as basis functions. Lower
  values are faster but less accurate. The default, `nothing`, chooses the
  number of basis functions from the 5% quantile of the lengthscale prior
  `ls`, as in R.
- `boundary_scale`: boundary factor of the approximation relative to the
  half-range of the time points. The default, `nothing`, chooses it from the
  95% quantile of `ls` (and uses 1.5 for the periodic kernel), as in R.
- `ls`: prior on the lengthscale in days. Defaults to a lognormal with mean
  21 and standard deviation 7, truncated at 60.
- `alpha`: prior on the magnitude, roughly the expected standard deviation
  of the process. Defaults to a half-normal with standard deviation 0.01.
- `kernel`: a [`GPKernel`](@ref): `matern` (default), `se` or `periodic`.
- `matern_order`: order of the Matérn kernel: 0.5, 1.5 (default) or 2.5.
- `w0`: fundamental frequency of the periodic kernel (default 1.0).

# Returns
A `GPOpts` object.

```@example gp_opts
using EpiNow2
gp_opts(basis_prop = 0.1, kernel = se)
```
"""
gp_opts(; kwargs...) = GPOpts(; kwargs...)

# ── Observation model ────────────────────────────────────────────────────

"""
    ObsOpts(; family, dispersion, weight, week_effect, week_length, scale,
            likelihood)

Observation model options. Construct with [`obs_opts`](@ref).
"""
Base.@kwdef struct ObsOpts{S <: Union{Float64, Distribution}}
    family::ObsFamily = negbin
    dispersion::Distribution = Normal(0.0, 0.25)
    weight::Float64 = 1.0
    week_effect::Bool = true
    week_length::Int = 7
    scale::S = 1.0
    likelihood::Bool = true
end

"""
    obs_opts(; family = negbin, dispersion = Normal(0, 0.25), weight = 1.0,
             week_effect = true, week_length = 7, scale = 1.0,
             likelihood = true)

Observation model options.

# Keyword Arguments
- `family`: the observation model, an [`ObsFamily`](@ref): `negbin`
  (default) or `poisson`.
- `dispersion`: prior on the reporting overdispersion, used only when
  `family = negbin`. It is one over the square root of the negative binomial
  `phi`. Defaults to a half-normal with standard deviation 0.25.
- `weight`: weight given to the observed data in the log density (default
  1).
- `week_effect`: should a day-of-week effect be used (default `true`)?
- `week_length`: assumed length of the week in days (default 7); change it
  for data with a non-weekly periodicity.
- `scale`: scaling from latent infections (convolved to date of report) to
  reports, as a fixed number or a prior distribution (default 1, i.e. no
  scaling).
- `likelihood`: should the likelihood be included in the model (default
  `true`)? Set to `false` to sample from the prior.

# Returns
An `ObsOpts` object.

```@example obs_opts
using EpiNow2
obs_opts(family = poisson, week_effect = false)
```
"""
obs_opts(; kwargs...) = ObsOpts(; kwargs...)

# ── Back-calculation ─────────────────────────────────────────────────────

"""
    BackcalcOpts(; prior, prior_window, rt_window)

Back-calculation options. Construct with [`backcalc_opts`](@ref).
"""
Base.@kwdef struct BackcalcOpts
    prior::BackcalcPrior = bc_infections
    prior_window::Int = 14
    rt_window::Int = 1
end

"""
    backcalc_opts(; prior = bc_infections, prior_window = 14, rt_window = 1)

Back-calculation options, used when infections are deconvolved without an
Rt model (`rt_opts(use_rt = false)`).

# Keyword Arguments
- `prior`: the prior used when deconvolving, a [`BackcalcPrior`](@ref):
  `bc_infections` (default), `bc_none` or `bc_growth_rate`.
- `prior_window`: the centred smoothing window, in days, for the
  delay-shifted reports used as a prior (default 14). Kept for parity with
  R; the Julia model does not yet smooth the prior.
- `rt_window`: the size of the centred rolling average used when
  calculating Rt from the back-calculated infections (default 1). Kept for
  parity with R; the Julia model does not yet apply it.

# Returns
A `BackcalcOpts` object.

```@example backcalc_opts
using EpiNow2
backcalc_opts(prior_window = 7)
```
"""
backcalc_opts(; kwargs...) = BackcalcOpts(; kwargs...)

# ── Forecasting ──────────────────────────────────────────────────────────

"""
    ForecastOpts(; horizon)

Forecast options. Construct with [`forecast_opts`](@ref).
"""
Base.@kwdef struct ForecastOpts
    horizon::Int = 7
end

"""
    forecast_opts(; horizon = 7)

Forecast options.

# Keyword Arguments
- `horizon`: number of days into the future to forecast (default 7).

# Returns
A `ForecastOpts` object.

```@example forecast_opts
using EpiNow2
forecast_opts(horizon = 14)
```
"""
forecast_opts(; kwargs...) = ForecastOpts(; kwargs...)

# ── Inference (replaces stan_opts) ───────────────────────────────────────

"""
    InferenceOpts(; sampler, samples, warmup, chains, seed, target_acceptance,
                  max_treedepth, progress, adtype)

Turing.jl inference options. Construct with [`inference_opts`](@ref).
"""
Base.@kwdef struct InferenceOpts
    sampler::InferenceSampler = nuts
    samples::Int = 2000
    warmup::Int = 500
    chains::Int = 4
    seed::Union{Int, Nothing} = nothing
    target_acceptance::Float64 = 0.9
    max_treedepth::Int = 12
    progress::Bool = true
    adtype::ADTypes.AbstractADType = ADTypes.AutoForwardDiff()
end

"""
    inference_opts(; sampler = nuts, samples = 2000, warmup = 500, chains = 4,
                   seed = nothing, target_acceptance = 0.9,
                   max_treedepth = 12, progress = true,
                   adtype = AutoForwardDiff())

Options for the Turing.jl inference backend, in place of EpiNow2's
`stan_opts()`.

# Keyword Arguments
- `sampler`: the sampler, an [`InferenceSampler`](@ref) (only `nuts`).
- `samples`: total number of posterior samples across all chains (default
  2000), as in R; each chain draws `ceil(samples / chains)`.
- `warmup`: number of warmup (adaptation) iterations per chain (default
  500).
- `chains`: number of MCMC chains (default 4), run in parallel across Julia
  threads.
- `seed`: random seed, or `nothing` (default) for none.
- `target_acceptance`: target acceptance rate of NUTS (default 0.9).
- `max_treedepth`: maximum tree depth of NUTS (default 12).
- `progress`: show a progress bar (default `true`).
- `adtype`: automatic differentiation backend used by NUTS, any
  `ADTypes.AbstractADType`. The default `AutoForwardDiff()` is the fastest
  for the parameter dimensions typical of EpiNow2 models.
  `AutoReverseDiff(compile = true)` can fail with a `DomainError` when
  sampling moves outside the domain recorded on its tape. `AutoMooncake()`
  is also supported.

# Returns
An `InferenceOpts` object.

```@example inference_opts
using EpiNow2
inference_opts(samples = 500, warmup = 250, chains = 2, seed = 1)
```
"""
inference_opts(; kwargs...) = InferenceOpts(; kwargs...)

# ── Secondary model ──────────────────────────────────────────────────────

@enum SecondaryType incidence prevalence

"""
    SecondaryType

Type of secondary observation, passed to [`secondary_opts`](@ref). Values:
[`incidence`](@ref) (the default) and [`prevalence`](@ref).
"""
SecondaryType

"""
    incidence

Secondary observations that equal a convolution of previous primary
observations, e.g. deaths predicted from reported cases (a
[`SecondaryType`](@ref) value), the default.

```@example incidence
using EpiNow2
secondary_opts(incidence)
```
"""
incidence

"""
    prevalence

Secondary observations that are cumulative: current primary observations
minus a convolution of secondary observations, e.g. hospital bed occupancy
predicted from admissions (a [`SecondaryType`](@ref) value).

```@example prevalence
using EpiNow2
secondary_opts(prevalence)
```
"""
prevalence

"""
    SecondaryOpts(; type, cumulative, historic, primary_hist_additive,
                  current, primary_current_additive)

Secondary observation model options. Construct with
[`secondary_opts`](@ref).
"""
Base.@kwdef struct SecondaryOpts
    type::SecondaryType = incidence
    cumulative::Bool = false
    historic::Bool = true
    primary_hist_additive::Bool = true
    current::Bool = false
    primary_current_additive::Bool = false
end

"""
    secondary_opts(type = incidence; kwargs...)

Secondary observation model options. The secondary observations are modelled
as a convolution of primary observations, optionally combined with current
primary observations (additively or subtractively) and optionally
cumulative.

# Arguments
- `type`: the type of secondary observation, a [`SecondaryType`](@ref).
  `incidence` (default) assumes secondary reports equal a convolution of
  previous primary reports, e.g. deaths from cases. `prevalence` assumes
  they are cumulative and equal current primary reports minus a convolution
  of secondary reports, e.g. hospital bed use from admissions.

# Keyword Arguments
Override the flags set by `type`:
- `cumulative`: should the secondary report be cumulative?
- `historic`: should a convolution of primary reports be used to predict
  secondary reports?
- `primary_hist_additive`: should that convolution be added (`true`) or
  subtracted (`false`)?
- `current`: should current primary reports contribute to current secondary
  reports?
- `primary_current_additive`: should current primary reports be added
  (`true`) or subtracted (`false`)?

# Returns
A `SecondaryOpts` object.

```@example secondary_opts
using EpiNow2
secondary_opts(prevalence)
```
"""
function secondary_opts(type::SecondaryType = incidence; kwargs...)
    return if type == incidence
        SecondaryOpts(;
            type = incidence, cumulative = false, historic = true,
            primary_hist_additive = true, current = false,
            primary_current_additive = false, kwargs...
        )
    elseif type == prevalence
        SecondaryOpts(;
            type = prevalence, cumulative = true, historic = true,
            primary_hist_additive = false, current = true,
            primary_current_additive = true, kwargs...
        )
    else
        SecondaryOpts(; type, kwargs...)
    end
end

# ── Show methods ─────────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", o::GTOpts)
    println(io, "Generation time options:")
    return _show_tree(io, o.dist; indent = 1)
end

function Base.show(io::IO, ::MIME"text/plain", o::DelayOpts)
    println(io, "Delay options:")
    return _show_tree(io, o.dist; indent = 1)
end

function Base.show(io::IO, ::MIME"text/plain", o::TruncOpts)
    println(io, "Truncation options:")
    return _show_tree(io, o.dist; indent = 1)
end

function Base.show(io::IO, ::MIME"text/plain", o::RtOpts)
    m = round(mean(o.prior), digits = 2)
    s = round(std(o.prior), digits = 2)
    println(io, "Rt options:")
    println(io, "  prior: $(_dist_family(o.prior))(mean=$m, sd=$s)")
    println(io, "  use_rt: $(o.use_rt)")
    o.rw > 0 && println(io, "  random walk period: $(o.rw)")
    println(io, "  gp_on: $(o.gp_on)")
    println(io, "  future: $(o.future)")
    return _has_pop(o) && println(io, "  population: $(o.pop)")
end

_has_pop(o::RtOpts) = o.pop isa Distribution || (o.pop isa Number && o.pop > 0)

function Base.show(io::IO, ::MIME"text/plain", o::ObsOpts)
    println(io, "Observation model options:")
    println(io, "  family: $(o.family)")
    println(io, "  week_effect: $(o.week_effect)")
    return o.scale isa Distribution && println(io, "  scale: $(o.scale)")
end

function Base.show(io::IO, ::MIME"text/plain", o::InferenceOpts)
    println(io, "Inference options:")
    println(io, "  sampler: $(o.sampler)")
    println(io, "  samples: $(o.samples), warmup: $(o.warmup), chains: $(o.chains)")
    return println(io, "  adtype: $(o.adtype)")
end
