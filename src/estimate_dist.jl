# ══════════════════════════════════════════════════════════════════════════
# estimate_dist: MCMC fit of a delay distribution accounting for primary
# and secondary event censoring (double interval censoring) and right
# truncation. Mirrors EpiNow2 R's v1.8 `estimate_dist()`.
#
# Likelihood is computed via `CensoredDistributions.jl`'s
# `double_interval_censored` constructor (vendored upstream from the
# `primarycensored` R package).
# ══════════════════════════════════════════════════════════════════════════


"""
    EstimateDistArgs

Configuration arguments for `estimate_dist()`. Mirrors EpiNow2 R v1.8's
`epinowfit` `args` field.
"""
struct EstimateDistArgs
    dist::Symbol
    priors::Vector{<:Distribution}
    primary::Symbol
    primary_params::Vector{Float64}
    inference::InferenceOpts
    obs_time_threshold::Float64
end

"""
    EstimateDistResult

Result of `estimate_dist()`. Holds the underlying FlexiChains `chain`,
the configuration `args`, the (preprocessed) `observations`, and a
posterior-mean `UncertainDistribution` for downstream use in
`delay_opts()` / `gt_opts()` / `trunc_opts()`.
"""
struct EstimateDistResult
    chain::VNChain
    args::EstimateDistArgs
    observations::DataFrame
    fitted::UncertainDistribution
    timing::Float64
end

# ── Default priors per family ───────────────────────────────────────────────

function _default_priors(dist::Symbol)
    return if dist == :lognormal
        # meanlog ~ Normal(1, 1); sdlog ~ HalfNormal(0.5)
        Distribution[
            Normal(1.0, 1.0),
            truncated(Normal(0.5, 0.5); lower = 0.01),
        ]
    elseif dist == :gamma
        # shape ~ HalfNormal(2); rate ~ HalfNormal(0.5)  (matches R)
        Distribution[
            truncated(Normal(2.0, 2.0); lower = 0.01),
            truncated(Normal(0.5, 0.5); lower = 0.01),
        ]
    elseif dist == :normal
        # mean ~ Normal(5, 5); sd ~ HalfNormal(1)
        Distribution[
            Normal(5.0, 5.0),
            truncated(Normal(1.0, 1.0); lower = 0.01),
        ]
    elseif dist == :exp
        # rate ~ HalfNormal(0.5)
        Distribution[truncated(Normal(0.5, 0.5); lower = 0.01)]
    elseif dist == :weibull
        # shape ~ HalfNormal(2); scale ~ HalfNormal(5)
        Distribution[
            truncated(Normal(2.0, 2.0); lower = 0.01),
            truncated(Normal(5.0, 5.0); lower = 0.01),
        ]
    else
        throw(
            ArgumentError(
                "estimate_dist supports :lognormal, :gamma, :normal, :exp, " *
                    "and :weibull"
            )
        )
    end
end

function _dist_constructor(dist::Symbol)
    return if dist == :lognormal
        (meanlog, sdlog) -> LogNormal(meanlog, sdlog)
    elseif dist == :gamma
        # R uses rate parameterisation; Distributions.jl's Gamma takes
        # (α=shape, θ=scale), so convert rate → 1/rate.
        (shape, rate) -> Gamma(shape, 1.0 / rate)
    elseif dist == :normal
        (μ, σ) -> Normal(μ, σ)
    elseif dist == :exp
        rate -> Exponential(1.0 / rate)
    elseif dist == :weibull
        (shape, scale) -> Weibull(shape, scale)
    else
        throw(ArgumentError("Unsupported dist: $dist"))
    end
end

# ── Data-frame normalisation ────────────────────────────────────────────────

# Accepts either:
#   - R-style: pdate_lwr, sdate_lwr, [pdate_upr, sdate_upr, obs_date, n]
#     (each row is one observation; columns are Dates)
#   - Simple: a `:delay` column with non-negative integer/numeric delays,
#     assumed to come from daily intervals.
#
# Returns a DataFrame with columns:
#   delay_lwr (Int), pwindow (Int), obs_time (Float64), n (Int)
#
# Where:
#   - delay_lwr = sdate_lwr - pdate_lwr (integer days)
#   - pwindow   = pdate_upr - pdate_lwr (length of primary event window;
#                                        defaults to 1 if upper missing)
#   - obs_time  = obs_date - pdate_lwr (days from primary to censoring)
#   - n         = aggregated count of identical observations
function _normalise_dist_data(data::DataFrame)
    cols = propertynames(data)
    if :delay in cols
        df = DataFrame(
            delay_lwr = Int.(data.delay),
            pwindow = ones(Int, nrow(data)),
            obs_time = fill(Inf, nrow(data)),
            n = ones(Int, nrow(data)),
        )
    elseif :pdate_lwr in cols && :sdate_lwr in cols
        n_row = nrow(data)
        pdate_upr = :pdate_upr in cols ? data.pdate_upr : data.pdate_lwr .+ Day(1)
        sdate_upr = :sdate_upr in cols ? data.sdate_upr : data.sdate_lwr .+ Day(1)
        obs_date = :obs_date in cols ? data.obs_date :
            fill(maximum(sdate_upr), n_row)
        n = :n in cols ? Int.(data.n) : ones(Int, n_row)
        df = DataFrame(
            delay_lwr = Dates.value.(data.sdate_lwr .- data.pdate_lwr),
            pwindow = Dates.value.(pdate_upr .- data.pdate_lwr),
            obs_time = Float64.(Dates.value.(obs_date .- data.pdate_lwr)),
            n = n,
        )
    else
        throw(
            ArgumentError(
                "data must have either a `:delay` column or `:pdate_lwr` " *
                    "and `:sdate_lwr` date columns"
            )
        )
    end

    # Aggregate identical (delay_lwr, pwindow, obs_time) rows
    return combine(
        groupby(df, [:delay_lwr, :pwindow, :obs_time]),
        :n => sum => :n,
    )
end

# ── Turing model ────────────────────────────────────────────────────────────

function _make_primary_event(
        primary::Symbol, pwindow::Real,
        primary_params::Vector{Float64}
    )
    if primary == :uniform
        return Uniform(0.0, Float64(pwindow))
    elseif primary == :expgrowth
        # Single fixed growth rate parameter, passed through (not estimated).
        return ExponentiallyTilted(
            0.0, Float64(pwindow),
            Float64(primary_params[1])
        )
    else
        throw(
            ArgumentError(
                "primary must be :uniform or :expgrowth, got $primary"
            )
        )
    end
end

@model function _dist_likelihood(
        delay_lwr::Vector{Int},
        pwindow::Vector{Int},
        obs_time::Vector{Float64},
        weight::Vector{Int},
        constructor::Function,
        priors::Vector{<:Distribution},
        primary::Symbol,
        primary_params::Vector{Float64},
    )
    n_params = length(priors)
    params = Vector{Real}(undef, n_params)
    for i in 1:n_params
        params[i] ~ priors[i]
    end
    base = constructor(params...)
    for i in eachindex(delay_lwr)
        cens = double_interval_censored(
            base;
            primary_event = _make_primary_event(
                primary, pwindow[i], primary_params
            ),
            upper = obs_time[i],
            interval = 1.0,
        )
        Turing.@addlogprob! weight[i] * logpdf(cens, delay_lwr[i])
    end
end

# ── Main entry point ────────────────────────────────────────────────────────

"""
    estimate_dist(data; dist = :lognormal, priors = nothing, primary = :uniform,
                  primary_params = Float64[], obs_time_threshold = 2.0,
                  inference = inference_opts(), verbose = false)

Estimate a delay distribution from censored and possibly right-truncated
delay data by MCMC, accounting for the interval censoring of both the
primary and the secondary event. Mirrors EpiNow2's `estimate_dist()`.

# Arguments
- `data`: a `DataFrame` in one of two formats. The R format has date columns
  `pdate_lwr` (and optionally `pdate_upr`) for the primary event,
  `sdate_lwr` (and optionally `sdate_upr`) for the secondary event, and
  optionally `obs_date` (date of observation) and `n` (count). The simple
  format has a single `:delay` column of non-negative integer delays, with
  daily censoring windows and no truncation.

# Keyword Arguments
- `dist`: the distribution family, `:lognormal` (default) or `:gamma`.
- `priors`: priors on the distribution parameters, as a vector in the order
  the constructor takes them (e.g. `[meanlog_prior, sdlog_prior]` for a
  lognormal); `nothing` (default) uses built-in priors.
- `primary`: distribution of the primary event within its window, `:uniform`
  (default) or `:expgrowth` (exponential growth).
- `primary_params`: the growth rate, as a one-element vector, when
  `primary = :expgrowth`.
- `obs_time_threshold`: observations whose observation time exceeds
  `obs_time_threshold` times the maximum delay are treated as not
  truncated, following the epidist convention (default 2).
- `inference`: inference options from [`inference_opts`](@ref).
- `verbose`: print progress messages (default `false`).

# Returns
An `EstimateDistResult` whose `fitted` field is an
[`UncertainDistribution`](@ref) with normal priors from the posterior mean
and standard deviation of each parameter, ready for [`delay_opts`](@ref),
[`gt_opts`](@ref) or [`trunc_opts`](@ref). Its `chain` field holds the
posterior samples.

```@example estimate_dist
using EpiNow2, DataFrames, Distributions
data = DataFrame(delay = round.(Int, rand(LogNormal(1.5, 0.5), 100)))
fit = estimate_dist(
    data;
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false)
)
fit.fitted
```
"""
function estimate_dist(
        data::DataFrame;
        dist::Symbol = :lognormal,
        priors::Union{Vector{<:Distribution}, Nothing} = nothing,
        primary::Symbol = :uniform,
        primary_params::Vector{Float64} = Float64[],
        obs_time_threshold::Float64 = 2.0,
        inference::InferenceOpts = inference_opts(),
        verbose::Bool = false,
    )
    primary in (:uniform, :expgrowth) || throw(
        ArgumentError(
            "primary must be :uniform or :expgrowth, got $primary"
        )
    )
    if primary == :expgrowth && length(primary_params) != 1
        throw(
            ArgumentError(
                "primary_params must contain exactly one growth rate when " *
                    "primary = :expgrowth"
            )
        )
    end

    obs = _normalise_dist_data(data)
    isempty(obs) && throw(ArgumentError("No observations after preprocessing"))

    # obs-time-to-Inf heuristic
    if isfinite(obs_time_threshold)
        cutoff = obs_time_threshold * maximum(obs.delay_lwr)
        obs.obs_time = ifelse.(obs.obs_time .> cutoff, Inf, obs.obs_time)
    end

    used_priors = isnothing(priors) ? _default_priors(dist) : priors
    constructor = _dist_constructor(dist)

    verbose && @info "Estimating delay distribution..." dist n_obs = nrow(obs)

    model = _dist_likelihood(
        obs.delay_lwr, obs.pwindow, obs.obs_time, obs.n,
        constructor, used_priors, primary, primary_params,
    )

    sampler = NUTS(
        inference.warmup, inference.target_acceptance;
        max_depth = inference.max_treedepth,
        adtype = inference.adtype,
    )
    rng = isnothing(inference.seed) ? Random.default_rng() :
        Random.MersenneTwister(inference.seed)

    t0 = time()
    chain = if inference.chains > 1
        Turing.sample(
            rng, model, sampler,
            MCMCThreads(),
            _samples_per_chain(inference),
            inference.chains;
            num_warmup = inference.warmup,
            progress = inference.progress,
            verbose = false,
            chain_type = VNChain,
        )
    else
        Turing.sample(
            rng, model, sampler,
            _samples_per_chain(inference);
            num_warmup = inference.warmup,
            progress = inference.progress,
            verbose = false,
            chain_type = VNChain,
        )
    end
    elapsed = time() - t0

    verbose && @info "estimate_dist complete" seconds = round(elapsed, digits = 1)

    # Posterior summary → UncertainDistribution
    n_params = length(used_priors)
    param_draws = [vec(Array(chain[@varname(params[i])])) for i in 1:n_params]
    param_means = mean.(param_draws)
    param_sds = std.(param_draws)

    fitted = UncertainDistribution(
        constructor,
        Distribution[
            Normal(param_means[i], max(param_sds[i], 1.0e-4))
                for i in 1:n_params
        ],
        Float64(maximum(obs.delay_lwr) * 2),
    )

    args = EstimateDistArgs(
        dist, used_priors, primary, primary_params,
        inference, obs_time_threshold,
    )
    return EstimateDistResult(chain, args, obs, fitted, elapsed)
end
