# ══════════════════════════════════════════════════════════════════════════
# estimate_dist: MCMC fit of a delay distribution accounting for primary
# and secondary event censoring (double interval censoring) and right
# truncation. Mirrors EpiNow2 R's v1.8 `estimate_dist()`.
#
# Likelihood is computed via `CensoredDistributions.jl`'s
# `double_interval_censored` constructor (vendored upstream from the
# `primarycensored` R package).
# ══════════════════════════════════════════════════════════════════════════

using CensoredDistributions: double_interval_censored

"""
    EstimateDistArgs

Configuration arguments for `estimate_dist()`. Mirrors EpiNow2 R v1.8's
`epinowfit` `args` field.
"""
struct EstimateDistArgs
    dist::Symbol
    priors::Vector{<:Distribution}
    primary::Symbol
    inference::InferenceOpts
    obs_time_threshold::Float64
end

"""
    EstimateDistResult

Result of `estimate_dist()`. Holds the underlying MCMCChains `chain`,
the configuration `args`, the (preprocessed) `observations`, and a
posterior-mean `UncertainDistribution` for downstream use in
`delay_opts()` / `gt_opts()` / `trunc_opts()`.
"""
struct EstimateDistResult
    chain::MCMCChains.Chains
    args::EstimateDistArgs
    observations::DataFrame
    fitted::UncertainDistribution
    timing::Float64
end

# ── Default priors per family ───────────────────────────────────────────────

function _default_priors(dist::Symbol)
    if dist == :lognormal
        Distribution[Normal(1.0, 1.0), truncated(Normal(0.5, 0.5); lower=0.01)]
    elseif dist == :gamma
        Distribution[truncated(Normal(2.0, 2.0); lower=0.01),
                     truncated(Normal(0.5, 0.5); lower=0.01)]
    else
        throw(ArgumentError(
            "estimate_dist currently supports :lognormal and :gamma. " *
            "For others use a custom Turing model with " *
            "CensoredDistributions.double_interval_censored."
        ))
    end
end

function _dist_constructor(dist::Symbol)
    if dist == :lognormal
        (a, b) -> LogNormal(a, b)
    elseif dist == :gamma
        (a, b) -> Gamma(a, b)
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
        obs_date  = :obs_date  in cols ? data.obs_date  :
                    fill(maximum(sdate_upr), n_row)
        n         = :n         in cols ? Int.(data.n) : ones(Int, n_row)
        df = DataFrame(
            delay_lwr = Dates.value.(data.sdate_lwr .- data.pdate_lwr),
            pwindow   = Dates.value.(pdate_upr .- data.pdate_lwr),
            obs_time  = Float64.(Dates.value.(obs_date .- data.pdate_lwr)),
            n         = n,
        )
    else
        throw(ArgumentError(
            "data must have either a `:delay` column or `:pdate_lwr` " *
            "and `:sdate_lwr` date columns"
        ))
    end

    # Aggregate identical (delay_lwr, pwindow, obs_time) rows
    return combine(
        groupby(df, [:delay_lwr, :pwindow, :obs_time]),
        :n => sum => :n,
    )
end

# ── Turing model ────────────────────────────────────────────────────────────

@model function _dist_likelihood(
    delay_lwr::Vector{Int},
    pwindow::Vector{Int},
    obs_time::Vector{Float64},
    weight::Vector{Int},
    constructor::Function,
    priors::Vector{<:Distribution},
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
            primary_event = Uniform(0.0, Float64(pwindow[i])),
            upper = obs_time[i],
            interval = 1.0,
        )
        Turing.@addlogprob! weight[i] * logpdf(cens, delay_lwr[i])
    end
end

# ── Main entry point ────────────────────────────────────────────────────────

"""
    estimate_dist(data; dist=:lognormal, priors=nothing, primary=:uniform,
                  obs_time_threshold=2.0, inference=inference_opts(),
                  verbose=false)

Estimate a delay distribution using interval-censored MCMC inference,
mirroring EpiNow2 R v1.8's `estimate_dist()`.

# Data formats

The input `DataFrame` may be either:

- **R-style**: columns `pdate_lwr` (and optional `pdate_upr`), `sdate_lwr`
  (and optional `sdate_upr`), optional `obs_date`, optional `n`. Dates are
  `Date` values.
- **Simple**: a single `:delay` column of non-negative integer delays.
  Daily primary/secondary windows are assumed and observations are treated
  as untruncated.

# Distribution choice

`dist` selects the parametric family. Currently `:lognormal` (default) and
`:gamma` are supported; for others write a custom Turing model using
`CensoredDistributions.double_interval_censored` directly.

`priors` optionally overrides the defaults — supply a `Vector{Distribution}`
in the same parameter order as the constructor (e.g. `[meanlog_prior,
sdlog_prior]` for LogNormal).

`obs_time_threshold` controls the obs-time-to-Inf heuristic
(observations whose `obs_time` exceeds `obs_time_threshold * max(delay)`
are treated as untruncated, matching the `epidist` convention).

# Returns

An `EstimateDistResult` with a posterior-mean `UncertainDistribution`
suitable for passing to `delay_opts()` / `gt_opts()` / `trunc_opts()`.
"""
function estimate_dist(
    data::DataFrame;
    dist::Symbol = :lognormal,
    priors::Union{Vector{<:Distribution}, Nothing} = nothing,
    primary::Symbol = :uniform,
    obs_time_threshold::Float64 = 2.0,
    inference::InferenceOpts = inference_opts(),
    verbose::Bool = false,
)
    primary == :uniform || throw(ArgumentError(
        "Only `primary=:uniform` is currently supported (R supports " *
        "`expgrowth` with a fixed growth rate)."
    ))

    obs = _normalise_dist_data(data)
    isempty(obs) && throw(ArgumentError("No observations after preprocessing"))

    # obs-time-to-Inf heuristic
    if isfinite(obs_time_threshold)
        cutoff = obs_time_threshold * maximum(obs.delay_lwr)
        obs.obs_time = ifelse.(obs.obs_time .> cutoff, Inf, obs.obs_time)
    end

    used_priors = isnothing(priors) ? _default_priors(dist) : priors
    constructor = _dist_constructor(dist)

    verbose && @info "Estimating delay distribution..." dist n_obs=nrow(obs)

    model = _dist_likelihood(
        obs.delay_lwr, obs.pwindow, obs.obs_time, obs.n,
        constructor, used_priors,
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
            inference.samples + inference.warmup,
            inference.chains;
            num_warmup = inference.warmup,
            progress = inference.progress,
            verbose = false,
        )
    else
        Turing.sample(
            rng, model, sampler,
            inference.samples + inference.warmup;
            num_warmup = inference.warmup,
            progress = inference.progress,
            verbose = false,
        )
    end
    elapsed = time() - t0

    verbose && @info "estimate_dist complete" seconds=round(elapsed, digits=1)

    # Posterior summary → UncertainDistribution
    n_params = length(used_priors)
    param_means = [mean(vec(Array(chain[Symbol("params[$i]")]))) for i in 1:n_params]
    param_sds   = [std(vec(Array(chain[Symbol("params[$i]")]))) for i in 1:n_params]

    fitted = UncertainDistribution(
        constructor,
        Distribution[
            Normal(param_means[i], max(param_sds[i], 1e-4))
            for i in 1:n_params
        ],
        Float64(maximum(obs.delay_lwr) * 2),
    )

    args = EstimateDistArgs(
        dist, used_priors, primary, inference, obs_time_threshold,
    )
    EstimateDistResult(chain, args, obs, fitted, elapsed)
end
