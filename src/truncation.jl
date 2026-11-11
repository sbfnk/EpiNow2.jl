# ── Truncation estimation ─────────────────────────────────────────────────

struct TruncationMetadata
    snapshot_dates::Vector{Vector{Date}}
    max_trunc::Int
end

"""
    EstimateTruncationArgs

Captures the configuration arguments passed to `estimate_truncation()`.
Mirrors the `args` field of EpiNow2 R's v1.8 `epinowfit` S3 class.
"""
struct EstimateTruncationArgs
    truncation::TruncOpts
    obs::ObsOpts
    noise::Distribution
    inference::InferenceOpts
    CrIs::Vector{Float64}
end

"""
    EstimateTruncationResult

Result of `estimate_truncation()`.
"""
struct EstimateTruncationResult
    fit::EpiNow2Fit
    args::EstimateTruncationArgs
    dist::DelayDistribution
    observations::Vector{EpiData}
    timing::Float64
end

"""
    estimate_truncation(data; truncation, obs = obs_opts(),
                        noise = Normal(0, 1), inference,
                        CrIs = [0.2, 0.5, 0.9], verbose = true)

Estimate a right-truncation distribution from several snapshots of the same
data taken at different times. Recent counts are incomplete in earlier
snapshots; comparing them with later snapshots gives the delay until counts
are complete, as a lognormal distribution.

# Arguments
- `data`: a vector of at least two `DataFrame` snapshots, each with `:date`
  and `:confirm` columns. Snapshots that end later should be more complete.

# Keyword Arguments
- `truncation`: truncation options from [`trunc_opts`](@ref), usually with an
  [`UncertainDistribution`](@ref) whose parameters are estimated. Defaults to
  a lognormal with `meanlog ~ Normal(0, 1)`, `sdlog ~ Normal(1, 1)` and a
  maximum of 10 days, as in R.
- `obs`: observation model options from [`obs_opts`](@ref): the `family`,
  the `dispersion` prior and whether to evaluate the `likelihood`.
- `noise`: prior on the additive noise `sigma`, truncated to be positive
  (default `Normal(0, 1)`).
- `inference`: inference options from [`inference_opts`](@ref).
- `CrIs`: credible intervals to summarise (default `[0.2, 0.5, 0.9]`).
- `verbose`: print progress messages (default `true`).

# Returns
An `EstimateTruncationResult` whose `dist` field holds the estimated
truncation distribution, to pass to [`trunc_opts`](@ref) for
[`epinow`](@ref) or [`estimate_infections`](@ref).

```@example estimate_truncation
using EpiNow2, DataFrames
full = example_confirmed()[1:30, :]
snapshots = map([20, 25, 30]) do last_day
    snap = full[1:last_day, :]
    # The most recent five days are only partly reported
    for d in 1:5
        snap.confirm[last_day - d + 1] = max(1, round(Int, snap.confirm[last_day - d + 1] * d / 6))
    end
    snap
end
fit = estimate_truncation(
    snapshots;
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
fit.dist
```
"""
function estimate_truncation(
        data::Vector{DataFrame};
        truncation::TruncOpts = trunc_opts(
            UncertainDistribution(
                (meanlog, sdlog) -> LogNormal(meanlog, sdlog),
                [Normal(0.0, 1.0), truncated(Normal(1.0, 1.0); lower = 0.0)],
                10.0
            )
        ),
        obs::ObsOpts = obs_opts(),
        noise::Distribution = Normal(0.0, 1.0),
        inference::InferenceOpts = inference_opts(),
        CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
        verbose::Bool = true
    )
    length(data) >= 2 || throw(ArgumentError("Need at least 2 snapshots"))

    # Sort snapshots by end date so the most complete one is last (the
    # reconstruction reference), and align all snapshots on a shared time axis
    # starting at the earliest observed date.
    sorted_snaps = sort([sort(df, :date) for df in data]; by = s -> s.date[end])
    epi_datas = [EpiData(df) for df in sorted_snaps]

    common_start = minimum(s.date[1] for s in sorted_snaps)
    latest_end = maximum(s.date[end] for s in sorted_snaps)
    n_times = Dates.value(latest_end - common_start) + 1

    obs_matrix = zeros(Float64, n_times, length(sorted_snaps))
    for (i, s) in enumerate(sorted_snaps)
        for (d, c) in zip(s.date, s.confirm)
            obs_matrix[Dates.value(d - common_start) + 1, i] = Float64(c)
        end
    end
    obs_dist = [Dates.value(latest_end - s.date[end]) for s in sorted_snaps]

    trunc_uncertain = _extract_uncertain(truncation.dist)
    length(trunc_uncertain) <= 1 || throw(
        ArgumentError("Only one uncertain truncation distribution is supported")
    )
    trunc = isempty(trunc_uncertain) ? nothing : only(trunc_uncertain)
    trunc_fixed_pmf = _fixed_pmf(truncation.dist)
    max_trunc = length(trunc_fixed_pmf) - 1 +
        (isnothing(trunc) ? 0 : Int(trunc.max))

    verbose && @info "Estimating truncation..." n_snapshots = length(data) max_trunc

    model = truncation_model(
        obs_matrix, obs_dist, trunc, trunc_fixed_pmf,
        obs.family, obs.dispersion, noise, obs.likelihood,
        truncation.weight_prior ? Float64(n_times) : 1.0
    )

    metadata = TruncationMetadata(
        [Date.(s.date) for s in sorted_snaps],
        max_trunc
    )

    t0 = time()
    fit = run_inference(model, metadata, inference)
    elapsed = time() - t0

    verbose && @info "Truncation estimation complete" seconds = round(elapsed, digits = 1)

    fitted_dist = _extract_truncation_dist(fit, truncation.dist, trunc)

    args = EstimateTruncationArgs(truncation, obs, noise, inference, CrIs)
    return EstimateTruncationResult(fit, args, fitted_dist, epi_datas, elapsed)
end

function _extract_truncation_dist(fit::EpiNow2Fit, dist, trunc)
    isnothing(trunc) && return dist
    params = reduce(hcat, [gq.trunc_params for gq in fit.generated_quantities])
    return trunc.constructor(vec(mean(params; dims = 2))...)
end
