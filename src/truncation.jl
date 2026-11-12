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
    estimate_truncation(data; truncation, inference, CrIs = [0.2, 0.5, 0.9],
                        verbose = true)

Estimate a right-truncation distribution from several snapshots of the same
data taken at different times. Recent counts are incomplete in earlier
snapshots; comparing them with later snapshots gives the delay until counts
are complete, as a lognormal distribution.

# Arguments
- `data`: a vector of at least two `DataFrame` snapshots, each with `:date`
  and `:confirm` columns. Snapshots that end later should be more complete.

# Keyword Arguments
- `truncation`: truncation options from [`trunc_opts`](@ref). Its
  `meanlog_prior` and `sdlog_prior` set the priors on the lognormal
  parameters, and the maximum truncation is taken from `dist` (default: a
  lognormal with meanlog 0 and sdlog 1).
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
        truncation::TruncOpts = trunc_opts(LogNormal(0.0, 1.0)),
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

    obs = zeros(Float64, n_times, length(sorted_snaps))
    for (i, s) in enumerate(sorted_snaps)
        for (d, c) in zip(s.date, s.confirm)
            obs[Dates.value(d - common_start) + 1, i] = Float64(c)
        end
    end
    obs_dist = [Dates.value(latest_end - s.date[end]) for s in sorted_snaps]

    max_trunc = if truncation.dist isa UncertainDistribution
        Int(truncation.dist.max)
    elseif truncation.dist isa Distribution && !(truncation.dist isa Dirac)
        min(30, Int(ceil(quantile(truncation.dist, 0.999))))
    else
        15
    end

    verbose && @info "Estimating truncation..." n_snapshots = length(data) max_trunc

    model = truncation_model(
        obs, obs_dist, max_trunc,
        truncation.meanlog_prior, truncation.sdlog_prior
    )

    metadata = TruncationMetadata(
        [Date.(s.date) for s in sorted_snaps],
        max_trunc
    )

    t0 = time()
    fit = run_inference(model, metadata, inference)
    elapsed = time() - t0

    verbose && @info "Truncation estimation complete" seconds = round(elapsed, digits = 1)

    fitted_dist = _extract_truncation_dist(fit)

    args = EstimateTruncationArgs(truncation, inference, CrIs)
    return EstimateTruncationResult(fit, args, fitted_dist, epi_datas, elapsed)
end

function _extract_truncation_dist(fit::EpiNow2Fit)
    gqs = fit.generated_quantities
    meanlog_samples = [gq.trunc_meanlog for gq in gqs]
    sdlog_samples = [gq.trunc_sdlog for gq in gqs]
    return LogNormal(mean(meanlog_samples), mean(sdlog_samples))
end
