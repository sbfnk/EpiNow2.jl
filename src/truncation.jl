# ── Truncation estimation ─────────────────────────────────────────────────

struct TruncationMetadata
    snapshot_dates::Vector{Vector{Date}}
    max_trunc::Int
end

"""
    EstimateTruncationResult

Result of `estimate_truncation()`.
"""
struct EstimateTruncationResult
    fit::EpiNow2Fit
    dist::DelayDistribution
    observations::Vector{EpiData}
    timing::Float64
end

"""
    estimate_truncation(data; truncation, inference, verbose)

Estimate the distribution of reporting delays from multiple snapshots of
the same data taken at different times.

# Arguments
- `data::Vector{DataFrame}` — multiple snapshots, each with `:date` and
  `:confirm` columns. Later snapshots should have more complete data.
- `truncation::TruncOpts` — prior on truncation distribution

# Returns
`EstimateTruncationResult` with fitted truncation distribution that can
be passed to `epinow()` or `estimate_infections()`.

# Example
```julia
snapshots = [early_data, mid_data, late_data]
trunc = estimate_truncation(snapshots)
result = epinow(data, truncation=trunc_opts(trunc.dist), ...)
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

    sorted_snaps = [sort(df, :date) for df in data]
    epi_datas = [EpiData(df) for df in sorted_snaps]

    # Extract confirm vectors; each snapshot covers a different date range
    snapshots = [Int.(s.confirm) for s in sorted_snaps]
    snapshot_lengths = [length(s) for s in snapshots]

    max_trunc = if truncation.dist isa UncertainDistribution
        Int(truncation.dist.max)
    elseif truncation.dist isa Distribution && !(truncation.dist isa Dirac)
        min(30, Int(ceil(quantile(truncation.dist, 0.999))))
    else
        15
    end

    verbose && @info "Estimating truncation..." n_snapshots=length(data) max_trunc

    model = truncation_model(
        snapshots, snapshot_lengths, max_trunc,
        truncation.meanlog_prior, truncation.sdlog_prior
    )

    metadata = TruncationMetadata(
        [Date.(s.date) for s in sorted_snaps],
        max_trunc
    )

    t0 = time()
    fit = run_inference(model, metadata, inference)
    elapsed = time() - t0

    verbose && @info "Truncation estimation complete" seconds=round(elapsed, digits=1)

    fitted_dist = _extract_truncation_dist(fit)

    EstimateTruncationResult(fit, fitted_dist, epi_datas, elapsed)
end

function _extract_truncation_dist(fit::EpiNow2Fit)
    gqs = fit.generated_quantities
    meanlog_samples = [gq.trunc_meanlog for gq in gqs]
    sdlog_samples = [gq.trunc_sdlog for gq in gqs]
    LogNormal(mean(meanlog_samples), mean(sdlog_samples))
end
