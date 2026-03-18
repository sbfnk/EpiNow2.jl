# ── Truncation estimation ─────────────────────────────────────────────────

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
# Three snapshots of the same epidemic, taken on different dates
snapshots = [early_data, mid_data, late_data]

trunc = estimate_truncation(
    snapshots,
    truncation = trunc_opts(LogNormal(0.0, 1.0))
)

# Use fitted truncation in main estimation
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
    verbose && @info "Estimating truncation..." n_snapshots=length(data)

    # TODO: Build truncation model
    # Compare snapshots to estimate how much recent data is missing

    nothing  # placeholder
end

struct EstimateTruncationResult
    fit::EpiNow2Fit
    dist::DelayDistribution  # fitted truncation distribution
    observations::Vector{EpiData}
    timing::Float64
end
