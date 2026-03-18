# ── Secondary observation modelling ───────────────────────────────────────

"""
    estimate_secondary(data; secondary, delays, truncation, obs,
                       inference, burn_in, verbose)

Estimate the relationship between primary and secondary observations
(e.g., cases → deaths, cases → hospitalisations).

# Arguments
- `data::DataFrame` — must contain `:date`, `:primary`, `:secondary` columns
- `secondary::SecondaryOpts` — model structure (:incidence or :prevalence)
- `delays::DelayOpts` — delay from primary to secondary observation
- `burn_in::Int` — days to discard from start (default: 14)

# Returns
`EstimateSecondaryResult` with fitted delay and scaling parameters.

# Example
```julia
data = DataFrame(
    date = dates,
    primary = cases,
    secondary = deaths
)

result = estimate_secondary(
    data,
    delays = delay_opts(LogNormal(2.5, 0.47))
)
```
"""
function estimate_secondary(
    data::DataFrame;
    secondary::SecondaryOpts = secondary_opts(),
    delays::DelayOpts = delay_opts(LogNormal(2.5, 0.47)),
    truncation::TruncOpts = trunc_opts(),
    obs::ObsOpts = obs_opts(),
    inference::InferenceOpts = inference_opts(),
    burn_in::Int = 14,
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    verbose::Bool = true
)
    sec_data = SecondaryData(data)

    # Assemble secondary-specific Turing model
    # Maps to EpiAware observation model with primary→secondary convolution

    verbose && @info "Estimating secondary observations..." type=secondary.type

    # TODO: Build and fit secondary model using EpiAware components
    # Key components:
    #   1. Convolve primary with delay → expected secondary
    #   2. For prevalence: cumulative sum
    #   3. Observation model (NegBin/Poisson)

    nothing  # placeholder
end

"""
    EstimateSecondaryResult

Result of `estimate_secondary()`.
"""
struct EstimateSecondaryResult
    fit::EpiNow2Fit
    observations::SecondaryData
    predictions::DataFrame
    timing::Float64
end
