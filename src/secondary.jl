# ── Secondary observation modelling ───────────────────────────────────────

struct SecondaryMetadata
    dates::Vector{Date}
    burn_in::Int
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

"""
    estimate_secondary(data; secondary, delays, obs, inference, burn_in, verbose)

Estimate the relationship between primary and secondary observations
(e.g., cases to deaths, cases to hospitalisations).

# Arguments
- `data::DataFrame` — must contain `:date`, `:primary`, `:secondary` columns
- `secondary::SecondaryOpts` — model structure (`:incidence` or `:prevalence`)
- `delays::DelayOpts` — delay from primary to secondary observation
- `burn_in::Int` — days to discard from start for likelihood (default: 14)

# Returns
`EstimateSecondaryResult` with fitted delay and scaling parameters.
"""
function estimate_secondary(
    data::DataFrame;
    secondary::SecondaryOpts = secondary_opts(),
    delays::DelayOpts = delay_opts(LogNormal(2.5, 0.47)),
    obs::ObsOpts = obs_opts(),
    inference::InferenceOpts = inference_opts(),
    burn_in::Int = 14,
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    verbose::Bool = true
)
    sec_data = SecondaryData(data)
    n_total = length(sec_data.primary)
    n_obs = n_total - burn_in

    delay_pmf = discretise(delays.dist).pmf

    verbose && @info "Estimating secondary observations..." type=secondary.type n_obs

    model = secondary_model(
        sec_data.primary,
        sec_data.secondary[(burn_in + 1):end],
        delay_pmf,
        n_obs,
        burn_in,
        secondary.type == :prevalence,
        obs.family
    )

    metadata = SecondaryMetadata(sec_data.date, burn_in)

    t0 = time()
    fit = run_inference(model, metadata, inference)
    elapsed = time() - t0

    verbose && @info "Secondary estimation complete" seconds=round(elapsed, digits=1)

    predictions = _summarise_secondary_gq(fit, sec_data.date, CrIs)

    EstimateSecondaryResult(fit, sec_data, predictions, elapsed)
end

function _summarise_secondary_gq(
    fit::EpiNow2Fit, dates::Vector{Date}, CrIs::Vector{Float64}
)
    gqs = fit.generated_quantities
    n_samples = length(gqs)
    n_times = length(first(gqs).expected)
    n_dates = min(n_times, length(dates))

    mat = Matrix{Float64}(undef, n_dates, n_samples)
    for (i, gq) in enumerate(gqs)
        for t in 1:n_dates
            mat[t, i] = Float64(gq.expected[t])
        end
    end

    _matrix_to_summary(mat, dates[1:n_dates], CrIs)
end
