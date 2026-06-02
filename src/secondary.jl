# ── Secondary observation modelling ───────────────────────────────────────

struct SecondaryMetadata
    dates::Vector{Date}
    burn_in::Int
end

"""
    EstimateSecondaryArgs

Captures the configuration arguments passed to `estimate_secondary()`.
Mirrors the `args` field of EpiNow2 R's v1.8 `epinowfit` S3 class.
"""
struct EstimateSecondaryArgs
    secondary::SecondaryOpts
    delays::DelayOpts
    obs::ObsOpts
    inference::InferenceOpts
    burn_in::Int
    CrIs::Vector{Float64}
end

"""
    EstimateSecondaryResult

Result of `estimate_secondary()`.
"""
struct EstimateSecondaryResult
    fit::EpiNow2Fit
    args::EstimateSecondaryArgs
    observations::SecondaryData
    predictions::DataFrame          # posterior predictive (incl. obs noise)
    predictions_expected::DataFrame # latent expected secondary (no obs noise)
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

    start_day = Dates.dayofweek(sec_data.date[1])
    frac_prior = obs.scale isa Distribution ? obs.scale : Beta(5.0, 5.0)

    model = secondary_model(
        sec_data.primary,
        sec_data.secondary[(burn_in + 1):end],
        delay_pmf,
        n_obs,
        burn_in,
        obs.family,
        secondary.cumulative,
        secondary.historic,
        secondary.primary_hist_additive,
        secondary.current,
        secondary.primary_current_additive,
        obs.week_effect,
        obs.week_length,
        start_day,
        frac_prior
    )

    metadata = SecondaryMetadata(sec_data.date, burn_in)

    t0 = time()
    fit = run_inference(model, metadata, inference)
    elapsed = time() - t0

    verbose && @info "Secondary estimation complete" seconds=round(elapsed, digits=1)

    # `predictions` is the posterior predictive (latent expectation + observation
    # noise), matching R's `get_predictions()`. The latent mean is kept separately.
    predictions_expected = _summarise_secondary_gq(fit, sec_data.date, CrIs)
    predictions = _summarise_secondary_predictive(
        fit, sec_data.date, CrIs, obs.family
    )

    args = EstimateSecondaryArgs(
        secondary, delays, obs, inference, burn_in, CrIs
    )
    EstimateSecondaryResult(
        fit, args, sec_data, predictions, predictions_expected, elapsed
    )
end

"""
    simulate_secondary(primary; delays, obs, secondary, frac)

Simulate secondary observations from a primary time series. Uses the
same model as `estimate_secondary` but with fixed parameters.

# Arguments
- `primary::DataFrame` — must contain `:date` and `:primary` columns
- `delays::DelayOpts` — delay from primary to secondary
- `obs::ObsOpts` — observation model (`:negbin` or `:poisson`)
- `secondary::SecondaryOpts` — `:incidence` or `:prevalence`
- `frac::Float64` — fraction of primary that become secondary (default: 0.1)

# Returns
A `DataFrame` with `:date`, `:primary`, and `:secondary` columns.
"""
function simulate_secondary(
    primary::DataFrame;
    delays::DelayOpts = delay_opts(LogNormal(2.5, 0.47)),
    obs::ObsOpts = obs_opts(family=poisson),
    secondary::SecondaryOpts = secondary_opts(),
    frac::Float64 = 0.1
)
    sorted = sort(primary, :date)
    dates = Date.(sorted.date)
    prim = Int.(sorted.primary)

    delay_pmf = discretise(delays.dist).pmf
    scaled = frac .* Float64.(prim)
    conv = convolve(scaled, delay_pmf)
    expected = calculate_secondary(
        scaled, conv,
        secondary.cumulative, secondary.historic, secondary.primary_hist_additive,
        secondary.current, secondary.primary_current_additive
    )

    sim = [max(0, rand(Poisson(max(e, 1e-6)))) for e in expected]

    DataFrame(date=dates, primary=prim, secondary=sim)
end

"""
    forecast_secondary(sec_result, inf_result)

Forecast secondary observations by applying the fitted secondary
relationship to the case forecast from `estimate_infections`.

# Returns
A `DataFrame` with secondary predictions for the forecast period.
"""
function forecast_secondary(
    sec_result::EstimateSecondaryResult,
    inf_result::EstimateInfectionsResult;
    delays::DelayOpts = delay_opts(LogNormal(2.5, 0.47)),
    secondary::SecondaryOpts = secondary_opts(),
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    # Extract posterior samples of frac from secondary fit
    sec_params = get_parameters(sec_result)
    frac_samples = sec_params[:frac]
    n_samples = length(frac_samples)

    delay_pmf = discretise(delays.dist).pmf

    # Get primary forecast samples
    primary_samples = get_samples(inf_result; variable=:reports)
    forecast_start = inf_result.observations.date[end]
    forecast_samples = filter(r -> r.date > forecast_start, primary_samples)

    if isempty(forecast_samples)
        return DataFrame()
    end

    forecast_dates = sort(unique(forecast_samples.date))
    sample_ids = sort(unique(forecast_samples.sample))

    # Build a date→index lookup for efficient access
    date_idx = Dict(d => i for (i, d) in enumerate(forecast_dates))
    n_dates = length(forecast_dates)
    n_out = min(n_samples, length(sample_ids))
    mat = Matrix{Float64}(undef, n_dates, n_out)

    for (si, s) in enumerate(sample_ids)
        si > n_out && break
        s_data = filter(r -> r.sample == s, forecast_samples)
        # Build primary vector in date order
        primary_vec = zeros(Float64, n_dates)
        for r in eachrow(s_data)
            idx = get(date_idx, r.date, nothing)
            !isnothing(idx) && (primary_vec[idx] = r.value)
        end

        frac_i = frac_samples[mod1(si, n_samples)]
        scaled = frac_i .* primary_vec
        conv = convolve(scaled, delay_pmf)
        expected = calculate_secondary(
            scaled, conv,
            secondary.cumulative, secondary.historic, secondary.primary_hist_additive,
            secondary.current, secondary.primary_current_additive
        )

        for t in 1:n_dates
            mat[t, si] = expected[t]
        end
    end

    _matrix_to_summary(mat, forecast_dates, CrIs)
end

_summarise_secondary_gq(fit::EpiNow2Fit, dates::Vector{Date}, CrIs::Vector{Float64}) =
    _summarise_gq(fit, :expected, dates, CrIs)

_summarise_secondary_predictive(
    fit::EpiNow2Fit, dates::Vector{Date}, CrIs::Vector{Float64}, family::ObsFamily
) = _summarise_gq_predictive(fit, :expected, dates, CrIs, family, :overdispersion)

"""
    get_predictions(result::EstimateSecondaryResult)

Return the posterior predictive of secondary observations (matching R's
`get_predictions()`). For the latent expectation without observation noise,
use `result.predictions_expected`.
"""
get_predictions(result::EstimateSecondaryResult) = result.predictions
