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
    obs::ObsOpts = obs_opts(family=:poisson),
    secondary::SecondaryOpts = secondary_opts(),
    frac::Float64 = 0.1
)
    sorted = sort(primary, :date)
    dates = Date.(sorted.date)
    prim = Int.(sorted.primary)

    delay_pmf = discretise(delays.dist).pmf
    scaled = frac .* Float64.(prim)
    expected = convolve(scaled, delay_pmf)

    if secondary.type == :prevalence
        expected = cumsum(expected)
    end

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
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    # Extract posterior samples of frac from secondary fit
    sec_params = get_parameters(sec_result)
    frac_samples = sec_params[:frac]
    n_samples = length(frac_samples)

    # Get primary forecast samples
    primary_samples = get_samples(inf_result; variable=:reports)
    forecast_start = inf_result.observations.date[end]
    forecast_samples = filter(r -> r.date > forecast_start, primary_samples)

    if isempty(forecast_samples)
        return DataFrame()
    end

    forecast_dates = sort(unique(forecast_samples.date))
    delay_pmf = discretise(sec_result.observations.primary |> _ -> Dirac(0.0)).pmf

    # For each sample, scale primary by frac
    n_dates = length(forecast_dates)
    mat = Matrix{Float64}(undef, n_dates, min(n_samples, length(unique(forecast_samples.sample))))

    for (si, s) in enumerate(unique(forecast_samples.sample))
        si > size(mat, 2) && break
        s_data = filter(r -> r.sample == s, forecast_samples)
        sort!(s_data, :date)
        frac_i = frac_samples[mod1(si, n_samples)]
        for (ti, d) in enumerate(forecast_dates)
            row = findfirst(r -> r.date == d, eachrow(s_data))
            val = isnothing(row) ? 0.0 : s_data.value[row]
            mat[ti, si] = frac_i * val
        end
    end

    _matrix_to_summary(mat, forecast_dates, CrIs)
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
