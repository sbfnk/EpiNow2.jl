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
    estimate_secondary(data; secondary, delays, obs, inference, burn_in = 14,
                       CrIs = [0.2, 0.5, 0.9], verbose = true)

Estimate the relationship between a primary and a secondary observation,
e.g. cases and deaths or admissions and bed occupancy: the delay between
them and the fraction of primary observations that lead to secondary ones.

# Arguments
- `data`: a `DataFrame` with `:date`, `:primary` and `:secondary` columns.

# Keyword Arguments
- `secondary`: the secondary model, from [`secondary_opts`](@ref)
  (default: incidence).
- `delays`: prior on the delay from primary to secondary observation, from
  [`delay_opts`](@ref) (default: a lognormal with meanlog 2.5 and sdlog
  0.47).
- `obs`: observation model options from [`obs_opts`](@ref).
- `inference`: inference options from [`inference_opts`](@ref).
- `burn_in`: number of initial days left out of the likelihood, while the
  convolution builds up (default 14).
- `CrIs`: credible intervals to summarise (default `[0.2, 0.5, 0.9]`).
- `verbose`: print progress messages (default `true`).

# Returns
An `EstimateSecondaryResult` with the posterior predictive of the secondary
observations in `predictions` (also returned by [`get_predictions`](@ref))
and the posterior samples in `fit`.

```@example estimate_secondary
using EpiNow2, DataFrames
cases = example_confirmed()[1:30, :]
deaths = [round(Int, 0.1 * cases.confirm[max(1, t - 3)]) for t in 1:30]
data = DataFrame(date = cases.date, primary = cases.confirm, secondary = deaths)
fit = estimate_secondary(
    data;
    obs = obs_opts(week_effect = false),
    burn_in = 7,
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
last(get_predictions(fit), 5)
```
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

    verbose && @info "Estimating secondary observations..." type = secondary.type n_obs

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

    verbose && @info "Secondary estimation complete" seconds = round(elapsed, digits = 1)

    # `predictions` is the posterior predictive (latent expectation + observation
    # noise), matching R's `get_predictions()`. The latent mean is kept separately.
    predictions_expected = _summarise_secondary_gq(fit, sec_data.date, CrIs)
    predictions = _summarise_secondary_predictive(
        fit, sec_data.date, CrIs, obs.family
    )

    args = EstimateSecondaryArgs(
        secondary, delays, obs, inference, burn_in, CrIs
    )
    return EstimateSecondaryResult(
        fit, args, sec_data, predictions, predictions_expected, elapsed
    )
end

"""
    simulate_secondary(primary; delays, obs, secondary, frac = 0.1)

Simulate secondary observations from primary observations, using the model
of [`estimate_secondary`](@ref) with fixed parameters.

# Arguments
- `primary`: a `DataFrame` with `:date` and `:primary` columns.

# Keyword Arguments
- `delays`: delay from primary to secondary observation, from
  [`delay_opts`](@ref) (default: a lognormal with meanlog 2.5 and sdlog
  0.47).
- `obs`: observation model options from [`obs_opts`](@ref). Secondary
  observations are currently always drawn from a Poisson distribution.
- `secondary`: the secondary model, from [`secondary_opts`](@ref) (default:
  incidence).
- `frac`: fraction of primary observations that become secondary
  observations (default 0.1).

# Returns
A `DataFrame` with `:date`, `:primary` and `:secondary` columns.

```@example simulate_secondary
using EpiNow2, DataFrames
cases = example_confirmed()
primary = DataFrame(date = cases.date, primary = cases.confirm)
last(simulate_secondary(primary; frac = 0.05), 5)
```
"""
function simulate_secondary(
        primary::DataFrame;
        delays::DelayOpts = delay_opts(LogNormal(2.5, 0.47)),
        obs::ObsOpts = obs_opts(family = poisson),
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

    sim = [max(0, rand(Poisson(max(e, 1.0e-6)))) for e in expected]

    return DataFrame(date = dates, primary = prim, secondary = sim)
end

"""
    forecast_secondary(sec_result, inf_result; delays, secondary,
                       CrIs = [0.2, 0.5, 0.9])

Forecast secondary observations by applying the fraction estimated by
[`estimate_secondary`](@ref) to the forecast of reported cases from
[`estimate_infections`](@ref).

# Arguments
- `sec_result`: a fit from [`estimate_secondary`](@ref).
- `inf_result`: a fit from [`estimate_infections`](@ref) with a forecast
  horizon.

# Keyword Arguments
- `delays`: delay from primary to secondary observation, from
  [`delay_opts`](@ref) (default: a lognormal with meanlog 2.5 and sdlog
  0.47). Use the delay the secondary model was fitted with.
- `secondary`: the secondary model, from [`secondary_opts`](@ref) (default:
  incidence).
- `CrIs`: credible intervals to summarise (default `[0.2, 0.5, 0.9]`).

# Returns
A summary `DataFrame` of expected secondary observations over the forecast
horizon, or an empty `DataFrame` if `inf_result` has no forecast.

```@example forecast_secondary
using EpiNow2, DataFrames
inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false)
cases = example_confirmed()[1:30, :]
deaths = [round(Int, 0.1 * cases.confirm[max(1, t - 3)]) for t in 1:30]
data = DataFrame(date = cases.date, primary = cases.confirm, secondary = deaths)
sec = estimate_secondary(
    data; obs = obs_opts(week_effect = false), burn_in = 7, inference, verbose = false
)
inf = estimate_infections(
    cases;
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference, verbose = false
)
forecast_secondary(sec, inf)
```
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
    primary_samples = get_samples(inf_result; variable = :reports)
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

    return _matrix_to_summary(mat, forecast_dates, CrIs)
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
