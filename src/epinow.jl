# ── Main entry points ─────────────────────────────────────────────────────

"""
    estimate_infections(data; generation_time, delays, truncation, rt,
                        backcalc, gp, obs, forecast, inference, CrIs,
                        verbose)

Estimate infections, the time-varying reproduction number and the growth
rate from reported cases, using a renewal equation (or back-calculation)
with delays from infection to report, an observation model and an
approximate Gaussian process prior.

# Arguments
- `data`: a `DataFrame` with `:date` and `:confirm` columns. Optional columns
  are `:accumulate` (see [`fill_missing`](@ref)) and `:breakpoints` (see
  [`add_breakpoints`](@ref)).

# Keyword Arguments
- `generation_time`: generation time options from [`gt_opts`](@ref).
- `delays`: delays from infection to report, from [`delay_opts`](@ref).
- `truncation`: right truncation of recent data, from
  [`trunc_opts`](@ref).
- `rt`: reproduction number options from [`rt_opts`](@ref). Use
  `rt_opts(use_rt = false)` to back-calculate infections instead.
- `backcalc`: back-calculation options from [`backcalc_opts`](@ref).
- `gp`: Gaussian process options from [`gp_opts`](@ref), or `nothing` to
  switch the Gaussian process off (for example with a random walk).
- `obs`: observation model options from [`obs_opts`](@ref).
- `forecast`: forecast options from [`forecast_opts`](@ref).
- `inference`: inference options from [`inference_opts`](@ref).
- `CrIs`: credible intervals to summarise (default `[0.2, 0.5, 0.9]`).
- `verbose`: print progress messages (default `true`).

# Returns
An `EstimateInfectionsResult` with posterior summaries by date in its
`infections`, `reports`, `rt` and `growth_rate` fields and the posterior
samples in `fit`. Use `summary(result)` for the latest estimates and
[`get_samples`](@ref) for the samples.

```@example estimate_infections
using EpiNow2
fit = estimate_infections(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(
        discretise(example_incubation_period() + example_reporting_delay())
    ),
    rt = rt_opts(rw = 7),
    gp = nothing,
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
summary(fit)
```
"""
function estimate_infections(
        data::DataFrame;
        generation_time::GTOpts = gt_opts(),
        delays::DelayOpts = delay_opts(),
        truncation::TruncOpts = trunc_opts(),
        rt::RtOpts = rt_opts(),
        backcalc::BackcalcOpts = backcalc_opts(),
        gp::Union{GPOpts, Nothing} = gp_opts(),
        obs::ObsOpts = obs_opts(),
        forecast::ForecastOpts = forecast_opts(),
        inference::InferenceOpts = inference_opts(),
        CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
        verbose::Bool = true
    )
    epi_data = EpiData(data)

    verbose && @info "Assembling model..." n_obs = length(epi_data)

    assembled = assemble_model(
        epi_data;
        generation_time, delays, truncation,
        rt, backcalc, gp, obs, forecast
    )

    verbose && @info "Running inference..." sampler = inference.sampler

    t0 = time()
    fit = run_inference(assembled.model, assembled.metadata, inference)
    elapsed = time() - t0

    verbose && @info "Inference complete" seconds = round(elapsed, digits = 1)

    if :gp_rho in names(fit.chain)
        check_gp_lengthscale(vec(Array(fit.chain[:gp_rho])), assembled.metadata)
    end

    args = EstimateInfectionsArgs(
        generation_time, delays, truncation, rt, backcalc, gp, obs,
        forecast, inference, CrIs,
    )
    return build_result(fit, epi_data, args, elapsed; CrIs)
end

"""
    epinow(data; generation_time, delays, truncation, rt, backcalc, gp, obs,
           forecast, inference, CrIs, target_folder = nothing,
           target_date = nothing, verbose)

Estimate infections, the time-varying reproduction number and the growth
rate, and forecast reported cases. A wrapper around
[`estimate_infections`](@ref) that can also save the results.

# Arguments
- `data`: a `DataFrame` with `:date` and `:confirm` columns, as for
  [`estimate_infections`](@ref).

# Keyword Arguments
- `generation_time`, `delays`, `truncation`, `rt`, `backcalc`, `gp`, `obs`,
  `forecast`, `inference`, `CrIs`, `verbose`: as for
  [`estimate_infections`](@ref).
- `target_folder`: directory in which to save the summaries of infections,
  reports, Rt and growth rate as CSV files; `nothing` (default) saves
  nothing.
- `target_date`: the `Date` the forecast horizon counts from. When it lies
  after the last date in `data`, the horizon is extended so the forecast
  still reaches `forecast.horizon` days beyond it, as in R. `nothing`
  (default) counts from the last date in `data`.

# Returns
An `EpinowResult` whose `estimates` field holds the
`EstimateInfectionsResult`.

```@example epinow
using EpiNow2
result = epinow(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
summary(result)
```
"""
function epinow(
        data::DataFrame;
        generation_time::GTOpts = gt_opts(),
        delays::DelayOpts = delay_opts(),
        truncation::TruncOpts = trunc_opts(),
        rt::RtOpts = rt_opts(),
        backcalc::BackcalcOpts = backcalc_opts(),
        gp::Union{GPOpts, Nothing} = gp_opts(),
        obs::ObsOpts = obs_opts(),
        forecast::ForecastOpts = forecast_opts(),
        inference::InferenceOpts = inference_opts(),
        CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
        target_folder::Union{String, Nothing} = nothing,
        target_date::Union{Date, Nothing} = nothing,
        verbose::Bool = true
    )
    t0 = time()

    # Forecast `forecast.horizon` days beyond the target date rather than
    # beyond the last date in the data, as R's `update_horizon()` does.
    if !isnothing(target_date) && forecast.horizon != 0
        forecast = forecast_opts(
            horizon = forecast.horizon +
                Dates.value(target_date - maximum(data.date))
        )
    end

    estimates = estimate_infections(
        data;
        generation_time, delays, truncation,
        rt, backcalc, gp, obs, forecast, inference, CrIs, verbose
    )

    elapsed = time() - t0
    result = EpinowResult(estimates, nothing, elapsed)

    if !isnothing(target_folder)
        _save_results(result, target_folder)
    end

    return result
end

function _save_results(result::EpinowResult, folder::String)
    mkpath(folder)
    est = result.estimates
    for (name, df) in [
            ("infections", est.infections), ("reports", est.reports),
            ("rt", est.rt), ("growth_rate", est.growth_rate),
        ]
        isempty(df) || CSV.write(joinpath(folder, "$name.csv"), df)
    end
    return
end
