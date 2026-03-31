# ── Main entry points ─────────────────────────────────────────────────────

"""
    estimate_infections(data; generation_time, delays, truncation, rt,
                        backcalc, gp, obs, forecast, inference, verbose)

Estimate infections, Rt, and growth rate from reported case data.

# Arguments
- `data::DataFrame` — must contain `:date` and `:confirm` columns
- `generation_time::GTOpts` — generation time distribution
- `delays::DelayOpts` — reporting delay distribution(s)
- `truncation::TruncOpts` — right-truncation distribution
- `rt::RtOpts` — reproduction number options
- `backcalc::BackcalcOpts` — back-calculation options
- `gp::GPOpts` — Gaussian process options for Rt smoothing
- `obs::ObsOpts` — observation model options
- `forecast::ForecastOpts` — forecasting options
- `inference::InferenceOpts` — Turing.jl inference options

# Returns
`EstimateInfectionsResult` with `infections`, `reports`, `rt`,
`growth_rate` DataFrames and the raw `fit`.

# Example
```julia
result = estimate_infections(
    data,
    generation_time = gt_opts(LogNormal(1.6, 0.5)),
    delays = delay_opts(LogNormal(0.5, 0.5))
)
summary(result)
```
"""
function estimate_infections(
    data::DataFrame;
    generation_time::GTOpts = gt_opts(),
    delays::DelayOpts = delay_opts(),
    truncation::TruncOpts = trunc_opts(),
    rt::RtOpts = rt_opts(),
    backcalc::BackcalcOpts = backcalc_opts(),
    gp::GPOpts = gp_opts(),
    obs::ObsOpts = obs_opts(),
    forecast::ForecastOpts = forecast_opts(),
    inference::InferenceOpts = inference_opts(),
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    verbose::Bool = true
)
    epi_data = EpiData(data)

    verbose && @info "Assembling model..." n_obs=length(epi_data)

    assembled = assemble_model(
        epi_data;
        generation_time, delays, truncation,
        rt, backcalc, gp, obs, forecast
    )

    verbose && @info "Running inference..." sampler=inference.sampler

    t0 = time()
    fit = run_inference(assembled.model, assembled.metadata, inference)
    elapsed = time() - t0

    verbose && @info "Inference complete" seconds=round(elapsed, digits=1)

    build_result(fit, epi_data, elapsed; CrIs)
end

"""
    epinow(data; generation_time, delays, ..., CrIs, target_folder)

Estimate infections, Rt, growth rate, and forecast reported cases.
Wrapper around `estimate_infections` that adds output management.

# Example
```julia
result = epinow(
    data,
    generation_time = gt_opts(LogNormal(1.6, 0.5)),
    delays = delay_opts(LogNormal(0.5, 0.5))
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
    gp::GPOpts = gp_opts(),
    obs::ObsOpts = obs_opts(),
    forecast::ForecastOpts = forecast_opts(),
    inference::InferenceOpts = inference_opts(),
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    target_folder::Union{String, Nothing} = nothing,
    verbose::Bool = true
)
    t0 = time()

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

    result
end

function _save_results(result::EpinowResult, folder::String)
    mkpath(folder)
    est = result.estimates
    for (name, df) in [("infections", est.infections), ("reports", est.reports),
                        ("rt", est.rt), ("growth_rate", est.growth_rate)]
        isempty(df) || CSV.write(joinpath(folder, "$name.csv"), df)
    end
end
