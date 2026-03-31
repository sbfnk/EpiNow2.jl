# ── Utility functions ─────────────────────────────────────────────────────

"""
    growth_to_R(r, gt_pmf)

Convert exponential growth rate r to reproduction number R.
R = 1 / Σ_k pmf[k] * exp(-r*k).
"""
function growth_to_R(r, gt_pmf)
    n = length(gt_pmf)
    1.0 / sum(gt_pmf[i] * exp(-r * i) for i in 1:n)
end

"""
    map_prob_change(prob_decrease)

Categorise the posterior probability that Rt is decreasing (i.e. Rt < 1).
Returns a descriptive string matching R's `map_prob_change()`.

- `prob_decrease < 0.05` → "Increasing"
- `prob_decrease < 0.4` → "Likely increasing"
- `prob_decrease < 0.6` → "Stable"
- `prob_decrease < 0.95` → "Likely decreasing"
- `prob_decrease ≤ 1.0` → "Decreasing"
"""
function map_prob_change(prob_decrease::Float64)
    if prob_decrease < 0.05
        "Increasing"
    elseif prob_decrease < 0.4
        "Likely increasing"
    elseif prob_decrease < 0.6
        "Stable"
    elseif prob_decrease < 0.95
        "Likely decreasing"
    else
        "Decreasing"
    end
end

"""
    prob_decrease(result; date=nothing)

Compute the posterior probability that Rt < 1 at a given date (default: latest).
"""
function prob_decrease(result::EstimateInfectionsResult; date=nothing)
    samples = get_samples(result; variable=:R)
    target_date = isnothing(date) ? result.observations.date[end] : date
    rt_samples = filter(r -> r.date == target_date, samples)
    isempty(rt_samples) && return NaN
    mean(rt_samples.value .< 1.0)
end

prob_decrease(result::EpinowResult; kwargs...) =
    prob_decrease(result.estimates; kwargs...)

"""
    simulate_infections(R_trajectory; generation_time, delays, obs,
                        initial_infections, pop, forecast, seed)

Forward-simulate infections and reports from a fixed Rt trajectory.
No inference — uses the renewal equation deterministically then draws
observations from the observation model.

# Arguments
- `R_trajectory::DataFrame` — must have `:date` and `:R` columns
- `generation_time::GTOpts` — generation time distribution
- `delays::DelayOpts` — reporting delay
- `obs::ObsOpts` — observation model
- `initial_infections::Float64` — initial infection count
- `pop::Float64` — population for depletion (0 = none)

# Returns
A `DataFrame` with `:date`, `:infections`, `:reports` columns.
"""
function simulate_infections(
    R_trajectory::DataFrame;
    generation_time::GTOpts = gt_opts(LogNormal(1.6, 0.5)),
    delays::DelayOpts = delay_opts(),
    obs::ObsOpts = obs_opts(family=:poisson),
    initial_infections::Float64 = 100.0,
    pop::Float64 = 0.0,
    seed::Union{Int, Nothing} = nothing
)
    !isnothing(seed) && Random.seed!(seed)

    sorted = sort(R_trajectory, :date)
    dates = Date.(sorted.date)
    R_vals = Float64.(sorted.R)
    n = length(dates)

    gt_pmf = _drop_zero_delay(discretise(generation_time.dist).pmf)
    delay_pmf = discretise(delays.dist).pmf
    gt_len = length(gt_pmf)

    # Seeding
    seeding_time = max(gt_len - 1, 1)
    growth = _R_to_r(R_vals[1], gt_pmf)
    initial = [initial_infections * exp(growth * (s - seeding_time))
               for s in 1:seeding_time]

    # Renewal equation
    infections = Vector{Float64}(undef, seeding_time + n)
    infections[1:seeding_time] .= initial
    cum_inf = sum(initial)

    for s in 1:n
        t = seeding_time + s
        infectiousness = sum(
            infections[t - j] * gt_pmf[j]
            for j in 1:min(t - 1, gt_len)
        )
        if pop > 0
            susceptible = max(1.0, pop - cum_inf)
            exp_adj = exp(-R_vals[s] * infectiousness / susceptible)
            infections[t] = susceptible * max(0.0, 1.0 - exp_adj)
        else
            infections[t] = R_vals[s] * infectiousness
        end
        cum_inf += infections[t]
    end

    post_seed = infections[(seeding_time + 1):end]

    # Convolve with delay
    all_reports = convolve(infections, delay_pmf)
    expected_reports = all_reports[(seeding_time + 1):end]

    # Sample observations
    reports = if obs.family == :negbin
        [max(0, rand(NegativeBinomial2(max(r, 1e-6), 5.0)))
         for r in expected_reports]
    else
        [max(0, rand(Poisson(max(r, 1e-6)))) for r in expected_reports]
    end

    DataFrame(date=dates, infections=post_seed, reports=reports)
end

"""
    forecast_infections(result, R_trajectory; generation_time, delays, obs, CrIs)

Forecast infections and reports by applying posterior parameter draws from a
fitted model to a new Rt trajectory. Uses the renewal equation with each
posterior sample's parameters.

# Arguments
- `result::EstimateInfectionsResult` — fitted model result
- `R_trajectory::DataFrame` — must have `:date` and `:R` columns for forecast period

# Returns
A `DataFrame` with summary statistics for forecasted reports.
"""
function forecast_infections(
    result::EstimateInfectionsResult,
    R_trajectory::DataFrame;
    generation_time::GTOpts = gt_opts(LogNormal(1.6, 0.5)),
    delays::DelayOpts = delay_opts(),
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    sorted_R = sort(R_trajectory, :date)
    forecast_dates = Date.(sorted_R.date)
    R_vals = Float64.(sorted_R.R)
    n_forecast = length(forecast_dates)

    gt_pmf = _drop_zero_delay(discretise(generation_time.dist).pmf)
    delay_pmf = discretise(delays.dist).pmf
    gt_len = length(gt_pmf)

    # Get the last infections from each posterior sample to seed the forecast
    gqs = result.fit.generated_quantities
    n_samples = length(gqs)
    params = get_parameters(result)

    mat = Matrix{Float64}(undef, n_forecast, n_samples)

    for (si, gq) in enumerate(gqs)
        # Use the tail of infections from this sample as history
        inf_history = Float64.(gq.infections)
        n_hist = length(inf_history)

        # Run renewal equation forward with the new R trajectory
        new_infections = Vector{Float64}(undef, n_forecast)
        for s in 1:n_forecast
            infectiousness = 0.0
            for j in 1:gt_len
                if s - j > 0
                    infectiousness += new_infections[s - j] * gt_pmf[j]
                else
                    # Reach back into history
                    hist_idx = n_hist + (s - j)
                    if hist_idx >= 1
                        infectiousness += inf_history[hist_idx] * gt_pmf[j]
                    end
                end
            end
            new_infections[s] = R_vals[s] * infectiousness
        end

        # Convolve with delay
        expected = convolve(new_infections, delay_pmf)

        for t in 1:n_forecast
            mat[t, si] = expected[t]
        end
    end

    _matrix_to_summary(mat, forecast_dates, CrIs)
end

forecast_infections(result::EpinowResult, args...; kwargs...) =
    forecast_infections(result.estimates, args...; kwargs...)

function _extract_delays(data::DataFrame; max_delay::Int)
    delays = if :delay in propertynames(data)
        Float64.(data.delay)
    elseif :date_onset in propertynames(data) && :date_report in propertynames(data)
        Float64.(Dates.value.(data.date_report .- data.date_onset))
    else
        throw(ArgumentError("Data must have :delay column or :date_onset/:date_report columns"))
    end
    valid = filter(d -> d > 0 && d <= max_delay, delays)
    isempty(valid) && throw(ArgumentError("No valid delays found"))
    valid
end

function _fit_family(family::Symbol, data::AbstractVector{Float64})
    if family == :lognormal
        fit(LogNormal, data)
    elseif family == :gamma
        fit(Gamma, data)
    else
        throw(ArgumentError("Unsupported family: $family. Use :lognormal or :gamma"))
    end
end

"""
    estimate_dist(data; family, max_delay)

Fit a delay distribution to line-list data using maximum likelihood.

# Arguments
- `data::DataFrame` — must have columns for event dates (e.g. `:date_onset`, `:date_report`)
  or a single `:delay` column with pre-computed delays
- `family::Symbol` — `:lognormal` (default) or `:gamma`
- `max_delay::Int` — maximum delay to consider (default: 30)

# Returns
A fitted `Distribution` (LogNormal or Gamma).
"""
function estimate_dist(
    data::DataFrame;
    family::Symbol = :lognormal,
    max_delay::Int = 30
)
    valid = _extract_delays(data; max_delay)
    _fit_family(family, valid)
end

"""
    bootstrapped_dist_fit(data; family, max_delay, n_bootstraps)

Fit a delay distribution with bootstrap uncertainty quantification.
Returns an `UncertainDistribution` with priors derived from the bootstrap
distribution of parameters.

# Arguments
- `data::DataFrame` — same format as `estimate_dist`
- `family::Symbol` — `:lognormal` or `:gamma`
- `n_bootstraps::Int` — number of bootstrap resamples (default: 100)

# Returns
An `UncertainDistribution` with Normal priors on parameters derived from
the bootstrap mean and standard deviation.
"""
function bootstrapped_dist_fit(
    data::DataFrame;
    family::Symbol = :lognormal,
    max_delay::Int = 30,
    n_bootstraps::Int = 100
)
    valid = _extract_delays(data; max_delay)
    n = length(valid)

    # Bootstrap
    param_samples = Vector{Vector{Float64}}(undef, n_bootstraps)
    for b in 1:n_bootstraps
        resample = valid[rand(1:n, n)]
        d = _fit_family(family, resample)
        param_samples[b] = collect(params(d))
    end

    n_params = length(param_samples[1])
    param_means = [mean(param_samples[b][i] for b in 1:n_bootstraps) for i in 1:n_params]
    param_sds = [std([param_samples[b][i] for b in 1:n_bootstraps]) for i in 1:n_params]

    constructor = if family == :lognormal
        (μ, σ) -> LogNormal(μ, σ)
    else
        (α, θ) -> Gamma(α, θ)
    end

    priors = Distribution[
        Normal(param_means[i], max(param_sds[i], 1e-4))
        for i in 1:n_params
    ]

    UncertainDistribution(constructor, priors, Float64(max_delay))
end

"""
    get_regional_results(folder; regions=nothing)

Load saved regional results from disk. Reads CSV files written by
`regional_epinow()` with `target_folder`.

# Arguments
- `folder::String` — path containing per-region subdirectories
- `regions` — specific regions to load (default: all subdirectories)

# Returns
A Dict mapping region names to Dict of DataFrames
(`:infections`, `:reports`, `:rt`, `:growth_rate`).
"""
function get_regional_results(
    folder::String;
    regions::Union{Vector{String}, Nothing} = nothing
)
    isdir(folder) || throw(ArgumentError("Directory not found: $folder"))

    available = filter(d -> isdir(joinpath(folder, d)), readdir(folder))
    load_regions = isnothing(regions) ? available : regions

    results = Dict{String, Dict{Symbol, DataFrame}}()
    for region in load_regions
        region_dir = joinpath(folder, region)
        isdir(region_dir) || continue

        region_data = Dict{Symbol, DataFrame}()
        for (name, file) in [
            (:infections, "infections.csv"),
            (:reports, "reports.csv"),
            (:rt, "rt.csv"),
            (:growth_rate, "growth_rate.csv")
        ]
            path = joinpath(region_dir, file)
            if isfile(path)
                region_data[name] = DataFrame(
                    CSV.File(path; dateformat="yyyy-mm-dd")
                )
            end
        end
        results[region] = region_data
    end

    results
end

"""
    opts_list(regions, opts; ...)

Create a Dict mapping region names to per-region options.

# Examples
```julia
# Same options for all regions
opts_list(["A", "B", "C"], rt_opts(rw=7))

# Different options per region
opts_list(Dict("A" => rt_opts(rw=7), "B" => rt_opts()))
```
"""
function opts_list(regions::AbstractVector, opts; kwargs...)
    Dict(string(r) => opts for r in regions)
end
function opts_list(d::Dict)
    Dict(string(k) => v for (k, v) in d)
end
