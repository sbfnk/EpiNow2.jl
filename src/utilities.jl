# ── Utility functions ─────────────────────────────────────────────────────

"""
    growth_to_R(r, gt_pmf)

Convert exponential growth rate r to reproduction number R.
R = 1 / G(r) where G is the negative MGF of the generation interval.
"""
growth_to_R(r, gt_pmf) = 1.0 / _neg_mgf(r, gt_pmf)

"""
    convert_to_logmean(mean, sd)

Convert a natural mean and standard deviation to the location parameter
(`meanlog`) of a lognormal distribution. Mirrors EpiNow2 R's
`convert_to_logmean()`.
"""
convert_to_logmean(mean, sd) = log(mean^2 / sqrt(sd^2 + mean^2))

"""
    convert_to_logsd(mean, sd)

Convert a natural mean and standard deviation to the scale parameter
(`sdlog`) of a lognormal distribution. Mirrors EpiNow2 R's
`convert_to_logsd()`.
"""
convert_to_logsd(mean, sd) = sqrt(log(1 + (sd^2 / mean^2)))

"""
    add_breakpoints(data; dates=Date[])

Return a copy of `data` with a `:breakpoints` column flagging each
listed date with `1` (others `0`). If a `:breakpoints` column already
exists it is preserved and the listed dates are merged in. Mirrors
EpiNow2 R's `add_breakpoints()`.
"""
function add_breakpoints(data::DataFrame; dates::AbstractVector{Date} = Date[])
    :date in propertynames(data) ||
        throw(ArgumentError("`data` must have a `:date` column"))
    out = copy(data)
    if !(:breakpoints in propertynames(out))
        out.breakpoints = zeros(Int, nrow(out))
    end
    if !isempty(dates)
        missing_dates = setdiff(dates, out.date)
        isempty(missing_dates) ||
            throw(ArgumentError(
                "Breakpoint date(s) not found in data: $(missing_dates)"
            ))
        for d in dates
            out.breakpoints[out.date .== d] .= 1
        end
    end
    return out
end

"""
    fill_missing(data; missing_dates=:ignore, missing_obs=:ignore,
                 initial_accumulate=nothing, obs_column=:confirm)

Fill date gaps and/or missing observations in a long-format counts
DataFrame, returning a copy with an `accumulate` column suitable for
passing to `estimate_infections()`. Mirrors EpiNow2 R's `fill_missing()`.

`missing_dates` and `missing_obs` each take one of:

- `:ignore`  — leave gaps / missing values as is
- `:accumulate` — flag the gap (or missing) row as `accumulate=true`,
  meaning the model rolls its expected report forward into the next
  non-accumulate observation
- `:zero`    — insert (or replace) with `0`

`initial_accumulate` extends the data backwards by that many days at
the start, all flagged as `accumulate=true`. If omitted and the data
have a single fixed date interval > 1 day, that interval is used
automatically.

# Example
```julia
# Weekly aggregate data → daily grid with the first six days of each
# week flagged as accumulated
fill_missing(
    DataFrame(date = Date(2024,1,1):Day(7):Date(2024,1,29),
              confirm = [50, 70, 65, 80, 60]);
    missing_dates = :accumulate,
)
```
"""
function fill_missing(
    data::DataFrame;
    missing_dates::Symbol = :ignore,
    missing_obs::Symbol = :ignore,
    initial_accumulate::Union{Int, Nothing} = nothing,
    obs_column::Symbol = :confirm,
)
    :date in propertynames(data) ||
        throw(ArgumentError("`data` must have a `:date` column"))
    obs_column in propertynames(data) ||
        throw(ArgumentError("`data` must have a `:$obs_column` column"))
    :accumulate in propertynames(data) &&
        throw(ArgumentError(
            "`data` already has an `accumulate` column"
        ))
    missing_dates in (:ignore, :accumulate, :zero) ||
        throw(ArgumentError(
            "`missing_dates` must be :ignore, :accumulate, or :zero"
        ))
    missing_obs in (:ignore, :accumulate, :zero) ||
        throw(ArgumentError(
            "`missing_obs` must be :ignore, :accumulate, or :zero"
        ))

    sorted = sort(data, :date)

    # Auto-detect a fixed reporting interval > 1 day for initial_accumulate
    init = initial_accumulate
    if isnothing(init) && nrow(sorted) > 1
        diffs = unique(diff(sorted.date))
        if length(diffs) == 1 && Dates.value(diffs[1]) > 1
            init = Dates.value(diffs[1])
        end
    end
    init_n = isnothing(init) ? 1 : init
    init_n >= 1 ||
        throw(ArgumentError("`initial_accumulate` must be ≥ 1"))

    start = sorted.date[1] - Day(init_n - 1)
    stop  = sorted.date[end]
    full_dates = collect(start:Day(1):stop)

    present = Set(sorted.date)
    obs_lookup = Dict(d => v for (d, v) in zip(sorted.date, sorted[!, obs_column]))

    obs_eltype = eltype(sorted[!, obs_column])
    fill_val = obs_eltype <: Union{Missing, Real} ? missing : zero(obs_eltype)

    out = DataFrame(date = full_dates)
    out[!, obs_column] = Vector{Union{obs_eltype, Missing}}(undef, length(full_dates))
    out.accumulate = falses(length(full_dates))

    for (i, d) in enumerate(full_dates)
        if d in present
            v = obs_lookup[d]
            out[i, obs_column] = v
            if ismissing(v)
                if missing_obs == :zero
                    out[i, obs_column] = zero(obs_eltype)
                elseif missing_obs == :accumulate
                    out.accumulate[i] = true
                end
            end
        else
            # Date was missing in input
            if missing_dates == :zero
                out[i, obs_column] = zero(obs_eltype)
            elseif missing_dates == :accumulate
                out[i, obs_column] = missing
                out.accumulate[i] = true
            else  # :ignore — but we already added the row; restore as missing
                out[i, obs_column] = missing
            end
        end
    end

    # When detecting interval automatically and init was applied, also
    # mark the initial padded days as accumulate (so they roll into the
    # first observation rather than being dropped from the likelihood).
    if !isnothing(init)
        for i in 1:(init_n - 1)
            out.accumulate[i] = true
        end
    end

    return out
end

"""
    filter_leading_zeros(data; obs_column=:confirm)

Drop rows from the start of `data` until the first day on which
`obs_column` is non-zero. Mirrors EpiNow2 R's `filter_leading_zeros()`
(without the `by =` group argument; add it when needed).
"""
function filter_leading_zeros(
    data::DataFrame;
    obs_column::Symbol = :confirm,
)
    :date in propertynames(data) ||
        throw(ArgumentError("`data` must have a `:date` column"))
    obs_column in propertynames(data) ||
        throw(ArgumentError("`data` must have a `:$obs_column` column"))
    sorted = sort(data, :date)
    first_pos = findfirst(x -> !ismissing(x) && x > 0,
                          sorted[!, obs_column])
    isnothing(first_pos) && return DataFrame(sorted[1:0, :])
    return DataFrame(sorted[first_pos:end, :])
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
    obs::ObsOpts = obs_opts(family=poisson),
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
    growth = R_to_growth(R_vals[1], gt_pmf)
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
    reports = if obs.family == negbin
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
    calc_CrI(samples; by=nothing, CrI=0.9, value_col=:value)

Compute a single credible interval for a long-format `samples` DataFrame
(columns include `value_col`, optionally a grouping column `by`).
Returns a DataFrame with the grouping columns plus `lower_X` and
`upper_X`, where `X = round(100*CrI)`. Mirrors EpiNow2 R's `calc_CrI()`.
"""
function calc_CrI(
    samples::DataFrame;
    by::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
    CrI::Float64 = 0.9,
    value_col::Symbol = :value,
)
    half = CrI / 2
    lo, hi = 0.5 - half, 0.5 + half
    pct = round(Int, 100 * CrI)
    lo_col, hi_col = Symbol("lower_$pct"), Symbol("upper_$pct")

    if isnothing(by)
        return DataFrame(
            (lo_col => quantile(samples[!, value_col], lo),
             hi_col => quantile(samples[!, value_col], hi))...
        )
    end
    by_vec = by isa Symbol ? [by] : by
    combine(groupby(samples, by_vec),
        value_col => (v -> quantile(v, lo)) => lo_col,
        value_col => (v -> quantile(v, hi)) => hi_col,
    )
end

"""
    calc_CrIs(samples; by=nothing, CrIs=[0.2, 0.5, 0.9], value_col=:value)

Compute multiple credible intervals at once, returning a DataFrame with
the grouping columns plus pairs of `lower_X` / `upper_X` columns for
each CrI in `CrIs`. Mirrors EpiNow2 R's `calc_CrIs()`.
"""
function calc_CrIs(
    samples::DataFrame;
    by::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    value_col::Symbol = :value,
)
    sorted_CrIs = sort(CrIs)
    pieces = [
        calc_CrI(samples; by = by, CrI = cri, value_col = value_col)
        for cri in sorted_CrIs
    ]
    if isnothing(by)
        return hcat(pieces...)
    end
    by_vec = by isa Symbol ? [by] : by
    out = pieces[1]
    for p in pieces[2:end]
        out = innerjoin(out, p, on = by_vec)
    end
    out
end

"""
    calc_summary_stats(samples; by=nothing, value_col=:value)

Compute median, mean and standard deviation for a long-format `samples`
DataFrame. Mirrors EpiNow2 R's `calc_summary_stats()`.
"""
function calc_summary_stats(
    samples::DataFrame;
    by::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
    value_col::Symbol = :value,
)
    if isnothing(by)
        v = samples[!, value_col]
        return DataFrame(median = median(v), mean = mean(v), sd = std(v))
    end
    by_vec = by isa Symbol ? [by] : by
    combine(groupby(samples, by_vec),
        value_col => median => :median,
        value_col => mean   => :mean,
        value_col => std    => :sd,
    )
end

"""
    calc_summary_measures(samples; by=nothing, CrIs=[0.2, 0.5, 0.9],
                          value_col=:value)

Join `calc_summary_stats()` and `calc_CrIs()` into a single summary
DataFrame. Mirrors EpiNow2 R's `calc_summary_measures()`.
"""
function calc_summary_measures(
    samples::DataFrame;
    by::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
    value_col::Symbol = :value,
)
    stats = calc_summary_stats(samples; by, value_col)
    cris  = calc_CrIs(samples; by, CrIs, value_col)
    if isnothing(by)
        return hcat(stats, cris)
    end
    by_vec = by isa Symbol ? [by] : by
    innerjoin(stats, cris; on = by_vec)
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
