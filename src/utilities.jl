# ── Utility functions ─────────────────────────────────────────────────────

"""
    growth_to_R(r, gt_pmf)

Convert an exponential growth rate to a reproduction number, using
`R = 1 / M(-r)` where `M` is the moment generating function of the
generation time. The inverse of [`R_to_growth`](@ref).

# Arguments
- `r`: the daily exponential growth rate.
- `gt_pmf`: generation time probability mass function, with `gt_pmf[i]` the
  probability of a generation time of `i` days.

```@example growth_to_R
using EpiNow2
growth_to_R(0.1, [0.2, 0.5, 0.3])
```
"""
growth_to_R(r, gt_pmf) = 1.0 / _neg_mgf(r, gt_pmf)

"""
    convert_to_logmean(mean, sd)

Convert the mean and standard deviation of a distribution on the natural
scale to the log-mean (`meanlog`) of a lognormal distribution with those
moments. Mirrors EpiNow2's `convert_to_logmean()`.

# Arguments
- `mean`: mean on the natural scale.
- `sd`: standard deviation on the natural scale.

```@example convert_to_logmean
using EpiNow2
convert_to_logmean(5.2, 2.8)
```
"""
convert_to_logmean(mean, sd) = log(mean^2 / sqrt(sd^2 + mean^2))

"""
    convert_to_logsd(mean, sd)

Convert the mean and standard deviation of a distribution on the natural
scale to the log-standard deviation (`sdlog`) of a lognormal distribution
with those moments. Mirrors EpiNow2's `convert_to_logsd()`.

# Arguments
- `mean`: mean on the natural scale.
- `sd`: standard deviation on the natural scale.

```@example convert_to_logsd
using EpiNow2
convert_to_logsd(5.2, 2.8)
```
"""
convert_to_logsd(mean, sd) = sqrt(log(1 + (sd^2 / mean^2)))

"""
    add_breakpoints(data; dates = Date[])

Add breakpoints to the given dates. Returns a copy of `data` with a
`:breakpoints` column that is 1 on each of `dates` and 0 otherwise; an
existing `:breakpoints` column is kept and `dates` are added to it. Mirrors
EpiNow2's `add_breakpoints()`. Breakpoints allow a step change in Rt (see
[`rt_opts`](@ref)).

# Arguments
- `data`: a `DataFrame` with a `:date` column.

# Keyword Arguments
- `dates`: dates at which to add breakpoints. Each must be in `data`.

```@example add_breakpoints
using EpiNow2, Dates
data = example_confirmed()
bp = add_breakpoints(data; dates = [Date(2024, 3, 15)])
bp[13:17, :]
```
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
            throw(
            ArgumentError(
                "Breakpoint date(s) not found in data: $(missing_dates)"
            )
        )
        for d in dates
            out.breakpoints[out.date .== d] .= 1
        end
    end
    return out
end

"""
    fill_missing(data; missing_dates = :ignore, missing_obs = :ignore,
                 initial_accumulate = nothing, obs_column = :confirm)

Fill missing dates and observations in count data. Returns a copy of `data`
on a complete daily grid with an `:accumulate` column for
[`estimate_infections`](@ref): a row with `accumulate = true` has its
expected count added to the next row's rather than being observed on its
own. Mirrors EpiNow2's `fill_missing()`.

# Arguments
- `data`: a `DataFrame` with a `:date` column and the `obs_column`.

# Keyword Arguments
- `missing_dates`: what to do with dates absent from `data`: `:ignore`
  (default) adds them as missing, `:accumulate` adds them flagged as
  accumulated into the next observation, `:zero` adds them with a count of 0.
- `missing_obs`: what to do with dates present in `data` but with a missing
  count, with the same options as `missing_dates`.
- `initial_accumulate`: number of days the first observation covers,
  itself included. The data are extended back so that the days before it are
  accumulated into it. If `nothing` (default) and the data have a single
  fixed interval of more than one day, that interval is used.
- `obs_column`: name of the column holding the counts (default `:confirm`).

```@example fill_missing
using EpiNow2, DataFrames, Dates
weekly = DataFrame(
    date = Date(2024, 1, 1):Day(7):Date(2024, 1, 29),
    confirm = [50, 70, 65, 80, 60]
)
fill_missing(weekly; missing_dates = :accumulate)
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
        throw(
        ArgumentError(
            "`data` already has an `accumulate` column"
        )
    )
    missing_dates in (:ignore, :accumulate, :zero) ||
        throw(
        ArgumentError(
            "`missing_dates` must be :ignore, :accumulate, or :zero"
        )
    )
    missing_obs in (:ignore, :accumulate, :zero) ||
        throw(
        ArgumentError(
            "`missing_obs` must be :ignore, :accumulate, or :zero"
        )
    )

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
    stop = sorted.date[end]
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
    filter_leading_zeros(data; obs_column = :confirm)

Drop the rows before the first non-zero count. Mirrors EpiNow2's
`filter_leading_zeros()`, without its `by` argument for grouped data.

# Arguments
- `data`: a `DataFrame` with a `:date` column and the `obs_column`.

# Keyword Arguments
- `obs_column`: name of the column holding the counts (default `:confirm`).

```@example filter_leading_zeros
using EpiNow2, DataFrames, Dates
data = DataFrame(date = Date(2024, 1, 1) .+ Day.(0:4), confirm = [0, 0, 3, 0, 5])
filter_leading_zeros(data)
```
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
    first_pos = findfirst(
        x -> !ismissing(x) && x > 0,
        sorted[!, obs_column]
    )
    isnothing(first_pos) && return DataFrame(sorted[1:0, :])
    return DataFrame(sorted[first_pos:end, :])
end

"""
    map_prob_change(prob_decrease)

Describe the posterior probability that Rt is below 1 in words, as
EpiNow2's `map_prob_change()` does:

- below 0.05: "Increasing"
- below 0.4: "Likely increasing"
- below 0.6: "Stable"
- below 0.95: "Likely decreasing"
- otherwise: "Decreasing"

# Arguments
- `prob_decrease`: probability that Rt is below 1, e.g. from
  [`prob_decrease`](@ref).

```@example map_prob_change
using EpiNow2
map_prob_change(0.8)
```
"""
function map_prob_change(prob_decrease::Float64)
    return if prob_decrease < 0.05
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
    prob_decrease(result; date = nothing)

Posterior probability that Rt is below 1 on a given date.

# Arguments
- `result`: a fit from [`estimate_infections`](@ref) or [`epinow`](@ref).

# Keyword Arguments
- `date`: the date to evaluate; `nothing` (default) uses the last date of
  the data. Returns `NaN` if the date has no Rt estimate.

```@example prob_decrease
using EpiNow2
fit = estimate_infections(
    example_confirmed()[1:30, :];
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
prob_decrease(fit)
```
"""
function prob_decrease(result::EstimateInfectionsResult; date = nothing)
    samples = get_samples(result; variable = :R)
    target_date = isnothing(date) ? result.observations.date[end] : date
    rt_samples = filter(r -> r.date == target_date, samples)
    isempty(rt_samples) && return NaN
    return mean(rt_samples.value .< 1.0)
end

prob_decrease(result::EpinowResult; kwargs...) =
    prob_decrease(result.estimates; kwargs...)

"""
    simulate_infections(R_trajectory; generation_time, delays, obs,
                        initial_infections = 100.0, pop = 0.0, seed = nothing)

Simulate infections and reported cases from a given Rt trajectory. Infections
follow the renewal equation deterministically; reports are then drawn from
the observation model. No inference is done.

# Arguments
- `R_trajectory`: a `DataFrame` with `:date` and `:R` columns.

# Keyword Arguments
- `generation_time`: generation time options from [`gt_opts`](@ref)
  (default: a lognormal with meanlog 1.6 and sdlog 0.5).
- `delays`: delay options from [`delay_opts`](@ref) (default: no delay).
- `obs`: observation model options from [`obs_opts`](@ref) (default:
  Poisson). With `negbin`, the overdispersion `phi` is fixed at 5.
- `initial_infections`: number of infections on the last seeding day
  (default 100).
- `pop`: susceptible population size for depletion; 0 (default) means none.
- `seed`: random seed, or `nothing` (default).

# Returns
A `DataFrame` with `:date`, `:infections` and `:reports` columns.

```@example simulate_infections
using EpiNow2, DataFrames, Dates
R = DataFrame(date = Date(2024, 1, 1) .+ Day.(0:29), R = [fill(1.3, 15); fill(0.8, 15)])
simulate_infections(
    R;
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    seed = 1
)
```
"""
function simulate_infections(
        R_trajectory::DataFrame;
        generation_time::GTOpts = gt_opts(LogNormal(1.6, 0.5)),
        delays::DelayOpts = delay_opts(),
        obs::ObsOpts = obs_opts(family = poisson),
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
    initial = [
        initial_infections * exp(growth * (s - seeding_time))
            for s in 1:seeding_time
    ]

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
        [
            max(0, rand(NegativeBinomial2(max(r, 1.0e-6), 5.0)))
                for r in expected_reports
        ]
    else
        [max(0, rand(Poisson(max(r, 1.0e-6)))) for r in expected_reports]
    end

    return DataFrame(date = dates, infections = post_seed, reports = reports)
end

"""
    forecast_infections(result, R_trajectory; generation_time, delays,
                        CrIs = [0.2, 0.5, 0.9])

Forecast reported cases under a given Rt trajectory, starting from the
infections in each posterior sample of a fitted model. Infections follow the
renewal equation and are convolved with the delay; observation noise is not
added.

# Arguments
- `result`: a fit from [`estimate_infections`](@ref) or [`epinow`](@ref).
- `R_trajectory`: a `DataFrame` with `:date` and `:R` columns covering the
  forecast period.

# Keyword Arguments
- `generation_time`: generation time options from [`gt_opts`](@ref)
  (default: a lognormal with meanlog 1.6 and sdlog 0.5). Use the options the
  model was fitted with.
- `delays`: delay options from [`delay_opts`](@ref) (default: no delay).
- `CrIs`: credible intervals to summarise (default `[0.2, 0.5, 0.9]`).

# Returns
A summary `DataFrame` of expected reports by date, with `mean`, `median`,
`sd` and the credible interval columns.

```@example forecast_infections
using EpiNow2, DataFrames, Dates
gt = gt_opts(example_generation_time())
delays = delay_opts(example_reporting_delay())
data = example_confirmed()[1:30, :]
fit = estimate_infections(
    data;
    generation_time = gt, delays = delays,
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    verbose = false
)
R = DataFrame(date = data.date[end] .+ Day.(1:7), R = fill(0.9, 7))
forecast_infections(fit, R; generation_time = gt, delays = delays)
```
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

    return _matrix_to_summary(mat, forecast_dates, CrIs)
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
    return valid
end

function _fit_family(family::Symbol, data::AbstractVector{Float64})
    return if family == :lognormal
        fit(LogNormal, data)
    elseif family == :gamma
        fit(Gamma, data)
    else
        throw(ArgumentError("Unsupported family: $family. Use :lognormal or :gamma"))
    end
end

"""
    calc_CrI(samples; by = nothing, CrI = 0.9, value_col = :value)

Calculate a credible interval from long-format samples. Returns a
`DataFrame` with the grouping columns and `lower_X` and `upper_X`, where `X`
is `CrI` as a percentage. Mirrors EpiNow2's `calc_CrI()`.

# Arguments
- `samples`: a `DataFrame` of samples, e.g. from [`get_samples`](@ref).

# Keyword Arguments
- `by`: column name or names to group by; `nothing` (default) summarises all
  rows.
- `CrI`: the credible interval, between 0 and 1 (default 0.9).
- `value_col`: column holding the sample values (default `:value`).

```@example calc_CrI
using EpiNow2, DataFrames
samples = DataFrame(group = repeat(["a", "b"], 50), value = randn(100))
calc_CrI(samples; by = :group, CrI = 0.5)
```
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
            (
                lo_col => quantile(samples[!, value_col], lo),
                hi_col => quantile(samples[!, value_col], hi),
            )...
        )
    end
    by_vec = by isa Symbol ? [by] : by
    return combine(
        groupby(samples, by_vec),
        value_col => (v -> quantile(v, lo)) => lo_col,
        value_col => (v -> quantile(v, hi)) => hi_col,
    )
end

"""
    calc_CrIs(samples; by = nothing, CrIs = [0.2, 0.5, 0.9], value_col = :value)

Calculate several credible intervals from long-format samples. Returns a
`DataFrame` with the grouping columns and a `lower_X`/`upper_X` pair for each
interval. Mirrors EpiNow2's `calc_CrIs()`.

# Arguments
- `samples`: a `DataFrame` of samples, e.g. from [`get_samples`](@ref).

# Keyword Arguments
- `by`: column name or names to group by; `nothing` (default) summarises all
  rows.
- `CrIs`: the credible intervals, between 0 and 1 (default
  `[0.2, 0.5, 0.9]`).
- `value_col`: column holding the sample values (default `:value`).

```@example calc_CrIs
using EpiNow2, DataFrames
samples = DataFrame(group = repeat(["a", "b"], 50), value = randn(100))
calc_CrIs(samples; by = :group)
```
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
    return out
end

"""
    calc_summary_stats(samples; by = nothing, value_col = :value)

Calculate the median, mean and standard deviation of long-format samples.
Mirrors EpiNow2's `calc_summary_stats()`.

# Arguments
- `samples`: a `DataFrame` of samples, e.g. from [`get_samples`](@ref).

# Keyword Arguments
- `by`: column name or names to group by; `nothing` (default) summarises all
  rows.
- `value_col`: column holding the sample values (default `:value`).

```@example calc_summary_stats
using EpiNow2, DataFrames
samples = DataFrame(group = repeat(["a", "b"], 50), value = randn(100))
calc_summary_stats(samples; by = :group)
```
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
    return combine(
        groupby(samples, by_vec),
        value_col => median => :median,
        value_col => mean => :mean,
        value_col => std => :sd,
    )
end

"""
    calc_summary_measures(samples; by = nothing, CrIs = [0.2, 0.5, 0.9],
                          value_col = :value)

Calculate summary statistics and credible intervals of long-format samples,
joining [`calc_summary_stats`](@ref) and [`calc_CrIs`](@ref). Mirrors
EpiNow2's `calc_summary_measures()`.

# Arguments
- `samples`: a `DataFrame` of samples, e.g. from [`get_samples`](@ref).

# Keyword Arguments
- `by`: column name or names to group by; `nothing` (default) summarises all
  rows.
- `CrIs`: the credible intervals, between 0 and 1 (default
  `[0.2, 0.5, 0.9]`).
- `value_col`: column holding the sample values (default `:value`).

```@example calc_summary_measures
using EpiNow2, DataFrames
samples = DataFrame(group = repeat(["a", "b"], 50), value = randn(100))
calc_summary_measures(samples; by = :group)
```
"""
function calc_summary_measures(
        samples::DataFrame;
        by::Union{Symbol, Vector{Symbol}, Nothing} = nothing,
        CrIs::Vector{Float64} = [0.2, 0.5, 0.9],
        value_col::Symbol = :value,
    )
    stats = calc_summary_stats(samples; by, value_col)
    cris = calc_CrIs(samples; by, CrIs, value_col)
    if isnothing(by)
        return hcat(stats, cris)
    end
    by_vec = by isa Symbol ? [by] : by
    return innerjoin(stats, cris; on = by_vec)
end

"""
    bootstrapped_dist_fit(data; family = :lognormal, max_delay = 30,
                          n_bootstraps = 100)

Fit a delay distribution by maximum likelihood on bootstrap resamples of
the data, to quantify the uncertainty in its parameters.

# Arguments
- `data`: a `DataFrame` with either a `:delay` column or `:date_onset` and
  `:date_report` columns. Only delays above 0 and up to `max_delay` are used.

# Keyword Arguments
- `family`: the distribution to fit, `:lognormal` (default) or `:gamma`.
- `max_delay`: maximum delay in days, also used as the maximum of the
  returned distribution (default 30).
- `n_bootstraps`: number of bootstrap resamples (default 100).

# Returns
An [`UncertainDistribution`](@ref) with normal priors on the parameters,
taken from the mean and standard deviation of the bootstrap estimates.

```@example bootstrapped_dist_fit
using EpiNow2, DataFrames, Distributions
data = DataFrame(delay = round.(Int, rand(LogNormal(1.5, 0.5), 100)))
bootstrapped_dist_fit(data; n_bootstraps = 20)
```
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
        Normal(param_means[i], max(param_sds[i], 1.0e-4))
            for i in 1:n_params
    ]

    return UncertainDistribution(constructor, priors, Float64(max_delay))
end

"""
    get_regional_results(folder; regions = nothing)

Read the results that [`regional_epinow`](@ref) saved with `target_folder`.

# Arguments
- `folder`: directory holding one subdirectory per region.

# Keyword Arguments
- `regions`: names of the regions to read; `nothing` (default) reads every
  subdirectory.

# Returns
A `Dict` from region name to a `Dict` of `DataFrame`s with keys
`:infections`, `:reports`, `:rt` and `:growth_rate`.

```@example get_regional_results
using EpiNow2, DataFrames
cases = example_confirmed()[1:30, :]
data = vcat(
    insertcols(cases, :region => "north"), insertcols(cases, :region => "south")
)
folder = mktempdir()
regional_epinow(
    data;
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false),
    target_folder = folder
)
results = get_regional_results(folder)
results["north"][:rt]
```
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
                (:growth_rate, "growth_rate.csv"),
            ]
            path = joinpath(region_dir, file)
            if isfile(path)
                region_data[name] = DataFrame(
                    CSV.File(path; dateformat = "yyyy-mm-dd")
                )
            end
        end
        results[region] = region_data
    end

    return results
end

"""
    opts_list(regions, opts)
    opts_list(opts::Dict)

Create a `Dict` from region names to options, for options that differ by
region.

# Arguments
- `regions`: region names that all get the same `opts`.
- `opts`: the options shared by all regions, e.g. from [`rt_opts`](@ref); or
  a `Dict` from region to options, whose keys are converted to strings.

```@example opts_list
using EpiNow2
opts_list(["north", "south"], rt_opts(rw = 7))
```

```@example opts_list
opts_list(Dict("north" => rt_opts(rw = 7), "south" => rt_opts()))
```
"""
function opts_list(regions::AbstractVector, opts; kwargs...)
    return Dict(string(r) => opts for r in regions)
end
function opts_list(d::Dict)
    return Dict(string(k) => v for (k, v) in d)
end
