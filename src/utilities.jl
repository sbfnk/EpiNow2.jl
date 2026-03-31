# ── Utility functions ─────────────────────────────────────────────────────

"""
    R_to_growth(R, gt_pmf; tol=1e-3, max_iter=100)

Convert reproduction number R to exponential growth rate r.
Solves R * Σ_k pmf[k] * exp(-r*k) = 1 via Newton's method.
"""
R_to_growth(R, gt_pmf; kwargs...) = _R_to_r(R, gt_pmf; kwargs...)

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
