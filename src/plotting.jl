# ── Plotting ──────────────────────────────────────────────────────────────
#
# CairoMakie-based plotting for EpiNow2 results.

using CairoMakie

# ── Distribution plots ────────────────────────────────────────────────────

"""
    plot(d::DelayDistribution; n_samples=50)

Plot a delay distribution. For fixed distributions, shows the discretised
PMF. For uncertain distributions, draws `n_samples` realisations from the
prior and shows the spread of PMFs.
"""
function CairoMakie.Makie.plot(d::Distribution; max::Union{Int, Nothing}=nothing)
    npd = discretise(d; max=max)
    _plot_pmf(npd)
end

function CairoMakie.Makie.plot(d::NonParametricDist)
    _plot_pmf(d)
end

function CairoMakie.Makie.plot(d::UncertainDistribution; n_samples::Int=50)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Uncertain $(_dist_family(d.constructor([mean(p) for p in d.param_priors]...))) distribution")

    max_val = Int(d.max)

    # Draw samples from priors and discretise each
    for _ in 1:n_samples
        sampled_params = [rand(p) for p in d.param_priors]
        try
            dist = d.constructor(sampled_params...)
            pmf = discretise(dist; max=max_val).pmf
            lines!(ax, 0:(length(pmf) - 1), pmf;
                   color=(:steelblue, 0.15), linewidth=1)
        catch
            continue
        end
    end

    # Show mean PMF
    mean_params = [mean(p) for p in d.param_priors]
    mean_dist = d.constructor(mean_params...)
    mean_pmf = discretise(mean_dist; max=max_val).pmf
    lines!(ax, 0:(length(mean_pmf) - 1), mean_pmf;
           color=:steelblue, linewidth=2.5)

    fig
end

function CairoMakie.Makie.plot(d::CompositeDelay; n_samples::Int=50)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Composite distribution")

    # For each sample, discretise all components and convolve
    for _ in 1:n_samples
        pmfs = Vector{Float64}[]
        ok = true
        for c in d.components
            if c isa UncertainDistribution
                sampled = [rand(p) for p in c.param_priors]
                try
                    dist = c.constructor(sampled...)
                    push!(pmfs, discretise(dist; max=Int(c.max)).pmf)
                catch
                    ok = false; break
                end
            else
                push!(pmfs, discretise(c).pmf)
            end
        end
        ok || continue
        conv = reduce(_convolve_pmfs, pmfs)
        lines!(ax, 0:(length(conv) - 1), conv;
               color=(:steelblue, 0.15), linewidth=1)
    end

    # Mean composite
    mean_pmf = discretise(d).pmf
    lines!(ax, 0:(length(mean_pmf) - 1), mean_pmf;
           color=:steelblue, linewidth=2.5)

    fig
end

function _plot_pmf(d::NonParametricDist)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Delay distribution")
    barplot!(ax, 0:(length(d.pmf) - 1), d.pmf; color=:steelblue)
    fig
end

# ── Result plots ─────────────────────────────────────────────────────────

"""
    plot(result::EstimateInfectionsResult; kwargs...)

Plot infections, reported cases, and Rt from an `estimate_infections` result.
Returns a Makie `Figure`.

# Keyword arguments
- `CrI::Float64=0.9` — credible interval level for the outer ribbon
- `forecast_date::Union{Date, Nothing}=nothing` — vertical line at forecast start
  (defaults to last observation date)
"""
function CairoMakie.Makie.plot(
    result::EstimateInfectionsResult;
    CrI::Float64=0.9,
    forecast_date::Union{Date, Nothing}=nothing
)
    fd = isnothing(forecast_date) ? result.observations.date[end] : forecast_date

    fig = Figure(size=(800, 700))

    _plot_panel!(fig[1, 1], result.infections, "Infections by date of infection";
                 CrI, forecast_date=fd)
    _plot_panel!(fig[2, 1], result.reports, "Reported cases by date of report";
                 CrI, forecast_date=fd,
                 observed=result.observations)
    _plot_panel!(fig[3, 1], result.rt, "Effective reproduction number (Rt)";
                 CrI, forecast_date=fd, hline=1.0)

    fig
end

function CairoMakie.Makie.plot(result::EpinowResult; kwargs...)
    plot(result.estimates; kwargs...)
end

"""
    _plot_panel!(pos, df, title; CrI, forecast_date, observed, hline)

Plot a single panel with median line, 50% and outer CrI ribbons.
"""
function _plot_panel!(
    pos, df::DataFrame, title::String;
    CrI::Float64=0.9,
    forecast_date::Union{Date, Nothing}=nothing,
    observed::Union{EpiData, Nothing}=nothing,
    hline::Union{Float64, Nothing}=nothing
)
    isempty(df) && return

    ax = Axis(pos;
        title=title,
        xlabel="Date",
        ylabel="",
        xticklabelrotation=π/6
    )

    dates = df.date
    date_nums = Dates.value.(dates .- dates[1])

    # Outer CrI ribbon
    cri_pct = round(Int, CrI * 100)
    lo_outer = Symbol("lower_$cri_pct")
    hi_outer = Symbol("upper_$cri_pct")
    if hasproperty(df, lo_outer) && hasproperty(df, hi_outer)
        band!(ax, date_nums, df[!, lo_outer], df[!, hi_outer];
              color=(:steelblue, 0.15), label="$(cri_pct)% CrI")
    end

    # 50% CrI ribbon
    if hasproperty(df, :lower_50) && hasproperty(df, :upper_50)
        band!(ax, date_nums, df.lower_50, df.upper_50;
              color=(:steelblue, 0.3), label="50% CrI")
    end

    # Median line
    lines!(ax, date_nums, df.median; color=:steelblue, linewidth=2, label="Median")

    # Observed data points
    if !isnothing(observed)
        obs_nums = Dates.value.(observed.date .- dates[1])
        scatter!(ax, obs_nums, Float64.(observed.confirm);
                 color=:black, markersize=4, label="Observed")
    end

    # Reference line
    if !isnothing(hline)
        hlines!(ax, [hline]; color=:grey40, linestyle=:dash, linewidth=1)
    end

    # Forecast line
    if !isnothing(forecast_date)
        fd_num = Dates.value(forecast_date - dates[1])
        vlines!(ax, [fd_num]; color=:grey40, linestyle=:dot, linewidth=1)
    end

    # Date tick labels
    tick_step = max(1, length(dates) ÷ 6)
    tick_idx = 1:tick_step:length(dates)
    ax.xticks = (date_nums[tick_idx], string.(dates[tick_idx]))

    ax
end
