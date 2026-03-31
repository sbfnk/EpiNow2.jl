module EpiNow2CairoMakieExt

using CairoMakie
using Dates
using DataFrames
using Distributions
using Statistics

using EpiNow2
using EpiNow2: _dist_family, _convolve_pmfs,
               EstimateInfectionsResult, EpinowResult, EpiData

# ── Comma-formatted tick labels ──────────────────────────────────────────

function _comma_format(x::Real)
    n = round(Int, x)
    s = string(abs(n))
    parts = String[]
    while length(s) > 3
        push!(parts, s[end-2:end])
        s = s[1:end-3]
    end
    push!(parts, s)
    result = join(reverse(parts), ",")
    n < 0 ? "-" * result : result
end

# ── Distribution plots ────────────────────────────────────────────────────

function CairoMakie.Makie.plot(d::Distribution; max::Union{Int, Nothing}=nothing)
    _plot_pmf(discretise(d; max=max))
end

function CairoMakie.Makie.plot(d::NonParametricDist)
    _plot_pmf(d)
end

function CairoMakie.Makie.plot(d::UncertainDistribution; n_samples::Int=50)
    fig = Figure(size=(600, 400))
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Uncertain $(_dist_family(example)) distribution")

    max_val = Int(d.max)

    for _ in 1:n_samples
        sampled_params = [rand(p) for p in d.param_priors]
        try
            dist = d.constructor(sampled_params...)
            pmf = discretise(dist; max=max_val).pmf
            stairs!(ax, -0.5:(length(pmf) - 0.5), vcat(pmf, [0.0]);
                    color=(:steelblue, 0.15), linewidth=1, step=:post)
        catch
            continue
        end
    end

    mean_pmf = discretise(example; max=max_val).pmf
    stairs!(ax, -0.5:(length(mean_pmf) - 0.5), vcat(mean_pmf, [0.0]);
            color=:steelblue, linewidth=2.5, step=:post)

    fig
end

function CairoMakie.Makie.plot(d::CompositeDelay; n_samples::Int=50)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Composite distribution")

    for _ in 1:n_samples
        pmfs = Vector{Float64}[]
        ok = true
        for c in d.components
            if c isa UncertainDistribution
                sampled = [rand(p) for p in c.param_priors]
                try
                    push!(pmfs, discretise(c.constructor(sampled...); max=Int(c.max)).pmf)
                catch
                    ok = false; break
                end
            else
                push!(pmfs, discretise(c).pmf)
            end
        end
        ok || continue
        conv = reduce(_convolve_pmfs, pmfs)
        stairs!(ax, -0.5:(length(conv) - 0.5), vcat(conv, [0.0]);
                color=(:steelblue, 0.15), linewidth=1, step=:post)
    end

    mean_pmf = discretise(d).pmf
    stairs!(ax, -0.5:(length(mean_pmf) - 0.5), vcat(mean_pmf, [0.0]);
            color=:steelblue, linewidth=2.5, step=:post)

    fig
end

function _plot_pmf(d::NonParametricDist)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1]; xlabel="Delay", ylabel="Probability",
              title="Delay distribution")
    stairs!(ax, -0.5:(length(d.pmf) - 0.5), vcat(d.pmf, [0.0]);
            color=:steelblue, linewidth=2, step=:post)
    fig
end

# ── Result plots ─────────────────────────────────────────────────────────

function CairoMakie.Makie.plot(
    result::EstimateInfectionsResult;
    CrI::Float64=0.9,
    forecast_date::Union{Date, Nothing}=nothing
)
    fd = isnothing(forecast_date) ? result.observations.date[end] : forecast_date

    fig = Figure(size=(600, 800))

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
    CairoMakie.Makie.plot(result.estimates; kwargs...)
end

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
        xticklabelrotation=π/6,
        ytickformat=vs -> [_comma_format(v) for v in vs]
    )

    dates = df.date
    date_nums = Dates.value.(dates .- dates[1])

    # Outer CrI ribbon
    cri_pct = round(Int, CrI * 100)
    lo_outer = Symbol("lower_$cri_pct")
    hi_outer = Symbol("upper_$cri_pct")
    if hasproperty(df, lo_outer) && hasproperty(df, hi_outer)
        band!(ax, date_nums, df[!, lo_outer], df[!, hi_outer];
              color=(:steelblue, 0.15))
    end

    # 50% CrI ribbon
    if hasproperty(df, :lower_50) && hasproperty(df, :upper_50)
        band!(ax, date_nums, df.lower_50, df.upper_50;
              color=(:steelblue, 0.3))
    end

    # Median line
    lines!(ax, date_nums, df.median; color=:steelblue, linewidth=2)

    # Observed data points
    if !isnothing(observed)
        obs_nums = Dates.value.(observed.date .- dates[1])
        scatter!(ax, obs_nums, Float64.(observed.confirm);
                 color=:black, markersize=4)
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

# ── Reporting utilities ──────────────────────────────────────────────────

function EpiNow2.report_plots(
    result::EstimateInfectionsResult;
    CrI::Float64=0.9,
    forecast_date::Union{Date, Nothing}=nothing
)
    fd = isnothing(forecast_date) ? result.observations.date[end] : forecast_date

    plots = Dict{Symbol, Figure}()

    for (name, df, title, hline) in [
        (:infections, result.infections, "Infections by date of infection", nothing),
        (:reports, result.reports, "Reported cases by date of report", nothing),
        (:rt, result.rt, "Effective reproduction number (Rt)", 1.0),
        (:growth_rate, result.growth_rate, "Growth rate", 0.0)
    ]
        isempty(df) && continue
        fig = Figure(size=(600, 400))
        obs = name == :reports ? result.observations : nothing
        _plot_panel!(fig[1, 1], df, title; CrI, forecast_date=fd,
                     observed=obs, hline=hline)
        plots[name] = fig
    end

    plots
end

EpiNow2.report_plots(result::EpinowResult; kwargs...) =
    EpiNow2.report_plots(result.estimates; kwargs...)

function EpiNow2.plot_summary(
    result::EstimateInfectionsResult;
    CrI::Float64=0.9,
    forecast_date::Union{Date, Nothing}=nothing
)
    fd = isnothing(forecast_date) ? result.observations.date[end] : forecast_date

    fig = Figure(size=(900, 400))

    _plot_panel!(fig[1, 1], result.rt,
                 "Effective reproduction number (Rt)";
                 CrI, forecast_date=fd, hline=1.0)
    _plot_panel!(fig[1, 2], result.reports,
                 "Reported cases";
                 CrI, forecast_date=fd,
                 observed=result.observations)

    fig
end

EpiNow2.plot_summary(result::EpinowResult; kwargs...) =
    EpiNow2.plot_summary(result.estimates; kwargs...)

end # module
