# ── Plotting ──────────────────────────────────────────────────────────────
#
# Optional plotting via Makie.jl (or a lightweight RecipesBase approach).
# Mirrors EpiNow2's plot.estimate_infections() output.

using RecipesBase

"""
    plot(result::EstimateInfectionsResult; type=:summary)

Plot estimation results.

# Types
- `:summary` — 3-panel plot: infections, Rt, reported cases
- `:infections` — infections by date of infection
- `:rt` — time-varying reproduction number
- `:reports` — reported cases with predictions
- `:growth_rate` — growth rate over time
"""
@recipe function f(
    result::EstimateInfectionsResult;
    type::Symbol = :summary
)
    # This uses RecipesBase so it works with any Plots.jl backend
    # (or can be adapted for Makie)

    if type == :summary
        layout := (3, 1)

        # Panel 1: Infections
        @series begin
            subplot := 1
            title := "Infections by date of infection"
            _plot_timeseries(result.infections)
        end

        # Panel 2: Rt
        @series begin
            subplot := 2
            title := "Effective reproduction number"
            # Add reference line at Rt = 1
            _plot_timeseries(result.rt; hline=1.0)
        end

        # Panel 3: Reported cases
        @series begin
            subplot := 3
            title := "Reported cases"
            _plot_timeseries(result.reports)
        end
    end
end

@recipe function f(result::EpinowResult; kwargs...)
    result.estimates
end

function _plot_timeseries(df; hline=nothing)
    # Returns data for ribbon plot with median + CrIs
    # Expects columns: date, median, lower_50, upper_50, lower_90, upper_90
    # TODO: implement
    nothing
end
