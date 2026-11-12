# [CairoMakie extension](@id extension-cairo-makie)

`EpiNow2CairoMakieExt` is loaded automatically when CairoMakie is available alongside EpiNow2.

## Scope

Loading CairoMakie (`using EpiNow2, CairoMakie`) adds plotting for delay distributions and fitted models:

- `plot` on a distribution draws its discretised probability mass function. For an `UncertainDistribution` or a `CompositeDelay` with uncertain parts, it draws PMFs sampled from the parameter priors.
- `plot` on a fit from [`estimate_infections`](@ref) or [`epinow`](@ref) draws infections, reported cases and Rt in one figure, with estimates and forecasts separated at the forecast date.
- [`report_plots`](@ref) returns one figure per quantity (infections, reported cases, Rt and growth rate).
- [`plot_summary`](@ref) draws Rt and reported cases side by side.

The extension defines no names of its own: `report_plots` and `plot_summary` are documented with the rest of the API.
