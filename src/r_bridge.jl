# ── R bridge helpers ──────────────────────────────────────────────────────
#
# Functions in this file exist solely to be called from R via
# JuliaConnectoR. Previously these were defined as multi-line Julia source
# strings inlined inside R/convert.R and R/get.R, which is fragile: any
# rename or signature change on the Julia side surfaces only at runtime
# from R, with no way to lint or unit-test the helper. Defining them here
# keeps them under Julia's normal compile-time checks and `Pkg.test()`.

"""
    _r_bridge_convert_samples_df(result)

Render `get_samples(result)` with `Date` and `Symbol` columns coerced to
`String`. JuliaConnectoR can serialise strings cleanly but stumbles on
`Date` / `Symbol`; doing the coercion on the Julia side avoids brittle
post-hoc fixups in R.
"""
function _r_bridge_convert_samples_df(result)
    df = get_samples(result)
    df2 = copy(df)
    if hasproperty(df2, :date)
        df2.date = string.(df2.date)
    end
    if hasproperty(df2, :variable)
        df2.variable = string.(df2.variable)
    end
    df2
end

"""
    _r_bridge_get_secondary_samples(result)

Flatten the `expected` field of each generated quantity in an
`EstimateSecondaryResult` into a long-format DataFrame with columns
(date, variable, sample, value). Called from R's
`get_samples.estimate_secondary()`.
"""
function _r_bridge_get_secondary_samples(result)
    gqs = result.fit.generated_quantities
    dates = result.observations.date
    n_times = length(dates)

    rows = Vector{
        NamedTuple{(:date, :variable, :sample, :value),
                   Tuple{String, String, Int, Float64}}
    }()
    for (si, gq) in enumerate(gqs)
        expected = gq.expected
        for t in 1:min(length(expected), n_times)
            push!(rows, (
                date = string(dates[t]),
                variable = "sim_secondary",
                sample = si,
                value = Float64(expected[t]),
            ))
        end
    end
    DataFrame(rows)
end

# ── String → enum lookups ─────────────────────────────────────────────────
#
# R passes enum choices as plain strings (Symbols don't survive the
# JuliaConnectoR round-trip cleanly). Each helper takes the string the R
# converter produces and returns the corresponding Julia enum value.
# Throws an explicit error on unknown values so the failure message points
# at the bad option.

function _r_bridge_forecast_mode(s::AbstractString)
    s == "latest"   ? latest   :
    s == "project"  ? project  :
    s == "estimate" ? estimate :
    error("Unknown forecast mode '$s' (expected latest|project|estimate)")
end

function _r_bridge_gp_target(s::AbstractString)
    s == "gp_Rt" ? gp_Rt :
    s == "gp_R0" ? gp_R0 :
    error("Unknown gp_on '$s' (expected gp_Rt|gp_R0)")
end

function _r_bridge_pop_period(s::AbstractString)
    s == "pop_forecast" ? pop_forecast :
    s == "pop_all"      ? pop_all      :
    error("Unknown pop_period '$s' (expected pop_forecast|pop_all)")
end

function _r_bridge_backcalc_prior(s::AbstractString)
    s == "bc_infections"  ? bc_infections  :
    s == "bc_none"        ? bc_none        :
    s == "bc_growth_rate" ? bc_growth_rate :
    error("Unknown backcalc prior '$s' (expected bc_infections|bc_none|bc_growth_rate)")
end

function _r_bridge_obs_family(s::AbstractString)
    s == "negbin"  ? negbin  :
    s == "poisson" ? poisson :
    error("Unknown obs family '$s' (expected negbin|poisson)")
end

function _r_bridge_gp_kernel(s::AbstractString)
    s == "matern"   ? matern   :
    s == "se"       ? se       :
    s == "periodic" ? periodic :
    error("Unknown GP kernel '$s' (expected matern|se|periodic)")
end

function _r_bridge_secondary_type(s::AbstractString)
    s == "incidence"  ? incidence  :
    s == "prevalence" ? prevalence :
    error("Unknown secondary type '$s' (expected incidence|prevalence)")
end

# ── Option-builder helpers ────────────────────────────────────────────────
#
# Replace the sprintf-templated `_make_*_opts` pattern previously inlined
# in R/convert.R. Each helper takes primitives (numbers, strings, plus
# Distribution objects passed through JuliaConnectoR) and constructs the
# corresponding *Opts struct.

function _r_bridge_rt_opts(prior::Distribution; rw::Integer, future::AbstractString,
                           gp_on::AbstractString, pop, pop_period::AbstractString,
                           pop_floor::Real)
    rt_opts(
        prior = prior,
        use_rt = true,
        rw = Int(rw),
        future = _r_bridge_forecast_mode(future),
        gp_on = _r_bridge_gp_target(gp_on),
        pop = pop,
        pop_period = _r_bridge_pop_period(pop_period),
        pop_floor = Float64(pop_floor),
    )
end

function _r_bridge_gp_opts(ls::Distribution, alpha::Distribution;
                           basis_prop::Real, boundary_scale::Real,
                           kernel::AbstractString, matern_order::Real, w0::Real)
    gp_opts(
        basis_prop = Float64(basis_prop),
        boundary_scale = Float64(boundary_scale),
        ls = ls,
        alpha = alpha,
        kernel = _r_bridge_gp_kernel(kernel),
        matern_order = Float64(matern_order),
        w0 = Float64(w0),
    )
end

function _r_bridge_obs_opts(dispersion::Distribution, scale;
                            family::AbstractString, weight::Real,
                            week_effect::Bool, week_length::Integer,
                            likelihood::Bool)
    obs_opts(
        family = _r_bridge_obs_family(family),
        dispersion = dispersion,
        weight = Float64(weight),
        week_effect = week_effect,
        week_length = Int(week_length),
        scale = scale,
        likelihood = likelihood,
    )
end

function _r_bridge_backcalc_opts(; prior::AbstractString,
                                 prior_window::Integer, rt_window::Integer)
    backcalc_opts(
        prior = _r_bridge_backcalc_prior(prior),
        prior_window = Int(prior_window),
        rt_window = Int(rt_window),
    )
end

function _r_bridge_inference_opts(; samples::Integer, warmup::Integer,
                                  chains::Integer, target_acceptance::Real,
                                  max_treedepth::Integer,
                                  seed::Union{Integer, Nothing}=nothing,
                                  adtype::Union{ADTypes.AbstractADType, Nothing}=nothing)
    kwargs = Dict{Symbol, Any}(
        :samples => Int(samples),
        :warmup => Int(warmup),
        :chains => Int(chains),
        :target_acceptance => Float64(target_acceptance),
        :max_treedepth => Int(max_treedepth),
    )
    isnothing(seed)   || (kwargs[:seed]   = Int(seed))
    isnothing(adtype) || (kwargs[:adtype] = adtype)
    inference_opts(; kwargs...)
end

function _r_bridge_secondary_opts(; type::AbstractString)
    secondary_opts(_r_bridge_secondary_type(type))
end
