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
