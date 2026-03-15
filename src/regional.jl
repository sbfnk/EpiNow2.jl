# ── Multi-region estimation ───────────────────────────────────────────────

"""
    regional_epinow(data; ..., kwargs...)

Run `epinow()` across multiple regions in parallel.

# Arguments
- `data::DataFrame` — must contain `:date`, `:confirm`, and `:region` columns
- All other keyword arguments are passed to `epinow()` for each region

# Parallelism
Uses Julia's built-in threading (`Threads.@threads`) or `Distributed`
for multi-region parallelism. Set `JULIA_NUM_THREADS` or add workers.

# Returns
`RegionalEpinowResult` with per-region results and cross-region summaries.

# Example
```julia
data = DataFrame(
    date = repeat(dates, 3),
    confirm = [cases_a; cases_b; cases_c],
    region = [fill("A", n); fill("B", n); fill("C", n)]
)

result = regional_epinow(
    data,
    generation_time = gt_opts(
        LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14)
    ),
    delays = delay_opts(
        LogNormalSpec(meanlog=0.5, sdlog=0.5, max=14)
    )
)

# Cross-region summary
summary(result)

# Individual region
result.regional[:A]
```
"""
function regional_epinow(
    data::DataFrame;
    non_zero_points::Int = 2,
    verbose::Bool = false,
    kwargs...
)
    @assert :region in propertynames(data) "Data must have a :region column"

    regions = unique(data.region)
    n_regions = length(regions)

    verbose && @info "Running epinow for $n_regions regions..."

    # Run in parallel using Julia threads
    results = Dict{String, Union{EpinowResult, Exception}}()
    timings = Dict{String, Float64}()

    Threads.@threads for region in regions
        region_data = filter(r -> r.region == region, data)

        # Check minimum non-zero data points
        if sum(region_data.confirm .> 0) < non_zero_points
            @warn "Skipping region $region: insufficient non-zero data points"
            continue
        end

        try
            t0 = time()
            result = epinow(
                select(region_data, Not(:region));
                verbose=false, kwargs...
            )
            results[region] = result
            timings[region] = time() - t0
        catch e
            @warn "Region $region failed" exception=e
            results[region] = e
        end
    end

    RegionalEpinowResult(results, timings)
end

struct RegionalEpinowResult
    regional::Dict{String, Union{EpinowResult, Exception}}
    timings::Dict{String, Float64}
end

function Base.summary(result::RegionalEpinowResult)
    # Cross-region summary: latest Rt, cases, growth rate per region
    rows = []
    for (region, res) in result.regional
        res isa Exception && continue
        s = summary(res.estimates; type=:snapshot)
        s[!, :region] .= region
        push!(rows, s)
    end
    vcat(rows...)
end
