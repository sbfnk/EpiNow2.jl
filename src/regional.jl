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
    generation_time = gt_opts(LogNormal(1.6, 0.5)),
    delays = delay_opts(LogNormal(0.5, 0.5))
)

# Cross-region summary
summary(result)

# Individual region
result.regional[:A]
```
"""
function regional_epinow(
    data::DataFrame;
    non_zero_points::Int = 7,
    target_folder::Union{String, Nothing} = nothing,
    verbose::Bool = false,
    kwargs...
)
    :region in propertynames(data) || throw(ArgumentError("Data must have a :region column"))

    regions = unique(data.region)
    n_regions = length(regions)

    verbose && @info "Running epinow for $n_regions regions..."

    # Run in parallel using Julia threads.
    # Collect into per-index vectors to avoid concurrent Dict writes.
    region_results = Vector{Union{EpinowResult, Exception, Nothing}}(
        nothing, n_regions
    )
    region_timings = Vector{Float64}(undef, n_regions)

    Threads.@threads for i in 1:n_regions
        region = regions[i]
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
            region_results[i] = result
            region_timings[i] = time() - t0
        catch e
            @warn "Region $region failed" exception=e
            region_results[i] = e
        end
    end

    # Collect into Dicts (single-threaded)
    results = Dict{String, Union{EpinowResult, Exception}}()
    timings = Dict{String, Float64}()
    for i in 1:n_regions
        isnothing(region_results[i]) && continue
        results[string(regions[i])] = region_results[i]
        timings[string(regions[i])] = region_timings[i]
    end

    # Save per-region results to disk
    if !isnothing(target_folder)
        for (region, res) in results
            res isa Exception && continue
            _save_results(res, joinpath(target_folder, region))
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
    # Includes probability of decrease and change category
    rows = []
    for (region, res) in result.regional
        res isa Exception && continue
        s = summary(res.estimates; type=:snapshot)
        s[!, :region] .= region

        # Add probability of decrease and change category for Rt
        p_dec = prob_decrease(res.estimates)
        s[!, :prob_decrease] .= p_dec
        s[!, :change] .= map_prob_change(p_dec)

        push!(rows, s)
    end
    isempty(rows) && return DataFrame()
    vcat(rows...)
end
