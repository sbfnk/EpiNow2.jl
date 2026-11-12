# ── Multi-region estimation ───────────────────────────────────────────────

struct RegionalEpinowResult
    regional::Dict{String, Union{EpinowResult, Exception}}
    timings::Dict{String, Float64}
end

"""
    regional_epinow(data; non_zero_points = 7, target_folder = nothing,
                    verbose = false, kwargs...)

Run [`epinow`](@ref) for each region in the data, in parallel across Julia
threads (start Julia with `--threads` to use more than one). A region whose
fit fails is reported with a warning and does not stop the others.

# Arguments
- `data`: a `DataFrame` with `:date`, `:confirm` and `:region` columns.

# Keyword Arguments
- `non_zero_points`: minimum number of days with non-zero cases for a region
  to be fitted (default 7); regions with fewer are skipped.
- `target_folder`: directory in which to save each region's results, one
  subdirectory per region; `nothing` (default) saves nothing. Read them back
  with [`get_regional_results`](@ref).
- `verbose`: print progress messages (default `false`).
- `kwargs...`: passed to [`epinow`](@ref) for every region, e.g.
  `generation_time`, `delays` or `inference`.

# Returns
A `RegionalEpinowResult` whose `regional` field maps each region name to
its `EpinowResult` (or the error it raised). `summary(result)` gives the
latest estimates of every region with the probability that Rt is
decreasing.

```@example regional_epinow
using EpiNow2, DataFrames
cases = example_confirmed()[1:30, :]
data = vcat(
    insertcols(cases, :region => "north"), insertcols(cases, :region => "south")
)
result = regional_epinow(
    data;
    generation_time = gt_opts(example_generation_time()),
    delays = delay_opts(example_reporting_delay()),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1, progress = false)
)
summary(result)
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
                verbose = false, kwargs...
            )
            region_results[i] = result
            region_timings[i] = time() - t0
        catch e
            @warn "Region $region failed" exception = e
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

    return RegionalEpinowResult(results, timings)
end

function Base.summary(result::RegionalEpinowResult)
    # Cross-region summary: latest Rt, cases, growth rate per region
    # Includes probability of decrease and change category
    rows = DataFrame[]
    for (region, res) in result.regional
        res isa Exception && continue
        s = summary(res.estimates; type = :snapshot)
        s[!, :region] .= region

        # Add probability of decrease and change category for Rt
        p_dec = prob_decrease(res.estimates)
        s[!, :prob_decrease] .= p_dec
        s[!, :change] .= map_prob_change(p_dec)

        push!(rows, s)
    end
    isempty(rows) && return DataFrame()
    return vcat(rows...)
end
