# ── Result extraction and summarisation ───────────────────────────────────
#
# Maps Turing generated quantities (NamedTuples of infections, R, reports)
# back to dated DataFrames with credible intervals.

"""
    EstimateInfectionsResult

Result of `estimate_infections()`.
"""
struct EstimateInfectionsResult
    fit::EpiNow2Fit
    observations::EpiData
    infections::DataFrame     # by date of infection
    reports::DataFrame        # by date of report
    rt::DataFrame             # reproduction number by date
    growth_rate::DataFrame    # growth rate by date
    timing::Float64           # seconds
end

"""
    EpinowResult

Result of `epinow()`.
"""
struct EpinowResult
    estimates::EstimateInfectionsResult
    forecasts::Union{DataFrame, Nothing}
    timing::Float64
end

# ── Show methods ──────────────────────────────────────────────────────────

function _latest_estimate(df::DataFrame)
    isempty(df) && return ""
    r = df[end, :]
    "$(round(r.median, digits=2)) (90% CrI: $(round(r.lower_90, digits=2))-$(round(r.upper_90, digits=2)))"
end

function Base.show(io::IO, ::MIME"text/plain", r::EstimateInfectionsResult)
    n_obs = length(r.observations)
    d1 = r.observations.date[1]
    d_end = r.observations.date[end]
    n_forecast = nrow(r.rt) - n_obs
    println(io, "EpiNow2 estimate_infections result")
    println(io, "  Data: $n_obs days ($d1 to $d_end)")
    n_forecast > 0 && println(io, "  Forecast: $n_forecast days")
    println(io, "  Timing: $(round(r.timing, digits=1))s")
    println(io)
    println(io, "  Latest Rt: $(_latest_estimate(r.rt))")
    println(io, "  Latest infections: $(_latest_estimate(r.infections))")
    println(io, "  Latest reports: $(_latest_estimate(r.reports))")
end

function Base.show(io::IO, r::EstimateInfectionsResult)
    print(io, "EstimateInfectionsResult($(length(r.observations)) obs, " *
          "Rt=$(_latest_estimate(r.rt)))")
end

function Base.show(io::IO, ::MIME"text/plain", r::EpinowResult)
    show(io, MIME("text/plain"), r.estimates)
end

# ── Accessors ────────────────────────────────────────────────────────────

"""
    get_samples(result; variable=nothing) -> DataFrame

Extract raw posterior samples as a long-format DataFrame.

Columns: `date`, `variable`, `sample`, `value`
"""
function get_samples(result::EstimateInfectionsResult; variable=nothing)
    fit = result.fit
    meta = fit.metadata
    gqs = fit.generated_quantities

    dates = _output_dates(meta)
    rows = NamedTuple{(:date, :variable, :sample, :value),
                      Tuple{Date, Symbol, Int, Float64}}[]

    for (i, gq) in enumerate(gqs)
        for (var, values) in pairs(gq)
            var_sym = Symbol(var)
            (!isnothing(variable) && var_sym != variable) && continue
            for (t, v) in enumerate(values)
                t > length(dates) && continue
                push!(rows, (
                    date=dates[t], variable=var_sym,
                    sample=i, value=Float64(v)
                ))
            end
        end
    end

    DataFrame(rows)
end

get_samples(result::EpinowResult; kwargs...) =
    get_samples(result.estimates; kwargs...)

"""
    get_predictions(result; format=:summary, CrIs=[0.2, 0.5, 0.9])

Extract case predictions.
"""
function get_predictions(
    result::EstimateInfectionsResult;
    format::Symbol=:summary,
    CrIs::Vector{Float64}=[0.2, 0.5, 0.9]
)
    if format == :summary
        return result.reports
    elseif format == :sample
        return get_samples(result; variable=:reports)
    elseif format == :quantile
        samples = get_samples(result; variable=:reports)
        return _samples_to_quantiles(samples, CrIs)
    end
end

"""
    get_parameters(result) -> Dict{Symbol, Any}

Extract fitted scalar parameters from the chain.
"""
get_parameters(result::EpinowResult) = get_parameters(result.estimates)

function get_parameters(result)
    chain = result.fit.chain
    params = Dict{Symbol, Any}()
    for name in names(chain)
        # Skip array parameters (infections[1], etc.)
        occursin("[", string(name)) && continue
        params[Symbol(name)] = vec(Array(chain[name]))
    end
    params
end

# ── Summary ──────────────────────────────────────────────────────────────

function Base.summary(
    result::EstimateInfectionsResult;
    type::Symbol=:snapshot,
    target_date::Union{Date, Nothing}=nothing,
    CrIs::Vector{Float64}=[0.2, 0.5, 0.9]
)
    if type == :snapshot
        date = isnothing(target_date) ? result.observations.date[end] :
            target_date
        _snapshot_summary(result, date, CrIs)
    elseif type == :parameters
        _parameter_summary(result, CrIs)
    end
end

function Base.summary(result::EpinowResult; kwargs...)
    summary(result.estimates; kwargs...)
end

# ── Internal: build result from generated quantities ─────────────────────

function build_result(
    fit::EpiNow2Fit, data::EpiData, elapsed::Float64;
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    dates = _output_dates(fit.metadata)

    infections = _summarise_gq(fit, :infections, dates, CrIs)
    reports = _summarise_gq(fit, :reports, dates, CrIs)
    rt_df = _summarise_gq(fit, :R, dates, CrIs)

    # Growth rate: log-diff of infections
    growth = _compute_growth_rate(fit, dates, CrIs)

    EstimateInfectionsResult(
        fit, data, infections, reports, rt_df, growth, elapsed
    )
end

function _output_dates(meta::ModelMetadata)
    n_obs = length(meta.dates)
    last_date = meta.dates[end]
    forecast_dates = [last_date + Day(i) for i in 1:meta.horizon]
    vcat(meta.dates, forecast_dates)
end

"""
Extract a named field from generated quantities across all samples,
compute summary statistics.
"""
function _summarise_gq(
    fit::EpiNow2Fit, field::Symbol,
    dates::Vector{Date}, CrIs::Vector{Float64}
)
    gqs = fit.generated_quantities
    n_samples = length(gqs)

    # Not all GQs may have the field (model variants)
    haskey(first(gqs), field) || return DataFrame()

    n_times = length(first(gqs)[field])
    n_dates = min(n_times, length(dates))

    # Collect samples into matrix (n_dates × n_samples)
    mat = Matrix{Float64}(undef, n_dates, n_samples)
    for (i, gq) in enumerate(gqs)
        vals = gq[field]
        for t in 1:n_dates
            mat[t, i] = Float64(vals[t])
        end
    end

    _matrix_to_summary(mat, dates[1:n_dates], CrIs)
end

function _matrix_to_summary(
    mat::Matrix{Float64}, dates::Vector{Date}, CrIs::Vector{Float64}
)
    n = size(mat, 1)
    rows = Vector{NamedTuple}(undef, n)

    for t in 1:n
        vals = mat[t, :]
        row = (
            date = dates[t],
            mean = mean(vals),
            median = median(vals),
            sd = std(vals)
        )

        # Add CrI columns
        for cri in CrIs
            lo = quantile(vals, (1 - cri) / 2)
            hi = quantile(vals, (1 + cri) / 2)
            pct = round(Int, cri * 100)
            row = merge(row, NamedTuple{(
                Symbol("lower_$pct"), Symbol("upper_$pct")
            )}((lo, hi)))
        end

        rows[t] = row
    end

    DataFrame(rows)
end

function _compute_growth_rate(
    fit::EpiNow2Fit, dates::Vector{Date}, CrIs::Vector{Float64}
)
    gqs = fit.generated_quantities
    n_samples = length(gqs)

    !haskey(first(gqs), :infections) && return DataFrame()

    n_times = length(first(gqs)[:infections])
    n_dates = min(n_times, length(dates))
    n_growth = n_dates - 1

    mat = Matrix{Float64}(undef, n_growth, n_samples)
    for (i, gq) in enumerate(gqs)
        inf = gq[:infections]
        for t in 1:n_growth
            mat[t, i] = log(Float64(inf[t + 1])) - log(Float64(inf[t]))
        end
    end

    _matrix_to_summary(mat, dates[2:n_dates], CrIs)
end

function _samples_to_quantiles(samples::DataFrame, CrIs::Vector{Float64})
    isempty(samples) && return DataFrame()
    probs = Float64[]
    for cri in CrIs
        push!(probs, (1 - cri) / 2)
        push!(probs, (1 + cri) / 2)
    end
    sort!(unique!(probs))

    grouped = groupby(samples, :date)
    rows = NamedTuple[]
    for g in grouped
        vals = g.value
        row = (date = g.date[1],)
        for p in probs
            pct = round(Int, p * 100)
            row = merge(row, NamedTuple{(Symbol("q$pct"),)}((quantile(vals, p),)))
        end
        push!(rows, row)
    end
    DataFrame(rows)
end

function _snapshot_summary(result, date, CrIs)
    # Filter summaries to target date
    rows = []
    for (name, df) in [
        (:infections, result.infections),
        (:rt, result.rt),
        (:growth_rate, result.growth_rate),
        (:reports, result.reports)
    ]
        isempty(df) && continue
        row = filter(r -> r.date == date, df)
        if !isempty(row)
            r = first(eachrow(row))
            push!(rows, merge((variable=name,), NamedTuple(r)))
        end
    end
    DataFrame(rows)
end

function _parameter_summary(result, CrIs)
    vcat(
        insertcols!(copy(result.infections), 1, :variable => :infections),
        insertcols!(copy(result.rt), 1, :variable => :rt),
        insertcols!(copy(result.growth_rate), 1, :variable => :growth_rate),
        insertcols!(copy(result.reports), 1, :variable => :reports);
        cols=:union
    )
end

"""
    get_imputed_reports(result; CrIs=[0.2, 0.5, 0.9])

Generate integer-valued imputed report draws by sampling from the fitted
observation model (NegBin or Poisson) for each posterior sample.

Returns a summary DataFrame matching the format of `result.reports`.
"""
function get_imputed_reports(
    result::EstimateInfectionsResult;
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    fit = result.fit
    gqs = fit.generated_quantities
    params = get_parameters(result)
    dates = _output_dates(fit.metadata)
    obs = fit.metadata.obs_opts
    n_samples = length(gqs)

    has_phi = haskey(params, :reporting_overdispersion)
    phi_samples = has_phi ? params[:reporting_overdispersion] : nothing

    n_times = length(first(gqs).reports)
    n_dates = min(n_times, length(dates))
    mat = Matrix{Float64}(undef, n_dates, n_samples)

    for (i, gq) in enumerate(gqs)
        reports = gq.reports
        for t in 1:n_dates
            μ = max(Float64(reports[t]), 1e-6)
            if has_phi
                φ = 1.0 / Float64(phi_samples[i])^2
                mat[t, i] = Float64(rand(NegativeBinomial2(μ, φ)))
            else
                mat[t, i] = Float64(rand(Poisson(μ)))
            end
        end
    end

    _matrix_to_summary(mat, dates[1:n_dates], CrIs)
end

get_imputed_reports(result::EpinowResult; kwargs...) =
    get_imputed_reports(result.estimates; kwargs...)
