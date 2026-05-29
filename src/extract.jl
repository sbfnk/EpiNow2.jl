# ── Result extraction and summarisation ───────────────────────────────────
#
# Maps Turing generated quantities (NamedTuples of infections, R, reports)
# back to dated DataFrames with credible intervals.

"""
    EstimateInfectionsArgs

Captures the configuration arguments passed to `estimate_infections()`.
Mirrors the `args` field of EpiNow2 R's v1.8 `epinowfit` S3 class so
that a fitted result can be inspected (and in principle replayed)
without referring back to the original call.
"""
struct EstimateInfectionsArgs
    generation_time::GTOpts
    delays::DelayOpts
    truncation::TruncOpts
    rt::RtOpts
    backcalc::BackcalcOpts
    gp::GPOpts
    obs::ObsOpts
    forecast::ForecastOpts
    inference::InferenceOpts
    CrIs::Vector{Float64}
end

"""
    EstimateInfectionsResult

Result of `estimate_infections()`. Fields:

- `fit`: the underlying `EpiNow2Fit` (chain + generated quantities + metadata)
- `args`: the `EstimateInfectionsArgs` capturing the call configuration
- `observations`: the input data as an `EpiData`
- `infections`, `reports`, `rt`, `growth_rate`: pre-computed posterior summaries.
  `reports` is the posterior predictive (includes observation noise); the latent
  expectation is in `reports_expected`.
- `timing`: elapsed inference seconds
"""
struct EstimateInfectionsResult
    fit::EpiNow2Fit
    args::EstimateInfectionsArgs
    observations::EpiData
    infections::DataFrame        # by date of infection
    reports::DataFrame           # posterior predictive reported cases (incl. obs noise)
    reports_expected::DataFrame  # latent expected reports (no obs noise)
    rt::DataFrame                # depletion-adjusted reproduction number
    rt_unadjusted::DataFrame     # transmission Rt (= rt when pop=0)
    growth_rate::DataFrame       # growth rate by date
    timing::Float64              # seconds
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
    get_samples(result; variable=nothing, predictive=false) -> DataFrame

Extract posterior samples as a long-format DataFrame.

Columns: `date`, `variable`, `sample`, `value`

By default the values are the latent generated quantities. With
`predictive=true`, `:reports` samples are drawn through the fitted observation
model (NegBin/Poisson), giving the posterior predictive of reported cases;
other variables are returned unchanged.
"""
function get_samples(
    result::EstimateInfectionsResult; variable=nothing, predictive=false
)
    fit = result.fit
    meta = fit.metadata
    gqs = fit.generated_quantities

    dates = _output_dates(meta)
    family = result.args.obs.family
    phi = predictive && family == negbin ?
        _chain_vector(fit, :reporting_overdispersion) : nothing
    rng = predictive ? Random.MersenneTwister(0) : nothing

    rows = NamedTuple{(:date, :variable, :sample, :value),
                      Tuple{Date, Symbol, Int, Float64}}[]

    for (i, gq) in enumerate(gqs)
        for (var, values) in pairs(gq)
            var_sym = Symbol(var)
            (!isnothing(variable) && var_sym != variable) && continue
            add_noise = predictive && var_sym == :reports
            for (t, v) in enumerate(values)
                t > length(dates) && continue
                val = add_noise ?
                    _draw_obs(rng, v, family, isnothing(phi) ? nothing : phi[i]) :
                    Float64(v)
                push!(rows, (
                    date=dates[t], variable=var_sym, sample=i, value=val
                ))
            end
        end
    end

    DataFrame(rows)
end

get_samples(result::EpinowResult; kwargs...) =
    get_samples(result.estimates; kwargs...)

"""
    get_predictions(result; format=:summary, CrIs=[0.2, 0.5, 0.9],
                    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])

Extract case predictions in one of three formats:

- `:summary` — pre-computed posterior summary (`result.reports`).
  `CrIs` controls the credible-interval columns surfaced.
- `:sample` — long-format posterior samples ready for
  `scoringutils::as_forecast_sample()`. Columns: `forecast_date`, `date`,
  `horizon`, `sample`, `predicted`.
- `:quantile` — quantile predictions ready for
  `scoringutils::as_forecast_quantile()`. Columns: `forecast_date`,
  `date`, `horizon`, `quantile_level`, `predicted`. `quantiles`
  controls the quantile levels.
"""
function get_predictions(
    result::EstimateInfectionsResult;
    format::Symbol=:summary,
    CrIs::Vector{Float64}=[0.2, 0.5, 0.9],
    quantiles::Vector{Float64}=[0.05, 0.25, 0.5, 0.75, 0.95],
)
    if format == :summary
        return result.reports
    elseif format == :sample
        samples = get_samples(result; variable=:reports, predictive=true)
        return _format_sample_predictions(samples, result)
    elseif format == :quantile
        samples = get_samples(result; variable=:reports, predictive=true)
        return _format_quantile_predictions(samples, quantiles, result)
    else
        throw(ArgumentError(
            "Unknown format: $format. Use :summary, :sample, or :quantile"
        ))
    end
end

function _format_sample_predictions(samples::DataFrame, result)
    forecast_date = result.observations.date[end]
    DataFrame(
        forecast_date = fill(forecast_date, nrow(samples)),
        date = samples.date,
        horizon = Dates.value.(samples.date .- forecast_date),
        sample = samples.sample,
        predicted = samples.value,
    )
end

function _format_quantile_predictions(
    samples::DataFrame, quantiles::Vector{Float64}, result,
)
    forecast_date = result.observations.date[end]
    grouped = groupby(samples, :date)
    rows = NamedTuple[]
    for g in grouped
        d = g.date[1]
        for q in quantiles
            push!(rows, (
                forecast_date = forecast_date,
                date = d,
                horizon = Dates.value(d - forecast_date),
                quantile_level = q,
                predicted = quantile(g.value, q),
            ))
        end
    end
    DataFrame(rows)
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
    fit::EpiNow2Fit, data::EpiData, args::EstimateInfectionsArgs,
    elapsed::Float64;
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    dates = _output_dates(fit.metadata)

    infections = _summarise_gq(fit, :infections, dates, CrIs)
    # `reports` is the posterior predictive (latent expectation + observation
    # noise), matching R's `reported_cases`. The latent mean is kept separately.
    reports_expected = _summarise_gq(fit, :reports, dates, CrIs)
    reports = _summarise_gq_predictive(
        fit, :reports, dates, CrIs, args.obs.family, :reporting_overdispersion
    )
    rt_df = _summarise_gq(fit, :R, dates, CrIs)
    rt_unadjusted = _summarise_gq(fit, :R_unadjusted, dates, CrIs)

    # Growth rate: log-diff of infections
    growth = _compute_growth_rate(fit, dates, CrIs)

    EstimateInfectionsResult(
        fit, args, data,
        infections, reports, reports_expected, rt_df, rt_unadjusted,
        growth, elapsed,
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
    mat, used_dates = _gq_matrix(fit, field, dates)
    isnothing(mat) && return DataFrame()
    _matrix_to_summary(mat, used_dates, CrIs)
end

"""
Collect a generated-quantities field into an `(n_dates × n_samples)` matrix.
Returns `(nothing, dates)` when the field is absent (model variant).
"""
function _gq_matrix(fit::EpiNow2Fit, field::Symbol, dates::Vector{Date})
    gqs = fit.generated_quantities
    n_samples = length(gqs)

    haskey(first(gqs), field) || return (nothing, dates)

    n_times = length(first(gqs)[field])
    n_dates = min(n_times, length(dates))

    mat = Matrix{Float64}(undef, n_dates, n_samples)
    for (i, gq) in enumerate(gqs)
        vals = gq[field]
        for t in 1:n_dates
            mat[t, i] = Float64(vals[t])
        end
    end
    (mat, dates[1:n_dates])
end

"""Extract a scalar chain parameter as a per-sample vector, or `nothing`."""
function _chain_vector(fit::EpiNow2Fit, name::Symbol)
    chain = fit.chain
    name in names(chain) || return nothing
    vec(Array(chain[name]))
end

"""
Draw one integer observation from the fitted observation model: NegBin when
`family == negbin` and an overdispersion is given, Poisson otherwise. `μ` is
clamped to a safe positive range, and extreme overdispersion falls back to
Poisson to avoid degenerate NegBin draws.
"""
function _draw_obs(
    rng, μ::Real, family::ObsFamily, overdisp::Union{Nothing, Real}
)
    μ = max(min(Float64(μ), 1e15), 1e-6)
    if family == negbin && !isnothing(overdisp)
        φ = 1.0 / Float64(overdisp)^2
        φ > 1e-4 && return Float64(rand(rng, NegativeBinomial2(μ, φ)))
    end
    Float64(rand(rng, Poisson(μ)))
end

"""
    _draw_obs_predictive(expected, family, phi_samples)

Draw an integer posterior-predictive matrix from a matrix of latent
expectations by sampling the fitted observation model per posterior sample.
`phi_samples[i]` is the overdispersion for sample `i` (NegBin only). A fixed
local RNG makes the draws reproducible for a given fit.
"""
function _draw_obs_predictive(
    expected::Matrix{Float64}, family::ObsFamily,
    phi_samples::Union{Nothing, AbstractVector}
)
    rng = Random.MersenneTwister(0)
    n_dates, n_samples = size(expected)
    out = Matrix{Float64}(undef, n_dates, n_samples)
    for i in 1:n_samples
        overdisp = isnothing(phi_samples) ? nothing : phi_samples[i]
        for t in 1:n_dates
            out[t, i] = _draw_obs(rng, expected[t, i], family, overdisp)
        end
    end
    out
end

"""
Summarise the posterior predictive of a generated-quantities field: add
observation noise to the latent expectation before computing summary stats.
"""
function _summarise_gq_predictive(
    fit::EpiNow2Fit, field::Symbol, dates::Vector{Date},
    CrIs::Vector{Float64}, family::ObsFamily, phi_param::Symbol
)
    mat, used_dates = _gq_matrix(fit, field, dates)
    isnothing(mat) && return DataFrame()
    phi = family == negbin ? _chain_vector(fit, phi_param) : nothing
    pp = _draw_obs_predictive(mat, family, phi)
    _matrix_to_summary(pp, used_dates, CrIs)
end

function _matrix_to_summary(
    mat::Matrix{Float64}, dates::Vector{Date}, CrIs::Vector{Float64}
)
    # Replace NaN/Inf with 0 for robust quantile computation
    clean = replace(mat, NaN => 0.0, Inf => 0.0, -Inf => 0.0)

    base = (
        date = dates,
        mean = vec(mean(clean; dims=2)),
        median = vec(mapslices(median, clean; dims=2)),
        sd = vec(mapslices(std, clean; dims=2))
    )
    cri_cols = NamedTuple()
    for cri in CrIs
        pct = round(Int, cri * 100)
        lo = vec(mapslices(v -> quantile(v, (1 - cri) / 2), clean; dims=2))
        hi = vec(mapslices(v -> quantile(v, (1 + cri) / 2), clean; dims=2))
        cri_cols = merge(cri_cols,
            NamedTuple{(Symbol("lower_$pct"), Symbol("upper_$pct"))}((lo, hi)))
    end
    DataFrame(merge(base, cri_cols))
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
            mat[t, i] = log(max(Float64(inf[t + 1]), 1e-10)) -
                        log(max(Float64(inf[t]), 1e-10))
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
    rows = NamedTuple[]
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
observation model (NegBin or Poisson) for each posterior sample. This is the
posterior predictive of reported cases — the same quantity surfaced as
`result.reports`, recomputable here at custom credible-interval levels.

Returns a summary DataFrame matching the format of `result.reports`.
"""
function get_imputed_reports(
    result::EstimateInfectionsResult;
    CrIs::Vector{Float64} = [0.2, 0.5, 0.9]
)
    fit = result.fit
    _summarise_gq_predictive(
        fit, :reports, _output_dates(fit.metadata), CrIs,
        result.args.obs.family, :reporting_overdispersion
    )
end

get_imputed_reports(result::EpinowResult; kwargs...) =
    get_imputed_reports(result.estimates; kwargs...)
