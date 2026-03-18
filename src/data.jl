# ── Data validation and preparation ──────────────────────────────────────
#
# EpiNow2 expects a DataFrame with :date and :confirm columns.
# We validate early and convert to internal representations.

"""
    EpiData

Validated epidemiological time series data for use with `estimate_infections`
and `epinow`.

# Required columns
- `date::Vector{Date}` — observation dates (must be contiguous)
- `confirm::Vector{Int}` — confirmed case counts

# Optional columns
- `accumulate::Vector{Bool}` — whether to accumulate to next data point

Construct via `EpiData(df::DataFrame)` which validates and sorts.
"""
struct EpiData
    date::Vector{Date}
    confirm::Vector{Int}
    accumulate::Vector{Bool}

    function EpiData(df::DataFrame)
        @assert :date in propertynames(df) "Data must have a :date column"
        @assert :confirm in propertynames(df) "Data must have a :confirm column"

        sorted = sort(df, :date)
        dates = Date.(sorted.date)
        confirm = Int.(sorted.confirm)

        # Check dates are contiguous (daily)
        diffs = diff(Dates.value.(dates))
        if !all(diffs .== 1)
            @warn "Non-contiguous dates detected; missing dates will be " *
                  "filled with zeros"
            dates, confirm = _fill_missing_dates(dates, confirm)
        end

        accumulate = if :accumulate in propertynames(sorted)
            Bool.(sorted.accumulate)
        else
            fill(false, length(dates))
        end

        new(dates, confirm, accumulate)
    end
end

Base.length(d::EpiData) = length(d.date)

"""
    SecondaryData

Data for `estimate_secondary`: primary and secondary time series.

# Required columns
- `date::Vector{Date}`
- `primary::Vector{Int}` — primary observations (e.g., cases)
- `secondary::Vector{Int}` — secondary observations (e.g., deaths)
"""
struct SecondaryData
    date::Vector{Date}
    primary::Vector{Int}
    secondary::Vector{Int}

    function SecondaryData(df::DataFrame)
        @assert :date in propertynames(df) "Must have :date column"
        @assert :primary in propertynames(df) "Must have :primary column"
        @assert :secondary in propertynames(df) "Must have :secondary column"

        sorted = sort(df, :date)
        new(Date.(sorted.date), Int.(sorted.primary), Int.(sorted.secondary))
    end
end

# ── Example data ─────────────────────────────────────────────────────────

"""
    example_confirmed() -> DataFrame

Return a DataFrame with ~60 days of synthetic case data for use in examples
and testing. Columns: `:date` and `:confirm`.
"""
function example_confirmed()
    n_days = 60
    dates = Date(2024, 3, 1):Day(1):Date(2024, 3, 1) + Day(n_days - 1)
    # Deterministic renewal process with time-varying Rt
    rt = [1.0 + 0.3 * sin(2π * t / 40) for t in 1:n_days]
    infections = zeros(Float64, n_days)
    infections[1] = 100.0
    gt_pmf = [0.0, 0.3, 0.4, 0.2, 0.1]
    for t in 2:n_days
        λ = rt[t] * sum(
            infections[max(t - s, 1)] * gt_pmf[min(s + 1, length(gt_pmf))]
            for s in 1:min(t - 1, length(gt_pmf) - 1)
        )
        infections[t] = max(λ, 1.0)
    end
    cases = [max(1, round(Int, x)) for x in infections]
    DataFrame(date=collect(dates), confirm=cases)
end

# ── Helpers ──────────────────────────────────────────────────────────────

function _fill_missing_dates(dates, confirm)
    full_range = dates[1]:Day(1):dates[end]
    full_confirm = zeros(Int, length(full_range))
    for (d, c) in zip(dates, confirm)
        idx = Dates.value(d - dates[1]) + 1
        full_confirm[idx] = c
    end
    (collect(full_range), full_confirm)
end
