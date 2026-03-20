# # Estimating infection dynamics with EpiNow2.jl
#
# This example mirrors the R EpiNow2 `estimate_infections` workflow vignette.
# It demonstrates the core pipeline: from reported case data to Rt estimates,
# infection trajectories, and forecasts.

using EpiNow2
using CSV
using DataFrames
using Dates
using Distributions

# ## 1. Load data
#
# We use the example case data from R's EpiNow2 package (UK COVID-19 cases,
# Feb-Jun 2020). For speed, we use the first 60 days.

pkgdir = dirname(dirname(pathof(EpiNow2)))
reported_cases = CSV.read(
    joinpath(pkgdir, "test", "reference", "example_confirmed_full.csv"),
    DataFrame
)
reported_cases.date = Date.(reported_cases.date)
reported_cases = first(reported_cases, 30)

println("Data: $(nrow(reported_cases)) days, " *
        "$(reported_cases.date[1]) to $(reported_cases.date[end])")
println("Cases range: $(minimum(reported_cases.confirm)) to " *
        "$(maximum(reported_cases.confirm))")

# ## 2. Define delay distributions
#
# EpiNow2 models the pipeline from infection → symptom onset → report.
# Each step has an associated delay distribution.

# Generation time: time between primary and secondary infection.
# Gamma distribution with uncertain shape and rate (matching R's
# example_generation_time).
generation_time = UncertainDistribution(
    (shape, rate) -> Gamma(shape, 1 / rate),
    [truncated(Normal(1.4, 0.48); lower=0.01),
     truncated(Normal(0.38, 0.25); lower=0.01)],
    14.0
)

# Incubation period: infection to symptom onset (uncertain).
incubation_period = UncertainDistribution(
    (μ, σ) -> LogNormal(μ, σ),
    [Normal(1.62, 0.064), truncated(Normal(0.418, 0.069); lower=0.01)],
    14.0
)

# Reporting delay: symptom onset to case report (fixed).
reporting_delay = LogNormal(0.5, 0.5)

# Compose incubation + reporting into a single delay by convolving PMFs.
combined_delay = discretise(incubation_period) + discretise(reporting_delay; max=10)

# ## 3. Run estimate_infections
#
# This is the core function: it estimates infections, Rt, and expected
# reports from the observed case data.

println("\nRunning estimate_infections...")
result = estimate_infections(
    reported_cases;
    generation_time = gt_opts(generation_time),
    delays = delay_opts(combined_delay),
    forecast = forecast_opts(horizon=7),
    inference = inference_opts(
        samples=1000,
        warmup=250,
        chains=2
    )
)

# ## 4. Examine results

println("\n── Summary ──")
println("Timing: $(round(result.timing, digits=1))s")

# Rt estimates
println("\n── Reproduction number ──")
println("Latest Rt ($(result.rt.date[end])): " *
        "median=$(round(result.rt.median[end], digits=2)) " *
        "(90% CI: $(round(result.rt.lower_90[end], digits=2))-" *
        "$(round(result.rt.upper_90[end], digits=2)))")

# At a few key dates
n_obs = nrow(reported_cases)
for i in [1, n_obs ÷ 4, n_obs ÷ 2, 3 * n_obs ÷ 4, n_obs]
    d = result.rt.date[i]
    m = round(result.rt.median[i], digits=2)
    lo = round(result.rt.lower_90[i], digits=2)
    hi = round(result.rt.upper_90[i], digits=2)
    println("  $d: Rt = $m ($lo-$hi)")
end

# Infections
println("\n── Estimated infections ──")
println("Peak infections: $(round(maximum(result.infections.median), digits=0)) " *
        "on $(result.infections.date[argmax(result.infections.median)])")

# Forecasts (last 7 days)
println("\n── 7-day forecast ──")
forecast_rows = filter(r -> r.date > reported_cases.date[end], result.reports)
for row in eachrow(forecast_rows)
    println("  $(row.date): $(round(row.median, digits=0)) " *
            "($(round(row.lower_90, digits=0))-$(round(row.upper_90, digits=0)))")
end

# ## 5. Access raw samples
#
# For custom analyses, extract posterior samples directly.
samples = get_samples(result; variable=:R)
println("\n── Raw samples ──")
println("$(nrow(samples)) Rt samples across $(length(unique(samples.date))) dates")

# ## 6. Parameter estimates
params = get_parameters(result)
println("\n── Fitted parameters ──")
for (k, v) in sort(collect(params), by=first)
    any(ismissing, v) && continue
    println("  $k: median=$(round(median(v), digits=4))")
end

println("\nDone!")
