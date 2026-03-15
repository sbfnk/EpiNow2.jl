# ── Turing model implementations ─────────────────────────────────────────
#
# Direct Turing.jl implementations of EpiNow2's epidemiological models.
# No EpiAware dependency — these are self-contained.

using Turing
using LinearAlgebra
using FFTW

# ══════════════════════════════════════════════════════════════════════════
# Core mathematical components
# ══════════════════════════════════════════════════════════════════════════

# ── Discrete convolution ─────────────────────────────────────────────────

"""
    convolve(signal, kernel)

Discrete convolution of a time series with a delay kernel (PMF).
Returns a vector of the same length as `signal`.
"""
function convolve(signal::AbstractVector, kernel::AbstractVector)
    n = length(signal)
    k = length(kernel)
    out = similar(signal)
    rev_kernel = reverse(kernel)
    for t in 1:n
        s = zero(eltype(signal))
        for j in 1:min(t, k)
            s += signal[t - j + 1] * rev_kernel[k - j + 1]
        end
        out[t] = s
    end
    out
end

# ── Renewal equation ─────────────────────────────────────────────────────

"""
    renewal_infections(R, initial_infections, gt_pmf, n_times)

Generate infections via the renewal equation:
    infections[t] = R[t] * sum(infections[t-s] * gt_pmf[s], s=1..max)

`initial_infections` seeds the first `seeding_time` days.
"""
function renewal_infections(
    R::AbstractVector,
    initial_infections::AbstractVector,
    gt_pmf::AbstractVector,
    n_times::Int
)
    seeding_time = length(initial_infections)
    total = seeding_time + n_times
    infections = similar(R, total)

    # Seed period
    infections[1:seeding_time] .= initial_infections

    # Renewal process
    gt_len = length(gt_pmf)
    for t in (seeding_time + 1):total
        infectiousness = zero(eltype(R))
        for s in 1:min(t - 1, gt_len)
            infectiousness += infections[t - s] * gt_pmf[s]
        end
        rt_idx = t - seeding_time
        infections[t] = R[rt_idx] * infectiousness
    end

    infections
end

# ── Gaussian process (Hilbert space approximation) ───────────────────────

"""
    spectral_density_matern(ω, α, ρ, ν)

Spectral density of the Matérn covariance function at frequency `ω`.
"""
function spectral_density_matern(ω::Real, α::Real, ρ::Real, ν::Real)
    # S(ω) = α² * (2ν)^ν * (2π)^(D/2) * Γ(ν + D/2) /
    #         (Γ(ν) * (2ν/ρ² + 4π²ω²)^(ν + D/2))
    # For D=1 (time series):
    c = 2 * α^2 * sqrt(π) * gamma(ν + 0.5) / gamma(ν)
    c * (2ν / ρ^2)^ν / (2ν / ρ^2 + 4π^2 * ω^2)^(ν + 0.5)
end

"""
    spectral_density_se(ω, α, ρ)

Spectral density of the squared exponential (RBF) kernel.
"""
function spectral_density_se(ω::Real, α::Real, ρ::Real)
    α^2 * sqrt(2π) * ρ * exp(-2π^2 * ρ^2 * ω^2)
end

"""
    hsgp_basis(n_basis, boundary, n_times)

Compute Hilbert space basis functions for GP approximation.
Returns matrix of shape (n_times, n_basis).
"""
function hsgp_basis(n_basis::Int, boundary::Float64, n_times::Int)
    L = boundary * n_times
    x = collect(1.0:n_times) ./ n_times  # normalised time
    basis = Matrix{Float64}(undef, n_times, n_basis)
    for j in 1:n_basis
        λ = j * π / (2L)
        basis[:, j] = sin.(λ .* (x .+ L)) ./ sqrt(L)
    end
    basis
end

"""
    hsgp_coefficients(n_basis, boundary, n_times, α, ρ, kernel, ν)

Compute spectral density weights for HSGP basis functions.
"""
function hsgp_coefficients(
    n_basis::Int, boundary::Float64, n_times::Int,
    α::Real, ρ::Real, kernel::Symbol, ν::Float64
)
    L = boundary * n_times
    weights = Vector{Float64}(undef, n_basis)
    for j in 1:n_basis
        ω = j / (2L)
        weights[j] = if kernel == :se
            sqrt(spectral_density_se(ω, α, ρ))
        else
            sqrt(spectral_density_matern(ω, α, ρ, ν))
        end
    end
    weights
end

# ── Day-of-week effect ───────────────────────────────────────────────────

"""
    apply_day_of_week(expected, effect, dates)

Scale expected values by day-of-week effects. `effect` is a simplex of
length 7 (sums to 7, so mean effect is 1).
"""
function apply_day_of_week(
    expected::AbstractVector,
    effect::AbstractVector,
    start_day::Int,  # day of week of first observation (1=Monday)
    n::Int
)
    scaled = similar(expected, n)
    for t in 1:n
        dow = mod1(start_day + t - 1, 7)
        scaled[t] = expected[t] * effect[dow]
    end
    scaled
end

# ══════════════════════════════════════════════════════════════════════════
# Turing models
# ══════════════════════════════════════════════════════════════════════════

"""
    infections_model(data, gt_pmf, delay_pmf, opts...)

The core Turing model for EpiNow2's `estimate_infections`.

Generative process:
1. Sample Rt trajectory (GP / random walk / fixed)
2. Generate infections via renewal equation
3. Convolve with reporting delay
4. Apply day-of-week effects
5. Observe with NegBin/Poisson likelihood
"""
@model function infections_model(
    cases::AbstractVector{Int},
    gt_pmf::AbstractVector{Float64},
    delay_pmf::AbstractVector{Float64},
    n_times::Int,
    n_forecast::Int,
    seeding_time::Int,
    # Model structure flags
    use_rt::Bool,
    use_gp::Bool,
    gp_n_basis::Int,
    gp_boundary::Float64,
    gp_kernel::Symbol,
    gp_matern_order::Float64,
    use_rw::Bool,
    rw_period::Int,
    use_week_effect::Bool,
    week_length::Int,
    start_day::Int,
    obs_family::Symbol,
    use_obs_scale::Bool,
    # Accumulation
    accumulate::AbstractVector{Bool},
    # Population adjustment
    pop::Float64
)
    total_times = n_times + n_forecast

    # ── Priors: initial conditions ───────────────────────────────────
    log_initial_infections ~ Normal(log(cases[1] + 1), 2.0)
    initial_growth ~ Normal(0.0, 0.1)

    # Seed infections with exponential growth/decay
    initial_infections = exp.(
        log_initial_infections .+
        initial_growth .* collect(0.0:(seeding_time - 1))
    )

    # ── Priors: Rt ───────────────────────────────────────────────────
    log_R0 ~ Normal(0.0, 1.0)  # prior on log(R0), mean ≈ 1

    if use_gp && !use_rw
        # Gaussian process on log(Rt)
        gp_alpha ~ truncated(Normal(0.0, 0.01); lower=0.0)
        gp_rho ~ truncated(
            LogNormal(log(21.0), 0.5); lower=1.0, upper=60.0
        )
        gp_z ~ filldist(Normal(0.0, 1.0), gp_n_basis)

        # HSGP: basis * diag(spectral_weights) * z
        basis = hsgp_basis(gp_n_basis, gp_boundary, total_times)
        spd_weights = hsgp_coefficients(
            gp_n_basis, gp_boundary, total_times,
            gp_alpha, gp_rho, gp_kernel, gp_matern_order
        )
        gp_f = basis * (spd_weights .* gp_z)

        log_R = log_R0 .+ gp_f
    elseif use_rw
        # Random walk on log(Rt)
        n_steps = div(total_times, rw_period)
        rw_sd ~ truncated(Normal(0.0, 0.1); lower=0.0)
        rw_noise ~ filldist(Normal(0.0, 1.0), n_steps)

        rw_values = cumsum(rw_sd .* rw_noise)
        # Expand to daily: each step covers rw_period days
        log_R = Vector{eltype(rw_values)}(undef, total_times)
        for t in 1:total_times
            step = min(div(t - 1, rw_period) + 1, n_steps)
            log_R[t] = log_R0 + rw_values[step]
        end
    else
        # Fixed Rt
        log_R = fill(log_R0, total_times)
    end

    R = exp.(log_R)

    # ── Generate infections (renewal equation) ───────────────────────
    all_infections = renewal_infections(
        R, initial_infections, gt_pmf, total_times
    )

    # Population adjustment (susceptible depletion)
    if pop > 0
        cumulative = cumsum(all_infections)
        susceptible_frac = max.((pop .- cumulative) ./ pop, 1e-6)
        all_infections = all_infections .* susceptible_frac
    end

    # Extract post-seeding infections
    infections = all_infections[(seeding_time + 1):end]

    # ── Map to expected reports ──────────────────────────────────────
    expected_reports = convolve(infections, delay_pmf)

    # ── Day-of-week effect ───────────────────────────────────────────
    if use_week_effect
        # Simplex prior: Dirichlet centred on uniform
        week_effect ~ Dirichlet(fill(1.0, week_length))
        scaled_effect = week_effect .* week_length  # scale so mean = 1
        expected_reports = apply_day_of_week(
            expected_reports, scaled_effect, start_day, total_times
        )
    end

    # ── Observation scale (fraction reported) ────────────────────────
    if use_obs_scale
        frac_observed ~ Beta(5.0, 5.0)
        expected_reports = expected_reports .* frac_observed
    end

    # ── Accumulation (e.g., weekly reporting) ────────────────────────
    if any(accumulate)
        expected_reports = _apply_accumulation(
            expected_reports, accumulate, n_times
        )
    end

    # ── Likelihood ───────────────────────────────────────────────────
    if obs_family == :negbin
        reporting_overdispersion ~ truncated(
            Normal(0.0, 0.25); lower=0.0
        )
        φ = 1.0 / reporting_overdispersion^2
        for t in 1:n_times
            μ = max(expected_reports[t], 1e-6)
            cases[t] ~ NegativeBinomial2(μ, φ)
        end
    else  # :poisson
        for t in 1:n_times
            μ = max(expected_reports[t], 1e-6)
            cases[t] ~ Poisson(μ)
        end
    end

    # ── Return generated quantities ──────────────────────────────────
    return (
        infections = infections,
        reports = expected_reports,
        R = R,
        log_R = log_R
    )
end

"""
    NegativeBinomial2(μ, φ)

Negative binomial parameterised by mean `μ` and precision `φ`.
Var = μ + μ²/φ. Equivalent to Stan's neg_binomial_2.
"""
function NegativeBinomial2(μ, φ)
    p = φ / (μ + φ)
    r = φ
    NegativeBinomial(r, p)
end

# ── Secondary observations model ─────────────────────────────────────────

@model function secondary_model(
    primary::AbstractVector{Int},
    secondary::AbstractVector{Int},
    delay_pmf::AbstractVector{Float64},
    n_times::Int,
    is_prevalence::Bool,
    obs_family::Symbol
)
    # Fraction of primary that become secondary
    frac ~ Beta(5.0, 5.0)

    # Scale primary
    scaled_primary = frac .* Float64.(primary)

    # Convolve with delay
    expected = convolve(scaled_primary, delay_pmf)

    # For prevalence: cumulative
    if is_prevalence
        expected = cumsum(expected)
    end

    # Likelihood
    if obs_family == :negbin
        overdispersion ~ truncated(Normal(0.0, 0.25); lower=0.0)
        φ = 1.0 / overdispersion^2
        for t in 1:n_times
            μ = max(expected[t], 1e-6)
            secondary[t] ~ NegativeBinomial2(μ, φ)
        end
    else
        for t in 1:n_times
            μ = max(expected[t], 1e-6)
            secondary[t] ~ Poisson(μ)
        end
    end

    return (expected = expected,)
end

# ── Truncation estimation model ──────────────────────────────────────────

@model function truncation_model(
    snapshots::Vector{Vector{Int}},
    snapshot_lengths::Vector{Int},
    max_trunc::Int
)
    # Parameters of truncation distribution (log-normal)
    trunc_meanlog ~ Normal(0.0, 1.0)
    trunc_sdlog ~ truncated(Normal(1.0, 1.0); lower=0.0)

    # Build truncation PMF
    d = Distributions.LogNormal(trunc_meanlog, trunc_sdlog)
    trunc_pmf = [cdf(d, k + 0.5) - cdf(d, k - 0.5) for k in 0:max_trunc]
    trunc_pmf ./= sum(trunc_pmf)

    # CDF for survival function
    trunc_cdf = cumsum(trunc_pmf)

    # The latest (most complete) snapshot is treated as truth
    final = last(snapshots)

    # Earlier snapshots are truncated versions of truth
    for (i, snap) in enumerate(snapshots[1:end-1])
        n = snapshot_lengths[i]
        days_before_final = snapshot_lengths[end] - n

        for t in 1:n
            days_truncated = n - t  # how many days of truncation
            if days_truncated < max_trunc
                reporting_prob = trunc_cdf[days_truncated + 1]
            else
                reporting_prob = 1.0
            end
            expected = final[t] * reporting_prob
            snap[t] ~ Poisson(max(expected, 1e-6))
        end
    end

    return (trunc_meanlog = trunc_meanlog, trunc_sdlog = trunc_sdlog)
end

# ── Model assembly (options → Turing model) ──────────────────────────────

"""
    assemble_model(data, gt, delays, truncation, rt, backcalc, gp, obs,
                   forecast)

Translate EpiNow2 options into a configured Turing model with all
structural choices baked in.
"""
function assemble_model(
    data::EpiData;
    generation_time::GTOpts,
    delays::DelayOpts,
    truncation::TruncOpts,
    rt::RtOpts,
    backcalc::BackcalcOpts,
    gp::GPOpts,
    obs::ObsOpts,
    forecast::ForecastOpts
)
    n_times = length(data)
    n_forecast = forecast.horizon
    total_times = n_times + n_forecast

    # Discretise distributions
    gt_pmf = discretise(generation_time.dist)
    delay_pmf = discretise(delays.dist)
    seeding_time = length(gt_pmf) - 1

    # GP configuration
    use_gp = rt.use_rt && gp.basis_prop > 0 && rt.rw == 0
    gp_n_basis = use_gp ? max(1, round(Int, gp.basis_prop * total_times)) : 0
    gp_boundary = gp.boundary_scale

    # Random walk configuration
    use_rw = rt.use_rt && rt.rw > 0

    # Day of week
    start_day = Dates.dayofweek(data.date[1])

    model = infections_model(
        data.confirm,
        gt_pmf,
        delay_pmf,
        n_times,
        n_forecast,
        seeding_time,
        rt.use_rt,
        use_gp,
        gp_n_basis,
        gp_boundary,
        gp.kernel,
        gp.matern_order,
        use_rw,
        rt.rw,
        obs.week_effect,
        obs.week_length,
        start_day,
        obs.family,
        !(obs.scale isa FixedSpec && obs.scale.value == 1.0),
        data.accumulate,
        rt.pop
    )

    metadata = ModelMetadata(
        dates = data.date,
        seeding_time = seeding_time,
        horizon = n_forecast,
        gt_pmf = gt_pmf,
        delay_pmf = delay_pmf,
        rt_opts = rt,
        obs_opts = obs
    )

    (model=model, metadata=metadata)
end

"""
    ModelMetadata

Bookkeeping for mapping posterior samples back to dated epidemiological
quantities.
"""
struct ModelMetadata
    dates::Vector{Date}
    seeding_time::Int
    horizon::Int
    gt_pmf::Vector{Float64}
    delay_pmf::Vector{Float64}
    rt_opts::RtOpts
    obs_opts::ObsOpts
end

# ── Helpers ──────────────────────────────────────────────────────────────

function _apply_accumulation(
    expected::AbstractVector, accumulate::AbstractVector{Bool}, n::Int
)
    result = similar(expected, n)
    buffer = zero(eltype(expected))
    for t in 1:n
        buffer += expected[t]
        if !accumulate[t]
            result[t] = buffer
            buffer = zero(eltype(expected))
        else
            result[t] = zero(eltype(expected))
        end
    end
    result
end
