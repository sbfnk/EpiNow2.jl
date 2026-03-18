# ── Turing model implementations ─────────────────────────────────────────
#
# Direct Turing.jl implementations of EpiNow2's epidemiological models.
# All helper functions are AD-compatible (no array mutation).

using Turing
using LinearAlgebra

# ══════════════════════════════════════════════════════════════════════════
# Core mathematical components (AD-safe)
# ══════════════════════════════════════════════════════════════════════════

# ── Discrete convolution ─────────────────────────────────────────────────

"""
    convolve(signal, kernel)

Discrete convolution of a time series with a delay kernel (PMF).
Returns a vector of the same length as `signal`. AD-safe (no mutation).
"""
function convolve(signal::AbstractVector, kernel::AbstractVector)
    n = length(signal)
    k = length(kernel)
    [sum(signal[t - j + 1] * kernel[j] for j in 1:min(t, k))
     for t in 1:n]
end

# ── Renewal equation ─────────────────────────────────────────────────────

"""
    renewal_infections(R, initial_infections, gt_pmf, n_times)

Generate infections via the renewal equation. AD-safe: uses vcat to
build the infection vector without mutation.
"""
function renewal_infections(
    R::AbstractVector,
    initial_infections::AbstractVector,
    gt_pmf::AbstractVector,
    n_times::Int
)
    seeding_time = length(initial_infections)
    total = seeding_time + n_times
    gt_len = length(gt_pmf)

    # Build infections sequentially via reduction (no mutation)
    infections = initial_infections
    for t in (seeding_time + 1):total
        infectiousness = sum(
            infections[t - s] * gt_pmf[s]
            for s in 1:min(t - 1, gt_len)
        )
        rt_idx = t - seeding_time
        new_inf = R[rt_idx] * infectiousness
        infections = vcat(infections, [new_inf])
    end

    infections
end

# ── HSGP: Hilbert space Gaussian process approximation ───────────────────

"""
    hsgp_basis(n_basis, boundary, n_times)

Compute HSGP basis functions matching Stan's implementation.
Returns matrix of shape (n_times, n_basis).

Stan normalisation: x = 2*(x - mean(x)) / (max(x) - 1), giving x ∈ [-1, 1].
L = boundary * (max(x) - min(x)) / 2 (half-range scaled by boundary).
"""
function hsgp_basis(n_basis::Int, boundary::Float64, n_times::Int)
    # Normalise x to [-1, 1] matching Stan
    x_raw = collect(1.0:n_times)
    x_mean = mean(x_raw)
    x_range = n_times > 1 ? (x_raw[end] - x_raw[1]) : 1.0
    x = 2.0 .* (x_raw .- x_mean) ./ x_range

    L = boundary  # boundary * half-range of normalised x (which is 1)

    basis = Matrix{Float64}(undef, n_times, n_basis)
    for j in 1:n_basis
        λ = j * π / (2.0 * L)
        @. basis[:, j] = sin(λ * (x + L)) / sqrt(L)
    end
    basis
end

"""
    diagSPD_Matern32(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 3/2 kernel.
Matches Stan: `2 * alpha * (sqrt(3)/rho)^1.5 / ((sqrt(3)/rho)^2 + ω²)`.
"""
function diagSPD_Matern32(alpha, rho, L, n_basis)
    factor = 2.0 * alpha * (sqrt(3.0) / rho)^1.5
    [factor / ((sqrt(3.0) / rho)^2 + (j * π / (2.0 * L))^2)
     for j in 1:n_basis]
end

"""
    diagSPD_Matern12(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 1/2 kernel (Ornstein-Uhlenbeck).
Matches Stan: `alpha * sqrt(2 / (rho * (1/rho² + ω²)))`.
"""
function diagSPD_Matern12(alpha, rho, L, n_basis)
    [alpha * sqrt(2.0 / (rho * (1.0 / rho^2 + (j * π / (2.0 * L))^2)))
     for j in 1:n_basis]
end

"""
    diagSPD_Matern52(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 5/2 kernel.
Matches Stan: `alpha * sqrt(16 * (sqrt(5)/rho)^5 / (3 * ((sqrt(5)/rho)² + ω²)³))`.
"""
function diagSPD_Matern52(alpha, rho, L, n_basis)
    factor = 16.0 * (sqrt(5.0) / rho)^5
    [alpha * sqrt(factor / (3.0 * ((sqrt(5.0) / rho)^2 + (j * π / (2.0 * L))^2)^3))
     for j in 1:n_basis]
end

"""
    diagSPD_EQ(alpha, rho, L, n_basis)

Diagonal spectral density for squared exponential (exponentiated quadratic) kernel.
"""
function diagSPD_EQ(alpha, rho, L, n_basis)
    [alpha * sqrt(sqrt(2.0 * π) * rho) *
     exp(-0.25 * (rho * j * π / (2.0 * L))^2)
     for j in 1:n_basis]
end

"""
    hsgp_coefficients(n_basis, boundary, alpha, rho, kernel, matern_order)

Compute spectral density weights for HSGP basis functions using
kernel-specific closed forms matching Stan.
"""
function hsgp_coefficients(
    n_basis::Int, boundary::Float64,
    alpha, rho, kernel::Symbol, matern_order::Float64
)
    if kernel == :se
        diagSPD_EQ(alpha, rho, boundary, n_basis)
    elseif kernel == :matern
        if matern_order ≈ 0.5
            diagSPD_Matern12(alpha, rho, boundary, n_basis)
        elseif matern_order ≈ 1.5
            diagSPD_Matern32(alpha, rho, boundary, n_basis)
        elseif matern_order ≈ 2.5
            diagSPD_Matern52(alpha, rho, boundary, n_basis)
        else
            error("Unsupported Matérn order: $matern_order. Use 0.5, 1.5, or 2.5")
        end
    else
        error("Unknown kernel: $kernel")
    end
end

# ── Day-of-week effect ───────────────────────────────────────────────────

"""
    apply_day_of_week(expected, effect, start_day, n)

Scale expected values by day-of-week effects. AD-safe (no mutation).
"""
function apply_day_of_week(
    expected::AbstractVector,
    effect::AbstractVector,
    start_day::Int,
    n::Int
)
    [expected[t] * effect[mod1(start_day + t - 1, 7)] for t in 1:n]
end

# ══════════════════════════════════════════════════════════════════════════
# Turing models
# ══════════════════════════════════════════════════════════════════════════

"""
    infections_model(data, gt_pmf, delay_pmf, ...)

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
    initial_infections_guess::Float64,
    # HSGP basis (precomputed)
    hsgp_basis_matrix::Union{Matrix{Float64}, Nothing},
    # Model structure flags
    use_rt::Bool,
    use_gp::Bool,
    gp_n_basis::Int,
    gp_boundary::Float64,
    gp_kernel::Symbol,
    gp_matern_order::Float64,
    gp_on::Symbol,
    n_noise_terms::Int,
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
    pop::Float64,
    # Priors (passed from options)
    rt_prior::Distribution,
    gp_alpha_prior::Distribution,
    gp_ls_prior::Distribution,
    obs_dispersion_prior::Distribution,
    obs_scale_prior::Union{Distribution, Nothing},
    # Uncertain delay distributions (nothing = use fixed gt_pmf/delay_pmf)
    gt_uncertain::Union{UncertainDistribution, Nothing},
    delay_uncertain::Union{UncertainDistribution, Nothing},
    # Truncation adjustment
    trunc_rev_cmf::Union{AbstractVector{Float64}, Nothing},
    trunc_uncertain::Union{UncertainDistribution, Nothing}
)
    total_times = n_times + n_forecast

    # ── Uncertain delay parameters ────────────────────────────────────
    # When delay distributions have uncertain parameters, sample them
    # and recompute the PMF. Otherwise use the precomputed fixed PMF.
    if !isnothing(gt_uncertain)
        gt_param_1 ~ gt_uncertain.param_priors[1]
        gt_param_2 ~ gt_uncertain.param_priors[2]
        gt_pmf = discretise_ad(
            gt_uncertain.constructor(gt_param_1, gt_param_2),
            Int(gt_uncertain.max)
        )
    end

    if !isnothing(delay_uncertain)
        delay_param_1 ~ delay_uncertain.param_priors[1]
        delay_param_2 ~ delay_uncertain.param_priors[2]
        delay_pmf = discretise_ad(
            delay_uncertain.constructor(delay_param_1, delay_param_2),
            Int(delay_uncertain.max)
        )
    end

    # ── Priors: Rt ───────────────────────────────────────────────────
    R0 ~ rt_prior
    log_R0 = log(R0)

    # ── Initial conditions (seeding) ─────────────────────────────────
    log_initial_infections ~ Normal(initial_infections_guess, 2.0)
    growth = _R_to_r(R0, gt_pmf)
    initial_infections = [
        exp(log_initial_infections + growth * (s - seeding_time))
        for s in 1:seeding_time
    ]

    if use_gp && !use_rw
        # Gaussian process on log(Rt)
        gp_alpha ~ truncated(gp_alpha_prior; lower=0.0)
        gp_rho ~ truncated(gp_ls_prior; lower=0.0)

        rescaled_rho = 2.0 * gp_rho / n_noise_terms

        gp_z ~ filldist(Normal(0.0, 1.0), gp_n_basis)

        spd_weights = hsgp_coefficients(
            gp_n_basis, gp_boundary,
            gp_alpha, rescaled_rho, gp_kernel, gp_matern_order
        )
        gp_f = hsgp_basis_matrix * (spd_weights .* gp_z)

        if gp_on == :R0
            # Stationary: GP values added directly, extend last value
            gp_full = if length(gp_f) < total_times
                vcat(gp_f, fill(gp_f[end], total_times - length(gp_f)))
            else
                gp_f[1:total_times]
            end
            log_R = log_R0 .+ gp_full
        else
            # Non-stationary: cumsum of GP increments, prepend 0
            gp_cumsum = vcat([zero(eltype(gp_f))], cumsum(gp_f))
            # Extend to total_times by holding last value constant
            gp_full = if length(gp_cumsum) < total_times
                vcat(gp_cumsum, fill(gp_cumsum[end], total_times - length(gp_cumsum)))
            else
                gp_cumsum[1:total_times]
            end
            log_R = log_R0 .+ gp_full
        end
    elseif use_rw
        # Random walk on log(Rt)
        n_steps = div(total_times, rw_period)
        rw_sd ~ truncated(Normal(0.0, 0.1); lower=0.0)
        rw_noise ~ filldist(Normal(0.0, 1.0), n_steps)

        rw_values = cumsum(rw_sd .* rw_noise)
        # Expand to daily (AD-safe comprehension)
        log_R = [
            log_R0 + rw_values[min(div(t - 1, rw_period) + 1, n_steps)]
            for t in 1:total_times
        ]
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
    # Convolve all infections (including seeding) then extract post-seeding
    all_reports = convolve(all_infections, delay_pmf)
    expected_reports = all_reports[(seeding_time + 1):end]

    # ── Day-of-week effect ───────────────────────────────────────────
    if use_week_effect
        week_effect ~ Dirichlet(fill(1.0, week_length))
        scaled_effect = week_effect .* week_length
        expected_reports = apply_day_of_week(
            expected_reports, scaled_effect, start_day, total_times
        )
    end

    # ── Observation scale (fraction reported) ────────────────────────
    if use_obs_scale
        frac_observed ~ obs_scale_prior
        expected_reports = expected_reports .* frac_observed
    end

    # ── Truncation adjustment ─────────────────────────────────────
    if !isnothing(trunc_uncertain)
        trunc_param_1 ~ trunc_uncertain.param_priors[1]
        trunc_param_2 ~ trunc_uncertain.param_priors[2]
        trunc_dist = trunc_uncertain.constructor(trunc_param_1, trunc_param_2)
        trunc_pmf = discretise_ad(trunc_dist, Int(trunc_uncertain.max))
        trunc_rev_cmf = reverse(cumsum(trunc_pmf))
    end

    if !isnothing(trunc_rev_cmf)
        trunc_max = length(trunc_rev_cmf)
        expected_reports = [
            let days_from_end = n_times - t
                if days_from_end < trunc_max
                    expected_reports[t] * trunc_rev_cmf[trunc_max - days_from_end]
                else
                    expected_reports[t]
                end
            end
            for t in 1:length(expected_reports)
        ]
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
            obs_dispersion_prior; lower=0.0
        )
        φ = 1.0 / reporting_overdispersion^2
        for t in 1:n_times
            μ = max(expected_reports[t], 1e-6)
            Turing.@addlogprob! _negbin2_logpmf(cases[t], μ, φ)
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
Only for use with non-AD-tracked arguments (e.g. for sampling).
"""
function NegativeBinomial2(μ, φ)
    p = φ / (μ + φ)
    r = φ
    NegativeBinomial(r, p)
end

"""
    _negbin2_logpmf(y, μ, φ)

Log-PMF of NegativeBinomial2(μ, φ) at y. AD-compatible: avoids
constructing a NegativeBinomial distribution with tracked parameters.
Uses the standard formula: log p(y|μ,φ) = logΓ(y+φ) - logΓ(φ) - logΓ(y+1)
    + φ log(φ/(μ+φ)) + y log(μ/(μ+φ))
"""
function _negbin2_logpmf(y::Int, μ, φ)
    loggamma(y + φ) - loggamma(φ) - loggamma(y + 1) +
        φ * log(φ / (μ + φ)) + y * log(μ / (μ + φ))
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
    frac ~ Beta(5.0, 5.0)
    scaled_primary = frac .* Float64.(primary)
    expected = convolve(scaled_primary, delay_pmf)

    if is_prevalence
        expected = cumsum(expected)
    end

    if obs_family == :negbin
        overdispersion ~ truncated(Normal(0.0, 0.25); lower=0.0)
        φ = 1.0 / overdispersion^2
        for t in 1:n_times
            μ = max(expected[t], 1e-6)
            Turing.@addlogprob! _negbin2_logpmf(secondary[t], μ, φ)
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
    trunc_meanlog ~ Normal(0.0, 1.0)
    trunc_sdlog ~ truncated(Normal(1.0, 1.0); lower=0.0)

    d = Distributions.LogNormal(trunc_meanlog, trunc_sdlog)
    trunc_pmf = [cdf(d, k + 0.5) - cdf(d, k - 0.5) for k in 0:max_trunc]
    trunc_pmf = trunc_pmf ./ sum(trunc_pmf)
    trunc_cdf = cumsum(trunc_pmf)

    final = last(snapshots)

    for (i, snap) in enumerate(snapshots[1:end-1])
        n = snapshot_lengths[i]
        for t in 1:n
            days_truncated = n - t
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

    # Discretise distributions (fixed case) or pass through (uncertain case)
    gt_uncertain = generation_time.dist isa UncertainDistribution ?
        generation_time.dist : nothing
    delay_uncertain = delays.dist isa UncertainDistribution ?
        delays.dist : nothing

    # For fixed distributions, discretise now. For uncertain, provide a
    # placeholder PMF (the model will recompute from sampled params).
    # Generation time PMF: drop P(GT=0) since the renewal equation
    # indexes gt_pmf[s] as the weight for infections[t-s], i.e.,
    # gt_pmf[1] = P(GT=1). This matches R/Stan convention.
    # Generation time PMF: drop P(GT=0) since gt_pmf[s] weights
    # infections[t-s] in the renewal equation (1-indexed).
    gt_pmf = if isnothing(gt_uncertain)
        _drop_zero_delay(discretise(generation_time.dist).pmf)
    else
        discretise_ad(
            gt_uncertain.constructor([mean(p) for p in gt_uncertain.param_priors]...),
            Int(gt_uncertain.max)
        )
    end

    delay_pmf = if isnothing(delay_uncertain)
        discretise(delays.dist).pmf
    else
        discretise_ad(
            delay_uncertain.constructor([mean(p) for p in delay_uncertain.param_priors]...),
            Int(delay_uncertain.max)
        )
    end

    # Truncation: fixed, uncertain, or none
    trunc_uncertain = truncation.dist isa UncertainDistribution ?
        truncation.dist : nothing
    trunc_rev_cmf = if !isnothing(trunc_uncertain)
        nothing  # model will compute from sampled params
    elseif truncation.dist isa Dirac && truncation.dist.value == 0.0
        nothing  # no truncation
    else
        trunc_pmf = discretise(truncation.dist).pmf
        reverse(cumsum(trunc_pmf))
    end

    # Seeding time: max of generation time and total delay, at least 1
    # Matches R: max(sum of delay means, max generation time)
    seeding_time = max(length(gt_pmf) - 1, length(delay_pmf) - 1, 1)

    # GP configuration
    use_gp = rt.use_rt && gp.basis_prop > 0 && rt.rw == 0
    stationary = rt.gp_on == :R0
    future_fixed = rt.future == :latest
    # Number of GP noise terms (matches Stan's setup_noise)
    noise_terms = if !use_gp
        0
    else
        nt = stationary ? total_times : total_times - 1
        future_fixed ? nt - n_forecast : nt
    end
    gp_n_basis = use_gp ? max(1, round(Int, gp.basis_prop * noise_terms)) : 0
    gp_boundary = gp.boundary_scale

    # Precompute HSGP basis on noise_terms dimensions
    basis_matrix = use_gp ?
        hsgp_basis(gp_n_basis, gp_boundary, noise_terms) : nothing

    # Random walk configuration
    use_rw = rt.use_rt && rt.rw > 0

    # Day of week
    start_day = Dates.dayofweek(data.date[1])

    # Observation scale
    use_obs_scale = obs.scale isa Distribution
    obs_scale_prior = use_obs_scale ? obs.scale : nothing

    # Empirical prior on initial infections: log(mean(first 7 cases))
    # Matches Stan's initial_infections_guess
    n_early = min(7, n_times)
    initial_infections_guess = max(0.0,
        log(Statistics.mean(data.confirm[1:n_early]) + 1e-6))

    model = infections_model(
        data.confirm,
        gt_pmf,
        delay_pmf,
        n_times,
        n_forecast,
        seeding_time,
        initial_infections_guess,
        basis_matrix,
        rt.use_rt,
        use_gp,
        gp_n_basis,
        gp_boundary,
        gp.kernel,
        gp.matern_order,
        rt.gp_on,
        noise_terms,
        use_rw,
        rt.rw,
        obs.week_effect,
        obs.week_length,
        start_day,
        obs.family,
        use_obs_scale,
        data.accumulate,
        rt.pop,
        # Priors
        rt.prior,
        gp.alpha,
        gp.ls,
        obs.dispersion,
        obs_scale_prior,
        gt_uncertain,
        delay_uncertain,
        trunc_rev_cmf,
        trunc_uncertain
    )

    metadata = ModelMetadata(
        data.date, seeding_time, n_forecast,
        gt_pmf, delay_pmf, rt, obs
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

"""
    _drop_zero_delay(pmf)

Drop the P(delay=0) element from a PMF and renormalise.
"""
function _drop_zero_delay(pmf::AbstractVector)
    stripped = pmf[2:end]
    stripped ./ sum(stripped)
end

"""
    _R_to_r(R, gt_pmf; tol=1e-3)

Convert reproduction number R to exponential growth rate r using
Newton's method. Solves R * Σ_k pmf[k] * exp(-r*k) = 1.
"""
function _R_to_r(R, gt_pmf; tol=1e-3)
    n = length(gt_pmf)
    k_series = collect(1.0:n)
    mean_gt = sum(gt_pmf[i] * k_series[i] for i in 1:n)
    r = max((R - 1) / (R * mean_gt), -1.0)
    step = tol + 1.0
    while abs(step) > tol
        exp_r = exp.(-r .* k_series)
        num = R * sum(gt_pmf .* exp_r) - 1.0
        den = -R * sum(gt_pmf .* k_series .* exp_r)
        step = num / den
        r -= step
    end
    r
end

function _apply_accumulation(
    expected::AbstractVector, accumulate::AbstractVector{Bool}, n::Int
)
    buf = zero(eltype(expected))
    [begin
        buf = buf + expected[t]
        if !accumulate[t]
            out = buf
            buf = zero(eltype(expected))
            out
        else
            zero(eltype(expected))
        end
    end for t in 1:n]
end
