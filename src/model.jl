# ── Turing model implementations ─────────────────────────────────────────
#
# Direct Turing.jl implementations of EpiNow2's epidemiological models.
# Helper functions are AD-compatible with ForwardDiff.

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

Generate infections via the renewal equation. Pre-allocates
the output vector and fills sequentially. Compatible with ForwardDiff.
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

    T = promote_type(eltype(R), eltype(initial_infections), eltype(gt_pmf))
    infections = Vector{T}(undef, total)
    infections[1:seeding_time] .= initial_infections

    for t in (seeding_time + 1):total
        infectiousness = sum(
            infections[t - s] * gt_pmf[s]
            for s in 1:min(t - 1, gt_len)
        )
        infections[t] = R[t - seeding_time] * infectiousness
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
    diagSPD_Periodic(alpha, rho, n_basis)

Diagonal spectral density for periodic kernel.
Uses modified Bessel function of the first kind.
Matches Stan: `q[j] = exp(log(alpha) + 0.5*(log(2) - a + log_besselI(j, a)))`.
Returns vector of length 2*n_basis (cos + sin components).
"""
function diagSPD_Periodic(alpha, rho, n_basis)
    a = 1.0 / rho^2
    q = [exp(log(alpha) + 0.5 * (log(2.0) - a +
         log(besseli(j, a)))) for j in 1:n_basis]
    vcat(q, q)
end

"""
    hsgp_periodic_basis(n_basis, w0, n_times)

Compute periodic HSGP basis functions: [cos(m*w0*x), sin(m*w0*x)]
for m = 1..n_basis. Returns matrix of shape (n_times, 2*n_basis).
"""
function hsgp_periodic_basis(n_basis::Int, w0::Float64, n_times::Int)
    x_raw = collect(1.0:n_times)
    x_mean = mean(x_raw)
    x_range = n_times > 1 ? (x_raw[end] - x_raw[1]) : 1.0
    x = 2.0 .* (x_raw .- x_mean) ./ x_range

    basis = Matrix{Float64}(undef, n_times, 2 * n_basis)
    for m in 1:n_basis
        mw0x = w0 .* x .* m
        basis[:, m] = cos.(mw0x)
        basis[:, n_basis + m] = sin.(mw0x)
    end
    basis
end

"""
    hsgp_coefficients(n_basis, boundary, alpha, rho, kernel, matern_order)

Compute spectral density weights for HSGP basis functions using
kernel-specific closed forms matching Stan.
"""
function hsgp_coefficients(
    n_basis::Int, boundary::Float64,
    alpha, rho, kernel::GPKernel, matern_order::Float64
)
    if kernel == se
        diagSPD_EQ(alpha, rho, boundary, n_basis)
    elseif kernel == matern
        if matern_order ≈ 0.5
            diagSPD_Matern12(alpha, rho, boundary, n_basis)
        elseif matern_order ≈ 1.5
            diagSPD_Matern32(alpha, rho, boundary, n_basis)
        elseif matern_order ≈ 2.5
            diagSPD_Matern52(alpha, rho, boundary, n_basis)
        else
            error("Unsupported Matérn order: $matern_order. Use 0.5, 1.5, or 2.5")
        end
    elseif kernel == periodic
        diagSPD_Periodic(alpha, rho, n_basis)
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
    gp_kernel::GPKernel,
    gp_matern_order::Float64,
    gp_on::GPTarget,
    n_noise_terms::Int,
    use_rw::Bool,
    rw_period::Int,
    use_week_effect::Bool,
    week_length::Int,
    start_day::Int,
    obs_family::ObsFamily,
    use_obs_scale::Bool,
    # Accumulation
    accumulate::AbstractVector{Bool},
    # Population adjustment
    pop::Float64,
    pop_prior::Union{Distribution, Nothing},
    pop_period::PopPeriod,
    pop_floor::Float64,
    n_non_horizon::Int,
    # Priors (passed from options)
    rt_prior::Distribution,
    gp_alpha_prior::Distribution,
    gp_ls_prior::Distribution,
    obs_dispersion_prior::Distribution,
    obs_scale_prior::Union{Distribution, Nothing},
    # Uncertain delay distributions (empty = use fixed gt_pmf/delay_pmf)
    gt_uncertain::Vector{UncertainDistribution},
    delay_uncertain::Vector{UncertainDistribution},
    # Truncation adjustment
    trunc_rev_cmf::Union{AbstractVector{Float64}, Nothing},
    trunc_uncertain::Union{UncertainDistribution, Nothing},
    # Likelihood weight (power-likelihood tempering)
    obs_weight::Float64,
    # Breakpoints (0 = no breakpoints, otherwise index array)
    bp_n::Int,
    bps::AbstractVector{Int},
    # Back-calculation
    shifted_cases::Union{AbstractVector{Float64}, Nothing},
    backcalc_prior::BackcalcPrior,
    # Prior weighting (1/n_obs when weight_prior=true)
    delay_prior_weight::Float64
)
    total_times = n_times + n_forecast

    # ── Uncertain delay parameters ────────────────────────────────────
    # When delay distributions have uncertain parameters, sample them
    # and recompute the PMF. Otherwise use the precomputed fixed PMF.
    if !isempty(gt_uncertain)
        gt_pmfs = Vector{Any}(undef, length(gt_uncertain))
        for (ci, ud) in enumerate(gt_uncertain)
            # Sample all params as one tracked vector via arraydist; this
            # gives ReverseDiff a typed storage to write gradients into.
            # (`Vector{Real}(undef, ...)` was abstract-eltype and broke
            # ReverseDiff's increment_deriv! propagation.)
            ud_params ~ arraydist(ud.param_priors)
            if !(delay_prior_weight ≈ 1.0)
                # Scale prior contribution: add (weight-1)*logpdf to get weight*logpdf total
                Turing.@addlogprob! (delay_prior_weight - 1.0) *
                    sum(logpdf(p, ud_params[i])
                        for (i, p) in enumerate(ud.param_priors))
            end
            gt_pmfs[ci] = discretise_ad(
                ud.constructor(ud_params...), Int(ud.max)
            )
        end
        gt_pmf = length(gt_pmfs) == 1 ? gt_pmfs[1] :
            reduce((a, b) -> _convolve_pmfs(a, b), gt_pmfs)
    end

    if !isempty(delay_uncertain)
        delay_pmfs = Vector{Any}(undef, length(delay_uncertain))
        for (ci, ud) in enumerate(delay_uncertain)
            ud_params ~ arraydist(ud.param_priors)
            if !(delay_prior_weight ≈ 1.0)
                Turing.@addlogprob! (delay_prior_weight - 1.0) *
                    sum(logpdf(p, ud_params[i])
                        for (i, p) in enumerate(ud.param_priors))
            end
            delay_pmfs[ci] = discretise_ad(
                ud.constructor(ud_params...), Int(ud.max)
            )
        end
        delay_pmf = length(delay_pmfs) == 1 ? delay_pmfs[1] :
            reduce((a, b) -> _convolve_pmfs(a, b), delay_pmfs)
    end

    # ── Priors: Rt ───────────────────────────────────────────────────
    R0 ~ rt_prior
    log_R0 = log(R0)

    # ── Initial conditions (seeding) ─────────────────────────────────
    log_initial_infections ~ Normal(initial_infections_guess, 2.0)
    growth = R_to_growth(R0, gt_pmf)
    initial_infections = [
        exp(log_initial_infections + growth * (s - seeding_time))
        for s in 1:seeding_time
    ]

    # ── Breakpoints ──────────────────────────────────────────────────
    bp_offset = if bp_n > 0
        bp_sd ~ truncated(Normal(0.0, 0.1); lower=0.0)
        bp_effects ~ filldist(Normal(0.0, bp_sd), bp_n)
        bp0 = vcat([zero(eltype(bp_effects))], cumsum(bp_effects))
        [bp0[bps[t]] for t in 1:total_times]
    else
        fill(0.0, total_times)
    end

    if use_rt && use_gp && !use_rw
        # Gaussian process on log(Rt)
        gp_alpha ~ truncated(gp_alpha_prior; lower=0.0)
        gp_rho ~ truncated(gp_ls_prior; lower=0.0)

        rescaled_rho = 2.0 * gp_rho / n_noise_terms

        # Periodic kernel uses 2*n_basis terms (cos + sin components)
        n_gp_terms = gp_kernel == periodic ? 2 * gp_n_basis : gp_n_basis
        gp_z ~ filldist(Normal(0.0, 1.0), n_gp_terms)

        spd_weights = hsgp_coefficients(
            gp_n_basis, gp_boundary,
            gp_alpha, rescaled_rho, gp_kernel, gp_matern_order
        )
        gp_f = hsgp_basis_matrix * (spd_weights .* gp_z)

        if gp_on == gp_R0
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
    elseif use_rt
        # Fixed Rt (no GP, no random walk)
        log_R = fill(log_R0, total_times)
    end

    if use_rt
        # Apply breakpoint offsets
        if bp_n > 0
            log_R = log_R .+ bp_offset
        end

        R = exp.(log_R)

        # ── Generate infections (renewal equation) ───────────────────
        # Sample population if uncertain
        if !isnothing(pop_prior)
            pop ~ pop_prior
        end

        if pop > 0
            # SIR-like depletion: infections = S * (1 - exp(-R * infectiousness / S))
            gt_len = length(gt_pmf)
            T = promote_type(eltype(R), eltype(initial_infections), eltype(gt_pmf))
            all_infections = Vector{T}(undef, seeding_time + total_times)
            all_infections[1:seeding_time] .= initial_infections
            cum_inf = sum(initial_infections)
            # Track depletion-adjusted Rt = R * (S / pop) for time steps
            # where the depletion model is active. R itself is the
            # transmission Rt assuming a fully susceptible population
            # (mirrors EpiNow2 R v1.8 R / R_unadjusted split).
            R_adjusted = Vector{T}(undef, total_times)

            for s in 1:total_times
                t = seeding_time + s
                infectiousness = sum(
                    all_infections[t - j] * gt_pmf[j]
                    for j in 1:min(t - 1, gt_len)
                )
                use_pop = (pop_period == pop_all) ||
                          (pop_period == pop_forecast && s > n_non_horizon)
                if use_pop
                    susceptible = max(pop_floor, pop - cum_inf)
                    exp_adj = exp(-R[s] * infectiousness / susceptible)
                    all_infections[t] = susceptible * max(0.0, 1.0 - exp_adj)
                    R_adjusted[s] = R[s] * (susceptible / pop)
                else
                    all_infections[t] = R[s] * infectiousness
                    R_adjusted[s] = R[s]
                end
                cum_inf += all_infections[t]
            end
        else
            all_infections = renewal_infections(
                R, initial_infections, gt_pmf, total_times
            )
            R_adjusted = R
        end

        # Extract post-seeding infections
        infections = all_infections[(seeding_time + 1):end]
    else
        # ── Back-calculation (deconvolution without Rt) ──────────────
        # GP noise applied to shifted cases to recover infections
        if use_gp
            gp_alpha ~ truncated(gp_alpha_prior; lower=0.0)
            gp_rho ~ truncated(gp_ls_prior; lower=0.0)
            rescaled_rho = 2.0 * gp_rho / n_noise_terms
            n_gp_terms = gp_kernel == periodic ? 2 * gp_n_basis : gp_n_basis
            gp_z ~ filldist(Normal(0.0, 1.0), n_gp_terms)
            spd_weights = hsgp_coefficients(
                gp_n_basis, gp_boundary,
                gp_alpha, rescaled_rho, gp_kernel, gp_matern_order
            )
            noise = hsgp_basis_matrix * (spd_weights .* gp_z)
        else
            noise = zeros(total_times)
        end

        exp_noise = exp.(noise)

        infections = if backcalc_prior == bc_infections
            # Multiplicative correction: infections = shifted_cases * exp(noise)
            [max(shifted_cases[t] * exp_noise[t], 1e-5) for t in 1:total_times]
        elseif backcalc_prior == bc_none
            # Pure GP: infections = exp(noise)
            [max(exp_noise[t], 1e-5) for t in 1:total_times]
        elseif backcalc_prior == bc_growth_rate
            # Random walk: infections[t] = infections[t-1] * exp(noise[t])
            inf_rw = Vector{eltype(exp_noise)}(undef, total_times)
            inf_rw[1] = max(shifted_cases[1] * exp_noise[1], 1e-5)
            for t in 2:total_times
                inf_rw[t] = max(inf_rw[t-1] * exp_noise[t], 1e-5)
            end
            inf_rw
        else
            error("Unknown backcalc prior: $backcalc_prior")
        end

        # Compute Rt post-hoc via Cori method: R[t] = infections[t] / infectiousness[t]
        gt_len = length(gt_pmf)
        R = [
            let infectiousness = sum(
                    infections[max(t - s, 1)] * gt_pmf[s]
                    for s in 1:min(t - 1, gt_len)
                )
                infectiousness > 1e-10 ? infections[t] / infectiousness : 1.0
            end
            for t in 1:total_times
        ]
        log_R = log.(R)
        # No depletion adjustment in the back-calculation branch.
        R_adjusted = R
    end

    # ── Map to expected reports ──────────────────────────────────────
    if use_rt
        # Convolve all infections (including seeding) then extract post-seeding
        all_reports = convolve(all_infections, delay_pmf)
        expected_reports = all_reports[(seeding_time + 1):end]
    else
        # Back-calculation: infections are already at the report-ready scale
        # (shifted_cases were the reports shifted back by delay)
        expected_reports = convolve(infections, delay_pmf)
    end

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
        trunc_params ~ arraydist(trunc_uncertain.param_priors)
        trunc_dist = trunc_uncertain.constructor(trunc_params...)
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
        # Extend accumulate pattern to cover the forecast period
        accum_full = if length(accumulate) < total_times
            vcat(accumulate, fill(false, total_times - length(accumulate)))
        else
            accumulate[1:total_times]
        end
        expected_reports = _apply_accumulation(
            expected_reports, accum_full, total_times
        )
    end

    # ── Likelihood ───────────────────────────────────────────────────
    if obs_family == negbin
        reporting_overdispersion ~ truncated(
            obs_dispersion_prior; lower=0.0
        )
        φ = 1.0 / reporting_overdispersion^2
        for t in 1:n_times
            μ = max(expected_reports[t], 1e-6)
            Turing.@addlogprob! obs_weight * _negbin2_logpmf(cases[t], μ, φ)
        end
    else  # poisson
        for t in 1:n_times
            μ = max(expected_reports[t], 1e-6)
            Turing.@addlogprob! obs_weight * logpdf(Poisson(μ), cases[t])
        end
    end

    # ── Return generated quantities ──────────────────────────────────
    # `R` is the depletion-adjusted (effective) reproduction number when
    # `pop > 0`; `R_unadjusted` is the transmission Rt assuming a fully
    # susceptible population (mirrors EpiNow2 R v1.8 split). When pop is
    # off, both are identical.
    return (
        infections = infections,
        reports = expected_reports,
        R = R_adjusted,
        R_unadjusted = R,
        log_R = log_R,
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
    n_obs::Int,
    burn_in::Int,
    obs_family::ObsFamily,
    # Secondary model structure flags
    cumulative::Bool,
    historic::Bool,
    primary_hist_additive::Bool,
    current::Bool,
    primary_current_additive::Bool,
    # Day-of-week
    use_week_effect::Bool,
    week_length::Int,
    start_day::Int,
    # Frac prior
    frac_prior::Distribution
)
    frac ~ frac_prior
    n_total = length(primary)
    scaled_primary = frac .* Float64.(primary)
    conv_primary = convolve(scaled_primary, delay_pmf)

    # Calculate secondary using R's calculate_secondary logic
    expected = Vector{eltype(conv_primary)}(undef, n_total)
    for i in 1:n_total
        s = 1e-6
        # Cumulative: carry forward
        if cumulative && i > 1
            s += expected[i - 1]
        end
        # Historic: add/subtract convolved history
        if historic
            if primary_hist_additive
                s += conv_primary[i]
            else
                s = max(1e-6, s - conv_primary[i])
            end
        end
        # Current: add/subtract current primary
        if current
            if primary_current_additive
                s += scaled_primary[i]
            else
                s = max(1e-6, s - scaled_primary[i])
            end
        end
        expected[i] = s
    end

    # Day-of-week effects
    if use_week_effect
        week_effect ~ Dirichlet(fill(1.0, week_length))
        scaled_effect = week_effect .* week_length
        expected = apply_day_of_week(
            expected, scaled_effect, start_day, n_total
        )
    end

    if obs_family == negbin
        overdispersion ~ truncated(Normal(0.0, 0.25); lower=0.0)
        φ = 1.0 / overdispersion^2
        for t in 1:n_obs
            μ = max(expected[burn_in + t], 1e-6)
            Turing.@addlogprob! _negbin2_logpmf(secondary[t], μ, φ)
        end
    else
        for t in 1:n_obs
            μ = max(expected[burn_in + t], 1e-6)
            secondary[t] ~ Poisson(μ)
        end
    end

    return (expected = expected,)
end

# ── Truncation estimation model ──────────────────────────────────────────

@model function truncation_model(
    snapshots::Vector{Vector{Int}},
    snapshot_lengths::Vector{Int},
    max_trunc::Int,
    meanlog_prior::Distribution,
    sdlog_prior::Distribution
)
    trunc_meanlog ~ meanlog_prior
    # Sample on log scale so sdlog is always positive
    log_trunc_sdlog ~ Normal(log(mean(sdlog_prior)), std(sdlog_prior) / mean(sdlog_prior))
    trunc_sdlog = exp(log_trunc_sdlog)

    d = Distributions.LogNormal(trunc_meanlog, trunc_sdlog)
    cd = CensoredDistributions.double_interval_censored(d; interval=1, upper=max_trunc + 1)
    trunc_pmf = [exp(logpdf(cd, k)) for k in 0:max_trunc]
    trunc_pmf = trunc_pmf ./ sum(trunc_pmf)
    trunc_cdf = cumsum(trunc_pmf)

    final = last(snapshots)
    n_final = length(final)

    for (i, snap) in enumerate(snapshots[1:end-1])
        n = snapshot_lengths[i]
        for t in 1:min(n, n_final)
            days_truncated = n - t
            if days_truncated < max_trunc
                reporting_prob = trunc_cdf[days_truncated + 1]
            else
                reporting_prob = 1.0
            end
            expected = final[t] * reporting_prob
            Turing.@addlogprob! logpdf(Poisson(max(expected, 1e-6)), snap[t])
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

    # Extract uncertain distribution components
    gt_uncertain = _extract_uncertain(generation_time.dist)
    delay_uncertain = _extract_uncertain(delays.dist)

    # For fixed distributions, discretise now. For uncertain, provide a
    # placeholder PMF (the model will recompute from sampled params).
    # Generation time PMF: drop P(GT=0) since gt_pmf[s] weights
    # infections[t-s] in the renewal equation (1-indexed).
    gt_pmf = if isempty(gt_uncertain)
        _drop_zero_delay(discretise(generation_time.dist).pmf)
    else
        # Placeholder PMF from prior means (model will recompute)
        pmfs = [discretise_ad(
            ud.constructor([mean(p) for p in ud.param_priors]...),
            Int(ud.max)
        ) for ud in gt_uncertain]
        reduce(_convolve_pmfs, pmfs)
    end

    delay_pmf = if isempty(delay_uncertain)
        discretise(delays.dist).pmf
    else
        pmfs = [discretise_ad(
            ud.constructor([mean(p) for p in ud.param_priors]...),
            Int(ud.max)
        ) for ud in delay_uncertain]
        reduce(_convolve_pmfs, pmfs)
    end

    # Truncation: fixed, uncertain, or none
    trunc_uncertain = truncation.dist isa UncertainDistribution ?
        truncation.dist : nothing
    trunc_rev_cmf = if trunc_uncertain !== nothing
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
    use_gp = gp.basis_prop > 0 && (rt.use_rt ? rt.rw == 0 : true)
    stationary = rt.gp_on == gp_R0
    future_fixed = rt.future in (latest, estimate)
    # Number of GP noise terms (matches Stan's setup_noise)
    noise_terms = if !use_gp
        0
    elseif !rt.use_rt
        # Back-calculation: GP covers full time series
        total_times
    else
        nt = stationary ? total_times : total_times - 1
        future_fixed ? nt - n_forecast + rt.fixed_from : nt
    end
    gp_n_basis = use_gp ? max(1, round(Int, gp.basis_prop * noise_terms)) : 0
    gp_boundary = gp.boundary_scale

    # Precompute HSGP basis on noise_terms dimensions
    # Periodic kernel uses cos/sin basis (2*n_basis columns)
    basis_matrix = if !use_gp
        nothing
    elseif gp.kernel == periodic
        hsgp_periodic_basis(gp_n_basis, gp.w0, noise_terms)
    else
        hsgp_basis(gp_n_basis, gp_boundary, noise_terms)
    end

    # Random walk configuration
    use_rw = rt.use_rt && rt.rw > 0

    # Day of week
    start_day = Dates.dayofweek(data.date[1])

    # Observation scale
    use_obs_scale = obs.scale isa Distribution
    obs_scale_prior = use_obs_scale ? obs.scale : nothing

    # Back-calculation: shift cases back by reporting delay
    shifted_cases = if !rt.use_rt
        delay_mean = sum((i - 1) * delay_pmf[i] for i in 1:length(delay_pmf))
        shift = round(Int, delay_mean)
        sc = Float64.(data.confirm)
        # Shift back: cases at time t represent infections at t - shift
        shifted = vcat(
            fill(max(1.0, sc[1]), shift),
            sc[1:end - min(shift, length(sc))]
        )
        # Extend for forecast
        if n_forecast > 0
            vcat(shifted, fill(shifted[end], n_forecast))
        else
            shifted
        end
    else
        nothing
    end

    # Breakpoints: convert 0/1 indicator column to cumulative group indices
    # bps[t] indexes into bp0 = [0; cumsum(bp_effects)] for t=1:total_times
    has_breakpoints = any(data.breakpoints .> 0)
    if has_breakpoints
        bp_cumsum = cumsum(data.breakpoints) .+ 1  # +1 for 1-indexing into bp0
        # Extend into forecast by holding last value
        bps = if n_forecast > 0
            vcat(bp_cumsum, fill(bp_cumsum[end], n_forecast))
        else
            bp_cumsum
        end
        bp_n = maximum(data.breakpoints |> cumsum)
    else
        bps = fill(1, total_times)
        bp_n = 0
    end

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
        rt.pop isa Distribution ? 0.0 : Float64(rt.pop),
        rt.pop isa Distribution ? rt.pop : nothing,
        rt.pop_period,
        rt.pop_floor,
        n_times,  # n_non_horizon (observations period)
        # Priors
        rt.prior,
        gp.alpha,
        gp.ls,
        obs.dispersion,
        obs_scale_prior,
        gt_uncertain,
        delay_uncertain,
        trunc_rev_cmf,
        trunc_uncertain,
        obs.likelihood ? obs.weight : 0.0,
        bp_n,
        bps,
        shifted_cases,
        backcalc.prior,
        # Prior weight: 1/n_obs when weight_prior=true, else 1.0
        (generation_time.weight_prior || delays.weight_prior) ?
            1.0 / n_times : 1.0
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

# R_to_growth and supporting functions adapted from EpiAware.jl
# (CDCgov/Rt-without-renewal), licensed under Apache License 2.0.
# See: https://github.com/CDCgov/Rt-without-renewal

"""
    _neg_mgf(r, w)

Negative moment generating function of the generation interval: Σ w[i] * exp(-r*i).
"""
_neg_mgf(r, w) = sum(w[i] * exp(-r * i) for i in eachindex(w))

"""
    _dneg_mgf_dr(r, w)

Derivative of the negative MGF w.r.t. r.
"""
_dneg_mgf_dr(r, w) = -sum(w[i] * i * exp(-r * i) for i in eachindex(w))

"""
    R_to_growth(R, gt_pmf; newton_steps=2)

Convert reproduction number R to exponential growth rate r.
Solves G(r) = 1/R where G is the negative MGF of the generation interval.

Uses a fixed number of Newton steps (default 2), which is AD-safe since the
loop count is deterministic. Based on the approach in EpiAware.jl.
"""
function R_to_growth(R, gt_pmf; newton_steps::Int=2)
    n = length(gt_pmf)
    mean_gt = sum(gt_pmf[i] * i for i in 1:n)
    # Small-r approximation as initial guess
    r = (R - 1) / (R * mean_gt)
    # Fixed Newton steps (AD-safe: deterministic loop count)
    for _ in 1:newton_steps
        r -= (R * _neg_mgf(r, gt_pmf) - 1) / (R * _dneg_mgf_dr(r, gt_pmf))
    end
    r
end

"""Extract UncertainDistribution components from any delay specification."""
function _extract_uncertain(d::UncertainDistribution)
    UncertainDistribution[d]
end
function _extract_uncertain(d::CompositeDelay)
    UncertainDistribution[c for c in d.components if c isa UncertainDistribution]
end
function _extract_uncertain(d)
    UncertainDistribution[]
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
