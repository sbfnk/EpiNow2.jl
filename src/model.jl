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
    return [
        sum(signal[t - j + 1] * kernel[j] for j in 1:min(t, k))
            for t in 1:n
    ]
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

    return infections
end

# ── HSGP: Hilbert space Gaussian process approximation ───────────────────

# ── HSGP approximation settings ──────────────────────────────────────────

# Half the range of the time points the GP is defined on.
gp_half_range(n::Int) = max(n - 1, 1) / 2

# Constants (m_factor, c_factor) from Riutort-Mayol et al. (2023) relating
# the lengthscale to the number of basis functions and the boundary that
# approximate it well; c_factor is unused for the periodic kernel.
function gp_approx_constants(kernel::GPKernel, matern_order::Float64)
    kernel == periodic && return (m_factor = 3.72, c_factor = NaN)
    (kernel == se || isinf(matern_order)) &&
        return (m_factor = 1.75, c_factor = 3.2)
    matern_order == 2.5 && return (m_factor = 2.65, c_factor = 4.1)
    matern_order == 0.5 && return (m_factor = 8.6, c_factor = 4.5)
    return (m_factor = 3.42, c_factor = 4.5)
end

# Quantiles of the lengthscale prior restricted to positive values.
function gp_ls_quantiles(ls::Distribution, probs = (0.05, 0.95))
    p_lower = cdf(ls, 0.0)
    return [quantile(ls, p_lower + p * (1 - p_lower)) for p in probs]
end

"""
    gp_basis_settings(gp::GPOpts, n::Int) -> (L, M)

Boundary factor `L` and number of basis functions `M` of the Hilbert space
approximation to a GP over `n` time points. Values left as `nothing` in `gp`
are chosen from the lengthscale prior so that lengthscales between its 5% and
95% quantiles are approximated well, as in R's `gp_basis_settings()`.
"""
function gp_basis_settings(gp::GPOpts, n::Int)
    S = gp_half_range(n)
    is_periodic = gp.kernel == periodic
    constants = gp_approx_constants(gp.kernel, gp.matern_order)
    q = gp_ls_quantiles(gp.ls)
    L = if is_periodic
        something(gp.boundary_scale, 1.5)
    else
        something(gp.boundary_scale, max(1.2, constants.c_factor * q[2] / S))
    end
    M = if isnothing(gp.basis_prop)
        ceil(Int, constants.m_factor * (is_periodic ? 1.0 : L) * S / q[1])
    else
        ceil(Int, n * gp.basis_prop)
    end
    return (Float64(L), max(1, M))
end

"""
    gp_ls_range(gp::GPOpts, n, L, M) -> (lower, upper)

Range of lengthscales (in days) that a Hilbert space GP approximation with
boundary factor `L` and `M` basis functions over `n` time points represents
accurately.
"""
function gp_ls_range(gp::GPOpts, n::Int, L::Float64, M::Int)
    constants = gp_approx_constants(gp.kernel, gp.matern_order)
    S = gp_half_range(n)
    is_periodic = gp.kernel == periodic
    return (
        constants.m_factor * (is_periodic ? 1.0 : L) * S / M,
        is_periodic ? Inf : L * S / constants.c_factor,
    )
end

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
    return basis
end

"""
    diagSPD_Matern32(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 3/2 kernel.
Matches Stan: `2 * alpha * (sqrt(3)/rho)^1.5 / ((sqrt(3)/rho)^2 + ω²)`.
"""
function diagSPD_Matern32(alpha, rho, L, n_basis)
    factor = 2.0 * alpha * (sqrt(3.0) / rho)^1.5
    return [
        factor / ((sqrt(3.0) / rho)^2 + (j * π / (2.0 * L))^2)
            for j in 1:n_basis
    ]
end

"""
    diagSPD_Matern12(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 1/2 kernel (Ornstein-Uhlenbeck).
Matches Stan: `alpha * sqrt(2 / (rho * (1/rho² + ω²)))`.
"""
function diagSPD_Matern12(alpha, rho, L, n_basis)
    return [
        alpha * sqrt(2.0 / (rho * (1.0 / rho^2 + (j * π / (2.0 * L))^2)))
            for j in 1:n_basis
    ]
end

"""
    diagSPD_Matern52(alpha, rho, L, n_basis)

Diagonal spectral density for Matérn 5/2 kernel.
Matches Stan: `alpha * sqrt(16 * (sqrt(5)/rho)^5 / (3 * ((sqrt(5)/rho)² + ω²)³))`.
"""
function diagSPD_Matern52(alpha, rho, L, n_basis)
    factor = 16.0 * (sqrt(5.0) / rho)^5
    return [
        alpha * sqrt(factor / (3.0 * ((sqrt(5.0) / rho)^2 + (j * π / (2.0 * L))^2)^3))
            for j in 1:n_basis
    ]
end

"""
    diagSPD_EQ(alpha, rho, L, n_basis)

Diagonal spectral density for squared exponential (exponentiated quadratic) kernel.
"""
function diagSPD_EQ(alpha, rho, L, n_basis)
    return [
        alpha * sqrt(sqrt(2.0 * π) * rho) *
            exp(-0.25 * (rho * j * π / (2.0 * L))^2)
            for j in 1:n_basis
    ]
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
    q = [
        exp(
            log(alpha) + 0.5 * (
                log(2.0) - a +
                    log(besseli(j, a))
            )
        ) for j in 1:n_basis
    ]
    return vcat(q, q)
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
    return basis
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
    return if kernel == se
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
    return [expected[t] * effect[mod1(start_day + t - 1, 7)] for t in 1:n]
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
        pop_start::Int,
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
        # Fixed components convolved with the sampled uncertain ones
        gt_fixed_pmf::AbstractVector{Float64},
        delay_fixed_pmf::AbstractVector{Float64},
        trunc_fixed_pmf::AbstractVector{Float64},
        # Prior weights: the number of observed time points when
        # `weight_prior` is set, as in R, otherwise 1
        gt_prior_weight::Float64,
        delay_prior_weight::Float64,
        trunc_prior_weight::Float64
    )
    total_times = n_times + n_forecast

    # ── Uncertain delay parameters ────────────────────────────────────
    # When delay distributions have uncertain parameters, sample them and
    # recompute the PMF, convolved with any fixed components. Each delay's
    # prior is weighted by `*_prior_weight`, as R's `weight_prior` does.
    if !isempty(gt_uncertain)
        gt_params = Vector{Any}(undef, length(gt_uncertain))
        for (ci, ud) in enumerate(gt_uncertain)
            gt_params[ci] ~ arraydist(ud.param_priors)
            if gt_prior_weight != 1
                Turing.@addlogprob! (gt_prior_weight - 1) *
                    _prior_logpdf(ud, gt_params[ci])
            end
        end
        gt_pmf = _drop_zero_delay(
            _uncertain_pmf(gt_uncertain, gt_params, gt_fixed_pmf)
        )
    end

    if !isempty(delay_uncertain)
        delay_params = Vector{Any}(undef, length(delay_uncertain))
        for (ci, ud) in enumerate(delay_uncertain)
            delay_params[ci] ~ arraydist(ud.param_priors)
            if delay_prior_weight != 1
                Turing.@addlogprob! (delay_prior_weight - 1) *
                    _prior_logpdf(ud, delay_params[ci])
            end
        end
        delay_pmf = _uncertain_pmf(
            delay_uncertain, delay_params, delay_fixed_pmf
        )
    end

    if use_rt
        # ── Rt ───────────────────────────────────────────────────────
        # The sampled intercept is the mean of log Rt over the observed
        # window, which keeps it from trading off against the GP and
        # breakpoint paths. The user's Rt prior applies to the derived
        # initial Rt, R[1], below.
        R_mean ~ Turing.FlatPos(0.0)

        gp_path = if use_gp
            gp_alpha ~ truncated(gp_alpha_prior; lower = 0.0)
            gp_rho ~ truncated(gp_ls_prior; lower = 0.0)
            rescaled_rho = 2.0 * gp_rho / max(n_noise_terms - 1, 1)
            # Periodic kernel uses 2*n_basis terms (cos + sin components)
            n_gp_terms = gp_kernel == periodic ? 2 * gp_n_basis : gp_n_basis
            gp_z ~ filldist(Normal(0.0, 1.0), n_gp_terms)
            spd_weights = hsgp_coefficients(
                gp_n_basis, gp_boundary,
                gp_alpha, rescaled_rho, gp_kernel, gp_matern_order
            )
            gp_f = hsgp_basis_matrix * (spd_weights .* gp_z)
            # A stationary GP enters log Rt directly; a non-stationary one
            # as the cumulative sum of daily increments starting from 0.
            raw_path = gp_on == gp_R0 ? gp_f :
                vcat([zero(eltype(gp_f))], cumsum(gp_f))
            _hold_forward(raw_path, total_times)
        else
            zeros(total_times)
        end

        bp_path = if bp_n > 0
            bp_sd ~ truncated(Normal(0.0, 0.1); lower = 0.0)
            bp_effects ~ filldist(Normal(0.0, bp_sd), bp_n)
            bp_levels = vcat([zero(eltype(bp_effects))], cumsum(bp_effects))
            [bp_levels[bps[t]] for t in 1:total_times]
        else
            zeros(total_times)
        end

        # Centre the paths over the observed window so that the mean of
        # log Rt there is log(R_mean). The stationary GP is not centred.
        log_intercept = log(R_mean)
        if use_gp && gp_on != gp_R0
            log_intercept -= mean(gp_path[1:n_non_horizon])
        end
        if bp_n > 0
            log_intercept -= mean(bp_path[1:n_non_horizon])
        end
        log_R = log_intercept .+ gp_path .+ bp_path

        # Prior on R[1]. The sampler moves on R_mean, whose positivity
        # transform already contributes log(R_mean), so the Jacobian of the
        # map to R[1] is log(R[1]) - log(R_mean).
        Turing.@addlogprob! logpdf(rt_prior, exp(log_R[1])) +
            log_R[1] - log(R_mean)

        # ── Initial conditions (seeding) ─────────────────────────────
        log_initial_infections ~ Normal(initial_infections_guess, 2.0)
        growth = R_to_growth(exp(log_R[1]), gt_pmf)
        if use_obs_scale
            frac_observed ~ obs_scale_prior
        end
        log_seed_scale = use_obs_scale ?
            log_initial_infections - log(frac_observed) :
            log_initial_infections
        initial_infections = [
            exp(log_seed_scale + growth * (s - 1 - seeding_time))
                for s in 1:seeding_time
        ]

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
                    (pop_period == pop_forecast && s > pop_start)
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
            gp_alpha ~ truncated(gp_alpha_prior; lower = 0.0)
            gp_rho ~ truncated(gp_ls_prior; lower = 0.0)
            rescaled_rho = 2.0 * gp_rho / max(n_noise_terms - 1, 1)
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
            [max(shifted_cases[t] * exp_noise[t], 1.0e-5) for t in 1:total_times]
        elseif backcalc_prior == bc_none
            # Pure GP: infections = exp(noise)
            [max(exp_noise[t], 1.0e-5) for t in 1:total_times]
        elseif backcalc_prior == bc_growth_rate
            # Random walk: infections[t] = infections[t-1] * exp(noise[t])
            inf_rw = Vector{eltype(exp_noise)}(undef, total_times)
            inf_rw[1] = max(shifted_cases[1] * exp_noise[1], 1.0e-5)
            for t in 2:total_times
                inf_rw[t] = max(inf_rw[t - 1] * exp_noise[t], 1.0e-5)
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
                infectiousness > 1.0e-10 ? infections[t] / infectiousness : 1.0
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
        if !use_rt
            frac_observed ~ obs_scale_prior
        end
        expected_reports = expected_reports .* frac_observed
    end

    # ── Truncation adjustment ─────────────────────────────────────
    if !isnothing(trunc_uncertain)
        trunc_params ~ arraydist(trunc_uncertain.param_priors)
        if trunc_prior_weight != 1
            Turing.@addlogprob! (trunc_prior_weight - 1) *
                _prior_logpdf(trunc_uncertain, trunc_params)
        end
        trunc_pmf = _uncertain_pmf(
            [trunc_uncertain], [trunc_params], trunc_fixed_pmf
        )
        trunc_rev_cmf = reverse(cumsum(trunc_pmf))
    end

    # As in R, truncation only enters the likelihood: the reports returned
    # stay untruncated, and only observed days are truncated.
    obs_reports = expected_reports
    if !isnothing(trunc_rev_cmf)
        trunc_max = length(trunc_rev_cmf)
        obs_reports = [
            let days_from_end = n_times - t
                if 0 <= days_from_end < trunc_max
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
        obs_reports = _apply_accumulation(obs_reports, accum_full, total_times)
    end

    # ── Likelihood ───────────────────────────────────────────────────
    if obs_family == negbin
        reporting_overdispersion ~ truncated(
            obs_dispersion_prior; lower = 0.0
        )
        φ = 1.0 / reporting_overdispersion^2
        for t in 1:n_times
            μ = max(obs_reports[t], 1.0e-6)
            Turing.@addlogprob! obs_weight * _negbin2_logpmf(cases[t], μ, φ)
        end
    else  # poisson
        for t in 1:n_times
            μ = max(obs_reports[t], 1.0e-6)
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
    return NegativeBinomial(r, p)
end

"""
    _negbin2_logpmf(y, μ, φ)

Log-PMF of NegativeBinomial2(μ, φ) at y. AD-compatible: avoids
constructing a NegativeBinomial distribution with tracked parameters.
Uses the standard formula: log p(y|μ,φ) = logΓ(y+φ) - logΓ(φ) - logΓ(y+1)
    + φ log(φ/(μ+φ)) + y log(μ/(μ+φ))
"""
function _negbin2_logpmf(y::Int, μ, φ)
    return loggamma(y + φ) - loggamma(φ) - loggamma(y + 1) +
        φ * log(φ / (μ + φ)) + y * log(μ / (μ + φ))
end

# ── Secondary observation recursion ──────────────────────────────────────

"""
    calculate_secondary(scaled_primary, conv_primary, cumulative, historic,
                        primary_hist_additive, current, primary_current_additive;
                        eps=1e-6)

Combine a scaled primary series and its delay-convolution into an expected
secondary series, following EpiNow2 R's `calculate_secondary` logic. The
boolean flags come from [`secondary_opts`](@ref):

- `cumulative`     — carry the previous value forward (stock, e.g. prevalence)
- `historic`       — add (`primary_hist_additive`) or subtract the convolved history
- `current`        — add (`primary_current_additive`) or subtract the current primary

For `incidence` this returns `conv_primary`; for `prevalence` it returns the
running balance `prev − outflow + inflow` (i.e. occupancy), not merely the
cumulative outflow. AD-safe (works with ForwardDiff dual numbers).
"""
function calculate_secondary(
        scaled_primary::AbstractVector,
        conv_primary::AbstractVector,
        cumulative::Bool,
        historic::Bool,
        primary_hist_additive::Bool,
        current::Bool,
        primary_current_additive::Bool;
        eps::Real = 1.0e-6
    )
    n = length(scaled_primary)
    T = promote_type(eltype(scaled_primary), eltype(conv_primary), typeof(eps))
    expected = Vector{T}(undef, n)
    for i in 1:n
        s = T(eps)
        if cumulative && i > 1
            s += expected[i - 1]
        end
        if historic
            s = primary_hist_additive ? s + conv_primary[i] :
                max(T(eps), s - conv_primary[i])
        end
        if current
            s = primary_current_additive ? s + scaled_primary[i] :
                max(T(eps), s - scaled_primary[i])
        end
        expected[i] = s
    end
    return expected
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

    # Calculate secondary using the shared calculate_secondary recursion
    expected = calculate_secondary(
        scaled_primary, conv_primary,
        cumulative, historic, primary_hist_additive,
        current, primary_current_additive
    )

    # Day-of-week effects
    if use_week_effect
        week_effect ~ Dirichlet(fill(1.0, week_length))
        scaled_effect = week_effect .* week_length
        expected = apply_day_of_week(
            expected, scaled_effect, start_day, n_total
        )
    end

    if obs_family == negbin
        overdispersion ~ truncated(Normal(0.0, 0.25); lower = 0.0)
        φ = 1.0 / overdispersion^2
        for t in 1:n_obs
            μ = max(expected[burn_in + t], 1.0e-6)
            Turing.@addlogprob! _negbin2_logpmf(secondary[t], μ, φ)
        end
    else
        for t in 1:n_obs
            μ = max(expected[burn_in + t], 1.0e-6)
            secondary[t] ~ Poisson(μ)
        end
    end

    return (expected = expected,)
end

# ── Truncation estimation model ──────────────────────────────────────────

"""
    _truncate_obs(reports, rev_cmf, reconstruct) -> Vector

Apply (or, when `reconstruct`, reverse) right-truncation to the final
`length(rev_cmf)` entries of `reports`, mirroring EpiNow2's Stan
`truncate_obs`. `rev_cmf` is the reversed cumulative truncation mass, so the
most recent entry is scaled by the smallest reporting fraction.
"""
function _truncate_obs(reports, rev_cmf, reconstruct::Bool)
    t = length(reports)
    trunc_max = length(rev_cmf)
    joint_max = min(t, trunc_max)
    first_t = t - joint_max + 1
    first_trunc = trunc_max - joint_max + 1
    return [
        k < first_t ? reports[k] :
            (
                reconstruct ? reports[k] / rev_cmf[first_trunc + k - first_t] :
                reports[k] * rev_cmf[first_trunc + k - first_t]
            )
            for k in 1:t
    ]
end

"""
    truncation_model(obs, obs_dist, trunc, trunc_fixed_pmf, family,
                     dispersion_prior, noise_prior, likelihood, prior_weight)

Turing model for `estimate_truncation()`, following R's
`estimate_truncation.stan`. The most complete (last) snapshot is
reconstructed without truncation, truncation is re-applied to map it onto
each earlier snapshot, and the earlier snapshots are observed with negative
binomial or Poisson noise around that, plus an additive `sigma`.

`trunc` is the uncertain truncation distribution, or `nothing` when the
truncation PMF `trunc_fixed_pmf` (from delay 0) is fixed.
"""
@model function truncation_model(
        obs::Matrix{Float64},
        obs_dist::Vector{Int},
        trunc::Union{UncertainDistribution, Nothing},
        trunc_fixed_pmf::AbstractVector{Float64},
        family::ObsFamily,
        dispersion_prior::Distribution,
        noise_prior::Distribution,
        likelihood::Bool,
        prior_weight::Float64
    )
    n_times, obs_sets = size(obs)

    trunc_pmf = if isnothing(trunc)
        trunc_fixed_pmf
    else
        trunc_params ~ arraydist(trunc.param_priors)
        if prior_weight != 1
            Turing.@addlogprob! (prior_weight - 1) *
                _prior_logpdf(trunc, trunc_params)
        end
        _uncertain_pmf([trunc], [trunc_params], trunc_fixed_pmf)
    end
    max_trunc = length(trunc_pmf) - 1
    trunc_rev_cmf = reverse(cumsum(trunc_pmf))

    if family == negbin
        reporting_overdispersion ~ truncated(dispersion_prior; lower = 0.0)
        φ = 1.0 / reporting_overdispersion^2
    end
    sigma ~ truncated(noise_prior; lower = 0.0)

    # Reconstruct the most complete (last) snapshot without truncation, then
    # re-apply truncation to map it back onto each earlier snapshot.
    last_obs = _truncate_obs(obs[:, obs_sets], trunc_rev_cmf, true)

    if likelihood
        for i in 1:(obs_sets - 1)
            end_t = n_times - obs_dist[i]
            start_t = max(1, end_t - max_trunc)
            expected = _truncate_obs(
                last_obs[start_t:end_t], trunc_rev_cmf, false
            ) .+ sigma
            for (j, idx) in enumerate(start_t:end_t)
                μ = max(expected[j], 1.0e-6)
                y = round(Int, obs[idx, i])
                Turing.@addlogprob! family == negbin ?
                    _negbin2_logpmf(y, μ, φ) : logpdf(Poisson(μ), y)
            end
        end
    end

    return (
        trunc_params = isnothing(trunc) ? Float64[] : trunc_params,
        trunc_pmf = trunc_pmf,
    )
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
        gp::Union{GPOpts, Nothing},
        obs::ObsOpts,
        forecast::ForecastOpts
    )
    n_times = length(data)
    n_forecast = forecast.horizon
    total_times = n_times + n_forecast

    # Extract uncertain distribution components
    gt_uncertain = _extract_uncertain(generation_time.dist)
    delay_uncertain = _extract_uncertain(delays.dist)

    # Fixed components are discretised now. With uncertain components the
    # PMFs below are placeholders at the prior means; the model recomputes
    # them from the sampled parameters, convolved with the fixed part.
    # The generation time PMF drops P(GT = 0) because gt_pmf[s] weights
    # infections[t - s] in the renewal equation.
    gt_fixed_pmf = _fixed_pmf(generation_time.dist)
    delay_fixed_pmf = _fixed_pmf(delays.dist)
    gt_pmf = _drop_zero_delay(
        _uncertain_pmf(
            gt_uncertain, _prior_means.(gt_uncertain), gt_fixed_pmf
        )
    )
    delay_pmf = _uncertain_pmf(
        delay_uncertain, _prior_means.(delay_uncertain), delay_fixed_pmf
    )

    # Truncation: fixed, uncertain, or none. Uncertain parts are sampled in
    # the model and convolved with the fixed parts, as for the delays.
    trunc_uncertain_parts = _extract_uncertain(truncation.dist)
    length(trunc_uncertain_parts) <= 1 || throw(
        ArgumentError("Only one uncertain truncation distribution is supported")
    )
    trunc_uncertain = isempty(trunc_uncertain_parts) ? nothing :
        only(trunc_uncertain_parts)
    trunc_fixed_pmf = _fixed_pmf(truncation.dist)
    trunc_rev_cmf = if trunc_uncertain !== nothing
        nothing  # model will compute from sampled params
    elseif truncation.dist isa Dirac && truncation.dist.value == 0.0
        nothing  # no truncation
    else
        trunc_pmf = discretise(truncation.dist).pmf
        reverse(cumsum(trunc_pmf))
    end

    # Seeding time as R's `get_seeding_time()`: the rounded sum of the delay
    # means, but at least the maximum generation time (the length of
    # `gt_pmf`, which starts at a delay of 1) and at least 1.
    seeding_time = max(
        round(Int, _delay_mean(delays.dist)), length(gt_pmf), 1
    )

    # GP configuration
    # `gp = nothing` switches the Gaussian process off; a random walk on
    # its own then needs `gp = nothing` as in R.
    use_gp = gp !== nothing
    gp = something(gp, gp_opts())
    stationary = rt.gp_on == gp_R0
    future_fixed = rt.future in (latest, estimate)
    # Offset of the last estimated Rt from the end of the data, as `from` in
    # R's `create_future_rt()`: 0 for `latest`, minus the hold-back for
    # `estimate`.
    fixed_offset = rt.future == estimate ?
        -something(rt.fixed_from, seeding_time) : 0
    # Number of GP noise terms (matches Stan's setup_noise)
    noise_terms = if !use_gp
        0
    elseif !rt.use_rt
        # Back-calculation: GP covers full time series
        total_times
    else
        nt = stationary ? total_times : total_times - 1
        future_fixed ? nt - n_forecast + fixed_offset : nt
    end
    gp_boundary, gp_n_basis = use_gp ? gp_basis_settings(gp, noise_terms) :
        (1.5, 0)

    # Precompute HSGP basis on noise_terms dimensions
    # Periodic kernel uses cos/sin basis (2*n_basis columns)
    basis_matrix = if !use_gp
        nothing
    elseif gp.kernel == periodic
        hsgp_periodic_basis(gp_n_basis, gp.w0, noise_terms)
    else
        hsgp_basis(gp_n_basis, gp_boundary, noise_terms)
    end

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
            sc[1:(end - min(shift, length(sc)))]
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

    # Breakpoints: bps[t] indexes the breakpoint levels
    # [0; cumsum(bp_effects)] for t = 1:total_times. A random walk is a
    # breakpoint every `rw` days, as in R's `create_rt_data()`.
    bps = if rt.use_rt && rt.rw > 0
        steps = [div(t, rt.rw) for t in 1:total_times]
        if rt.future != project
            max_bps = total_times - n_forecast + fixed_offset
            if max_bps < total_times
                steps[(max_bps + 1):end] .= steps[max_bps]
            end
        end
        steps
    else
        bp_cumsum = cumsum(data.breakpoints)
        vcat(bp_cumsum, fill(bp_cumsum[end], n_forecast))
    end
    bp_n = maximum(bps)
    bps = bps .+ 1

    # Empirical prior on initial infections: log(mean(first 7 cases))
    # Matches Stan's initial_infections_guess
    n_early = min(7, n_times)
    initial_infections_guess = max(
        0.0,
        log(Statistics.mean(data.confirm[1:n_early]) + 1.0e-6)
    )

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
        # Depletion for `pop_forecast` starts where Rt stops being
        # estimated, as R's `ot - ht`.
        n_times + fixed_offset,  # pop_start
        # Priors
        _positive_prior(rt.prior),
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
        gt_fixed_pmf,
        delay_fixed_pmf,
        trunc_fixed_pmf,
        generation_time.weight_prior ? Float64(n_times) : 1.0,
        delays.weight_prior ? Float64(n_times) : 1.0,
        truncation.weight_prior ? Float64(n_times) : 1.0
    )

    metadata = ModelMetadata(
        data.date, seeding_time, n_forecast,
        gt_pmf, delay_pmf, rt, obs,
        use_gp ? gp_ls_range(gp, noise_terms, gp_boundary, gp_n_basis) :
            nothing,
        gp_half_range(noise_terms)
    )

    return (model = model, metadata = metadata)
end

"""
    check_gp_lengthscale(rho, metadata)

Warn when the posterior median GP lengthscale `rho` falls outside the range
the Hilbert space approximation represents accurately, as R's
`check_gp_lengthscale()` does.
"""
function check_gp_lengthscale(rho::AbstractVector, metadata)
    range = metadata.gp_ls_range
    isnothing(range) && return nothing
    ls_median = median(rho)
    if ls_median + 0.01 * metadata.gp_half_range < range[1]
        @warn "The posterior median GP lengthscale " *
            "($(round(ls_median; sigdigits = 3)) days) is shorter than the " *
            "$(round(range[1]; sigdigits = 3)) days the approximate GP can " *
            "represent accurately. Increase `basis_prop` in `gp_opts()`, or " *
            "use a lengthscale prior with more weight on shorter lengthscales."
    end
    if ls_median > range[2]
        @warn "The posterior median GP lengthscale " *
            "($(round(ls_median; sigdigits = 3)) days) is longer than the " *
            "$(round(range[2]; sigdigits = 3)) days the approximate GP " *
            "boundary supports. Increase `boundary_scale` in `gp_opts()`, " *
            "or use a lengthscale prior with more weight on longer " *
            "lengthscales."
    end
    return range
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
    # Lengthscale range the GP approximation represents accurately, and the
    # half-range of its time points; `nothing` without a GP.
    gp_ls_range::Union{Nothing, Tuple{Float64, Float64}}
    gp_half_range::Float64
end

# ── Helpers ──────────────────────────────────────────────────────────────

"""
    _drop_zero_delay(pmf)

Drop the P(delay=0) element from a PMF and renormalise.
"""
# R truncates the Rt prior to positive values (`T[0, ]`); a lognormal or
# gamma prior is already positive.
_positive_prior(d::Distribution) = minimum(d) < 0 ? truncated(d; lower = 0.0) : d

# Extend `x` to length `n` by repeating its last value.
function _hold_forward(x::AbstractVector, n::Int)
    length(x) >= n && return x[1:n]
    return vcat(x, fill(x[end], n - length(x)))
end

function _drop_zero_delay(pmf::AbstractVector)
    stripped = pmf[2:end]
    return stripped ./ sum(stripped)
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
    R_to_growth(R, gt_pmf; newton_steps = 2)

Convert a reproduction number to an exponential growth rate by solving
`M(-r) = 1 / R`, where `M` is the moment generating function of the
generation time. The inverse of [`growth_to_R`](@ref).

A fixed number of Newton steps from a small-growth-rate approximation keeps
the calculation differentiable with automatic differentiation.

# Arguments
- `R`: the reproduction number.
- `gt_pmf`: generation time probability mass function, with `gt_pmf[i]` the
  probability of a generation time of `i` days.

# Keyword Arguments
- `newton_steps`: number of Newton steps (default 2).

```@example R_to_growth
using EpiNow2
R_to_growth(1.5, [0.2, 0.5, 0.3])
```
"""
function R_to_growth(R, gt_pmf; newton_steps::Int = 2)
    n = length(gt_pmf)
    mean_gt = sum(gt_pmf[i] * i for i in 1:n)
    # Small-r approximation as initial guess
    r = (R - 1) / (R * mean_gt)
    # Fixed Newton steps (AD-safe: deterministic loop count)
    for _ in 1:newton_steps
        r -= (R * _neg_mgf(r, gt_pmf) - 1) / (R * _dneg_mgf_dr(r, gt_pmf))
    end
    return r
end

# Log density of an uncertain distribution's parameter priors.
_prior_logpdf(ud::UncertainDistribution, θ) =
    sum(logpdf(p, θ[i]) for (i, p) in enumerate(ud.param_priors))

_prior_means(ud::UncertainDistribution) = [mean(p) for p in ud.param_priors]

# PMF (from delay 0) of the fixed components of a delay specification.
_fixed_pmf(d::UncertainDistribution) = [1.0]
function _fixed_pmf(d::CompositeDelay)
    fixed = [c for c in d.components if !(c isa UncertainDistribution)]
    isempty(fixed) && return [1.0]
    return discretise(CompositeDelay(fixed)).pmf
end
_fixed_pmf(d) = discretise(d).pmf

# PMF (from delay 0) of uncertain components at parameters `θs`,
# convolved with the PMF of the fixed components.
function _uncertain_pmf(uncertain, θs, fixed_pmf)
    pmf = fixed_pmf
    for (ud, θ) in zip(uncertain, θs)
        pmf = _convolve_pmfs(pmf, discretise_ad(ud.constructor(θ...), Int(ud.max)))
    end
    return pmf
end

# Mean of a delay ignoring parameter uncertainty, as R's
# `mean(delays, ignore_uncertainty = TRUE)`: uncertain parts at the mean of
# each parameter prior, composites as the sum of their parts. R applies
# parameter bounds separately from the prior, so a truncated prior counts
# with the mean of the distribution it truncates.
_delay_mean(d::Distribution) = mean(d)
_delay_mean(d::NonParametricDist) = sum((0:(length(d.pmf) - 1)) .* d.pmf)
_delay_mean(d::UncertainDistribution) = mean(
    d.constructor(
        [p isa Truncated ? mean(p.untruncated) : mean(p) for p in d.param_priors]...
    )
)
_delay_mean(d::CompositeDelay) = sum(_delay_mean, d.components)

"""Extract UncertainDistribution components from any delay specification."""
function _extract_uncertain(d::UncertainDistribution)
    return UncertainDistribution[d]
end
function _extract_uncertain(d::CompositeDelay)
    return UncertainDistribution[c for c in d.components if c isa UncertainDistribution]
end
function _extract_uncertain(d)
    return UncertainDistribution[]
end

function _apply_accumulation(
        expected::AbstractVector, accumulate::AbstractVector{Bool}, n::Int
    )
    buf = zero(eltype(expected))
    return [
        begin
            buf = buf + expected[t]
            if !accumulate[t]
                out = buf
                buf = zero(eltype(expected))
                out
            else
                zero(eltype(expected))
            end
        end for t in 1:n
    ]
end
