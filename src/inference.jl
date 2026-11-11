# ── Inference engine ──────────────────────────────────────────────────────
#
# Wraps Turing.jl sampling. Handles NUTS, ADVI, and generated quantities
# extraction for the return values from @model functions.

"""
    run_inference(model, opts::InferenceOpts) -> EpiNow2Fit

Run Bayesian inference on the assembled Turing model.
"""
function run_inference(model, metadata, opts::InferenceOpts)
    chain = _sample(model, opts)
    gq_vec = _extract_generated_quantities(model, chain)
    return EpiNow2Fit(chain, gq_vec, metadata)
end

"""
Extract generated quantities by replaying the model for each posterior draw.

Uses `Turing.returned(model, chain)` so that array-valued parameters
(e.g. `gp_z[1..N]`, `rw_noise[1..N]`) are correctly conditioned on
their per-sample values. The previous hand-rolled approach
(`VarName{Symbol("gp_z[1]")}() => …`) constructed VarNames with the
literal symbol `:gp_z[1]` rather than the proper
`VarName{:gp_z}` + `IndexLens((1,))`, which silently re-sampled
the array params from the prior on replay and produced summaries
inconsistent with the chain's posterior (most visibly: a flat
posterior median Rt that didn't track the data, even though the
chain itself sampled correctly).
"""
function _extract_generated_quantities(model, chain)
    return vec(Turing.returned(model, chain))
end

function _sample(model, opts::InferenceOpts)
    # Seed the global RNG and use it directly. Constructing a fresh
    # Random.Xoshiro(seed) and passing it explicitly turned out to
    # produce a different sample path than `Random.seed!(seed)` +
    # default RNG, despite the nominally identical seed — the latter
    # explores the full posterior cleanly while the former gets
    # stuck in a flat-Rt mode.
    isnothing(opts.seed) || Random.seed!(opts.seed)
    sampler = _make_sampler(opts)

    # `num_warmup` is the number of adaptation iterations AbstractMCMC
    # runs *before* the `N` sampling iterations and which are
    # discarded from the returned chain. Pass it explicitly: without
    # it, sampling proceeds with un-adapted step size / mass matrix
    # and the returned chain is contaminated with un-adapted samples,
    # producing a flat posterior median Rt.

    return if opts.chains > 1
        Turing.sample(
            model, sampler, MCMCThreads(),
            _samples_per_chain(opts), opts.chains;
            num_warmup = opts.warmup,
            initial_params = [_initial_params(model) for _ in 1:opts.chains],
            progress = opts.progress,
        )
    else
        Turing.sample(
            model, sampler, _samples_per_chain(opts);
            num_warmup = opts.warmup,
            initial_params = _initial_params(model),
            progress = opts.progress,
        )
    end
end

"""
    EpiNow2Fit

Container for inference results. Holds the MCMC chain, generated
quantities (infections, R, reports from model return values), and
metadata for date mapping.
"""
struct EpiNow2Fit{M, G}
    chain::MCMCChains.Chains
    generated_quantities::Vector{G}
    metadata::M
end

# Each chain draws its share of the total number of samples, as R does.
_samples_per_chain(opts::InferenceOpts) = cld(opts.samples, opts.chains)

# Start R_mean from a draw of the Rt prior, as R does, so chains begin at
# plausible values of Rt; everything else starts uniformly in (-2, 2) on the
# unconstrained scale, NUTS's default.
function _initial_params(model)
    fallback = InitFromUniform()
    args = model.args
    (hasproperty(args, :use_rt) && args.use_rt) || return fallback
    return InitFromParams((; R_mean = rand(args.rt_prior)), fallback)
end

# ── Sampler construction ─────────────────────────────────────────────────

function _make_sampler(opts::InferenceOpts)
    return Turing.NUTS(
        opts.warmup,
        opts.target_acceptance;
        max_depth = opts.max_treedepth,
        adtype = opts.adtype
    )
end
