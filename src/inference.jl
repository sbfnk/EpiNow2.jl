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

    # Extract generated quantities (infections, R, reports) from the
    # return values of the @model function.
    # Flatten to a Vector for consistent handling (multi-chain returns matrix).
    gq_vec = vec(Turing.returned(model, chain))

    EpiNow2Fit(chain, gq_vec, metadata)
end

function _sample(model, opts::InferenceOpts)
    rng = isnothing(opts.seed) ? Random.default_rng() : Random.Xoshiro(opts.seed)
    sampler = _make_sampler(opts)

    if opts.chains > 1
        Turing.sample(
            rng, model, sampler, MCMCThreads(),
            opts.samples, opts.chains;
            discard_initial=0, progress=opts.progress
        )
    else
        Turing.sample(
            rng, model, sampler, opts.samples;
            discard_initial=0, progress=opts.progress
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

# ── Sampler construction ─────────────────────────────────────────────────

function _make_sampler(opts::InferenceOpts)
    Turing.NUTS(
        opts.warmup,
        opts.target_acceptance;
        max_depth=opts.max_treedepth
    )
end

