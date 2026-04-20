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
    EpiNow2Fit(chain, gq_vec, metadata)
end

"""
Extract generated quantities by replaying the model for each posterior draw.
Uses DynamicPPL's low-level API to condition the model on chain values,
avoiding the re-initialisation that can fail with domain-constrained parameters.
"""
function _extract_generated_quantities(model, chain)
    n_samples = size(chain, 1)
    n_chains = size(chain, 3)
    param_names = names(chain, :parameters)

    gqs = Vector{Any}(undef, n_samples * n_chains)
    idx = 1
    for c in 1:n_chains
        for s in 1:n_samples
            # Build conditioning pairs from this posterior draw
            pairs = [
                Turing.DynamicPPL.VarName{Symbol(name)}() => chain[s, name, c]
                for name in param_names
            ]
            conditioned = Turing.DynamicPPL.condition(model, pairs...)
            gqs[idx] = conditioned()
            idx += 1
        end
    end
    gqs
end

function _sample(model, opts::InferenceOpts)
    rng = isnothing(opts.seed) ? Random.default_rng() : Random.Xoshiro(opts.seed)
    sampler = _make_sampler(opts)

    if opts.chains > 1
        Turing.sample(
            rng, model, sampler, MCMCThreads(),
            opts.samples, opts.chains;
            discard_initial=0, progress=opts.progress,
            check_model=false
        )
    else
        Turing.sample(
            rng, model, sampler, opts.samples;
            discard_initial=0, progress=opts.progress,
            check_model=false
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
        max_depth=opts.max_treedepth,
        adtype=opts.adtype
    )
end

