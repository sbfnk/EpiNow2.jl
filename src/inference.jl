# ── Inference engine ──────────────────────────────────────────────────────
#
# Wraps Turing.jl sampling. Handles NUTS, ADVI, and generated quantities
# extraction for the return values from @model functions.

"""
    run_inference(model, opts::InferenceOpts) -> EpiNow2Fit

Run Bayesian inference on the assembled Turing model.
"""
function run_inference(model, metadata::ModelMetadata, opts::InferenceOpts)
    sampler = _make_sampler(opts)

    chain = if opts.sampler == :nuts
        Turing.sample(
            model,
            sampler,
            MCMCThreads(),
            opts.samples,
            opts.chains;
            discard_initial=opts.warmup,
            progress=opts.progress
        )
    elseif opts.sampler == :advi
        q = Turing.vi(model, sampler)
        _vi_to_chains(q, model, opts.samples)
    else
        error("Sampler :$(opts.sampler) not yet supported")
    end

    # Extract generated quantities (infections, R, reports) from the
    # return values of the @model function
    gen_quants = generated_quantities(model, chain)

    EpiNow2Fit(chain, gen_quants, metadata)
end

"""
    EpiNow2Fit

Container for inference results. Holds the MCMC chain, generated
quantities (infections, R, reports from model return values), and
metadata for date mapping.
"""
struct EpiNow2Fit
    chain::MCMCChains.Chains
    generated_quantities::Vector  # vector of NamedTuples per sample
    metadata::ModelMetadata
end

# ── Sampler construction ─────────────────────────────────────────────────

function _make_sampler(opts::InferenceOpts)
    if opts.sampler == :nuts
        Turing.NUTS(
            opts.warmup,
            opts.target_acceptance;
            max_depth=opts.max_treedepth
        )
    elseif opts.sampler == :advi
        Turing.ADVI(10, 10_000)
    else
        error("Unknown sampler: $(opts.sampler)")
    end
end

function _vi_to_chains(q, model, n_samples)
    samples = rand(q, n_samples)
    # TODO: convert to Chains with proper parameter names
    samples
end
