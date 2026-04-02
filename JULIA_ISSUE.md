# Issue: `Turing.returned()` fails with "failed to find valid initial parameters"

## Summary

After MCMC sampling completes successfully, `Turing.returned(model, chain)` in
`inference.jl:17` fails with "failed to find valid initial parameters in 1000
tries". This blocks generated quantities extraction (infections, R, reports).

## How to reproduce

```julia
using EpiNow2, DataFrames, Distributions

result = estimate_infections(
    example_confirmed(),
    generation_time = gt_opts(EpiNow2.UncertainDistribution(
        (α, θ) -> Gamma(max(1e-6, α), max(1e-6, 1.0 / θ)),
        Distribution[truncated(Normal(1.3, 0.3); lower=0.0),
                     truncated(Normal(0.37, 0.09); lower=0.0)],
        14.0
    )),
    rt = rt_opts(prior = LogNormal(0.69, 0.05)),
    inference = inference_opts(samples = 50, warmup = 50, chains = 1)
)
```

## Error

```
Sampling (1 thread) 100%|████████████████████| Time: 0:00:35

TaskFailedException
    nested task error: failed to find valid initial parameters in 1000 tries
```

The sampling step completes fine. The error occurs on `inference.jl:17`:

```julia
gq_vec = vec(Turing.returned(model, chain))
```

## What `Turing.returned()` does

`Turing.returned(model, chain)` replays the model for each posterior sample to
extract the return value (the `(infections=..., reports=..., R=..., log_R=...)`
named tuple). Internally it calls `DynamicPPL.generated_quantities(model, chain)`
which:

1. For each draw in the chain, creates a `VarInfo` from the draw's parameter
   values
2. Evaluates the model with those values
3. Collects the return value

The "failed to find valid initial parameters" error suggests step 1 or 2 fails
because the parameter values from the chain produce invalid model states (e.g.
negative Gamma shape, NaN from AD, etc.).

## Likely causes

### 1. Uncertain distribution constructors hit domain errors

The `UncertainDistribution` constructor `(α, θ) -> Gamma(α, 1/θ)` can fail if
the sampled `α` or `θ` values are outside the valid domain. Even with
`truncated(Normal(...); lower=0)` priors, the sampler might explore values that
cause issues when replayed.

The `max(1e-6, ...)` guard in the constructor helps during sampling (where AD
handles it), but during `Turing.returned()` replay, the raw parameter values
from the chain are used directly, and these might include edge cases.

### 2. The `@addlogprob!` pattern

The model uses `Turing.@addlogprob!` for the negative binomial likelihood
rather than `y ~ NegBinomial(...)`. This is fine for sampling but might
interact poorly with `Turing.returned()` if the replay mechanism expects
standard `~` statements.

### 3. `frac_observed ~ Dirac(...)` or similar

If `obs_scale_prior` is a `Dirac` distribution (from `obs_opts(scale = 1.0)`
being a Float64 that gets wrapped), the `~` statement with a non-differentiable
distribution can cause issues during replay.

## Suggested fix

### Option A: Use `DynamicPPL.generated_quantities` with explicit parameter mapping

Instead of `Turing.returned(model, chain)`, manually extract parameter values
from the chain and evaluate the model deterministically:

```julia
function _extract_gqs(model, chain)
    gqs = Vector{Any}(undef, size(chain, 1) * size(chain, 3))
    idx = 1
    for c in 1:size(chain, 3)
        for s in 1:size(chain, 1)
            params = chain[s, :, c]
            vi = DynamicPPL.VarInfo(model)
            DynamicPPL.setval!(vi, params)
            gqs[idx] = DynamicPPL.evaluate!!(model, vi)[2]
            idx += 1
        end
    end
    gqs
end
```

### Option B: Separate the generative model from the inference model

Split `infections_model` into:
1. A pure inference `@model` that only samples parameters and computes the
   log-likelihood
2. A plain Julia function that takes parameter values and computes generated
   quantities (infections, R, reports)

This avoids `Turing.returned()` entirely:

```julia
function run_inference(model, metadata, opts)
    chain = _sample(model, opts)
    gqs = [compute_gq(chain[s, :, c], metadata) for s in ..., c in ...]
    EpiNow2Fit(chain, gqs, metadata)
end
```

### Option C: Wrap `Turing.returned` in error handling

```julia
function run_inference(model, metadata, opts)
    chain = _sample(model, opts)
    gq_vec = try
        vec(Turing.returned(model, chain))
    catch e
        @warn "Turing.returned failed: $e. Falling back to manual GQ extraction."
        _manual_gq_extraction(model, chain)
    end
    EpiNow2Fit(chain, gq_vec, metadata)
end
```

## Context

This was discovered while testing the R package's Julia backend (via
JuliaConnectoR). The R bridge successfully converts all opts and data, and MCMC
sampling completes. The failure is entirely within `Turing.returned()`.
