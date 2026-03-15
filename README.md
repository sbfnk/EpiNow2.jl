# EpiNow2.jl

**Estimate real-time case counts and time-varying epidemiological parameters.**

A Julia implementation of the [EpiNow2](https://epiforecasts.io/EpiNow2/) methodology, providing the same opinionated, batteries-included interface for Rt estimation. Uses [Turing.jl](https://turinglang.org/) directly for Bayesian inference.

## Motivation

EpiNow2 in R has a well-tested, simple interface that lets epidemiologists go from case data to Rt estimates in a single function call. This package brings that same interface to Julia with:

- **Familiar API** for users of the R package
- **Single-function entry point** — `epinow(data, generation_time=..., delays=...)`
- **Sensible defaults** — GP-smoothed Rt, negative binomial observation model, day-of-week effects
- **Julia-native performance** — Turing.jl inference, native threading for multi-region

```
┌─────────────────────────────────────────────┐
│  User code                                  │
│  epinow(data, generation_time=..., ...)     │
├─────────────────────────────────────────────┤
│  EpiNow2.jl                                │
│  Options → Turing @model → Post-processing  │
├─────────────────────────────────────────────┤
│  Turing.jl + Distributions.jl              │
│  NUTS / ADVI inference                      │
└─────────────────────────────────────────────┘
```

## Quick start

```julia
using EpiNow2
using DataFrames

data = DataFrame(
    date = Date(2023,1,1):Day(1):Date(2023,3,1),
    confirm = reported_cases
)

result = epinow(
    data,
    generation_time = gt_opts(
        LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14)
    ),
    delays = delay_opts(
        LogNormalSpec(meanlog=0.5, sdlog=0.5, max=14)
    )
)

summary(result)
plot(result)
```

## Interface mapping from R

| R (EpiNow2)             | Julia (EpiNow2.jl)        | Notes                          |
|--------------------------|---------------------------|--------------------------------|
| `epinow()`               | `epinow()`                | Same interface                 |
| `estimate_infections()`  | `estimate_infections()`   | Same interface                 |
| `estimate_secondary()`   | `estimate_secondary()`    | Same interface                 |
| `estimate_truncation()`  | `estimate_truncation()`   | Same interface                 |
| `regional_epinow()`      | `regional_epinow()`       | Uses Julia threads             |
| `stan_opts()`            | `inference_opts()`        | Turing.jl instead of Stan      |
| `LogNormal()`            | `LogNormalSpec()`          | Avoids clash with Distributions.jl |
| `Gamma()`                | `GammaSpec()`             | Avoids clash with Distributions.jl |
| `dist_spec`              | `DistSpec` subtypes        | Julia type hierarchy           |
| `data.table` output      | `DataFrame` output         |                                |
| `ggplot2` plots          | `RecipesBase` / Makie      |                                |

## Architecture

The package is structured as:

1. **`distributions.jl`** — `DistSpec` type hierarchy for specifying delay distributions (with optional uncertain parameters)
2. **`options.jl`** — Julia structs mirroring `*_opts()` functions with identical defaults
3. **`model.jl`** — Turing `@model` functions implementing the renewal equation, HSGP, and observation model
4. **`inference.jl`** — Turing.jl sampling wrapper (NUTS, ADVI)
5. **`extract.jl`** — maps generated quantities back to dated DataFrames with credible intervals
6. **`epinow.jl`** — user-facing `epinow()` and `estimate_infections()` entry points

### The Turing model

The core `infections_model` implements the same generative process as EpiNow2's Stan model:

```
log(Rt) = log(R₀) + GP(t)           # or random walk, or fixed
infections[t] = R[t] × Σₛ infections[t-s] × g[s]   # renewal equation
expected[t] = infections ⊛ delay_pmf  # convolution with reporting delay
expected[t] *= day_of_week[t]          # weekly reporting pattern
cases[t] ~ NegBin(expected[t], φ)      # observation model
```

Key components:
- **Hilbert space GP approximation** for efficient Rt smoothing (Matérn / SE kernels)
- **Discrete convolution** for delay composition
- **`NegativeBinomial2(μ, φ)`** matching Stan's mean-precision parameterisation
- **Population depletion** adjustment for susceptible dynamics

## Advantages over the R package

- **No Stan toolchain** — no C++ compilation, no CmdStan installation
- **Faster inference** — Turing.jl with ForwardDiff/ReverseDiff AD
- **Native parallelism** — `regional_epinow()` uses `Threads.@threads`
- **Multiple AD backends** — ForwardDiff, ReverseDiff, Enzyme
- **Composability** — extract the Turing model and extend it directly
