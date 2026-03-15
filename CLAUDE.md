# EpiNow2.jl — Julia Port of EpiNow2

You are building a Julia package that replicates the interface of the EpiNow2 R package (https://epiforecasts.io/EpiNow2/) for estimating time-varying reproduction numbers, infection dynamics, and forecasting. This package uses Turing.jl directly for Bayesian inference — **not** Stan, and **not** EpiAware.jl.

## Context

EpiNow2 is a widely-used R package for real-time Rt estimation. Its key strength is a simple, opinionated interface: a single function call (`epinow(data, generation_time=..., delays=...)`) takes you from case data to Rt estimates with sensible defaults. The user (Sebastian Funk, LSHTM) is the primary author of EpiNow2.

EpiAware.jl (CDCgov/Rt-without-renewal) is a separate Julia package that provides a composable DSL for similar models. EpiAware is likely to be redesigned, so **do not depend on it**. EpiNow2.jl goes directly to Turing.jl. PrimaryCensored.jl is stable and can be used.

An initial sketch exists in this repo with the following files:
- `src/EpiNow2.jl` — module definition
- `src/distributions.jl` — `DistSpec` type hierarchy (LogNormalSpec, GammaSpec, etc.)
- `src/options.jl` — option structs mirroring R's `*_opts()` functions
- `src/data.jl` — data validation types
- `src/model.jl` — Turing `@model` functions (renewal equation, HSGP, observation model)
- `src/inference.jl` — Turing.jl sampling wrapper
- `src/extract.jl` — posterior → dated DataFrames with credible intervals
- `src/epinow.jl` — `epinow()` and `estimate_infections()` entry points
- `src/secondary.jl` — `estimate_secondary()`
- `src/truncation.jl` — `estimate_truncation()`
- `src/regional.jl` — `regional_epinow()` with Julia threads
- `src/plotting.jl` — RecipesBase plot methods

This is the **long-term flagship** in a broader Julia epidemiology ecosystem plan, but was deprioritised because it is the most complex port (the Stan model is ~1500 lines). The branching process framework (`~/code/simulist.jl`) is the starting point.

Related projects in `~/code/`:
- `simulist.jl` — branching process / line list simulation (starting point)
- `outbreak-analytics-jl` — course materials and supporting packages
- `messy-line-lists` — typo challenge / data quality tools

## What the R package does

EpiNow2's core model (`estimate_infections`) implements:

1. **Rt estimation via renewal equation**: `infections[t] = R[t] * Σ infections[t-s] * g[s]` where `g` is the generation time PMF
2. **Gaussian process smoothing**: Hilbert space GP approximation on log(Rt) with Matérn or squared exponential kernels
3. **Alternative: random walk** on log(Rt) with configurable step size
4. **Alternative: back-calculation** (Bayesian deconvolution without Rt)
5. **Delay composition**: convolve infections with one or more delay distributions (incubation, reporting) to get expected reports
6. **Observation model**: negative binomial or Poisson likelihood with day-of-week effects, observation scaling, and right-truncation correction
7. **Population depletion**: optional susceptible dynamics adjustment
8. **Forecasting**: project Rt and infections forward

Secondary models: `estimate_secondary` (cases→deaths convolution), `estimate_truncation` (right-truncation estimation from multiple data snapshots).

## Interface requirements

The Julia interface must mirror the R interface closely:

| R function | Julia function | Notes |
|---|---|---|
| `epinow()` | `epinow()` | Main entry point |
| `estimate_infections()` | `estimate_infections()` | Core model |
| `estimate_secondary()` | `estimate_secondary()` | Primary→secondary |
| `estimate_truncation()` | `estimate_truncation()` | Truncation fitting |
| `regional_epinow()` | `regional_epinow()` | Multi-region, Julia threads |
| `gt_opts()` | `gt_opts()` | Generation time options |
| `delay_opts()` | `delay_opts()` | Reporting delay options |
| `rt_opts()` | `rt_opts()` | Rt estimation options |
| `gp_opts()` | `gp_opts()` | Gaussian process options |
| `obs_opts()` | `obs_opts()` | Observation model options |
| `stan_opts()` | `inference_opts()` | Turing.jl instead of Stan |
| `LogNormal()` | `LogNormalSpec()` | Avoids Distributions.jl clash |
| `Gamma()` | `GammaSpec()` | Avoids Distributions.jl clash |

All defaults should match the R package exactly (e.g., GP basis_prop=0.2, Matérn 3/2 kernel, NegBin observation model, day-of-week effects on by default).

## Key implementation details

### The Turing model

The sketch in `src/model.jl` contains the core `infections_model` as a `@model` function. Key components:

- **HSGP**: Hilbert space basis functions with spectral density weighting. Must support Matérn (1/2, 3/2, 5/2) and squared exponential kernels
- **Renewal equation**: forward simulation from initial infections using generation time PMF
- **NegativeBinomial2(μ, φ)**: mean-precision parameterisation matching Stan's `neg_binomial_2`
- **Day-of-week**: Dirichlet simplex prior scaled so mean effect = 1
- **Generated quantities**: the `@model` function returns a NamedTuple of (infections, reports, R, log_R) which are extracted via `Turing.generated_quantities`

### Distribution specifications

`DistSpec` types allow uncertain parameters (priors on delay distribution params):
```julia
# Fixed
LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14)

# Uncertain (prior on parameters)
LogNormalSpec(
    meanlog=NormalSpec(mean=1.6, sd=0.2),
    sdlog=NormalSpec(mean=0.5, sd=0.1),
    max=14
)
```

### Output format

All results are DataFrames with columns: `date`, `mean`, `median`, `sd`, `lower_20`, `upper_20`, `lower_50`, `upper_50`, `lower_90`, `upper_90`. This mirrors EpiNow2's data.table output structure.

## Outstanding TODOs from the sketch

- Back-calculation mode (deconvolution without Rt)
- Uncertain delay parameters within the Turing model (priors on delay dist params need to generate PMFs inside the model)
- Truncation adjustment within the main infections model
- Plotting implementation
- Tests and numerical validation against the R package

## Research before coding

1. **Read the R package source**: especially `inst/stan/estimate_infections.stan` and `inst/stan/functions/` for the exact mathematical implementation
2. **Read the sketch**: all files in `/home/sebfunk/code/EpiNow2.jl-sketch/src/`
3. **EpiNow2 documentation**: https://epiforecasts.io/EpiNow2/ for user-facing behaviour
4. **Turing.jl generated quantities**: understand how to extract return values from `@model` functions across all MCMC samples
5. **AD compatibility**: ensure all model code is compatible with ForwardDiff and ReverseDiff (no mutation of tracked arrays, etc.)

## Style and conventions

- Use British English ("modelling", "behaviour", etc.)
- No dependency on EpiAware.jl
- PrimaryCensored.jl is available for censored delay fitting
- DataFrames for all tabular output
- RecipesBase for plotting (works with any Plots.jl backend)
- Test against EpiNow2 R package outputs for numerical validation
