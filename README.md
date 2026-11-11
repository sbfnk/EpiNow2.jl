# EpiNow2.jl <img src="docs/src/assets/logo.svg" width="150" alt="EpiNow2 logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://epiaware.org/EpiNow2.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://epiaware.org/EpiNow2.jl/dev/) | [![Test](https://github.com/sbfnk/EpiNow2.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sbfnk/EpiNow2.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg)](https://codecov.io/gh/sbfnk/EpiNow2.jl) [![AD](https://github.com/sbfnk/EpiNow2.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/sbfnk/EpiNow2.jl/actions/workflows/ad.yaml) | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FEpiNow2&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/EpiNow2) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FEpiNow2&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/EpiNow2) |

| ForwardDiff | ReverseDiff (tape) | ReverseDiff (compiled) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-reversediff) | [![cov ReverseDiff compiled](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-reversediff-compiled)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-reversediff-compiled) | [![cov Enzyme forward](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/sbfnk/EpiNow2.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/sbfnk/EpiNow2.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

**Estimate real-time case counts and time-varying epidemiological parameters.**

A Julia implementation of the [EpiNow2](https://epiforecasts.io/EpiNow2/) methodology, with the same opinionated interface for Rt estimation. It uses [Turing.jl](https://turinglang.org/) directly for Bayesian inference.

## Why EpiNow2.jl?

- **Familiar interface** for users of the R package: `epinow()`, `estimate_infections()`, `estimate_secondary()`, `estimate_truncation()` and the `*_opts()` constructors keep their R names and defaults.
- **Same model**: renewal equation with Gaussian process (HSGP), random walk or back-calculation for Rt, uncertain delay distributions, negative binomial or Poisson observations with day-of-week effects, right-truncation adjustment, population depletion and forecasting.
- **No Stan toolchain**: inference runs in Julia through Turing.jl, with a choice of automatic differentiation backends.
- **Multi-region estimation**: `regional_epinow()` fits regions in parallel on Julia threads.

Which R features are ported, and against which EpiNow2 commit, is tracked in [`PARITY.md`](https://github.com/sbfnk/EpiNow2.jl/blob/main/PARITY.md).

## Quick start

```julia
using EpiNow2
using Distributions

cases = first(example_confirmed(), 60)

result = epinow(
    cases;
    generation_time = gt_opts(discretise(LogNormal(1.6, 0.5); max = 14)),
    delays = delay_opts(discretise(LogNormal(0.5, 0.5); max = 14)),
)

summary(result)
```

## Installation

EpiNow2.jl is not yet registered. Install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/sbfnk/EpiNow2.jl")
```

## Interface mapping from R

| R (EpiNow2) | Julia (EpiNow2.jl) | Notes |
|---|---|---|
| `epinow()` | `epinow()` | |
| `estimate_infections()` | `estimate_infections()` | |
| `estimate_secondary()` | `estimate_secondary()` | |
| `estimate_truncation()` | `estimate_truncation()` | |
| `estimate_dist()` | `estimate_dist()` | |
| `regional_epinow()` | `regional_epinow()` | Uses Julia threads |
| `stan_opts()` | `inference_opts()` | Turing.jl instead of Stan |
| `LogNormal()`, `Gamma()`, ... | `LogNormal`, `Gamma`, ... from Distributions.jl | Uncertain parameters via `UncertainDistribution` |
| `NonParametric()` | `NonParametricDist` | |
| `data.table` output | `DataFrame` output | |
| `ggplot2` plots | Makie, through the CairoMakie extension | |

## Related packages

- [EpiNow2](https://github.com/epiforecasts/EpiNow2) is the R package this one ports.
- [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl) provides the censored delay distributions used to discretise delays and fit them with `estimate_dist()`.
- [Turing.jl](https://turinglang.org/) runs the inference.

## Where to learn more

- The [documentation](https://sbfnk.github.io/EpiNow2.jl/dev/) has a worked example of the full workflow and the API reference.
- The [R package documentation](https://epiforecasts.io/EpiNow2/) explains the model in depth; the Julia port follows it.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/sbfnk/EpiNow2.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and is formatted with [Runic](https://github.com/fredrikekre/Runic.jl).

## How to cite

If you use EpiNow2 in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/sbfnk/EpiNow2.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the EpiNow2 project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->

## License

Released under the [MIT License](https://github.com/sbfnk/EpiNow2.jl/blob/main/LICENSE).
