# ── Distribution system ───────────────────────────────────────────────────
#
# Uses Distributions.jl directly for fixed delay distributions.
# Only adds thin wrappers for PMF vectors and uncertain parameters.

"""
    NonParametricDist(pmf)

Non-parametric distribution specified as a probability mass function.
`pmf[1]` is the probability of delay 0, `pmf[2]` of delay 1, etc.

Supports `+` for convolution:
```julia
discretise(LogNormal(1.6, 0.5); max=14) + discretise(LogNormal(0.5, 0.3); max=7)
```
"""
struct NonParametricDist
    pmf::Vector{Float64}

    function NonParametricDist(pmf::AbstractVector{<:Real})
        p = Vector{Float64}(pmf)
        @assert sum(p) ≈ 1.0 "PMF must sum to 1"
        @assert all(p .>= 0) "PMF values must be non-negative"
        new(p)
    end
end

"""
    UncertainDistribution

A delay distribution with priors on its parameters. Used when the delay
distribution itself is uncertain and parameters should be estimated.

# Examples
```julia
# Uncertain log-normal delay
UncertainDistribution(
    (μ, σ) -> LogNormal(μ, σ),
    [Normal(1.6, 0.2), truncated(Normal(0.5, 0.1); lower=0.0)],
    14.0
)
```
"""
struct UncertainDistribution
    constructor::Function
    param_priors::Vector{<:Distribution}
    max::Float64
end

"""
    DelayDistribution

Union type for all delay distribution specifications:
- `Distribution` — a fixed Distributions.jl distribution
- `NonParametricDist` — a discretised PMF
- `UncertainDistribution` — distribution with priors on parameters
"""
const DelayDistribution = Union{Distribution, NonParametricDist, UncertainDistribution}

"""
    CompositeDelay

Multiple delay distributions to be convolved sequentially
(e.g., incubation period + reporting delay).
"""
struct CompositeDelay
    components::Vector{DelayDistribution}
end

# ── Show methods (matching R's tree format) ──────────────────────────────

function _dist_family(d::Distribution)
    d isa LogNormal && return "lognormal"
    d isa Gamma && return "gamma"
    d isa Normal && return "normal"
    d isa Dirac && return "fixed"
    d isa Truncated && return _dist_family(d.untruncated)
    lowercase(string(typeof(d).name.name))
end

function _dist_name(d::Distribution)
    replace(string(typeof(d).name.name), r"\{.*\}" => "")
end

function _param_names(d::Distribution)
    d isa LogNormal && return ["meanlog", "sdlog"]
    d isa Gamma && return ["shape", "rate"]
    d isa Normal && return ["mean", "sd"]
    ["param_$i" for i in 1:length(params(d))]
end

function _param_values(d::Distribution)
    d isa LogNormal && return [d.μ, d.σ]
    d isa Gamma && return [shape(d), 1 / scale(d)]
    d isa Normal && return [d.μ, d.σ]
    collect(params(d))
end

"""Print a delay distribution in R's indented tree format."""
function _show_tree(io::IO, d::Distribution; indent=0)
    prefix = "  " ^ indent
    if d isa Dirac
        println(io, "$(prefix)- fixed (value: $(d.value))")
    elseif d isa Truncated
        _show_tree(io, d.untruncated; indent=indent)
    else
        println(io, "$(prefix)- $(_dist_family(d)) distribution:")
        for (name, val) in zip(_param_names(d), _param_values(d))
            println(io, "$(prefix)  $(name):")
            println(io, "$(prefix)    $(round(val, digits=4))")
        end
    end
end

function _show_tree(io::IO, d::UncertainDistribution; indent=0)
    prefix = "  " ^ indent
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)

    println(io, "$(prefix)- $(_dist_family(example)) distribution (max: $(Int(d.max))):")
    for (name, prior) in zip(_param_names(example), d.param_priors)
        println(io, "$(prefix)  $(name):")
        prior_inner = prior isa Truncated ? prior.untruncated : prior
        _show_tree(io, prior_inner; indent=indent + 2)
    end
end

function _show_tree(io::IO, d::NonParametricDist; indent=0)
    prefix = "  " ^ indent
    n = length(d.pmf)
    m = sum(i * d.pmf[i + 1] for i in 0:(n - 1) if i + 1 <= n)
    println(io, "$(prefix)- non-parametric distribution (max: $(n - 1), mean: $(round(m, digits=2)))")
end

# ── Base.show for each type ──

function Base.show(io::IO, d::NonParametricDist)
    n = length(d.pmf)
    m = sum(i * d.pmf[i + 1] for i in 0:(n - 1) if i + 1 <= n)
    print(io, "NonParametricDist(max=$(n - 1), mean=$(round(m, digits=2)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::NonParametricDist)
    _show_tree(io, d)
end

function Base.show(io::IO, d::UncertainDistribution)
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)
    print(io, "Uncertain $(_dist_family(example))(max=$(Int(d.max)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::UncertainDistribution)
    _show_tree(io, d)
end

function Base.show(io::IO, d::CompositeDelay)
    parts = [sprint(show, c) for c in d.components]
    print(io, join(parts, " + "))
end

function Base.show(io::IO, ::MIME"text/plain", d::CompositeDelay)
    println(io, "Composite distribution:")
    for c in d.components
        _show_tree(io, c; indent=0)
    end
end

# ── Operators ────────────────────────────────────────────────────────────

# NonParametricDist + NonParametricDist: convolve PMFs directly
function Base.:+(a::NonParametricDist, b::NonParametricDist)
    NonParametricDist(_convolve_pmfs(a.pmf, b.pmf))
end

# General DelayDistribution composition
Base.:+(a::DelayDistribution, b::DelayDistribution) =
    CompositeDelay(DelayDistribution[a, b])
Base.:+(a::CompositeDelay, b::DelayDistribution) =
    CompositeDelay(DelayDistribution[a.components; b])
Base.:+(a::DelayDistribution, b::CompositeDelay) =
    CompositeDelay(DelayDistribution[a; b.components])

# ── Discretisation ───────────────────────────────────────────────────────

"""
    discretise(d::Distribution; max=nothing, cdf_cutoff=0.001) -> NonParametricDist

Discretise a continuous distribution into a PMF using double interval
censoring (matching R's `discretise()`). Returns a `NonParametricDist`
that supports `+` for convolution.

The PMF is 0-indexed: `pmf[1]` = P(delay=0), `pmf[2]` = P(delay=1), etc.

# Examples
```julia
discretise(LogNormal(1.6, 0.5))              # auto max from cdf_cutoff
discretise(LogNormal(1.6, 0.5); max=14)      # explicit max
discretise(LogNormal(1.0, 0.5); max=14) +    # compose with +
    discretise(LogNormal(0.5, 0.3); max=7)
```
"""
function discretise(
    d::Distribution;
    max::Union{Int, Nothing}=nothing,
    cdf_cutoff::Float64=0.001
)
    max_val = if !isnothing(max)
        max
    else
        Int(ceil(quantile(d, 1.0 - cdf_cutoff)))
    end

    cd = double_interval_censored(d; interval=1, upper=max_val + 1)
    pmf = [pdf(cd, k) for k in 0:max_val]

    # Remove trailing near-zeros
    last_nonzero = findlast(x -> x > 1e-10, pmf)
    if !isnothing(last_nonzero)
        pmf = pmf[1:last_nonzero]
    end

    NonParametricDist(pmf)
end

"""
    discretise(d::NonParametricDist) -> NonParametricDist

Return a copy (already discretised).
"""
discretise(d::NonParametricDist; kwargs...) = NonParametricDist(copy(d.pmf))

"""
    discretise(d::UncertainDistribution; kwargs...) -> NonParametricDist

Discretise at the prior mean parameter values.
"""
function discretise(d::UncertainDistribution; max::Union{Int, Nothing}=nothing, kwargs...)
    mean_params = [mean(p) for p in d.param_priors]
    max_val = isnothing(max) ? Int(d.max) : max
    dist = d.constructor(mean_params...)
    discretise(dist; max=max_val, kwargs...)
end

"""
    discretise(d::Dirac) -> NonParametricDist

Discretise a point mass distribution.
"""
function discretise(d::Dirac; kwargs...)
    v = Int(d.value)
    pmf = v == 0 ? [1.0] : [zeros(v); 1.0]
    NonParametricDist(pmf)
end

"""
    discretise(d::CompositeDelay; kwargs...) -> NonParametricDist

Discretise a composite delay by convolving component PMFs.
"""
function discretise(d::CompositeDelay; kwargs...)
    pmfs = [discretise(c; kwargs...).pmf for c in d.components]
    NonParametricDist(reduce(_convolve_pmfs, pmfs))
end

# ── PMF convolution ──────────────────────────────────────────────────────

"""
    convolve_pmfs(a, b) -> NonParametricDist or Vector{Float64}

Convolve two discretised distributions or raw PMF vectors.
Equivalent to `a + b` for `NonParametricDist`.
"""
convolve_pmfs(a::NonParametricDist, b::NonParametricDist) = a + b
convolve_pmfs(a::AbstractVector{Float64}, b::AbstractVector{Float64}) =
    _convolve_pmfs(a, b)

function _convolve_pmfs(a::AbstractVector{Float64}, b::AbstractVector{Float64})
    na, nb = length(a), length(b)
    n_out = na + nb - 1
    out = zeros(Float64, n_out)
    for i in 1:na, j in 1:nb
        out[i + j - 1] += a[i] * b[j]
    end
    out
end

# ── AD-safe discretisation ───────────────────────────────────────────────

"""
    discretise_ad(d::Distribution, max_val::Int)

Discretise a continuous distribution into a 1-indexed PMF
(delay 1..max_val) using double interval censoring. AD-compatible
for use inside Turing `@model` functions.

Uses CensoredDistributions.jl which supports ForwardDiff through
both LogNormal and Gamma distribution parameters.
"""
function discretise_ad(d::Distribution, max_val::Int)
    cd = double_interval_censored(d; interval=1, upper=max_val + 1)
    # 0-indexed PMF from CensoredDistributions, then drop delay 0
    pmf_full = [exp(logpdf(cd, k)) for k in 0:max_val]
    pmf = pmf_full[2:end]
    pmf ./ sum(pmf)
end

# ── Internal helpers ─────────────────────────────────────────────────────

function _moments_to_lognormal(mean, sd)
    σ² = log(1 + (sd / mean)^2)
    μ = log(mean) - σ² / 2
    (μ, sqrt(σ²))
end
