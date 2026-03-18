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

# ── Show methods ─────────────────────────────────────────────────────────

function Base.show(io::IO, d::NonParametricDist)
    n = length(d.pmf)
    m = sum(i * d.pmf[i + 1] for i in 0:(n - 1) if i + 1 <= n)
    print(io, "NonParametricDist(max=$(n - 1), mean=$(round(m, digits=2)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::NonParametricDist)
    n = length(d.pmf)
    m = sum(i * d.pmf[i + 1] for i in 0:(n - 1) if i + 1 <= n)
    println(io, "Non-parametric distribution (max: $(n - 1), mean: $(round(m, digits=2)))")
    # Show PMF as a sparkline-style bar
    peak = maximum(d.pmf)
    for i in 1:min(n, 20)
        bar = repeat("█", round(Int, d.pmf[i] / peak * 15))
        println(io, "  $(lpad(i - 1, 2)): $(rpad(bar, 15)) $(round(d.pmf[i], digits=4))")
    end
    n > 20 && println(io, "  ...")
end

function _dist_name(d::Distribution)
    n = string(typeof(d).name.name)
    # Strip type parameters
    replace(n, r"\{.*\}" => "")
end

function _show_dist(io::IO, d::Distribution; indent=0)
    prefix = "  " ^ indent
    if d isa Dirac
        println(io, "$(prefix)Fixed($(d.value))")
    elseif d isa LogNormal
        println(io, "$(prefix)LogNormal distribution:")
        println(io, "$(prefix)  meanlog: $(round(d.μ, digits=4))")
        println(io, "$(prefix)  sdlog: $(round(d.σ, digits=4))")
    elseif d isa Gamma
        println(io, "$(prefix)Gamma distribution:")
        println(io, "$(prefix)  shape: $(round(shape(d), digits=4))")
        println(io, "$(prefix)  rate: $(round(1/scale(d), digits=4))")
    elseif d isa Truncated
        _show_dist(io, d.untruncated; indent=indent)
        lo = (d.lower isa Number && d.lower != -Inf) ? "lower=$(round(d.lower, digits=4))" : ""
        hi = (d.upper isa Number && d.upper != Inf) ? "upper=$(round(d.upper, digits=4))" : ""
        bounds = join(filter(!isempty, [lo, hi]), ", ")
        !isempty(bounds) && println(io, "$(prefix)  truncated: $bounds")
    elseif d isa Normal
        println(io, "$(prefix)Normal(mean=$(round(d.μ, digits=4)), sd=$(round(d.σ, digits=4)))")
    else
        println(io, "$(prefix)$(_dist_name(d))($(params(d)))")
    end
end

function Base.show(io::IO, ::MIME"text/plain", d::UncertainDistribution)
    # Infer distribution family from constructor at prior means
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)
    family = _dist_name(example)
    println(io, "$family distribution (max: $(Int(d.max))):")
    param_names = if example isa LogNormal
        ["meanlog", "sdlog"]
    elseif example isa Gamma
        ["shape", "rate"]
    else
        ["param_$i" for i in 1:length(d.param_priors)]
    end
    for (name, prior) in zip(param_names, d.param_priors)
        println(io, "  $name:")
        _show_dist(io, prior; indent=2)
    end
end

function Base.show(io::IO, d::UncertainDistribution)
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)
    print(io, "Uncertain $(_dist_name(example))(max=$(Int(d.max)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::CompositeDelay)
    println(io, "Composite distribution:")
    for (i, c) in enumerate(d.components)
        println(io, "  [$i]")
        _show_delay(io, c; indent=2)
    end
end

function _show_delay(io::IO, d::Distribution; indent=0)
    _show_dist(io, d; indent=indent)
end

function _show_delay(io::IO, d::NonParametricDist; indent=0)
    prefix = "  " ^ indent
    n = length(d.pmf)
    m = sum(i * d.pmf[i + 1] for i in 0:(n - 1) if i + 1 <= n)
    println(io, "$(prefix)Non-parametric (max: $(n - 1), mean: $(round(m, digits=2)))")
end

function _show_delay(io::IO, d::UncertainDistribution; indent=0)
    prefix = "  " ^ indent
    mean_params = [mean(p) for p in d.param_priors]
    example = d.constructor(mean_params...)
    println(io, "$(prefix)Uncertain $(_dist_name(example)) (max: $(Int(d.max)))")
end

function Base.show(io::IO, d::CompositeDelay)
    parts = [sprint(show, c) for c in d.components]
    print(io, join(parts, " + "))
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
