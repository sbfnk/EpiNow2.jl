# ── Distribution specification system ────────────────────────────────────
#
# Maps EpiNow2's dist_spec to Julia types. These are *specifications* that
# describe a distribution (possibly with uncertain parameters), not
# Distributions.jl objects directly. They get converted to EpiAware
# components at model-assembly time.

"""
    DistSpec

Abstract type for all distribution specifications. A DistSpec describes a
delay distribution that may have uncertain parameters (priors on parameters).
"""
abstract type DistSpec end

"""
    LogNormalSpec(; meanlog, sdlog, mean, sd, max, cdf_cutoff)

Log-normal distribution specification. Provide either natural parameters
(`meanlog`, `sdlog`) or moment parameters (`mean`, `sd`).

Parameters can be:
- A number (fixed)
- A `NormalSpec` (uncertain, with prior)

# Examples
```julia
# Fixed parameters
LogNormalSpec(meanlog=1.6, sdlog=0.5, max=14)

# Uncertain parameters (priors on delay distribution params)
LogNormalSpec(
    meanlog=NormalSpec(mean=1.6, sd=0.2),
    sdlog=NormalSpec(mean=0.5, sd=0.1),
    max=14
)

# Moment parameterisation
LogNormalSpec(mean=5.0, sd=3.0, max=14)
```
"""
struct LogNormalSpec <: DistSpec
    meanlog::Union{Float64, NormalSpec}
    sdlog::Union{Float64, NormalSpec}
    max::Float64
    cdf_cutoff::Float64

    function LogNormalSpec(;
        meanlog=nothing, sdlog=nothing,
        mean=nothing, sd=nothing,
        max=Inf, cdf_cutoff=0.0
    )
        if !isnothing(meanlog) && !isnothing(sdlog)
            new(meanlog, sdlog, Float64(max), Float64(cdf_cutoff))
        elseif !isnothing(mean) && !isnothing(sd)
            # Convert moments to natural parameters
            μ, σ = _moments_to_lognormal(Float64(mean), Float64(sd))
            new(μ, σ, Float64(max), Float64(cdf_cutoff))
        else
            throw(ArgumentError(
                "Provide either (meanlog, sdlog) or (mean, sd)"
            ))
        end
    end
end

"""
    GammaSpec(; shape, rate, scale, mean, sd, max, cdf_cutoff)

Gamma distribution specification.
"""
struct GammaSpec <: DistSpec
    shape::Union{Float64, NormalSpec}
    rate::Union{Float64, NormalSpec}
    max::Float64
    cdf_cutoff::Float64

    function GammaSpec(;
        shape=nothing, rate=nothing, scale=nothing,
        mean=nothing, sd=nothing,
        max=Inf, cdf_cutoff=0.0
    )
        if !isnothing(shape) && (!isnothing(rate) || !isnothing(scale))
            r = isnothing(rate) ? 1.0 / Float64(scale) : Float64(rate)
            new(shape, r, Float64(max), Float64(cdf_cutoff))
        elseif !isnothing(mean) && !isnothing(sd)
            s, r = _moments_to_gamma(Float64(mean), Float64(sd))
            new(s, r, Float64(max), Float64(cdf_cutoff))
        else
            throw(ArgumentError(
                "Provide either (shape, rate/scale) or (mean, sd)"
            ))
        end
    end
end

"""
    NormalSpec(; mean, sd)

Normal distribution specification. Used both as a standalone distribution
and as a prior on parameters of other distributions.
"""
struct NormalSpec <: DistSpec
    mean::Float64
    sd::Float64
    max::Float64
    cdf_cutoff::Float64

    function NormalSpec(; mean, sd, max=Inf, cdf_cutoff=0.0)
        new(Float64(mean), Float64(sd), Float64(max), Float64(cdf_cutoff))
    end
end

"""
    FixedSpec(value)

Point mass (delta) distribution at `value`.
"""
struct FixedSpec <: DistSpec
    value::Float64
end

"""
    NonParametricSpec(pmf)

Non-parametric distribution specified as a probability mass function.
`pmf[1]` is the probability of delay 0, `pmf[2]` of delay 1, etc.
"""
struct NonParametricSpec <: DistSpec
    pmf::Vector{Float64}

    function NonParametricSpec(pmf::Vector{Float64})
        @assert sum(pmf) ≈ 1.0 "PMF must sum to 1"
        @assert all(pmf .>= 0) "PMF values must be non-negative"
        new(pmf)
    end
end

"""
    CompositeDistSpec(components)

Multiple distributions to be convolved (e.g., incubation + reporting delay).
Created via `+` operator on DistSpec objects.
"""
struct CompositeDistSpec <: DistSpec
    components::Vector{DistSpec}
end

# ── Operators ────────────────────────────────────────────────────────────

Base.:+(a::DistSpec, b::DistSpec) = CompositeDistSpec([a, b])
Base.:+(a::CompositeDistSpec, b::DistSpec) = CompositeDistSpec([a.components; b])
Base.:+(a::DistSpec, b::CompositeDistSpec) = CompositeDistSpec([a; b.components])

# ── Conversion to Distributions.jl ──────────────────────────────────────

"""
    to_distribution(spec::DistSpec) -> Distribution

Convert a fixed DistSpec to a Distributions.jl distribution.
Errors if parameters are uncertain (use `fix_parameters` first).
"""
function to_distribution(spec::LogNormalSpec)
    spec.meanlog isa Float64 || error("Cannot convert uncertain dist")
    spec.sdlog isa Float64 || error("Cannot convert uncertain dist")
    Distributions.LogNormal(spec.meanlog, spec.sdlog)
end

function to_distribution(spec::GammaSpec)
    spec.shape isa Float64 || error("Cannot convert uncertain dist")
    spec.rate isa Float64 || error("Cannot convert uncertain dist")
    Distributions.Gamma(spec.shape, 1.0 / spec.rate)
end

function to_distribution(spec::NormalSpec)
    Distributions.Normal(spec.mean, spec.sd)
end

"""
    has_uncertain_params(spec::DistSpec) -> Bool

Check whether a distribution specification has uncertain (prior) parameters.
"""
has_uncertain_params(::FixedSpec) = false
has_uncertain_params(::NonParametricSpec) = false
has_uncertain_params(spec::NormalSpec) = false
function has_uncertain_params(spec::LogNormalSpec)
    spec.meanlog isa NormalSpec || spec.sdlog isa NormalSpec
end
function has_uncertain_params(spec::GammaSpec)
    spec.shape isa NormalSpec || spec.rate isa NormalSpec
end

"""
    discretise(spec::DistSpec; remove_trailing_zeros=true) -> Vector{Float64}

Discretise a distribution specification into a PMF vector.
Only works for fixed-parameter distributions.
"""
function discretise(spec::DistSpec; remove_trailing_zeros=true)
    d = to_distribution(spec)
    max_val = isfinite(spec.max) ? Int(spec.max) :
        Int(ceil(quantile(d, 1.0 - spec.cdf_cutoff)))

    pmf = [cdf(d, k + 0.5) - cdf(d, k - 0.5) for k in 0:max_val]
    pmf ./= sum(pmf)

    if remove_trailing_zeros
        last_nonzero = findlast(x -> x > 1e-10, pmf)
        pmf = pmf[1:last_nonzero]
    end
    pmf
end

discretise(spec::NonParametricSpec; kwargs...) = copy(spec.pmf)
discretise(spec::FixedSpec; kwargs...) =
    spec.value == 0 ? [1.0] : [zeros(Int(spec.value)); 1.0]

# ── Internal helpers ─────────────────────────────────────────────────────

function _moments_to_lognormal(mean, sd)
    σ² = log(1 + (sd / mean)^2)
    μ = log(mean) - σ² / 2
    (μ, sqrt(σ²))
end

function _moments_to_gamma(mean, sd)
    shape = (mean / sd)^2
    rate = mean / sd^2
    (shape, rate)
end
