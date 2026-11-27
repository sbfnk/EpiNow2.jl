# Run with: julia --project=test benchmark/infections_gradient.jl
using EpiNow2, Distributions, ForwardDiff, Profile

gt = UncertainDistribution(
    (shape, rate) -> Gamma(max(1.0e-6, shape), max(1.0e-6, 1 / rate)),
    [
        truncated(Normal(1.4, 0.48); lower = 0.01),
        truncated(Normal(0.38, 0.25); lower = 0.01),
    ],
    14.0
)
inc = UncertainDistribution(
    (μ, σ) -> LogNormal(μ, σ),
    [Normal(1.62, 0.064), truncated(Normal(0.418, 0.069); lower = 0.01)],
    14.0
)
assembled = EpiNow2.assemble_model(
    EpiNow2.EpiData(first(EpiNow2.example_confirmed(), 60));
    generation_time = gt_opts(gt),
    delays = delay_opts(discretise(inc + LogNormal(0.5, 0.5))),
    truncation = trunc_opts(), rt = rt_opts(), backcalc = backcalc_opts(),
    gp = gp_opts(), obs = obs_opts(), forecast = forecast_opts(horizon = 14)
)
dynamicppl = EpiNow2.Turing.DynamicPPL
ldp = dynamicppl.LogDensityProblems
ldf = dynamicppl.LogDensityFunction(
    assembled.model, dynamicppl.getlogjoint_internal, dynamicppl.LinkAll()
)
θ = fill(0.1, ldp.dimension(ldf))
f(x) = ldp.logdensity(ldf, x)
ForwardDiff.gradient(f, θ)
@time ForwardDiff.gradient(f, θ)
Profile.clear()
@profile for _ in 1:1000
    ForwardDiff.gradient(f, θ)
end
Profile.print(; format = :flat, sortedby = :count, mincount = 20)
