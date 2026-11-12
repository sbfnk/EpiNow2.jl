# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract: scenarios (each with a ForwardDiff reference), a backend list, and
# broken/skip bookkeeping, consumed by the shared harness (`test/ad/setup.jl`).
# The scenarios are the log density of the `estimate_infections` Turing model,
# the function NUTS differentiates during a fit.
module ADFixtures

using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake,
    AutoMooncakeForward
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Mooncake
using Distributions: LogNormal
using DynamicPPL: DynamicPPL
using EpiNow2: EpiNow2
using LogDensityProblems: LogDensityProblems

export scenarios, backends, broken_scenario_names,
    backend_broken_scenarios, backend_skip_scenarios

# ForwardDiff reference gradient for a scenario function.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...
    )
end

# A short window of the bundled example data keeps each gradient cheap while
# still exercising the renewal equation, delays and observation model.
_ad_data() = first(EpiNow2.example_confirmed(), 30)

function _model_scenario(name; rt = EpiNow2.rt_opts(), with_reference)
    assembled = EpiNow2.assemble_model(
        EpiNow2.EpiData(_ad_data());
        generation_time = EpiNow2.gt_opts(
            EpiNow2.discretise(LogNormal(1.6, 0.5); max = 14)
        ),
        delays = EpiNow2.delay_opts(
            EpiNow2.discretise(LogNormal(0.5, 0.5); max = 14)
        ),
        truncation = EpiNow2.trunc_opts(),
        rt = rt,
        backcalc = EpiNow2.backcalc_opts(),
        gp = EpiNow2.gp_opts(),
        obs = EpiNow2.obs_opts(),
        forecast = EpiNow2.forecast_opts(horizon = 0)
    )
    ldf = DynamicPPL.LogDensityFunction(
        assembled.model, DynamicPPL.getlogjoint_internal, DynamicPPL.LinkAll()
    )
    θ = fill(0.1, LogDensityProblems.dimension(ldf))
    f = x -> LogDensityProblems.logdensity(ldf, x)
    return DIT.Scenario{:gradient, :out}(
        f, θ; name = name,
        res1 = with_reference ? _reference(f, θ, ()) : nothing
    )
end

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios: the joint log density, on the unconstrained scale,
of the `estimate_infections` model with a Gaussian process prior on Rt and
with a random walk prior on Rt. Each is a `DIT.Scenario{:gradient, :out}`
whose `res1` carries a ForwardDiff reference when `with_reference = true`.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    category == :marginal || return DIT.Scenario{:gradient, :out}[]
    return DIT.Scenario{:gradient, :out}[
        _model_scenario("infections model (GP)"; with_reference),
        _model_scenario(
            "infections model (random walk)";
            rt = EpiNow2.rt_opts(rw = 7), with_reference
        ),
    ]
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples. Enzyme is left
out: it cannot be loaded alongside Turing (see `test/ad/scenarios.jl`).
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)", backend = AutoReverseDiff(compile = false)),
        (name = "ReverseDiff (compiled)", backend = AutoReverseDiff(compile = true)),
        (name = "Mooncake reverse", backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward()),
    ]
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

"""
Per-backend broken scenario names (`Dict{String, Set{String}}`).

Mooncake's forward mode returns NaN gradients for the Gaussian process model,
while its reverse mode agrees with ForwardDiff.
"""
backend_broken_scenarios() = Dict{String, Set{String}}(
    "Mooncake forward" => Set(["infections model (GP)"]),
)

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
