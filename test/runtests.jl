using Test
using EpiNow2
using DataFrames
using Dates
using Distributions
using Random

@testset "EpiNow2.jl" begin

    @testset "Distributions" begin
        # Discretise a LogNormal
        d = discretise(LogNormal(1.6, 0.5); max=14)
        @test d isa NonParametricDist
        @test length(d.pmf) > 0
        @test sum(d.pmf) ≈ 1.0
        @test all(d.pmf .>= 0)

        # Discretise a Dirac at 0
        d0 = discretise(Dirac(0.0))
        @test d0.pmf == [1.0]

        # Discretise a Dirac at 3
        d3 = discretise(Dirac(3.0))
        @test length(d3.pmf) == 4
        @test d3.pmf[end] == 1.0

        # NonParametricDist round-trip
        npd = NonParametricDist([0.2, 0.5, 0.3])
        @test discretise(npd).pmf == [0.2, 0.5, 0.3]

        # PMF convolution via +
        a = discretise(Dirac(1.0))
        b = discretise(Dirac(2.0))
        c = a + b
        @test c isa NonParametricDist
        @test length(c.pmf) == 4  # support 0:3
        @test c.pmf[4] ≈ 1.0     # all mass at delay 3

        # Raw vector convolution
        cv = convolve_pmfs([0.5, 0.3, 0.2], [0.4, 0.6])
        @test length(cv) == 4
        @test sum(cv) ≈ 1.0

        # CompositeDelay (Distribution + Distribution)
        cd = LogNormal(1.0, 0.5) + LogNormal(0.5, 0.3)
        @test cd isa CompositeDelay
        pmf_cd = discretise(cd; max=10)
        @test pmf_cd isa NonParametricDist
        @test sum(pmf_cd.pmf) ≈ 1.0

        # Discretise then compose (per-component max)
        composed = discretise(LogNormal(1.0, 0.5); max=8) +
                   discretise(LogNormal(0.5, 0.3); max=5)
        @test composed isa NonParametricDist
        @test sum(composed.pmf) ≈ 1.0
    end

    @testset "Options" begin
        # Default options should construct without error
        @test gt_opts() isa EpiNow2.GTOpts
        @test delay_opts() isa EpiNow2.DelayOpts
        @test rt_opts() isa EpiNow2.RtOpts
        @test gp_opts() isa EpiNow2.GPOpts
        @test obs_opts() isa EpiNow2.ObsOpts
        @test inference_opts() isa EpiNow2.InferenceOpts

        # With distribution argument
        gt = gt_opts(LogNormal(1.6, 0.5))
        @test gt.dist isa LogNormal
    end

    @testset "Model helpers" begin
        # convolve
        signal = [1.0, 2.0, 3.0, 4.0, 5.0]
        kernel = [0.5, 0.3, 0.2]
        result = EpiNow2.convolve(signal, kernel)
        @test length(result) == 5
        @test result[1] ≈ 1.0 * 0.5

        # HSGP basis
        basis = EpiNow2.hsgp_basis(5, 1.5, 30)
        @test size(basis) == (30, 5)

        # HSGP coefficients
        spd = EpiNow2.hsgp_coefficients(5, 1.5, 0.1, 0.5, :matern, 1.5)
        @test length(spd) == 5
        @test all(isfinite.(spd))

        # NegativeBinomial2
        nb = EpiNow2.NegativeBinomial2(10.0, 5.0)
        @test nb isa NegativeBinomial
        @test mean(nb) ≈ 10.0
    end

    @testset "EpiData" begin
        df = DataFrame(
            date = Date(2024, 1, 1):Day(1):Date(2024, 1, 30),
            confirm = round.(Int, 100 .* exp.(0.05 .* (1:30)))
        )
        ed = EpiNow2.EpiData(df)
        @test length(ed) == 30
    end

    @testset "estimate_infections end-to-end" begin
        Random.seed!(42)

        # Generate synthetic exponential growth data
        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        result = estimate_infections(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=100,
                warmup=100,
                chains=1,
                progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateInfectionsResult

        # Check DataFrames have expected columns
        for df in [result.infections, result.reports, result.rt]
            @test :date in propertynames(df)
            @test :mean in propertynames(df)
            @test :median in propertynames(df)
            @test :lower_50 in propertynames(df)
            @test :upper_50 in propertynames(df)
            @test :lower_90 in propertynames(df)
            @test :upper_90 in propertynames(df)
            @test nrow(df) > 0
        end

        # Rt posterior median should be plausible for exponential growth
        rt_median = result.rt[end, :median]
        @test 0.5 < rt_median < 3.0

        # All infections should be positive
        @test all(result.infections.mean .> 0)
    end

    @testset "estimate_infections with uncertain delays" begin
        Random.seed!(123)

        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        # Uncertain generation time: priors on LogNormal params
        uncertain_gt = UncertainDistribution(
            (μ, σ) -> LogNormal(μ, σ),
            [Normal(1.6, 0.1), truncated(Normal(0.5, 0.05); lower=0.01)],
            14.0
        )

        result = estimate_infections(
            data;
            generation_time = gt_opts(uncertain_gt),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=50,
                warmup=50,
                chains=1,
                progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateInfectionsResult
        @test nrow(result.rt) > 0
        @test all(result.infections.mean .> 0)
    end

    @testset "example_confirmed" begin
        data = example_confirmed()
        @test data isa DataFrame
        @test :date in propertynames(data)
        @test :confirm in propertynames(data)
        @test nrow(data) == 60
        @test all(data.confirm .> 0)
    end

    @testset "estimate_infections with fixed truncation" begin
        Random.seed!(42)

        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        result = estimate_infections(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            truncation = trunc_opts(LogNormal(0.5, 0.3)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=100,
                warmup=100,
                chains=1,
                progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateInfectionsResult
        @test nrow(result.infections) > 0
        @test all(result.infections.mean .> 0)
    end

    @testset "estimate_infections with uncertain truncation" begin
        Random.seed!(42)

        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        uncertain_trunc = UncertainDistribution(
            (μ, σ) -> LogNormal(μ, σ),
            [Normal(0.5, 0.2), truncated(Normal(0.3, 0.1); lower=0.01)],
            10.0
        )

        result = estimate_infections(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            truncation = trunc_opts(uncertain_trunc),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=50,
                warmup=50,
                chains=1,
                progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateInfectionsResult
        @test nrow(result.infections) > 0
        @test all(result.infections.mean .> 0)
    end

    @testset "estimate_secondary" begin
        Random.seed!(42)

        n_days = 40
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 1) + Day(n_days - 1)
        cases = round.(Int, 100 .* exp.(0.03 .* (1:n_days)))
        deaths = [max(1, round(Int, 0.02 * cases[max(1, i - 5)])) for i in 1:n_days]
        data = DataFrame(date=collect(dates), primary=cases, secondary=deaths)

        result = estimate_secondary(
            data;
            delays = delay_opts(LogNormal(1.5, 0.5)),
            obs = obs_opts(week_effect=false),
            inference = inference_opts(
                samples=100, warmup=100, chains=1, progress=false
            ),
            burn_in=10,
            verbose=false
        )

        @test result isa EpiNow2.EstimateSecondaryResult
        @test nrow(result.predictions) > 0
    end

    @testset "estimate_truncation" begin
        Random.seed!(42)

        # Create synthetic snapshots: each later one has more complete recent data
        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        full_cases = round.(Int, 100 .* exp.(0.03 .* (1:n_days)))

        snapshots = DataFrame[]
        for snap_day in [20, 25, 30]
            snap_dates = dates[1:snap_day]
            snap_cases = copy(full_cases[1:snap_day])
            # Truncate recent days
            for d in 1:min(5, snap_day)
                snap_cases[snap_day - d + 1] = max(1,
                    round(Int, snap_cases[snap_day - d + 1] * (d / 6)))
            end
            push!(snapshots, DataFrame(date=collect(snap_dates), confirm=snap_cases))
        end

        result = estimate_truncation(
            snapshots;
            inference = inference_opts(
                samples=100, warmup=100, chains=1, seed=42, progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateTruncationResult
        @test result.dist isa LogNormal
    end

    @testset "epinow end-to-end" begin
        Random.seed!(42)

        data = example_confirmed()

        result = epinow(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=7),
            inference = inference_opts(
                samples=100, warmup=100, chains=1, progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EpinowResult
        @test result.estimates isa EpiNow2.EstimateInfectionsResult
        @test nrow(result.estimates.rt) > 60  # obs + forecast
    end

    @testset "accessors" begin
        Random.seed!(42)

        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        result = estimate_infections(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=50, warmup=50, chains=1, progress=false
            ),
            verbose=false
        )

        # get_samples
        samples = get_samples(result)
        @test samples isa DataFrame
        @test :date in propertynames(samples)
        @test :variable in propertynames(samples)
        @test :sample in propertynames(samples)
        @test :value in propertynames(samples)
        @test nrow(samples) > 0

        # get_samples with variable filter
        rt_samples = get_samples(result; variable=:R)
        @test all(rt_samples.variable .== :R)

        # get_predictions
        preds = get_predictions(result)
        @test preds isa DataFrame
        @test :date in propertynames(preds)

        # get_parameters
        params = get_parameters(result)
        @test params isa Dict
        @test :R0 in keys(params)

        # summary
        s = summary(result)
        @test s isa DataFrame
        @test :variable in propertynames(s)
    end

    @testset "simulate_secondary" begin
        data = example_confirmed()
        primary = rename(data, :confirm => :primary)

        result = simulate_secondary(
            primary;
            delays = delay_opts(LogNormal(1.5, 0.5)),
            frac = 0.1
        )

        @test result isa DataFrame
        @test :secondary in propertynames(result)
        @test nrow(result) == nrow(data)
        @test all(result.secondary .>= 0)
    end

    @testset "R_to_growth / growth_to_R" begin
        gt_pmf = [0.3, 0.4, 0.2, 0.1]
        # Normal case
        r = R_to_growth(1.5, gt_pmf)
        @test isfinite(r)
        @test r > 0

        # R=1 should give r≈0
        r1 = R_to_growth(1.0, gt_pmf)
        @test abs(r1) < 0.01

        # Round-trip: R → r → R
        R_back = growth_to_R(r, gt_pmf)
        @test R_back ≈ 1.5 atol=0.05
    end

    @testset "error paths" begin
        # Missing columns
        @test_throws ArgumentError EpiNow2.EpiData(DataFrame(x=[1,2]))
        @test_throws ArgumentError EpiNow2.SecondaryData(DataFrame(x=[1,2]))

        # Invalid PMF
        @test_throws ArgumentError NonParametricDist([0.5, 0.3])  # doesn't sum to 1

        # ADVI not supported
        @test_throws ErrorException inference_opts(sampler=:advi)  |>
            opts -> EpiNow2._make_sampler(opts)
    end

    @testset "CrIs parameter passthrough" begin
        Random.seed!(42)

        n_days = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        data = DataFrame(date=collect(dates), confirm=cases)

        result = estimate_infections(
            data;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            CrIs = [0.5, 0.9],
            inference = inference_opts(
                samples=50, warmup=50, chains=1, progress=false
            ),
            verbose=false
        )

        # Should have 50% and 90% CrIs but NOT 20%
        @test :lower_50 in propertynames(result.rt)
        @test :lower_90 in propertynames(result.rt)
        @test !(:lower_20 in propertynames(result.rt))
    end
end
