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
        spd = EpiNow2.hsgp_coefficients(5, 1.5, 0.1, 0.5, matern, 1.5)
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
        # v1.8 epinowfit parity: args round-trip
        @test result.args isa EpiNow2.EstimateSecondaryArgs
        @test result.args.burn_in == 10
        @test result.args.obs.week_effect == false
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
                samples=100, warmup=100, chains=1, seed=123, progress=false
            ),
            verbose=false
        )

        @test result isa EpiNow2.EstimateTruncationResult
        @test result.dist isa LogNormal
        # v1.8 epinowfit parity: args round-trip
        @test result.args isa EpiNow2.EstimateTruncationArgs
        @test result.args.inference.samples == 100
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

        # Only nuts sampler is supported — invalid symbols throw MethodError
        @test_throws MethodError inference_opts(sampler=:advi)
    end

    # ── Helper: shared test data ─────────────────────────────────────
    _test_data() = let
        n = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n)
        DataFrame(date=collect(dates),
                  confirm=round.(Int, 100 .* exp.(0.05 .* (1:n))))
    end

    _fast_inference() = inference_opts(
        samples=50, warmup=50, chains=1, seed=42, progress=false
    )

    # ── New feature tests ────────────────────────────────────────────

    @testset "random walk mode" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            rt = rt_opts(rw=7),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        @test result isa EpiNow2.EstimateInfectionsResult
        @test nrow(result.rt) == 30
    end

    @testset "forecasting with future=project" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            rt = rt_opts(future=project),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=7),
            inference = _fast_inference(),
            verbose=false
        )
        @test nrow(result.rt) == 37  # 30 obs + 7 forecast
    end

    @testset "population depletion" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            rt = rt_opts(pop=10000.0, pop_period=pop_all),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        @test result isa EpiNow2.EstimateInfectionsResult
        @test all(result.infections.mean .> 0)
        # v1.8: result.rt is depletion-adjusted, result.rt_unadjusted is
        # the transmission Rt. With non-trivial depletion they should
        # not be identical (adjusted ≤ unadjusted). Use the median.
        @test :median in propertynames(result.rt_unadjusted)
        @test all(result.rt.median .<= result.rt_unadjusted.median .+ 1e-9)
        # Without depletion the two are identical.
        result_nopop = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        @test result_nopop.rt.median == result_nopop.rt_unadjusted.median
    end

    @testset "prior-predictive mode" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false, likelihood=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        @test result isa EpiNow2.EstimateInfectionsResult
    end

    @testset "get_imputed_reports" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        imputed = get_imputed_reports(result)
        @test imputed isa DataFrame
        @test :date in propertynames(imputed)
        @test :mean in propertynames(imputed)
        @test nrow(imputed) > 0
        # Imputed reports should be non-negative integers on average
        @test all(imputed.mean .>= 0)
    end

    @testset "simulate_infections" begin
        n = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n)
        R_traj = DataFrame(date=collect(dates), R=fill(1.2, n))

        result = simulate_infections(
            R_traj;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(family=poisson),
            seed = 42
        )
        @test result isa DataFrame
        @test :infections in propertynames(result)
        @test :reports in propertynames(result)
        @test nrow(result) == n
        @test all(result.infections .> 0)
    end

    @testset "estimate_dist (MCMC, censored)" begin
        Random.seed!(42)
        # 200 noisy daily-grid lognormal delays
        delays = round.(Int, rand(LogNormal(1.5, 0.5), 200))
        delays = clamp.(delays, 0, 30)
        data = DataFrame(delay=delays)

        result = estimate_dist(
            data;
            dist=:lognormal,
            inference=inference_opts(
                samples=200, warmup=200, chains=1, progress=false
            ),
            verbose=false,
        )
        @test result isa EpiNow2.EstimateDistResult
        @test result.fitted isa UncertainDistribution
        @test length(result.fitted.param_priors) == 2
        # Posterior mean of meanlog should land near the truth (1.5).
        meanlog_prior = result.fitted.param_priors[1]
        @test 1.0 < mean(meanlog_prior) < 2.0
        # args round-trip
        @test result.args.dist == :lognormal

        # Gamma family — using rate parameterisation matching R
        Random.seed!(42)
        gam_delays = round.(Int, rand(Gamma(2.0, 2.0), 200))  # mean = shape*scale = 4
        gam_data = DataFrame(delay=clamp.(gam_delays, 0, 30))
        gres = estimate_dist(
            gam_data; dist=:gamma,
            inference=inference_opts(samples=200, warmup=200,
                                     chains=1, progress=false),
            verbose=false,
        )
        @test gres.args.dist == :gamma
        @test length(gres.fitted.param_priors) == 2

        # expgrowth primary — fixed growth rate, single arg
        Random.seed!(42)
        expg_data = DataFrame(delay=clamp.(round.(Int,
            rand(LogNormal(1.5, 0.5), 100)), 0, 30))
        eres = estimate_dist(
            expg_data; dist=:lognormal,
            primary=:expgrowth, primary_params=[0.1],
            inference=inference_opts(samples=200, warmup=200,
                                     chains=1, progress=false),
            verbose=false,
        )
        @test eres.args.primary == :expgrowth
        @test eres.args.primary_params == [0.1]

        # Validation: expgrowth without primary_params errors
        @test_throws ArgumentError estimate_dist(
            data; dist=:lognormal, primary=:expgrowth,
            inference=inference_opts(samples=2, warmup=2, chains=1,
                                     progress=false),
        )
        # Unknown primary errors
        @test_throws ArgumentError estimate_dist(
            data; primary=:bogus,
            inference=inference_opts(samples=2, warmup=2, chains=1,
                                     progress=false),
        )
        # Unknown dist errors
        @test_throws ArgumentError estimate_dist(
            data; dist=:bogus,
            inference=inference_opts(samples=2, warmup=2, chains=1,
                                     progress=false),
        )
    end

    @testset "bootstrapped_dist_fit" begin
        Random.seed!(42)
        delays = rand(LogNormal(1.5, 0.5), 200)
        data = DataFrame(delay=delays)

        ud = bootstrapped_dist_fit(data; family=:lognormal, max_delay=30,
                                    n_bootstraps=20)
        @test ud isa UncertainDistribution
        @test length(ud.param_priors) == 2
    end

    @testset "map_prob_change" begin
        @test map_prob_change(0.01) == "Increasing"
        @test map_prob_change(0.2) == "Likely increasing"
        @test map_prob_change(0.5) == "Stable"
        @test map_prob_change(0.8) == "Likely decreasing"
        @test map_prob_change(0.99) == "Decreasing"
    end

    @testset "example delay distributions" begin
        @test example_generation_time() isa LogNormal
        @test example_incubation_period() isa LogNormal
        @test example_reporting_delay() isa LogNormal
    end

    @testset "enum types" begin
        # Enums work as option values
        @test rt_opts(future=latest).future == latest
        @test rt_opts(future=project).future == project
        @test rt_opts(gp_on=gp_R0).gp_on == gp_R0
        @test gp_opts(kernel=se).kernel == se
        @test obs_opts(family=poisson).family == poisson
        @test secondary_opts(prevalence).type == prevalence

        # Invalid symbols rejected
        @test_throws MethodError rt_opts(future=:invalid)
        @test_throws MethodError obs_opts(family=:invalid)
    end

    @testset "secondary prevalence mode" begin
        Random.seed!(42)
        n_days = 40
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 1) + Day(n_days - 1)
        cases = round.(Int, 100 .* exp.(0.03 .* (1:n_days)))
        hosp = [max(1, round(Int, 0.05 * sum(cases[max(1,i-5):i]))) for i in 1:n_days]
        data = DataFrame(date=collect(dates), primary=cases, secondary=hosp)

        result = estimate_secondary(
            data;
            secondary = secondary_opts(prevalence),
            delays = delay_opts(LogNormal(1.5, 0.5)),
            obs = obs_opts(week_effect=false),
            inference = _fast_inference(),
            burn_in=10,
            verbose=false
        )
        @test result isa EpiNow2.EstimateSecondaryResult
        @test nrow(result.predictions) > 0
    end

    @testset "opts_list" begin
        ol = opts_list(["A", "B", "C"], rt_opts(rw=7))
        @test length(ol) == 3
        @test ol["A"].rw == 7
    end

    @testset "get_predictions formats" begin
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = _fast_inference(),
            verbose=false
        )
        # :summary format
        preds = get_predictions(result; format=:summary)
        @test preds isa DataFrame

        # :sample format — scoringutils-compatible shape
        samp = get_predictions(result; format=:sample)
        @test names(samp) == ["forecast_date", "date", "horizon",
                              "sample", "predicted"]
        @test all(samp.forecast_date .== result.observations.date[end])
        @test all(samp.horizon .== Dates.value.(samp.date .- samp.forecast_date))

        # :quantile format — scoringutils-compatible shape
        quant = get_predictions(result; format=:quantile)
        @test names(quant) == ["forecast_date", "date", "horizon",
                               "quantile_level", "predicted"]
        # Default quantile levels match R: 5/25/50/75/95
        @test sort(unique(quant.quantile_level)) ==
              [0.05, 0.25, 0.5, 0.75, 0.95]

        # Custom quantile levels
        custom_q = get_predictions(
            result; format=:quantile, quantiles=[0.1, 0.5, 0.9]
        )
        @test sort(unique(custom_q.quantile_level)) == [0.1, 0.5, 0.9]

        # invalid format
        @test_throws ArgumentError get_predictions(result; format=:invalid)
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

    @testset "result has args field (v1.8 epinowfit parity)" begin
        Random.seed!(42)
        n_days = 30
        dates = collect(Date(2024,1,1):Day(1):Date(2024,1,n_days))
        cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
        df = DataFrame(date=dates, confirm=cases)

        result = estimate_infections(
            df;
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            delays = delay_opts(Dirac(0.0)),
            obs = obs_opts(week_effect=false),
            forecast = forecast_opts(horizon=0),
            inference = inference_opts(
                samples=20, warmup=20, chains=1, progress=false
            ),
            verbose = false,
        )
        @test hasproperty(result, :args)
        @test result.args isa EpiNow2.EstimateInfectionsArgs
        # Round-trip a few key configuration choices
        @test result.args.forecast.horizon == 0
        @test result.args.obs.week_effect == false
        @test result.args.inference.samples == 20
    end

    @testset "calc_CrI / calc_CrIs / calc_summary_*" begin
        Random.seed!(42)
        # Two groups, each with 1000 samples, known parameters
        n = 1000
        df = vcat(
            DataFrame(group = fill(:a, n), value = randn(n)),
            DataFrame(group = fill(:b, n), value = randn(n) .+ 5),
        )

        # Single CrI, no grouping
        ci = calc_CrI(df; CrI = 0.9)
        @test names(ci) == ["lower_90", "upper_90"]
        @test ci.lower_90[1] ≈ quantile(df.value, 0.05) atol = 1e-10
        @test ci.upper_90[1] ≈ quantile(df.value, 0.95) atol = 1e-10

        # Single CrI, by group
        ci_g = calc_CrI(df; by = :group, CrI = 0.5)
        @test sort(names(ci_g)) == sort(["group", "lower_50", "upper_50"])
        @test nrow(ci_g) == 2

        # Multiple CrIs
        cis = calc_CrIs(df; by = :group, CrIs = [0.5, 0.9])
        @test "lower_50" in names(cis) && "upper_90" in names(cis)
        @test nrow(cis) == 2

        # Summary stats
        stats = calc_summary_stats(df; by = :group)
        @test sort(names(stats)) == sort(["group", "median", "mean", "sd"])
        @test nrow(stats) == 2
        # Group :b mean should be near 5
        b_row = stats[stats.group .== :b, :]
        @test b_row.mean[1] ≈ 5.0 atol = 0.2

        # Summary measures (joined)
        meas = calc_summary_measures(df; by = :group, CrIs = [0.5, 0.9])
        for col in ["group", "median", "mean", "sd",
                    "lower_50", "upper_50", "lower_90", "upper_90"]
            @test col in names(meas)
        end
        @test nrow(meas) == 2
    end

    @testset "convert_to_logmean / convert_to_logsd" begin
        # Round-trip a known LogNormal(meanlog=1.0, sdlog=0.5)
        d = LogNormal(1.0, 0.5)
        m = mean(d); s = std(d)
        @test convert_to_logmean(m, s) ≈ 1.0 atol = 1e-10
        @test convert_to_logsd(m, s) ≈ 0.5 atol = 1e-10
    end

    @testset "add_breakpoints" begin
        dates = collect(Date(2024,1,1):Day(1):Date(2024,1,5))
        df = DataFrame(date=dates, confirm=[10, 12, 15, 18, 20])
        # No breakpoints: column added, all zero
        out = add_breakpoints(df)
        @test :breakpoints in propertynames(out)
        @test all(out.breakpoints .== 0)
        # Marked breakpoint
        out2 = add_breakpoints(df; dates=[Date(2024,1,3)])
        @test out2.breakpoints == [0, 0, 1, 0, 0]
        # Multiple breakpoints
        out3 = add_breakpoints(df; dates=[Date(2024,1,2), Date(2024,1,5)])
        @test out3.breakpoints == [0, 1, 0, 0, 1]
        # Date not in data → error
        @test_throws ArgumentError add_breakpoints(df; dates=[Date(2030,1,1)])
        # Missing :date column → error
        @test_throws ArgumentError add_breakpoints(DataFrame(confirm=[1,2]))
    end

    @testset "fill_missing" begin
        # Weekly data → daily grid. Auto-detected interval = 7 days, so
        # the start is padded with 6 days, all marked accumulate=true.
        # Each weekly gap contributes 6 more accumulate=true days.
        weekly = DataFrame(
            date = collect(Date(2024,1,1):Day(7):Date(2024,1,29)),
            confirm = [50, 70, 65, 80, 60],
        )
        out = fill_missing(weekly; missing_dates = :accumulate)
        @test nrow(out) == 29 + 6                       # initial padding
        @test out.date[1] == Date(2023,12,26)            # 6 days before first obs
        @test count(out.accumulate) == 6 + 4 * 6         # padding + gaps
        @test out[out.date .== Date(2024,1,8), :confirm][1] == 70
        @test ismissing(out[out.date .== Date(2024,1,2), :confirm][1])

        # Same but with explicit initial_accumulate=1 (no leading pad)
        out_no_pad = fill_missing(weekly; missing_dates = :accumulate,
                                   initial_accumulate = 1)
        @test nrow(out_no_pad) == 29
        @test out_no_pad.date[1] == Date(2024,1,1)
        @test count(out_no_pad.accumulate) == 4 * 6      # only gaps

        # missing_dates = :zero fills gap with zero (no accumulate flag for
        # the gap rows). Initial pad still applies to give 6 leading
        # accumulate=true rows.
        out_zero = fill_missing(weekly; missing_dates = :zero)
        @test count(out_zero.accumulate) == 6            # only leading pad
        @test out_zero[out_zero.date .== Date(2024,1,2), :confirm][1] == 0

        # initial_accumulate pads the start
        daily = DataFrame(
            date = collect(Date(2024,1,1):Day(1):Date(2024,1,5)),
            confirm = [10, 12, 15, 18, 20],
        )
        padded = fill_missing(daily; initial_accumulate = 3)
        @test nrow(padded) == 7
        @test padded.date[1] == Date(2023,12,30)
        # First two padded days marked accumulate
        @test padded.accumulate[1:2] == [true, true]
        @test padded.accumulate[3:end] == fill(false, 5)

        # missing_obs = :zero on already-present rows
        with_na = DataFrame(
            date = collect(Date(2024,1,1):Day(1):Date(2024,1,5)),
            confirm = Union{Int, Missing}[10, missing, 15, missing, 20],
        )
        out_obs = fill_missing(with_na; missing_obs = :zero)
        @test out_obs.confirm == [10, 0, 15, 0, 20]

        # Pre-existing accumulate column → error
        bad = DataFrame(date = [Date(2024,1,1)], confirm = [10],
                        accumulate = [false])
        @test_throws ArgumentError fill_missing(bad)
    end

    @testset "filter_leading_zeros" begin
        dates = collect(Date(2024,1,1):Day(1):Date(2024,1,5))
        df = DataFrame(date=dates, confirm=[0, 0, 5, 7, 9])
        out = filter_leading_zeros(df)
        @test nrow(out) == 3
        @test out.date[1] == Date(2024,1,3)
        @test out.confirm == [5, 7, 9]
        # All zero → empty
        df0 = DataFrame(date=dates, confirm=zeros(Int, 5))
        @test nrow(filter_leading_zeros(df0)) == 0
        # First positive is row 1 → no rows dropped
        df_clean = DataFrame(date=dates, confirm=[1, 2, 3, 4, 5])
        @test nrow(filter_leading_zeros(df_clean)) == 5
        # Custom obs column
        df_custom = DataFrame(date=dates,
                              counts=[0, 0, 0, 4, 8])
        out_c = filter_leading_zeros(df_custom; obs_column=:counts)
        @test nrow(out_c) == 2
    end
end
