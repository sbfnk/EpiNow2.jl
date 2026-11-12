@testitem "estimate_infections end-to-end" setup = [TestSetup] begin
    Random.seed!(42)

    # Generate synthetic exponential growth data
    n_days = 30
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
    cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
    data = DataFrame(date = collect(dates), confirm = cases)

    result = estimate_infections(
        data;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = inference_opts(
            samples = 100,
            warmup = 100,
            chains = 1,
            progress = false,
            max_treedepth = 8
        ),
        verbose = false
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

@testitem "estimate_infections with uncertain delays" setup = [TestSetup] begin
    Random.seed!(123)

    n_days = 30
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
    cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
    data = DataFrame(date = collect(dates), confirm = cases)

    # Uncertain generation time: priors on LogNormal params
    uncertain_gt = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(1.6, 0.1), truncated(Normal(0.5, 0.05); lower = 0.01)],
        14.0
    )

    result = estimate_infections(
        data;
        generation_time = gt_opts(uncertain_gt),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = inference_opts(
            samples = 50,
            warmup = 50,
            chains = 1,
            progress = false,
            max_treedepth = 8
        ),
        verbose = false
    )

    @test result isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.rt) > 0
    @test all(result.infections.mean .> 0)
end

@testitem "estimate_infections with fixed truncation" setup = [TestSetup] begin
    Random.seed!(42)

    n_days = 30
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
    cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
    data = DataFrame(date = collect(dates), confirm = cases)

    result = estimate_infections(
        data;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        truncation = trunc_opts(LogNormal(0.5, 0.3)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = inference_opts(
            samples = 100,
            warmup = 100,
            chains = 1,
            progress = false,
            max_treedepth = 8
        ),
        verbose = false
    )

    @test result isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.infections) > 0
    @test all(result.infections.mean .> 0)
end

@testitem "estimate_infections with uncertain truncation" setup = [TestSetup] begin
    Random.seed!(42)

    n_days = 30
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n_days)
    cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
    data = DataFrame(date = collect(dates), confirm = cases)

    uncertain_trunc = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(0.5, 0.2), truncated(Normal(0.3, 0.1); lower = 0.01)],
        10.0
    )

    result = estimate_infections(
        data;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        truncation = trunc_opts(uncertain_trunc),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = inference_opts(
            samples = 50,
            warmup = 50,
            chains = 1,
            progress = false,
            max_treedepth = 8
        ),
        verbose = false
    )

    @test result isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.infections) > 0
    @test all(result.infections.mean .> 0)
end

@testitem "random walk mode" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        rt = rt_opts(rw = 7),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    @test result isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.rt) == 30
end

@testitem "forecasting with future=project" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        rt = rt_opts(future = project),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 7),
        inference = _fast_inference(),
        verbose = false
    )
    @test nrow(result.rt) == 37  # 30 obs + 7 forecast
end

@testitem "population depletion" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        rt = rt_opts(pop = 10000.0, pop_period = pop_all),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    @test result isa EpiNow2.EstimateInfectionsResult
    @test all(result.infections.mean .> 0)
    # v1.8: result.rt is depletion-adjusted, result.rt_unadjusted is
    # the transmission Rt. With non-trivial depletion they should
    # not be identical (adjusted ≤ unadjusted). Use the median.
    @test :median in propertynames(result.rt_unadjusted)
    @test all(result.rt.median .<= result.rt_unadjusted.median .+ 1.0e-9)
    # Without depletion the two are identical.
    result_nopop = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    @test result_nopop.rt.median == result_nopop.rt_unadjusted.median
end

@testitem "prior-predictive mode" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false, likelihood = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    @test result isa EpiNow2.EstimateInfectionsResult
end

@testitem "simulate_infections" setup = [TestSetup] begin
    n = 30
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n)
    R_traj = DataFrame(date = collect(dates), R = fill(1.2, n))

    result = simulate_infections(
        R_traj;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(family = poisson),
        seed = 42
    )
    @test result isa DataFrame
    @test :infections in propertynames(result)
    @test :reports in propertynames(result)
    @test nrow(result) == n
    @test all(result.infections .> 0)
end

@testitem "epinow end-to-end" setup = [TestSetup] begin
    Random.seed!(42)

    data = example_confirmed()

    result = epinow(
        data;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 7),
        inference = inference_opts(
            samples = 100, warmup = 100, chains = 1, progress = false,
            max_treedepth = 8
        ),
        verbose = false
    )

    @test result isa EpiNow2.EpinowResult
    @test result.estimates isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.estimates.rt) > 60  # obs + forecast
    @test get_predictions(result) == get_predictions(result.estimates)
end

@testitem "result has args field (v1.8 epinowfit parity)" setup = [TestSetup] begin
    Random.seed!(42)
    n_days = 30
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, n_days))
    cases = round.(Int, 100 .* exp.(0.05 .* (1:n_days)))
    df = DataFrame(date = dates, confirm = cases)

    result = estimate_infections(
        df;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = inference_opts(
            samples = 20, warmup = 20, chains = 1, progress = false,
            max_treedepth = 8
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
