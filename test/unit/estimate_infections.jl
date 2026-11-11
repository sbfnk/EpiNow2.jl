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
        gp = nothing,
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    @test result isa EpiNow2.EstimateInfectionsResult
    @test nrow(result.rt) == 30
    # A breakpoint every 7 days over 30 days gives 4 random walk steps
    @test count(n -> startswith(string(n), "bp_effects"), names(result.fit.chain)) == 4
end

@testitem "log Rt is centred on R_mean over the observed window" setup = [TestSetup] begin
    for (rt, gp) in (
            (rt_opts(), gp_opts()),
            (rt_opts(rw = 7), nothing),
            (rt_opts(rw = 7, future = project), gp_opts()),
        )
        result = estimate_infections(
            _test_data();
            generation_time = gt_opts(LogNormal(1.6, 0.5)),
            rt = rt, gp = gp,
            obs = obs_opts(week_effect = false),
            forecast = forecast_opts(horizon = 7),
            inference = _fast_inference(),
            verbose = false
        )
        R_mean = vec(Array(result.fit.chain[:R_mean]))
        for (i, gq) in enumerate(result.fit.generated_quantities)
            @test mean(gq.log_R[1:30]) ≈ log(R_mean[i]) atol = 1.0e-8
        end
    end
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

@testitem "epinow target_date extends the forecast horizon" setup = [TestSetup] begin
    data = _test_data()
    result = epinow(
        data;
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        rt = rt_opts(rw = 7), gp = nothing,
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 3),
        target_date = maximum(data.date) + Day(2),
        inference = _fast_inference(),
        verbose = false
    )
    @test result.estimates.args.forecast.horizon == 5
    @test maximum(result.estimates.rt.date) == maximum(data.date) + Day(5)
end

@testitem "composite delays mixing fixed and uncertain parts" setup = [TestSetup] begin
    incubation = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(1.6, 0.1), truncated(Normal(0.5, 0.05); lower = 0.0)],
        14.0
    )
    reporting = discretise(LogNormal(0.5, 0.3); max = 7)
    delays = delay_opts(incubation + reporting)
    assembled = EpiNow2.assemble_model(
        EpiNow2.EpiData(_test_data());
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delays, truncation = trunc_opts(), rt = rt_opts(),
        backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts()
    )
    # The placeholder delay PMF convolves the incubation period at its prior
    # means with the fixed reporting delay, so it starts at delay 0 and is
    # as long as both supports combined.
    expected = EpiNow2.convolve_pmfs(
        EpiNow2.discretise_ad(LogNormal(1.6, 0.5), 14), reporting.pmf
    )
    @test assembled.metadata.delay_pmf ≈ expected
    @test length(assembled.metadata.delay_pmf) == 15 + 8 - 1
end

@testitem "future = estimate holds Rt from before the end of the data" setup = [TestSetup] begin
    data = EpiNow2.EpiData(_test_data())
    n_obs = length(data)
    assemble(rt) = EpiNow2.assemble_model(
        data;
        generation_time = gt_opts(discretise(LogNormal(1.6, 0.5); max = 10)),
        delays = delay_opts(Dirac(0.0)), truncation = trunc_opts(), rt = rt,
        backcalc = EpiNow2.backcalc_opts(), gp = nothing,
        obs = obs_opts(), forecast = forecast_opts(horizon = 7)
    )
    # A weekly random walk changes level every 7 days, so a held window
    # shows up as a constant run of breakpoint indices.
    held_from(bps) = findfirst(t -> all(==(bps[t]), bps[t:end]), eachindex(bps))

    fixed = assemble(rt_opts(rw = 7, future = EpiNow2.estimate, fixed_from = 4))
    @test held_from(fixed.model.args.bps) <= n_obs - 4

    # By default R holds Rt from the seeding time before the end.
    default = assemble(rt_opts(rw = 7, future = EpiNow2.estimate))
    seeding = default.metadata.seeding_time
    @test held_from(default.model.args.bps) <= n_obs - seeding
    @test default.model.args.bps[n_obs - seeding - 7] <
        default.model.args.bps[n_obs - seeding]

    latest_fit = assemble(rt_opts(rw = 7, future = EpiNow2.latest))
    @test held_from(latest_fit.model.args.bps) > n_obs - 7
end

@testitem "composite truncation keeps its uncertain part" setup = [TestSetup] begin
    uncertain = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(0.5, 0.2), truncated(Normal(0.3, 0.1); lower = 0.01)],
        5.0
    )
    fixed = discretise(LogNormal(0.2, 0.3); max = 3)
    assembled = EpiNow2.assemble_model(
        EpiNow2.EpiData(_test_data());
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        truncation = trunc_opts(uncertain + fixed), rt = rt_opts(),
        backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts(horizon = 0)
    )
    @test assembled.model.args.trunc_uncertain === uncertain
    @test assembled.model.args.trunc_fixed_pmf ≈ fixed.pmf
    @test assembled.model() isa NamedTuple
end

@testitem "truncation leaves forecast reports alone" setup = [TestSetup] begin
    assembled = EpiNow2.assemble_model(
        EpiNow2.EpiData(_test_data());
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        truncation = trunc_opts(LogNormal(0.5, 0.3)), rt = rt_opts(),
        backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts(horizon = 7)
    )
    @test assembled.model() isa NamedTuple
end

@testitem "seeding time follows R's get_seeding_time" setup = [TestSetup] begin
    seeding(delay) = EpiNow2.assemble_model(
        EpiNow2.EpiData(_test_data());
        generation_time = gt_opts(discretise(LogNormal(1.6, 0.5); max = 10)),
        delays = delay_opts(delay), truncation = trunc_opts(),
        rt = rt_opts(), backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts()
    ).metadata.seeding_time
    # max(round(sum of delay means), maximum generation time, 1)
    @test seeding(Dirac(0.0)) == 10
    @test seeding(LogNormal(1.6, 0.5)) == 10
    @test seeding(LogNormal(2.5, 0.5)) == round(Int, exp(2.5 + 0.5^2 / 2))
    @test seeding(LogNormal(1.6, 0.5) + LogNormal(1.6, 0.5)) ==
        round(Int, 2 * exp(1.6 + 0.5^2 / 2))
    # A truncated parameter prior counts at the untruncated location, as R
    # applies the bound separately: here sdlog = 1, giving a mean of 4.48
    # where the truncated prior's mean (sdlog 1.29) would give 6.23.
    uncertain = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(1.0, 0.5), truncated(Normal(1.0, 1.0); lower = 0.0)],
        30.0
    )
    @test seeding(uncertain + LogNormal(2.0, 0.5)) ==
        round(Int, exp(1.5) + exp(2.0 + 0.5^2 / 2))
end

@testitem "forecast depletion starts where Rt is held" setup = [TestSetup] begin
    data = EpiNow2.EpiData(_test_data())
    assemble(future) = EpiNow2.assemble_model(
        data;
        generation_time = gt_opts(discretise(LogNormal(1.6, 0.5); max = 10)),
        delays = delay_opts(Dirac(0.0)), truncation = trunc_opts(),
        rt = rt_opts(future = future, pop = 1.0e6),
        backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts(horizon = 7)
    )
    held = assemble(EpiNow2.estimate)
    @test held.model.args.pop_start ==
        length(data) - held.metadata.seeding_time
    @test assemble(EpiNow2.latest).model.args.pop_start == length(data)
end

@testitem "returned reports are not truncated" setup = [TestSetup] begin
    assemble(truncation) = EpiNow2.assemble_model(
        EpiNow2.EpiData(_test_data());
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)), truncation = truncation,
        rt = rt_opts(), backcalc = EpiNow2.backcalc_opts(), gp = gp_opts(),
        obs = obs_opts(), forecast = forecast_opts(horizon = 7)
    ).model
    # A fixed truncation adds no parameters, so the same seed draws the same
    # parameters for both models; only the likelihood should differ.
    Random.seed!(1)
    plain = assemble(trunc_opts())()
    Random.seed!(1)
    truncated_fit = assemble(trunc_opts(LogNormal(0.5, 0.3)))()
    @test truncated_fit.reports ≈ plain.reports
end
