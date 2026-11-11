@testitem "Distributions" setup = [TestSetup] begin
    # Discretise a LogNormal
    d = discretise(LogNormal(1.6, 0.5); max = 14)
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
    pmf_cd = discretise(cd; max = 10)
    @test pmf_cd isa NonParametricDist
    @test sum(pmf_cd.pmf) ≈ 1.0

    # Discretise then compose (per-component max)
    composed = discretise(LogNormal(1.0, 0.5); max = 8) +
        discretise(LogNormal(0.5, 0.3); max = 5)
    @test composed isa NonParametricDist
    @test sum(composed.pmf) ≈ 1.0
end

@testitem "example delay distributions" setup = [TestSetup] begin
    @test example_generation_time() isa LogNormal
    @test example_incubation_period() isa LogNormal
    @test example_reporting_delay() isa LogNormal
end

@testitem "enum types" setup = [TestSetup] begin
    # Enums work as option values
    @test rt_opts(future = latest).future == latest
    @test rt_opts(future = project).future == project
    @test rt_opts(gp_on = gp_R0).gp_on == gp_R0
    @test gp_opts(kernel = se).kernel == se
    @test obs_opts(family = poisson).family == poisson
    @test secondary_opts(prevalence).type == prevalence

    # Invalid symbols rejected
    @test_throws MethodError rt_opts(future = :invalid)
    @test_throws MethodError obs_opts(family = :invalid)
end

@testitem "convert_to_logmean / convert_to_logsd" setup = [TestSetup] begin
    # Round-trip a known LogNormal(meanlog=1.0, sdlog=0.5)
    d = LogNormal(1.0, 0.5)
    m = mean(d); s = std(d)
    @test convert_to_logmean(m, s) ≈ 1.0 atol = 1.0e-10
    @test convert_to_logsd(m, s) ≈ 0.5 atol = 1.0e-10
end

@testitem "estimate_dist (MCMC, censored)" setup = [TestSetup] begin
    Random.seed!(42)
    # 200 noisy daily-grid lognormal delays
    delays = round.(Int, rand(LogNormal(1.5, 0.5), 200))
    delays = clamp.(delays, 0, 30)
    data = DataFrame(delay = delays)

    result = estimate_dist(
        data;
        dist = :lognormal,
        inference = inference_opts(
            samples = 200, warmup = 200, chains = 1, progress = false
        ),
        verbose = false,
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
    gam_data = DataFrame(delay = clamp.(gam_delays, 0, 30))
    gres = estimate_dist(
        gam_data; dist = :gamma,
        inference = inference_opts(
            samples = 200, warmup = 200,
            chains = 1, progress = false
        ),
        verbose = false,
    )
    @test gres.args.dist == :gamma
    @test length(gres.fitted.param_priors) == 2

    # expgrowth primary — fixed growth rate, single arg
    Random.seed!(42)
    expg_data = DataFrame(
        delay = clamp.(
            round.(
                Int,
                rand(LogNormal(1.5, 0.5), 100)
            ), 0, 30
        )
    )
    eres = estimate_dist(
        expg_data; dist = :lognormal,
        primary = :expgrowth, primary_params = [0.1],
        inference = inference_opts(
            samples = 200, warmup = 200,
            chains = 1, progress = false
        ),
        verbose = false,
    )
    @test eres.args.primary == :expgrowth
    @test eres.args.primary_params == [0.1]

    # Validation: expgrowth without primary_params errors
    @test_throws ArgumentError estimate_dist(
        data; dist = :lognormal, primary = :expgrowth,
        inference = inference_opts(
            samples = 2, warmup = 2, chains = 1,
            progress = false
        ),
    )
    # Unknown primary errors
    @test_throws ArgumentError estimate_dist(
        data; primary = :bogus,
        inference = inference_opts(
            samples = 2, warmup = 2, chains = 1,
            progress = false
        ),
    )
    # Unknown dist errors
    @test_throws ArgumentError estimate_dist(
        data; dist = :bogus,
        inference = inference_opts(
            samples = 2, warmup = 2, chains = 1,
            progress = false
        ),
    )
end

@testitem "bootstrapped_dist_fit" setup = [TestSetup] begin
    Random.seed!(42)
    delays = rand(LogNormal(1.5, 0.5), 200)
    data = DataFrame(delay = delays)

    ud = bootstrapped_dist_fit(
        data; family = :lognormal, max_delay = 30,
        n_bootstraps = 20
    )
    @test ud isa UncertainDistribution
    @test length(ud.param_priors) == 2
end

@testitem "options display composite delays" setup = [TestSetup] begin
    delays = delay_opts(LogNormal(1.6, 0.5) + discretise(LogNormal(0.5, 0.3); max = 7))
    shown = sprint(show, MIME"text/plain"(), delays)
    @test occursin("Composite distribution:", shown)
    @test occursin("lognormal", shown)
end
