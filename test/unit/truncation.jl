@testitem "estimate_truncation" setup = [TestSetup] begin
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
            snap_cases[snap_day - d + 1] = max(
                1,
                round(Int, snap_cases[snap_day - d + 1] * (d / 6))
            )
        end
        push!(snapshots, DataFrame(date = collect(snap_dates), confirm = snap_cases))
    end

    result = estimate_truncation(
        snapshots;
        inference = inference_opts(
            samples = 100, warmup = 100, chains = 1, seed = 123, progress = false
        ),
        verbose = false
    )

    @test result isa EpiNow2.EstimateTruncationResult
    @test result.dist isa LogNormal
    @test result.args.obs.family == negbin
    # v1.8 epinowfit parity: args round-trip
    @test result.args isa EpiNow2.EstimateTruncationArgs
    @test result.args.inference.samples == 100
end

@testitem "estimate_truncation with a Poisson model and gamma truncation" setup = [TestSetup] begin
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 30)
    full_cases = round.(Int, 100 .* exp.(0.03 .* (1:30)))
    snapshots = [
        DataFrame(date = collect(dates[1:n]), confirm = full_cases[1:n])
            for n in (20, 25, 30)
    ]
    result = estimate_truncation(
        snapshots;
        truncation = trunc_opts(
            UncertainDistribution(
                (shape, rate) -> Gamma(shape, 1 / rate),
                [
                    truncated(Normal(2.0, 0.5); lower = 0.0),
                    truncated(Normal(1.0, 0.2); lower = 0.0),
                ],
                10.0
            )
        ),
        obs = obs_opts(family = poisson),
        inference = _fast_inference(),
        verbose = false
    )
    @test result.dist isa Gamma
    @test !(:reporting_overdispersion in names(result.fit.chain))
end

@testitem "estimate_truncation keeps the fixed part of a composite" setup = [TestSetup] begin
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 30)
    full_cases = round.(Int, 100 .* exp.(0.03 .* (1:30)))
    snapshots = [
        DataFrame(date = collect(dates[1:n]), confirm = full_cases[1:n])
            for n in (20, 25, 30)
    ]
    fixed = discretise(LogNormal(0.2, 0.3); max = 3)
    uncertain = UncertainDistribution(
        (μ, σ) -> LogNormal(μ, σ),
        [Normal(0.0, 1.0), truncated(Normal(1.0, 1.0); lower = 0.0)],
        6.0
    )
    result = estimate_truncation(
        snapshots;
        truncation = trunc_opts(uncertain + fixed),
        inference = _fast_inference(),
        verbose = false
    )
    @test result.dist isa EpiNow2.CompositeDelay
    @test any(c -> c isa LogNormal, result.dist.components)
    @test fixed in result.dist.components
    @test !any(c -> c isa UncertainDistribution, result.dist.components)
end

@testitem "truncation observations are trimmed as in R" setup = [TestSetup] begin
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 30)
    snapshots = [
        DataFrame(date = collect(dates[1:n]), confirm = fill(10, n))
            for n in (20, 25, 30)
    ]
    obs_matrix, obs_dist = EpiNow2._truncation_obs(snapshots, 10)
    # R keeps the last `trunc_max` days of the earliest-ending snapshot and
    # the days after it: rows 11 to 30 here.
    @test size(obs_matrix) == (20, 3)
    @test obs_dist == [10, 5, 0]
    @test obs_matrix[1, :] == [10.0, 10.0, 10.0]
    @test obs_matrix[end, :] == [0.0, 0.0, 10.0]

    # A truncation longer than the data keeps every row.
    @test size(first(EpiNow2._truncation_obs(snapshots, 40)), 1) == 30
end
