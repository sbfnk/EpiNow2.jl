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
    # v1.8 epinowfit parity: args round-trip
    @test result.args isa EpiNow2.EstimateTruncationArgs
    @test result.args.inference.samples == 100
end
