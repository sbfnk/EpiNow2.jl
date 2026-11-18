@testitem "estimate_secondary" setup = [TestSetup] begin
    Random.seed!(42)

    n_days = 40
    dates = Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(n_days - 1))
    cases = round.(Int, 100 .* exp.(0.03 .* (1:n_days)))
    deaths = [max(1, round(Int, 0.02 * cases[max(1, i - 5)])) for i in 1:n_days]
    data = DataFrame(date = collect(dates), primary = cases, secondary = deaths)

    result = estimate_secondary(
        data;
        delays = delay_opts(LogNormal(1.5, 0.5)),
        obs = obs_opts(week_effect = false),
        inference = inference_opts(
            samples = 100, warmup = 100, chains = 1, progress = false
        ),
        burn_in = 10,
        verbose = false
    )

    @test result isa EpiNow2.EstimateSecondaryResult
    @test nrow(result.predictions) > 0
    # v1.8 epinowfit parity: args round-trip
    @test result.args isa EpiNow2.EstimateSecondaryArgs
    @test result.args.burn_in == 10
    @test result.args.obs.week_effect == false
end

@testitem "simulate_secondary" setup = [TestSetup] begin
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

@testitem "calculate_secondary recursion (incidence vs prevalence)" setup = [TestSetup] begin
    scaled = fill(5.0, 15)
    pmf = [0.2, 0.5, 0.3]
    conv = EpiNow2.convolve(scaled, pmf)

    # incidence: expected == convolved primary (up to the 1e-6 floor)
    inc = EpiNow2.calculate_secondary(scaled, conv, false, true, true, false, false)
    @test maximum(abs.(inc .- conv)) < 1.0e-5

    # prevalence: occupancy balance Δ = inflow − outflow after the first step
    prev = EpiNow2.calculate_secondary(scaled, conv, true, true, false, true, true)
    @test maximum(abs.(diff(prev) .- (scaled[2:end] .- conv[2:end]))) < 1.0e-5
    @test all(prev .>= 0)
    # must NOT equal the cumulative-outflow-only series (the former bug)
    @test !isapprox(prev, cumsum(conv))
end

@testitem "simulate_secondary prevalence is a stock not cumulative outflow (regression)" setup = [TestSetup] begin
    Random.seed!(11)
    n = 60
    prim = DataFrame(date = Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(n - 1)), primary = fill(100, n))
    sp = simulate_secondary(
        prim; delays = delay_opts(LogNormal(1.5, 0.5)),
        secondary = secondary_opts(prevalence), frac = 0.3
    )
    # For constant admissions, occupancy (a stock) reaches a bounded steady state.
    # The former bug returned cumsum(convolve(...)), which grows ~ frac*primary*t
    # (into the hundreds-to-thousands by day 60).
    tail = mean(sp.secondary[(n - 9):n])
    early = mean(sp.secondary[div(n, 2):(div(n, 2) + 9)])
    @test tail < 600                      # nowhere near the runaway cumulative-outflow level
    @test tail > 0.3 * 100                # it is a stock (above one day's inflow)
    @test abs(tail - early) < 0.5 * early # roughly flat, not still climbing like a cumsum
end

@testitem "secondary prevalence mode" setup = [TestSetup] begin
    Random.seed!(42)
    n_days = 40
    dates = Date(2024, 1, 1):Day(1):(Date(2024, 1, 1) + Day(n_days - 1))
    cases = round.(Int, 100 .* exp.(0.03 .* (1:n_days)))
    hosp = [max(1, round(Int, 0.05 * sum(cases[max(1, i - 5):i]))) for i in 1:n_days]
    data = DataFrame(date = collect(dates), primary = cases, secondary = hosp)

    result = estimate_secondary(
        data;
        secondary = secondary_opts(prevalence),
        delays = delay_opts(LogNormal(1.5, 0.5)),
        obs = obs_opts(week_effect = false),
        inference = _fast_inference(),
        burn_in = 10,
        verbose = false
    )
    @test result isa EpiNow2.EstimateSecondaryResult
    @test nrow(result.predictions) > 0
end
