@testitem "accessors" setup = [TestSetup] begin
    Random.seed!(42)

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
            samples = 50, warmup = 50, chains = 1, progress = false
        ),
        verbose = false
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
    rt_samples = get_samples(result; variable = :R)
    @test all(rt_samples.variable .== :R)

    # get_predictions
    preds = get_predictions(result)
    @test preds isa DataFrame
    @test :date in propertynames(preds)

    # get_parameters
    params = get_parameters(result)
    @test params isa Dict
    @test :R_mean in keys(params)

    # summary
    s = summary(result)
    @test s isa DataFrame
    @test :variable in propertynames(s)
end

@testitem "get_imputed_reports" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    imputed = get_imputed_reports(result)
    @test imputed isa DataFrame
    @test :date in propertynames(imputed)
    @test :mean in propertynames(imputed)
    @test nrow(imputed) > 0
    # Imputed reports should be non-negative integers on average
    @test all(imputed.mean .>= 0)
end

@testitem "map_prob_change" setup = [TestSetup] begin
    @test map_prob_change(0.01) == "Increasing"
    @test map_prob_change(0.2) == "Likely increasing"
    @test map_prob_change(0.5) == "Stable"
    @test map_prob_change(0.8) == "Likely decreasing"
    @test map_prob_change(0.99) == "Decreasing"
end

@testitem "get_predictions formats" setup = [TestSetup] begin
    result = estimate_infections(
        _test_data();
        generation_time = gt_opts(LogNormal(1.6, 0.5)),
        delays = delay_opts(Dirac(0.0)),
        obs = obs_opts(week_effect = false),
        forecast = forecast_opts(horizon = 0),
        inference = _fast_inference(),
        verbose = false
    )
    # :summary format
    preds = get_predictions(result; format = :summary)
    @test preds isa DataFrame

    # :sample format — scoringutils-compatible shape
    samp = get_predictions(result; format = :sample)
    @test names(samp) == [
        "forecast_date", "date", "horizon",
        "sample", "predicted",
    ]
    @test all(samp.forecast_date .== result.observations.date[end])
    @test all(samp.horizon .== Dates.value.(samp.date .- samp.forecast_date))

    # :quantile format — scoringutils-compatible shape
    quant = get_predictions(result; format = :quantile)
    @test names(quant) == [
        "forecast_date", "date", "horizon",
        "quantile_level", "predicted",
    ]
    # Default quantile levels match R: 5/25/50/75/95
    @test sort(unique(quant.quantile_level)) ==
        [0.05, 0.25, 0.5, 0.75, 0.95]

    # Custom quantile levels
    custom_q = get_predictions(
        result; format = :quantile, quantiles = [0.1, 0.5, 0.9]
    )
    @test sort(unique(custom_q.quantile_level)) == [0.1, 0.5, 0.9]

    # invalid format
    @test_throws ArgumentError get_predictions(result; format = :invalid)
end

@testitem "CrIs parameter passthrough" setup = [TestSetup] begin
    Random.seed!(42)

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
        CrIs = [0.5, 0.9],
        inference = inference_opts(
            samples = 50, warmup = 50, chains = 1, progress = false
        ),
        verbose = false
    )

    # Should have 50% and 90% CrIs but NOT 20%
    @test :lower_50 in propertynames(result.rt)
    @test :lower_90 in propertynames(result.rt)
    @test !(:lower_20 in propertynames(result.rt))
end

@testitem "calc_CrI / calc_CrIs / calc_summary_*" setup = [TestSetup] begin
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
    @test ci.lower_90[1] ≈ quantile(df.value, 0.05) atol = 1.0e-10
    @test ci.upper_90[1] ≈ quantile(df.value, 0.95) atol = 1.0e-10

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
    for col in [
            "group", "median", "mean", "sd",
            "lower_50", "upper_50", "lower_90", "upper_90",
        ]
        @test col in names(meas)
    end
    @test nrow(meas) == 2
end
