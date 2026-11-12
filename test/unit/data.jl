@testitem "EpiData" setup = [TestSetup] begin
    df = DataFrame(
        date = Date(2024, 1, 1):Day(1):Date(2024, 1, 30),
        confirm = round.(Int, 100 .* exp.(0.05 .* (1:30)))
    )
    ed = EpiNow2.EpiData(df)
    @test length(ed) == 30
end

@testitem "example_confirmed" setup = [TestSetup] begin
    data = example_confirmed()
    @test data isa DataFrame
    @test :date in propertynames(data)
    @test :confirm in propertynames(data)
    @test nrow(data) == 60
    @test all(data.confirm .> 0)
end

@testitem "add_breakpoints" setup = [TestSetup] begin
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 5))
    df = DataFrame(date = dates, confirm = [10, 12, 15, 18, 20])
    # No breakpoints: column added, all zero
    out = add_breakpoints(df)
    @test :breakpoints in propertynames(out)
    @test all(out.breakpoints .== 0)
    # Marked breakpoint
    out2 = add_breakpoints(df; dates = [Date(2024, 1, 3)])
    @test out2.breakpoints == [0, 0, 1, 0, 0]
    # Multiple breakpoints
    out3 = add_breakpoints(df; dates = [Date(2024, 1, 2), Date(2024, 1, 5)])
    @test out3.breakpoints == [0, 1, 0, 0, 1]
    # Date not in data → error
    @test_throws ArgumentError add_breakpoints(df; dates = [Date(2030, 1, 1)])
    # Missing :date column → error
    @test_throws ArgumentError add_breakpoints(DataFrame(confirm = [1, 2]))
end

@testitem "fill_missing" setup = [TestSetup] begin
    # Weekly data → daily grid. Auto-detected interval = 7 days, so
    # the start is padded with 6 days, all marked accumulate=true.
    # Each weekly gap contributes 6 more accumulate=true days.
    weekly = DataFrame(
        date = collect(Date(2024, 1, 1):Day(7):Date(2024, 1, 29)),
        confirm = [50, 70, 65, 80, 60],
    )
    out = fill_missing(weekly; missing_dates = :accumulate)
    @test nrow(out) == 29 + 6                       # initial padding
    @test out.date[1] == Date(2023, 12, 26)            # 6 days before first obs
    @test count(out.accumulate) == 6 + 4 * 6         # padding + gaps
    @test out[out.date .== Date(2024, 1, 8), :confirm][1] == 70
    @test ismissing(out[out.date .== Date(2024, 1, 2), :confirm][1])

    # Same but with explicit initial_accumulate=1 (no leading pad)
    out_no_pad = fill_missing(
        weekly; missing_dates = :accumulate,
        initial_accumulate = 1
    )
    @test nrow(out_no_pad) == 29
    @test out_no_pad.date[1] == Date(2024, 1, 1)
    @test count(out_no_pad.accumulate) == 4 * 6      # only gaps

    # missing_dates = :zero fills gap with zero (no accumulate flag for
    # the gap rows). Initial pad still applies to give 6 leading
    # accumulate=true rows.
    out_zero = fill_missing(weekly; missing_dates = :zero)
    @test count(out_zero.accumulate) == 6            # only leading pad
    @test out_zero[out_zero.date .== Date(2024, 1, 2), :confirm][1] == 0

    # initial_accumulate pads the start
    daily = DataFrame(
        date = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 5)),
        confirm = [10, 12, 15, 18, 20],
    )
    padded = fill_missing(daily; initial_accumulate = 3)
    @test nrow(padded) == 7
    @test padded.date[1] == Date(2023, 12, 30)
    # First two padded days marked accumulate
    @test padded.accumulate[1:2] == [true, true]
    @test padded.accumulate[3:end] == fill(false, 5)

    # missing_obs = :zero on already-present rows
    with_na = DataFrame(
        date = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 5)),
        confirm = Union{Int, Missing}[10, missing, 15, missing, 20],
    )
    out_obs = fill_missing(with_na; missing_obs = :zero)
    @test out_obs.confirm == [10, 0, 15, 0, 20]

    # Pre-existing accumulate column → error
    bad = DataFrame(
        date = [Date(2024, 1, 1)], confirm = [10],
        accumulate = [false]
    )
    @test_throws ArgumentError fill_missing(bad)
end

@testitem "filter_leading_zeros" setup = [TestSetup] begin
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 5))
    df = DataFrame(date = dates, confirm = [0, 0, 5, 7, 9])
    out = filter_leading_zeros(df)
    @test nrow(out) == 3
    @test out.date[1] == Date(2024, 1, 3)
    @test out.confirm == [5, 7, 9]
    # All zero → empty
    df0 = DataFrame(date = dates, confirm = zeros(Int, 5))
    @test nrow(filter_leading_zeros(df0)) == 0
    # First positive is row 1 → no rows dropped
    df_clean = DataFrame(date = dates, confirm = [1, 2, 3, 4, 5])
    @test nrow(filter_leading_zeros(df_clean)) == 5
    # Custom obs column
    df_custom = DataFrame(
        date = dates,
        counts = [0, 0, 0, 4, 8]
    )
    out_c = filter_leading_zeros(df_custom; obs_column = :counts)
    @test nrow(out_c) == 2
end
