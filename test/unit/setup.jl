@testsnippet TestSetup begin
    using EpiNow2
    using DataFrames
    using Dates
    using Distributions
    using Random
    using Statistics

    _test_data() = let
        n = 30
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, n)
        DataFrame(
            date = collect(dates),
            confirm = round.(Int, 100 .* exp.(0.05 .* (1:n)))
        )
    end

    _fast_inference() = inference_opts(
        samples = 50, warmup = 50, chains = 1, seed = 42, progress = false
    )
end
